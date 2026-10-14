"""Indic Document Question-Answering Engine with zero-copy prefix rewinding.

Enables answering multiple sequential Indic-language questions against a pre-ingested
PrefixState (TurboQuantCache) using zero-copy cache rewinding and offset RoPE position_ids.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

from turboquant.cache import TurboQuantCache
from turboquant.ingest import PrefixState
from turboquant.patch import _is_attention_module, turboquant_context
from turboquant.quantizer import DynamicKVQuantizer

DEFAULT_INDIC_QA_TEMPLATE = "\n\nप्रश्न: {question}\nउत्तर:"


@dataclass
class QAResult:
    """Result container for a single question-answering generation run.

    Attributes:
        question: Input question string.
        answer: Generated answer text.
        tokens_generated: Count of new tokens generated during autoregressive decoding.
        latency_sec: End-to-end wall-clock time in seconds.
        tokens_per_sec: Generation throughput in tokens per second.
        cache_bytes: Memory footprint of the compressed TurboQuantCache in bytes.
    """

    question: str
    answer: str
    tokens_generated: int
    latency_sec: float
    tokens_per_sec: float
    cache_bytes: int

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Invalid key '{key}' for QAResult")


class IndicDocumentQAEngine:
    """Answers sequential Indic questions against a pre-ingested document PrefixState.

    Exploits zero-copy cache rewinding (crop) to discard previous query/answer turns
    and reuse the exact compressed document prefix without memory duplication.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        prompt_template: str = DEFAULT_INDIC_QA_TEMPLATE,
        device: Optional[Union[str, torch.device]] = None,
        quantizer: Optional[DynamicKVQuantizer] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize IndicDocumentQAEngine.

        Args:
            model: HuggingFace LLM or attention module.
            tokenizer: Tokenizer corresponding to the model.
            prompt_template: Template string containing '{question}' placeholder.
            device: Torch device (defaults to model parameter device or CPU).
            quantizer: Optional DynamicKVQuantizer instance.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.quantizer = quantizer

        if device is not None:
            self.device = torch.device(device) if isinstance(device, str) else device
        else:
            try:
                self.device = next(model.parameters()).device
            except (StopIteration, AttributeError):
                self.device = torch.device("cpu")

    def _tokenize_suffix(self, question: str) -> torch.Tensor:
        """Format and tokenize question suffix with add_special_tokens=False."""
        if "{question}" in self.prompt_template:
            formatted_prompt = self.prompt_template.format(question=question)
        else:
            formatted_prompt = f"{self.prompt_template}{question}"

        if callable(self.tokenizer):
            res = self.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
            if isinstance(res, dict) and "input_ids" in res:
                suffix_ids = res["input_ids"]
            elif isinstance(res, torch.Tensor):
                suffix_ids = res
            elif hasattr(res, "input_ids"):
                suffix_ids = res.input_ids
            else:
                suffix_ids = torch.tensor(res, device=self.device)
        elif hasattr(self.tokenizer, "encode"):
            res = self.tokenizer.encode(formatted_prompt, return_tensors="pt", add_special_tokens=False)
            if isinstance(res, torch.Tensor):
                suffix_ids = res
            else:
                suffix_ids = torch.tensor(res, device=self.device)
        else:
            raise TypeError(f"Tokenizer {type(self.tokenizer)} must be callable or define encode()")

        if suffix_ids.ndim == 1:
            suffix_ids = suffix_ids.unsqueeze(0)

        return suffix_ids.to(device=self.device, dtype=torch.long)

    def _forward_step(
        self,
        input_ids: torch.Tensor,
        cache: TurboQuantCache,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch a single forward pass, returning logits for the last token."""
        if _is_attention_module(self.model):
            dim = getattr(self.model, "hidden_size", 128)
            if hasattr(self.model, "q_proj"):
                dim = self.model.q_proj.in_features
            embeds = torch.zeros(input_ids.shape[0], input_ids.shape[1], dim, device=self.device)
            outputs, _ = self.model(embeds, past_key_values=cache, position_ids=position_ids)
            if hasattr(self.model, "lm_head"):
                logits = self.model.lm_head(outputs)
            else:
                logits = outputs
            return logits[:, -1, :]
        else:
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=cache,
                position_ids=position_ids,
                use_cache=True,
            )
            if hasattr(outputs, "logits"):
                return outputs.logits[:, -1, :]
            elif isinstance(outputs, tuple):
                return outputs[0][:, -1, :]
            elif isinstance(outputs, torch.Tensor):
                return outputs[:, -1, :]
            raise ValueError(f"Unsupported model output type: {type(outputs)}")

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Select next token via greedy argmax or temperature sampling."""
        if temperature <= 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        scaled_logits = logits / temperature
        if top_k is not None and top_k > 0:
            k = min(top_k, scaled_logits.shape[-1])
            val, _ = torch.topk(scaled_logits, k)
            scaled_logits = torch.where(
                scaled_logits < val[..., -1, None],
                torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype),
                scaled_logits,
            )
        probs = torch.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def answer(
        self,
        prefix_state: PrefixState,
        question: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> QAResult:
        """Answer a question against prefix_state with zero-copy prefix rewinding.

        Args:
            prefix_state: Pre-ingested document PrefixState containing TurboQuantCache.
            question: Question string.
            max_new_tokens: Maximum number of autoregressive tokens to generate.
            temperature: Sampling temperature (0.0 denotes deterministic greedy decoding).
            top_k: Optional top-k filter for sampling.
            eos_token_id: Optional end-of-sequence token ID to stop generation early.

        Returns:
            QAResult container with answer, metrics, and memory statistics.
        """
        # 1. Zero-copy prefix rewind to remove previous turns
        prefix_state.crop()

        # 2. Tokenize question suffix
        suffix_ids = self._tokenize_suffix(question)
        l_doc = prefix_state.prefix_length
        l_q = suffix_ids.shape[1]

        # 3. Construct offset position_ids starting at L_doc
        position_ids = torch.arange(
            l_doc, l_doc + l_q, device=self.device, dtype=torch.long
        ).unsqueeze(0)

        # Resolve EOS token
        if eos_token_id is None:
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            config = getattr(self.model, "config", None)
            if config is not None:
                eos_token_id = getattr(config, "eos_token_id", None)

        active_quantizer = self.quantizer or prefix_state.cache.quantizer
        t0 = time.perf_counter()
        generated_tokens: List[int] = []

        with torch.no_grad(), turboquant_context(self.model, quantizer=active_quantizer):
            # 4. Question prefill pass
            logits = self._forward_step(suffix_ids, prefix_state.cache, position_ids)
            next_token = self._sample_token(logits, temperature=temperature, top_k=top_k)
            first_tok = next_token.item()
            generated_tokens.append(first_tok)

            # 5. Autoregressive decode loop
            curr_pos = l_doc + l_q
            while len(generated_tokens) < max_new_tokens:
                if eos_token_id is not None and generated_tokens[-1] == eos_token_id:
                    break

                curr_pos_ids = torch.tensor([[curr_pos]], device=self.device, dtype=torch.long)
                logits = self._forward_step(next_token, prefix_state.cache, curr_pos_ids)
                curr_pos += 1

                next_token = self._sample_token(logits, temperature=temperature, top_k=top_k)
                generated_tokens.append(next_token.item())

        latency = time.perf_counter() - t0
        tps = len(generated_tokens) / latency if latency > 0 else 0.0

        # Decode answer text
        if hasattr(self.tokenizer, "decode"):
            answer_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        else:
            answer_text = " ".join(str(t) for t in generated_tokens)

        cache_bytes = prefix_state.cache.get_cache_size_bytes()

        return QAResult(
            question=question,
            answer=answer_text,
            tokens_generated=len(generated_tokens),
            latency_sec=latency,
            tokens_per_sec=tps,
            cache_bytes=cache_bytes,
        )

    def __call__(
        self,
        prefix_state: PrefixState,
        question: str,
        **kwargs: Any,
    ) -> QAResult:
        """Callable interface forwarding to answer()."""
        return self.answer(prefix_state, question, **kwargs)


__all__ = [
    "DEFAULT_INDIC_QA_TEMPLATE",
    "IndicDocumentQAEngine",
    "QAResult",
]
