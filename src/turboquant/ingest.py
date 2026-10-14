"""Low-memory document prefix ingestion pipeline for TurboQuant.

Enables efficient chunked prefill of long (4,000+ token) Indic documents into
a 2-bit compressed TurboQuantCache on CPU, and zero-copy prefix reuse across queries.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn

from turboquant.cache import TurboQuantCache
from turboquant.patch import _is_attention_module, apply_turboquant
from turboquant.quantizer import DynamicKVQuantizer


@dataclass
class PrefixState:
    """Container holding the state of an ingested document prefix.

    Attributes:
        cache: Populated TurboQuantCache containing compressed KV states.
        prefix_length: Total number of tokens ingested into the prefix cache.
        cache_bytes: Memory footprint of the compressed KV cache in bytes.
        prefill_time_sec: Wall-clock time in seconds taken to ingest the prefix.
    """

    cache: TurboQuantCache
    prefix_length: int
    cache_bytes: int
    prefill_time_sec: float

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Invalid key '{key}' for PrefixState")

    def crop(self, max_length: Optional[int] = None) -> None:
        """Crop underlying TurboQuantCache to max_length tokens (defaults to self.prefix_length).

        Args:
            max_length: Target sequence length to retain. If None, resets back to prefix_length.
        """
        target_len = self.prefix_length if max_length is None else max_length
        self.cache.crop(target_len)
        self.prefix_length = target_len
        self.cache_bytes = self.cache.get_cache_size_bytes()


def _forward_step(
    model: nn.Module,
    chunk: torch.Tensor,
    cache: TurboQuantCache,
    is_embeddings: bool,
) -> Any:
    """Execute a forward pass on a chunk across attention blocks, decoder layers, or full models."""
    if _is_attention_module(model):
        if is_embeddings:
            return model(chunk, past_key_values=cache)
        else:
            dim = getattr(model, "hidden_size", None)
            if dim is None and hasattr(model, "q_proj"):
                dim = model.q_proj.in_features
            dim = dim or 128
            embeds = torch.zeros(
                chunk.shape[0], chunk.shape[1], dim,
                dtype=torch.float32, device=chunk.device
            )
            return model(embeds, past_key_values=cache)
    elif hasattr(model, "self_attn") and not hasattr(model, "model") and not hasattr(model, "get_input_embeddings"):
        if is_embeddings:
            return model(chunk, past_key_value=cache, use_cache=True)
        else:
            dim = getattr(model, "hidden_size", 128)
            embeds = torch.zeros(
                chunk.shape[0], chunk.shape[1], dim,
                dtype=torch.float32, device=chunk.device
            )
            return model(embeds, past_key_value=cache, use_cache=True)
    else:
        if is_embeddings:
            return model(inputs_embeds=chunk, past_key_values=cache, use_cache=True)
        else:
            return model(input_ids=chunk, past_key_values=cache, use_cache=True)


class DocumentPrefixEngine:
    """Low-memory document prefix ingestion engine using chunked prefill into TurboQuantCache.

    Supports chunked prefill (or full-sequence prefill) under torch.no_grad() into
    TurboQuantCache, enabling long Indic documents (4,000+ tokens) to be compressed into
    2-bit RAM on CPU.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        document_text: Optional[Union[str, torch.Tensor]] = None,
        quantizer: Optional[DynamicKVQuantizer] = None,
        chunk_size: int = 512,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize DocumentPrefixEngine.

        Args:
            model: HuggingFace LLM or attention module.
            tokenizer: Optional tokenizer instance.
            document_text: Optional initial document text or synthetic tensor.
            quantizer: Optional DynamicKVQuantizer instance.
            chunk_size: Chunk size for chunked prefill (default: 512).
            device: Target torch device.
        """
        # Support positional argument variation where quantizer is passed as 3rd arg
        if isinstance(document_text, DynamicKVQuantizer):
            quantizer = document_text
            document_text = kwargs.get("document_text", None)

        self.model = model
        self.tokenizer = tokenizer
        self.document_text = document_text
        self.quantizer = quantizer
        self.chunk_size = chunk_size

        if device is not None:
            self.device = torch.device(device) if isinstance(device, str) else device
        else:
            try:
                self.device = next(model.parameters()).device
            except (StopIteration, AttributeError):
                self.device = torch.device("cpu")

    def ingest(
        self,
        document_text: Optional[Union[str, torch.Tensor]] = None,
        tokenizer: Optional[Any] = None,
        quantizer: Optional[DynamicKVQuantizer] = None,
        chunk_size: Optional[int] = None,
        cache: Optional[TurboQuantCache] = None,
    ) -> PrefixState:
        """Ingest document into TurboQuantCache using chunked prefill.

        Args:
            document_text: Text string to ingest, or pre-tokenized / synthetic tensor.
            tokenizer: Optional tokenizer overriding instance tokenizer.
            quantizer: Optional quantizer overriding instance quantizer.
            chunk_size: Optional chunk size overriding instance chunk_size.
            cache: Optional existing TurboQuantCache to ingest into.

        Returns:
            PrefixState holding cache, prefix_length, cache_bytes, and prefill_time_sec.
        """
        doc = document_text if document_text is not None else self.document_text
        if doc is None:
            raise ValueError("document_text must be provided to ingest()")

        tok = tokenizer or self.tokenizer
        active_chunk_size = chunk_size if chunk_size is not None else self.chunk_size
        active_quantizer = quantizer or self.quantizer

        # Resolve input tensor and whether it is embeddings
        if isinstance(doc, torch.Tensor):
            input_tensor = doc
        elif isinstance(doc, str):
            if tok is None:
                raise ValueError("tokenizer must be provided when document_text is a string")
            if callable(tok):
                res = tok(doc, return_tensors="pt")
                if isinstance(res, dict) and "input_ids" in res:
                    input_tensor = res["input_ids"]
                elif isinstance(res, torch.Tensor):
                    input_tensor = res
                elif hasattr(res, "input_ids"):
                    input_tensor = res.input_ids
                else:
                    input_tensor = torch.tensor(res).unsqueeze(0)
            elif hasattr(tok, "encode"):
                res = tok.encode(doc, return_tensors="pt")
                if isinstance(res, torch.Tensor):
                    input_tensor = res
                else:
                    input_tensor = torch.tensor(res).unsqueeze(0)
            else:
                raise TypeError(f"Tokenizer {type(tok)} has no call or encode method")
        else:
            raise TypeError(f"Unsupported document_text type: {type(doc)}")

        if input_tensor.ndim == 1:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(device=self.device)

        if input_tensor.ndim == 2:
            total_tokens = input_tensor.shape[1]
            is_embeddings = False
        elif input_tensor.ndim == 3:
            total_tokens = input_tensor.shape[1]
            is_embeddings = True
        else:
            raise ValueError(f"Expected 2D input_ids or 3D hidden states, got shape {input_tensor.shape}")

        if cache is None:
            cache = TurboQuantCache(quantizer=active_quantizer)

        # Ensure model attention layers are monkey-patched
        apply_turboquant(self.model, quantizer=active_quantizer)
        self.model.eval()

        t0 = time.perf_counter()

        with torch.no_grad():
            if active_chunk_size <= 0 or total_tokens <= active_chunk_size:
                _forward_step(self.model, input_tensor, cache, is_embeddings)
            else:
                for start_idx in range(0, total_tokens, active_chunk_size):
                    end_idx = min(start_idx + active_chunk_size, total_tokens)
                    chunk = input_tensor[:, start_idx:end_idx]
                    _forward_step(self.model, chunk, cache, is_embeddings)

        elapsed = time.perf_counter() - t0
        cache_bytes = cache.get_cache_size_bytes()
        prefix_len = cache.get_seq_length(0) if len(cache) > 0 else total_tokens

        return PrefixState(
            cache=cache,
            prefix_length=prefix_len,
            cache_bytes=cache_bytes,
            prefill_time_sec=elapsed,
        )

    def __call__(
        self,
        document_text: Optional[Union[str, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> PrefixState:
        """Callable interface forwarding to ingest()."""
        return self.ingest(document_text=document_text, **kwargs)


def ingest_document_prefix(
    model: nn.Module,
    tokenizer: Optional[Any] = None,
    document_text: Union[str, torch.Tensor] = "",
    quantizer: Optional[DynamicKVQuantizer] = None,
    chunk_size: int = 512,
    device: Optional[Union[str, torch.device]] = None,
    cache: Optional[TurboQuantCache] = None,
) -> PrefixState:
    """Convenience functional wrapper for document prefix ingestion.

    Args:
        model: HuggingFace model or attention block.
        tokenizer: Optional tokenizer instance.
        document_text: Document text string or pre-tokenized / synthetic tensor.
        quantizer: Optional DynamicKVQuantizer instance.
        chunk_size: Ingestion chunk size (default: 512).
        device: Device to run ingestion on.
        cache: Optional existing TurboQuantCache.

    Returns:
        PrefixState holding cache, prefix_length, cache_bytes, and prefill_time_sec.
    """
    engine = DocumentPrefixEngine(
        model=model,
        tokenizer=tokenizer,
        quantizer=quantizer,
        chunk_size=chunk_size,
        device=device,
    )
    return engine.ingest(document_text=document_text, cache=cache)


__all__ = [
    "DocumentPrefixEngine",
    "PrefixState",
    "ingest_document_prefix",
]
