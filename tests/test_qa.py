"""Validation test suite for IndicDocumentQAEngine and zero-copy prefix rewinding."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM

from turboquant.ingest import DocumentPrefixEngine, PrefixState
from turboquant.qa import IndicDocumentQAEngine, QAResult
from turboquant.quantizer import DynamicKVQuantizer


class MockIndicTokenizer:
    """Deterministic character-mapping tokenizer for Indic QA unit testing."""

    def __init__(self, vocab_size: int = 256, eos_token_id: int = 2) -> None:
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = 0

    def __call__(
        self,
        text: str,
        return_tensors: Optional[str] = "pt",
        add_special_tokens: bool = False,
    ) -> Dict[str, Any]:
        ids = [3 + (ord(c) % (self.vocab_size - 4)) for c in text]
        if not ids:
            ids = [3]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def encode(
        self,
        text: str,
        return_tensors: Optional[str] = "pt",
        add_special_tokens: bool = False,
    ) -> Union[torch.Tensor, List[int]]:
        res = self(text, return_tensors=return_tensors, add_special_tokens=add_special_tokens)
        return res["input_ids"]

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        chars = []
        for t in token_ids:
            if skip_special_tokens and t == self.eos_token_id:
                continue
            chars.append(chr(33 + (t % 60)))
        return "".join(chars)


@pytest.fixture
def qa_setup() -> Dict[str, Any]:
    """Fixture providing a lightweight 2-layer Llama model, tokenizer, and quantizer."""
    torch.manual_seed(42)
    vocab_size = 128
    hidden_size = 64
    head_dim = 32

    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=head_dim,
        pad_token_id=0,
        eos_token_id=2,
        attention_bias=False,
    )
    model = LlamaForCausalLM(cfg)
    model.eval()

    tokenizer = MockIndicTokenizer(vocab_size=vocab_size, eos_token_id=2)
    quantizer = DynamicKVQuantizer(head_dim=head_dim, use_qjl=False)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "quantizer": quantizer,
        "head_dim": head_dim,
    }


def test_qa_cache_rewind_across_queries(qa_setup: Dict[str, Any]) -> None:
    """Ask 3 sequential questions against the same PrefixState and verify the cache rewinds to prefix_length before each query."""
    model = qa_setup["model"]
    tokenizer = qa_setup["tokenizer"]
    quantizer = qa_setup["quantizer"]

    # 1. Ingest a 128-token document prefix
    doc_tokens = torch.randint(3, 90, (1, 128))
    prefix_engine = DocumentPrefixEngine(
        model=model,
        tokenizer=tokenizer,
        quantizer=quantizer,
        chunk_size=64,
    )
    prefix_state: PrefixState = prefix_engine.ingest(doc_tokens)
    prefix_length = prefix_state.prefix_length
    assert prefix_length == 128
    assert prefix_state.cache.get_seq_length(0) == 128

    # 2. Instantiate QA Engine
    qa_engine = IndicDocumentQAEngine(
        model=model,
        tokenizer=tokenizer,
        quantizer=quantizer,
    )

    questions = [
        "भारत की राजधानी क्या है?",
        "गंगा नदी कहाँ से निकलती है?",
        "ताजमहल किस शहर में स्थित है?",
    ]

    for i, q in enumerate(questions):
        # Before calling answer, cache might have had previous answer tokens, but at start of answer(),
        # prefix_state.crop() rewinds cache to prefix_length
        res = qa_engine.answer(prefix_state, q, max_new_tokens=12, temperature=0.0)

        assert isinstance(res, QAResult)
        assert res.question == q
        assert res.tokens_generated > 0
        assert res.latency_sec >= 0.0
        assert res.tokens_per_sec >= 0.0
        assert res.cache_bytes > 0

        # After question prefill + generation, the cache has grown
        assert prefix_state.cache.get_seq_length(0) > prefix_length

        # Explicitly verify prefix_state.crop() restores it back to exact prefix_length
        prefix_state.crop()
        assert prefix_state.cache.get_seq_length(0) == prefix_length
        assert prefix_state.cache.get_seq_length(1) == prefix_length


def test_qa_deterministic_consistency(qa_setup: Dict[str, Any]) -> None:
    """Verify that asking Question A, then Question B, then Question A again produces the exact same output tokens for Question A both times."""
    model = qa_setup["model"]
    tokenizer = qa_setup["tokenizer"]
    quantizer = qa_setup["quantizer"]

    doc_tokens = torch.randint(3, 90, (1, 128))
    prefix_engine = DocumentPrefixEngine(
        model=model,
        tokenizer=tokenizer,
        quantizer=quantizer,
        chunk_size=64,
    )
    prefix_state: PrefixState = prefix_engine.ingest(doc_tokens)

    qa_engine = IndicDocumentQAEngine(
        model=model,
        tokenizer=tokenizer,
        quantizer=quantizer,
    )

    question_a = "भारत का राष्ट्रीय पक्षी कौन सा है?"
    question_b = "सूर्य किस दिशा में उगता है?"

    # 1. Ask Question A first time
    res_a1 = qa_engine.answer(prefix_state, question_a, max_new_tokens=16, temperature=0.0)

    # 2. Ask Question B
    res_b = qa_engine.answer(prefix_state, question_b, max_new_tokens=16, temperature=0.0)

    # 3. Ask Question A second time
    res_a2 = qa_engine.answer(prefix_state, question_a, max_new_tokens=16, temperature=0.0)

    # Verify deterministic output consistency for Question A
    assert res_a1.answer == res_a2.answer, (
        f"Mismatch in answer for Question A!\nRun 1: '{res_a1.answer}'\nRun 2: '{res_a2.answer}'"
    )
    assert res_a1.tokens_generated == res_a2.tokens_generated
    assert res_a1.tokens_generated > 0


def test_qa_sampling_and_dict_access(qa_setup: Dict[str, Any]) -> None:
    """Validate QA sampling with temperature > 0 and dict-like indexing on QAResult."""
    model = qa_setup["model"]
    tokenizer = qa_setup["tokenizer"]
    quantizer = qa_setup["quantizer"]

    doc_tokens = torch.randint(3, 90, (1, 64))
    prefix_engine = DocumentPrefixEngine(model=model, tokenizer=tokenizer, quantizer=quantizer)
    prefix_state = prefix_engine.ingest(doc_tokens)

    qa_engine = IndicDocumentQAEngine(model=model, tokenizer=tokenizer, quantizer=quantizer)
    res = qa_engine(prefix_state, "परीक्षण प्रश्न?", max_new_tokens=8, temperature=0.7, top_k=10)

    assert res["question"] == "परीक्षण प्रश्न?"
    assert "answer" in res.__dict__
    assert res["tokens_generated"] > 0
    assert res["cache_bytes"] > 0
