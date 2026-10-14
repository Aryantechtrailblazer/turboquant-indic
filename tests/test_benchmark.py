"""Unit tests for benchmark_evaluation harness."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.benchmark_evaluation import (
    benchmark_decode_latency,
    benchmark_kv_scaling,
    eval_indic_fidelity,
)


def test_benchmark_kv_scaling_smoke() -> None:
    """Verify that KV scaling returns valid metrics and compression ratios."""
    res = benchmark_kv_scaling(
        seq_lengths=[64, 128],
        n_layers=2,
        h_q=4,
        h_kv=2,
        d_h=128,
        use_qjl=False,
    )
    assert len(res) == 2
    for r in res:
        assert r["bf16_theo_bytes"] > r["tq_theo_bytes"]
        assert r["theo_compression_ratio"] > 5.0
        assert r["tq_actual_tensor_bytes"] > 0


def test_benchmark_decode_latency_smoke() -> None:
    """Verify latency benchmark measures steps and returns throughput."""
    res = benchmark_decode_latency(
        seq_lengths=[32],
        h_q=4,
        h_kv=2,
        d_h=128,
        num_warmup=1,
        num_steps=5,
    )
    assert len(res) == 1
    assert res[0]["base_latency_ms"] > 0
    assert res[0]["tq_latency_ms"] > 0
    assert res[0]["base_tokens_sec"] > 0
    assert res[0]["tq_tokens_sec"] > 0


def test_eval_indic_fidelity_smoke() -> None:
    """Verify generation fidelity evaluation on synthetic Indic prompt."""
    tok = AutoTokenizer.from_pretrained("sarvamai/sarvam-1")
    cfg = AutoConfig.from_pretrained("sarvamai/sarvam-1")
    cfg.num_hidden_layers = 1
    cfg.hidden_size = 256
    cfg.intermediate_size = 512
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.vocab_size = len(tok)
    model = LlamaForCausalLM(cfg).to(torch.bfloat16)

    prompts = [
        {"language": "Hindi", "prompt": "भारत की राजधानी"}
    ]

    res = eval_indic_fidelity(
        model=model,
        tokenizer=tok,
        prompts=prompts,
        max_new_tokens=4,
    )
    assert len(res) == 1
    assert "avg_cosine_similarity" in res[0]
    assert "token_match_rate" in res[0]
    assert len(res[0]["base_decoded"]) >= 0
    assert len(res[0]["tq_decoded"]) >= 0
