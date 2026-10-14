"""Validation tests for scripts/indic_assistant_app.py telemetry and helper logic."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import pytest
import torch

from scripts.indic_assistant_app import (
    SAMPLE_DOCS,
    compute_doc_hash,
    compute_telemetry_metrics,
    create_synthetic_model,
    ingest_document_if_needed,
    process_question,
)
from turboquant.qa import IndicDocumentQAEngine, QAResult
from turboquant.quantizer import DynamicKVQuantizer


def test_telemetry_math_baseline_vs_compressed() -> None:
    """Verify uncompressed bfloat16 baseline vs compressed TurboQuant KV calculations."""
    prefix_length = 4096
    num_layers = 28
    num_kv_heads = 8
    head_dim = 128

    # Expected baseline formula: 2 (K, V) * n_layers * H_kv * d_h * L_doc * 2 bytes (bf16)
    expected_baseline_bytes = 2 * num_layers * num_kv_heads * head_dim * prefix_length * 2
    # 2 * 28 * 8 * 128 * 4096 * 2 = 469,762,048 bytes = 448.0 MiB
    assert expected_baseline_bytes == 469762048

    # Hypothetical 2-bit TurboQuant footprint: ~56 MiB
    tq_cache_bytes = 58720256  # 56 MiB
    prefill_time_sec = 2.0

    metrics = compute_telemetry_metrics(
        prefix_length=prefix_length,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        cache_bytes=tq_cache_bytes,
        prefill_time_sec=prefill_time_sec,
    )

    assert metrics["baseline_bytes"] == expected_baseline_bytes
    assert abs(metrics["baseline_mib"] - 448.0) < 1e-3
    assert metrics["turboquant_bytes"] == tq_cache_bytes
    assert metrics["memory_saved_bytes"] == expected_baseline_bytes - tq_cache_bytes
    assert metrics["memory_saved_pct"] > 80.0
    assert metrics["compression_ratio"] > 7.0
    assert metrics["prefill_tokens_per_sec"] == prefix_length / prefill_time_sec


def test_doc_hash_and_ingestion_state_transitions() -> None:
    """Verify document hashing, caching, and state transitions during ingestion."""
    text1 = "यह भारत की राजधानी दिल्ली के बारे में एक दस्तावेज़ है।"
    text2 = "यह तमिलनाडु की राजधानी चेन्नई के बारे में एक दस्तावेज़ है।"

    hash1 = compute_doc_hash(text1)
    hash2 = compute_doc_hash(text2)
    assert hash1 != hash2
    assert compute_doc_hash(text1) == hash1

    model, tokenizer = create_synthetic_model(vocab_size=128, hidden_size=64, head_dim=32)
    quantizer = DynamicKVQuantizer(head_dim=32)

    # 1. First ingestion (cold state)
    state1, h1, did_ingest1 = ingest_document_if_needed(
        document_text=text1,
        model=model,
        tokenizer=tokenizer,
        quantizer=quantizer,
        chunk_size=32,
        current_hash=None,
        current_state=None,
    )
    assert did_ingest1 is True
    assert h1 == hash1
    assert state1.prefix_length > 0
    assert state1.cache.get_seq_length(0) == state1.prefix_length

    # 2. Second ingestion with identical text (should be a no-op hit)
    state2, h2, did_ingest2 = ingest_document_if_needed(
        document_text=text1,
        model=model,
        tokenizer=tokenizer,
        quantizer=quantizer,
        chunk_size=32,
        current_hash=h1,
        current_state=state1,
    )
    assert did_ingest2 is False
    assert h2 == hash1
    assert state2 is state1  # exact instance preserved

    # 3. Third ingestion with modified text (should re-ingest)
    state3, h3, did_ingest3 = ingest_document_if_needed(
        document_text=text2,
        model=model,
        tokenizer=tokenizer,
        quantizer=quantizer,
        chunk_size=32,
        current_hash=h1,
        current_state=state1,
    )
    assert did_ingest3 is True
    assert h3 == hash2
    assert state3 is not state1


def test_multiturn_qa_helper_execution() -> None:
    """Verify multi-turn QA helper execution with zero-copy prefix rewinding."""
    model, tokenizer = create_synthetic_model(vocab_size=128, hidden_size=64, head_dim=32)
    quantizer = DynamicKVQuantizer(head_dim=32)
    doc_text = "चंद्रयान-3 मिशन ने चंद्रमा के दक्षिणी ध्रुव पर सफल लैंडिंग की।"

    state, _, _ = ingest_document_if_needed(
        document_text=doc_text,
        model=model,
        tokenizer=tokenizer,
        quantizer=quantizer,
        chunk_size=32,
    )
    prefix_length = state.prefix_length

    qa_engine = IndicDocumentQAEngine(model=model, tokenizer=tokenizer, quantizer=quantizer)

    # Question 1
    res1 = process_question(
        qa_engine=qa_engine,
        prefix_state=state,
        question="चंद्रयान-3 कहाँ उतरा?",
        max_new_tokens=8,
        temperature=0.0,
    )
    assert isinstance(res1, QAResult)
    assert res1.tokens_generated > 0
    assert res1.latency_sec >= 0.0
    assert res1.tokens_per_sec >= 0.0

    # Question 2 (rewinds state back to prefix_length before generating)
    res2 = process_question(
        qa_engine=qa_engine,
        prefix_state=state,
        question="मिशन कब लॉन्च हुआ था?",
        max_new_tokens=8,
        temperature=0.0,
    )
    assert isinstance(res2, QAResult)
    assert res2.tokens_generated > 0

    # After state.crop(), cache length matches prefix_length
    state.crop()
    assert state.cache.get_seq_length(0) == prefix_length


def test_sample_indic_documents() -> None:
    """Verify sample Indic documents for Hindi, Tamil, and Telugu exist and are non-empty."""
    assert "Hindi" in [k.split(" - ")[0] for k in SAMPLE_DOCS.keys()]
    assert "Tamil" in [k.split(" - ")[0] for k in SAMPLE_DOCS.keys()]
    assert "Telugu" in [k.split(" - ")[0] for k in SAMPLE_DOCS.keys()]

    for title, content in SAMPLE_DOCS.items():
        assert len(content.strip()) > 50, f"Sample doc '{title}' is too short."
