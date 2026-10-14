"""Validation tests for zero-copy prefix rewinding (crop) and document ingestion."""

from __future__ import annotations

import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaConfig

from turboquant.cache import TurboQuantCache
from turboquant.ingest import DocumentPrefixEngine, PrefixState, ingest_document_prefix
from turboquant.patch import apply_turboquant
from turboquant.quantizer import DynamicKVQuantizer


@pytest.fixture
def llama_attn_config() -> LlamaConfig:
    """Fixture providing Llama / Sarvam attention configuration with head_dim=128."""
    return LlamaConfig(
        hidden_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA ratio: 4 / 2 = 2
        head_dim=128,
        attention_bias=False,
    )


def test_cache_crop_zero_copy() -> None:
    """Populate a cache with 128 tokens, append 32 query/answer tokens, call cache.crop(128), and verify shapes and get_seq_length() == 128 across all layers."""
    torch.manual_seed(42)
    batch_size = 1
    num_kv_heads = 2
    head_dim = 128
    num_layers = 2

    quantizer = DynamicKVQuantizer(head_dim=head_dim, use_qjl=True)
    cache = TurboQuantCache(quantizer=quantizer)

    # 1. Populate cache with 128 tokens across all layers
    k_init = torch.randn(batch_size, num_kv_heads, 128, head_dim)
    v_init = torch.randn(batch_size, num_kv_heads, 128, head_dim)

    for layer_idx in range(num_layers):
        cache.update(k_init, v_init, layer_idx=layer_idx)
        assert cache.get_seq_length(layer_idx) == 128

    # Store base data pointers to verify zero-copy slicing
    k_ptrs_init = [cache.k_packed[i].data_ptr() for i in range(num_layers)]
    v_ptrs_init = [cache.v_packed[i].data_ptr() for i in range(num_layers)]

    # 2. Append 32 query/answer tokens
    k_append = torch.randn(batch_size, num_kv_heads, 32, head_dim)
    v_append = torch.randn(batch_size, num_kv_heads, 32, head_dim)

    for layer_idx in range(num_layers):
        cache.update(k_append, v_append, layer_idx=layer_idx)
        assert cache.get_seq_length(layer_idx) == 160

    # 3. Call cache.crop(128)
    cache.crop(128)

    # 4. Verify sequence length and shapes across all layers
    assert len(cache) == num_layers
    for layer_idx in range(num_layers):
        assert cache.get_seq_length(layer_idx) == 128
        assert cache[layer_idx].get_seq_length() == 128

        # Sliced buffers should share underlying data pointers (zero-copy)
        assert cache.k_packed[layer_idx] is not None
        assert cache.v_packed[layer_idx] is not None
        assert cache.k_scales[layer_idx] is not None
        assert cache.v_scales[layer_idx] is not None

        assert cache.k_packed[layer_idx].shape == (batch_size, num_kv_heads, 128, head_dim // 4)
        assert cache.v_packed[layer_idx].shape == (batch_size, num_kv_heads, 128, head_dim // 4)
        assert cache.k_scales[layer_idx].shape == (batch_size, num_kv_heads, 128, 1)
        assert cache.v_scales[layer_idx].shape == (batch_size, num_kv_heads, 128, 1)

        # Verify QJL buffers if present
        if cache.k_qjl[layer_idx] is not None:
            assert cache.k_qjl[layer_idx].signs.shape == (batch_size, num_kv_heads, 128, head_dim)
            assert cache.k_qjl[layer_idx].scale.shape == (batch_size, num_kv_heads, 128, 1)
        if cache.v_qjl[layer_idx] is not None:
            assert cache.v_qjl[layer_idx].signs.shape == (batch_size, num_kv_heads, 128, head_dim)
            assert cache.v_qjl[layer_idx].scale.shape == (batch_size, num_kv_heads, 128, 1)

    # 5. Verify subsequent tokens append from position 128
    k_next = torch.randn(batch_size, num_kv_heads, 1, head_dim)
    v_next = torch.randn(batch_size, num_kv_heads, 1, head_dim)
    for layer_idx in range(num_layers):
        cache.update(k_next, v_next, layer_idx=layer_idx)
        assert cache.get_seq_length(layer_idx) == 129


def test_chunked_prefix_ingestion(llama_attn_config: LlamaConfig) -> None:
    """Ingest a 512-token synthetic sequence in 128-token chunks through a patched attention block and confirm the final TurboQuantCache length equals 512."""
    torch.manual_seed(42)
    batch_size = 1
    seq_len = 512
    chunk_size = 128
    hidden_size = llama_attn_config.hidden_size
    head_dim = llama_attn_config.head_dim

    # 1. Instantiate attention block and quantizer
    attn = LlamaAttention(llama_attn_config, layer_idx=0)
    quantizer = DynamicKVQuantizer(head_dim=head_dim, use_qjl=False)
    apply_turboquant(attn, quantizer=quantizer)

    # 2. Ingest 512 synthetic tokens using DocumentPrefixEngine
    x_synthetic = torch.randn(batch_size, seq_len, hidden_size)
    engine = DocumentPrefixEngine(
        model=attn,
        quantizer=quantizer,
        chunk_size=chunk_size,
    )

    state: PrefixState = engine.ingest(x_synthetic)

    # 3. Confirm the final TurboQuantCache length equals 512
    assert state.prefix_length == 512
    assert state.cache.get_seq_length(0) == 512
    assert state.cache_bytes > 0
    assert state.prefill_time_sec >= 0.0

    k_buf = state.cache.k_packed[0]
    v_buf = state.cache.v_packed[0]
    assert k_buf is not None
    assert v_buf is not None
    assert k_buf.shape == (batch_size, llama_attn_config.num_key_value_heads, 512, head_dim // 4)
    assert v_buf.shape == (batch_size, llama_attn_config.num_key_value_heads, 512, head_dim // 4)


def test_ingest_document_prefix_function(llama_attn_config: LlamaConfig) -> None:
    """Validate convenience functional wrapper ingest_document_prefix."""
    attn = LlamaAttention(llama_attn_config, layer_idx=0)
    quantizer = DynamicKVQuantizer(head_dim=llama_attn_config.head_dim)
    apply_turboquant(attn, quantizer=quantizer)

    seq_len = 256
    x = torch.randn(1, seq_len, llama_attn_config.hidden_size)

    state = ingest_document_prefix(
        model=attn,
        document_text=x,
        quantizer=quantizer,
        chunk_size=64,
    )

    assert isinstance(state, PrefixState)
    assert state.prefix_length == seq_len
    assert state.cache.get_seq_length(0) == seq_len
    assert state["prefix_length"] == seq_len
    assert state["cache_bytes"] == state.cache.get_cache_size_bytes()


def test_prefix_crop_and_multiquery_reuse(llama_attn_config: LlamaConfig) -> None:
    """Validate prefix ingestion followed by multi-turn question answer generation with prefix rewinding."""
    attn = LlamaAttention(llama_attn_config, layer_idx=0)
    quantizer = DynamicKVQuantizer(head_dim=llama_attn_config.head_dim)
    apply_turboquant(attn, quantizer=quantizer)

    # Ingest document prefix of 256 tokens
    doc = torch.randn(1, 256, llama_attn_config.hidden_size)
    engine = DocumentPrefixEngine(model=attn, quantizer=quantizer, chunk_size=128)
    state = engine.ingest(doc)
    assert state.cache.get_seq_length(0) == 256

    # Turn 1: Question 1 (16 tokens) + decode answer (8 tokens)
    q1 = torch.randn(1, 16, llama_attn_config.hidden_size)
    attn(q1, past_key_values=state.cache)
    assert state.cache.get_seq_length(0) == 256 + 16

    for _ in range(8):
        ans_tok = torch.randn(1, 1, llama_attn_config.hidden_size)
        attn(ans_tok, past_key_values=state.cache)
    assert state.cache.get_seq_length(0) == 256 + 16 + 8

    # Rewind to document prefix (256 tokens)
    state.crop(256)
    assert state.cache.get_seq_length(0) == 256
    assert state.prefix_length == 256

    # Turn 2: Question 2 (20 tokens) reuses exact same cached document prefix
    q2 = torch.randn(1, 20, llama_attn_config.hidden_size)
    attn(q2, past_key_values=state.cache)
    assert state.cache.get_seq_length(0) == 256 + 20


def test_cpu_long_document_ingestion_4000_tokens(llama_attn_config: LlamaConfig) -> None:
    """Validate low-memory chunked ingestion of 4,000+ token sequence on CPU."""
    attn = LlamaAttention(llama_attn_config, layer_idx=0)
    quantizer = DynamicKVQuantizer(head_dim=llama_attn_config.head_dim)
    apply_turboquant(attn, quantizer=quantizer)

    total_tokens = 4096
    chunk_size = 512
    x_long = torch.randn(1, total_tokens, llama_attn_config.hidden_size)

    engine = DocumentPrefixEngine(
        model=attn,
        quantizer=quantizer,
        chunk_size=chunk_size,
    )
    state = engine.ingest(x_long)

    assert state.prefix_length == total_tokens
    assert state.cache.get_seq_length(0) == total_tokens
    # 2-bit compressed footprint: 4096 tokens * 2 KV heads * (128 / 4 bytes packed) = 262,144 bytes
    assert state.cache_bytes > 0
    assert state.prefill_time_sec > 0.0
