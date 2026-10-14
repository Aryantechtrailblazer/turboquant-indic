"""Validation test suite for TurboQuant attention monkey-patching and 2-bit KV cache."""

from __future__ import annotations

import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaConfig

from turboquant.cache import TurboQuantCache
from turboquant.patch import (
    apply_turboquant,
    restore_turboquant,
    turboquant_context,
)
from turboquant.quantizer import DynamicKVQuantizer


@pytest.fixture
def llama_attn_config() -> LlamaConfig:
    """Fixture providing Sarvam-1 / Llama attention configuration with head_dim=128."""
    return LlamaConfig(
        hidden_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA ratio: 4 / 2 = 2
        head_dim=128,
        attention_bias=False,
    )


def test_prefill_and_decode_step(llama_attn_config: LlamaConfig) -> None:
    """Run a 2-step synthetic forward pass (S=16 prefill, followed by S=1 decode step)."""
    torch.manual_seed(42)
    batch_size = 1
    head_dim = 128
    hidden_size = llama_attn_config.hidden_size

    # 1. Instantiate attention block and quantizer
    attn = LlamaAttention(llama_attn_config, layer_idx=0)
    quantizer = DynamicKVQuantizer(head_dim=head_dim, use_qjl=False)

    # 2. Apply monkey patch in-place
    apply_turboquant(attn, quantizer=quantizer)
    assert getattr(attn, "_is_turboquant_patched", False) is True

    # 3. Create compressed cache container
    cache = TurboQuantCache(quantizer=quantizer)

    # Step 1: Prefill phase (S = 16)
    s_prefill = 16
    x_prefill = torch.randn(batch_size, s_prefill, hidden_size)
    out_prefill, _ = attn(x_prefill, past_key_values=cache)

    assert out_prefill.shape == (batch_size, s_prefill, hidden_size)
    assert not torch.isnan(out_prefill).any()
    assert not torch.isinf(out_prefill).any()
    assert cache.get_seq_length(0) == s_prefill
    assert cache.k_packed[0] is not None
    assert cache.k_packed[0].shape == (batch_size, 2, s_prefill, head_dim // 4)

    # Step 2: Autoregressive decode phase (S = 1)
    s_decode = 1
    x_decode = torch.randn(batch_size, s_decode, hidden_size)
    out_decode, _ = attn(x_decode, past_key_values=cache)

    assert out_decode.shape == (batch_size, s_decode, hidden_size)
    assert not torch.isnan(out_decode).any()
    assert not torch.isinf(out_decode).any()
    assert cache.get_seq_length(0) == s_prefill + s_decode
    assert cache.k_packed[0].shape == (batch_size, 2, s_prefill + s_decode, head_dim // 4)
    assert cache.v_packed[0].shape == (batch_size, 2, s_prefill + s_decode, head_dim // 4)


def test_cache_memory_footprint(llama_attn_config: LlamaConfig) -> None:
    """Verify that the internal cache retains uint8 tensors (occupying exactly 32 bytes per 128-dim head vector per token)."""
    torch.manual_seed(42)
    head_dim = 128
    hidden_size = llama_attn_config.hidden_size

    attn = LlamaAttention(llama_attn_config, layer_idx=0)
    quantizer = DynamicKVQuantizer(head_dim=head_dim, use_qjl=False)
    apply_turboquant(attn, quantizer=quantizer)

    cache = TurboQuantCache(quantizer=quantizer)

    # Run prefill (S=16) and decode (S=1)
    attn(torch.randn(1, 16, hidden_size), past_key_values=cache)
    attn(torch.randn(1, 1, hidden_size), past_key_values=cache)

    k_packed = cache.k_packed[0]
    v_packed = cache.v_packed[0]

    assert k_packed is not None
    assert v_packed is not None

    # Verify physical uint8 data types
    assert k_packed.dtype == torch.uint8
    assert v_packed.dtype == torch.uint8

    # Packed dimension for 128-dim head is 128 / 4 = 32 bytes
    packed_dim = k_packed.shape[-1]
    assert packed_dim == head_dim // 4 == 32

    # Verify memory footprint: uint8 tensor element size is 1 byte
    element_size = k_packed.element_size()
    assert element_size == 1

    bytes_per_head_vector = packed_dim * element_size
    assert bytes_per_head_vector == 32, f"Expected 32 bytes per head vector, got {bytes_per_head_vector}"

    # Verify 8x compression compared to 16-bit baseline (128 elements * 2 bytes = 256 bytes)
    baseline_bytes_16bit = head_dim * 2
    compression_ratio = baseline_bytes_16bit / bytes_per_head_vector
    assert compression_ratio == 8.0


def test_context_manager_restoration(llama_attn_config: LlamaConfig) -> None:
    """Confirm that exiting turboquant_context restores original attention methods without side effects."""
    torch.manual_seed(42)
    hidden_size = llama_attn_config.hidden_size

    attn = LlamaAttention(llama_attn_config, layer_idx=0)
    original_forward = attn.forward
    quantizer = DynamicKVQuantizer(head_dim=128)
    cache = TurboQuantCache(quantizer=quantizer)

    # Inside context manager: forward should be patched
    with turboquant_context(attn, quantizer=quantizer):
        assert attn.forward != original_forward
        assert getattr(attn, "_is_turboquant_patched", False) is True

        # Forward pass functions properly under patched context
        out, _ = attn(torch.randn(1, 16, hidden_size), past_key_values=cache)
        assert out.shape == (1, 16, hidden_size)

    # Outside context manager: forward must be restored
    assert attn.forward == original_forward
    assert getattr(attn, "_is_turboquant_patched", False) is False

    # Verify exception safety: restore still occurs even on error
    with pytest.raises(RuntimeError):
        with turboquant_context(attn, quantizer=quantizer):
            assert attn.forward != original_forward
            raise RuntimeError("Synthetic error within context")

    assert attn.forward == original_forward
    assert getattr(attn, "_is_turboquant_patched", False) is False


def test_cache_layer_view_and_reset(llama_attn_config: LlamaConfig) -> None:
    """Validate cache layer indexing, dictionary access, and reset mechanisms."""
    quantizer = DynamicKVQuantizer(head_dim=128)
    cache = TurboQuantCache(quantizer=quantizer)

    k = torch.randn(1, 2, 8, 128)
    v = torch.randn(1, 2, 8, 128)
    k_quant, v_quant = cache.update(k, v, layer_idx=0)

    # Check return types
    assert k_quant.packed_codes.dtype == torch.uint8
    assert v_quant.packed_codes.dtype == torch.uint8

    # Check layer view indexing
    layer = cache[0]
    assert layer.k_packed is not None
    assert layer["k_packed"].shape == (1, 2, 8, 32)
    assert layer.get_seq_length() == 8

    # Check get_k and get_v
    k_obj = cache.get_k(0)
    assert k_obj.packed_codes.shape == (1, 2, 8, 32)

    # Check reset
    cache.reset()
    assert len(cache) == 0
    assert cache.get_seq_length(0) == 0
