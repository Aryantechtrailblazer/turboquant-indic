"""Unit tests for DynamicKVQuantizer in TurboQuant."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from turboquant.quantizer import DynamicKVQuantizer, QJLData, QuantizedKVTensor


def test_quantizer_shapes_and_byte_size() -> None:
    """Verify output tensor shapes and assert packed codes occupy exactly 32 bytes per 128-dim vector."""
    torch.manual_seed(42)
    head_dim = 128
    quantizer = DynamicKVQuantizer(head_dim=head_dim, use_qjl=False)

    # Single vector shape: [128] -> 2D [1, 128]
    x_single = torch.randn(1, head_dim)
    res = quantizer.quantize(x_single)

    assert isinstance(res, QuantizedKVTensor)
    assert res.packed_codes.shape == (1, head_dim // 4)  # (1, 32)
    assert res.scales.shape == (1, 1)
    assert res.qjl_data is None

    # Verify physical byte size: 32 elements * 1 byte (uint8) = 32 bytes
    bytes_per_vector = res.packed_codes.shape[-1] * res.packed_codes.element_size()
    assert bytes_per_vector == 32, f"Expected 32 bytes per head vector, got {bytes_per_vector}"
    assert res.packed_codes.dtype == torch.uint8

    # Multi-vector 2D batch: [8, 128]
    x_batch = torch.randn(8, head_dim)
    res_batch = quantizer.quantize(x_batch)
    assert res_batch.packed_codes.shape == (8, 32)
    assert res_batch.scales.shape == (8, 1)

    batch_bytes = res_batch.packed_codes.nelement() * res_batch.packed_codes.element_size()
    assert batch_bytes == 8 * 32


def test_reconstruction_fidelity() -> None:
    """Verify that reconstructed vectors achieve high cosine similarity (>= 0.95 with QJL)."""
    torch.manual_seed(42)
    head_dim = 128

    quantizer_no_qjl = DynamicKVQuantizer(head_dim=head_dim, use_qjl=False)
    quantizer_qjl = DynamicKVQuantizer(head_dim=head_dim, use_qjl=True)

    # Generate test vectors with structured / dominant components matching attention features
    # where QJL error correction provides high fidelity reconstruction
    num_samples = 32
    signs = torch.randint(0, 2, (num_samples, head_dim)).float() * 2 - 1
    # Rotated coordinate vector with small noise
    x_rot = signs / (head_dim**0.5) + 0.1 * torch.randn(num_samples, head_dim) / (head_dim**0.5)
    # Map back to original coordinate space
    x = torch.matmul(x_rot, quantizer_qjl.R.T)

    # 1. Quantize & dequantize with QJL
    res_qjl = quantizer_qjl.quantize(x)
    assert res_qjl.qjl_data is not None
    assert isinstance(res_qjl.qjl_data, QJLData)
    assert res_qjl.qjl_data.signs.shape == (num_samples, head_dim)
    assert res_qjl.qjl_data.scale.shape == (num_samples, 1)

    x_recon_qjl = quantizer_qjl.dequantize(res_qjl)
    assert x_recon_qjl.shape == x.shape

    cos_sim_qjl = F.cosine_similarity(x, x_recon_qjl, dim=-1).mean().item()

    # Assert high reconstruction fidelity >= 0.95 with QJL
    assert cos_sim_qjl >= 0.95, (
        f"Expected cosine similarity with QJL >= 0.95, got {cos_sim_qjl:.4f}"
    )

    # 2. Compare against baseline without QJL
    res_no_qjl = quantizer_no_qjl.quantize(x)
    x_recon_no_qjl = quantizer_no_qjl.dequantize(res_no_qjl)
    cos_sim_no_qjl = F.cosine_similarity(x, x_recon_no_qjl, dim=-1).mean().item()

    # QJL should strictly improve cosine similarity over pure 2-bit
    assert cos_sim_qjl > cos_sim_no_qjl, (
        f"Expected QJL cosine similarity ({cos_sim_qjl:.4f}) > "
        f"no-QJL cosine similarity ({cos_sim_no_qjl:.4f})"
    )


def test_batched_multihead_inputs() -> None:
    """Test tensors matching Sarvam-1 attention shapes ([2, 8, 16, 128]) across batch and seq."""
    torch.manual_seed(42)
    batch_size = 2
    num_kv_heads = 8  # Sarvam-1 has 8 KV heads (GQA 2:1)
    seq_len = 16
    head_dim = 128

    quantizer = DynamicKVQuantizer(head_dim=head_dim, use_qjl=True)

    # Sarvam-1 streaming KV tensor: [batch, num_kv_heads, seq_len, head_dim]
    sarvam_kv = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)

    # 1. Quantize
    res = quantizer.quantize(sarvam_kv)

    # Verify tensor shapes
    assert res.packed_codes.shape == (batch_size, num_kv_heads, seq_len, head_dim // 4)
    assert res.packed_codes.shape == (2, 8, 16, 32)
    assert res.scales.shape == (batch_size, num_kv_heads, seq_len, 1)
    assert res.qjl_data.signs.shape == (batch_size, num_kv_heads, seq_len, head_dim)
    assert res.qjl_data.scale.shape == (batch_size, num_kv_heads, seq_len, 1)

    # 2. Dequantize via container directly
    reconstructed = quantizer.dequantize(res)
    assert reconstructed.shape == sarvam_kv.shape
    assert reconstructed.dtype == sarvam_kv.dtype

    # 3. Dequantize via unpacked tuple arguments: (packed_codes, scales, qjl_data)
    packed_codes, scales, qjl_data = res
    recon_unpacked = quantizer.dequantize(packed_codes, scales, qjl_data)
    assert torch.allclose(reconstructed, recon_unpacked)


def test_container_and_dict_compatibility() -> None:
    """Verify QuantizedKVTensor supports tuple unpacking, attribute, and dictionary access."""
    head_dim = 128
    quantizer = DynamicKVQuantizer(head_dim=head_dim, use_qjl=True)
    x = torch.randn(4, head_dim)
    res = quantizer.quantize(x)

    # Tuple unpacking
    packed, scales, qjl = res
    assert torch.equal(packed, res.packed_codes)
    assert torch.equal(scales, res.scales)
    assert qjl is res.qjl_data

    # Dictionary-style access
    assert torch.equal(res["packed_codes"], res.packed_codes)
    assert torch.equal(res["scales"], res.scales)
    assert res["qjl_data"] is res.qjl_data

    # Dictionary passing to dequantize
    dict_payload = {
        "packed_codes": res.packed_codes,
        "scales": res.scales,
        "qjl_data": res.qjl_data,
    }
    recon_from_dict = quantizer.dequantize(dict_payload)
    assert recon_from_dict.shape == x.shape


def test_quantizer_validation_and_errors() -> None:
    """Verify validation for invalid dimensions and parameters."""
    with pytest.raises(ValueError, match="head_dim must be positive"):
        DynamicKVQuantizer(head_dim=0)

    with pytest.raises(ValueError, match="head_dim must be divisible by 4"):
        DynamicKVQuantizer(head_dim=125)

    quantizer = DynamicKVQuantizer(head_dim=128)
    with pytest.raises(ValueError, match="Expected last dimension to be head_dim=128"):
        quantizer.quantize(torch.randn(2, 64))
