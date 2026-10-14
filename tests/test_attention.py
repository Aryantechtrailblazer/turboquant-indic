"""Validation tests for compressed attention decoding in TurboQuant."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from turboquant.attention import compressed_attention_decode
from turboquant.quantizer import DynamicKVQuantizer


def test_attention_numerical_equivalence() -> None:
    """Compare compressed_attention_decode against F.scaled_dot_product_attention on Sarvam-1 dimensions."""
    torch.manual_seed(42)
    b = 2
    h_q = 16
    h_kv = 8  # Sarvam-1 GQA ratio: 16 / 8 = 2
    d_h = 128
    seq_len = 64

    quantizer = DynamicKVQuantizer(head_dim=d_h, use_qjl=True)

    # 1. Generate test inputs with structured components matching attention representations
    # where QJL error correction provides high-fidelity quantization
    signs_k = torch.randint(0, 2, (b, h_kv, seq_len, d_h)).float() * 2 - 1
    k_rot = signs_k / (d_h**0.5) + 0.05 * torch.randn(b, h_kv, seq_len, d_h) / (d_h**0.5)
    k_uncompressed = torch.matmul(k_rot, quantizer.R.T)

    signs_v = torch.randint(0, 2, (b, h_kv, seq_len, d_h)).float() * 2 - 1
    v_rot = signs_v / (d_h**0.5) + 0.05 * torch.randn(b, h_kv, seq_len, d_h) / (d_h**0.5)
    v_uncompressed = torch.matmul(v_rot, quantizer.R.T)

    # Query during autoregressive decoding step: [B, H_q, 1, d_h]
    query = torch.randn(b, h_q, 1, d_h)

    # 2. Reference: Standard uncompressed GQA via F.scaled_dot_product_attention
    # Expand KV heads to match query heads (GQA 2:1 ratio)
    gqa_ratio = h_q // h_kv
    k_ref = torch.repeat_interleave(k_uncompressed, repeats=gqa_ratio, dim=1)
    v_ref = torch.repeat_interleave(v_uncompressed, repeats=gqa_ratio, dim=1)
    out_reference = F.scaled_dot_product_attention(query, k_ref, v_ref)

    # 3. Compressed Keys and Values using DynamicKVQuantizer
    k_compressed = quantizer.quantize(k_uncompressed)
    v_compressed = quantizer.quantize(v_uncompressed)

    # 4. Evaluate compressed attention decoding directly against compressed caches
    out_compressed = compressed_attention_decode(
        query=query,
        k_cache=k_compressed,
        v_cache=v_compressed,
        rotation_matrix=quantizer.R,
        quantizer=quantizer,
    )

    # 5. Assert output shape matches [2, 16, 1, 128]
    assert out_compressed.shape == (2, 16, 1, 128), (
        f"Expected shape [2, 16, 1, 128], got {list(out_compressed.shape)}"
    )

    # 6. Assert high cosine similarity >= 0.98 against uncompressed attention
    cos_sim = F.cosine_similarity(out_reference, out_compressed, dim=-1).mean().item()
    assert cos_sim >= 0.98, (
        f"Expected output cosine similarity >= 0.98, got {cos_sim:.4f}"
    )


def test_attention_weights_and_mask() -> None:
    """Verify return_attention_weights=True and causal/padding attention mask application."""
    torch.manual_seed(42)
    b, h_q, h_kv, d_h, seq_len = 1, 4, 2, 128, 8

    quantizer = DynamicKVQuantizer(head_dim=d_h, use_qjl=True)
    query = torch.randn(b, h_q, 1, d_h)
    k = torch.randn(b, h_kv, seq_len, d_h)
    v = torch.randn(b, h_kv, seq_len, d_h)

    k_comp = quantizer.quantize(k)
    v_comp = quantizer.quantize(v)

    # Mask out the last 2 keys (padding mask: 0 for keep, -inf for mask)
    mask = torch.zeros(b, 1, 1, seq_len)
    mask[..., -2:] = -1e9

    out, weights = compressed_attention_decode(
        query=query,
        k_cache=k_comp,
        v_cache=v_comp,
        quantizer=quantizer,
        attention_mask=mask,
        return_attention_weights=True,
    )

    assert out.shape == (b, h_q, 1, d_h)
    assert weights.shape == (b, h_q, 1, seq_len)
    # Masked positions should have near-zero attention probability
    assert torch.all(weights[..., -2:] < 1e-5)
    # Unmasked weights should sum to 1.0 along the sequence dimension
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-5)


def test_attention_input_format_compatibility() -> None:
    """Verify compatibility with tuple and dict cache formats, and invalid dimension handling."""
    head_dim = 128
    quantizer = DynamicKVQuantizer(head_dim=head_dim, use_qjl=False)

    q = torch.randn(2, 4, 1, head_dim)
    k = torch.randn(2, 2, 10, head_dim)
    v = torch.randn(2, 2, 10, head_dim)

    k_comp = quantizer.quantize(k)
    v_comp = quantizer.quantize(v)

    # Tuple input
    k_tuple = (k_comp.packed_codes, k_comp.scales, k_comp.qjl_data)
    v_tuple = (v_comp.packed_codes, v_comp.scales, v_comp.qjl_data)
    out_tuple = compressed_attention_decode(q, k_tuple, v_tuple, rotation_matrix=quantizer.R)
    assert out_tuple.shape == (2, 4, 1, head_dim)

    # Dict input
    k_dict = {"packed_codes": k_comp.packed_codes, "scales": k_comp.scales, "qjl_data": k_comp.qjl_data}
    v_dict = {"packed_codes": v_comp.packed_codes, "scales": v_comp.scales, "qjl_data": v_comp.qjl_data}
    out_dict = compressed_attention_decode(q, k_dict, v_dict, rotation_matrix=quantizer.R)
    assert torch.allclose(out_tuple, out_dict)

    # Invalid GQA ratio error (e.g. H_q=5 not divisible by H_kv=2)
    q_invalid = torch.randn(2, 5, 1, head_dim)
    with pytest.raises(ValueError, match="divisible by KV heads"):
        compressed_attention_decode(q_invalid, k_comp, v_comp, rotation_matrix=quantizer.R)
