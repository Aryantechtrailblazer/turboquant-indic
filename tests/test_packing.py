"""Unit tests for physical 2-bit tensor packing and unpacking module."""

from __future__ import annotations

import pytest
import torch

from turboquant.packing import pack_2bit, unpack_2bit


def test_shape_and_dtype() -> None:
    """📐 Verify [N, 768] packs to [N, 192] of dtype torch.uint8 and unpacks back."""
    torch.manual_seed(42)
    n = 10
    d = 768

    # 1. 2D Tensor [N, 768]
    codes = torch.randint(0, 4, (n, d), dtype=torch.uint8)
    packed = pack_2bit(codes)

    assert isinstance(packed, torch.Tensor)
    assert packed.shape == (n, 192), f"Expected shape ({n}, 192), got {packed.shape}"
    assert packed.dtype == torch.uint8, f"Expected dtype torch.uint8, got {packed.dtype}"

    # Unpack back to uint8
    unpacked_uint8 = unpack_2bit(packed, target_dtype=torch.uint8)
    assert unpacked_uint8.shape == (n, d)
    assert unpacked_uint8.dtype == torch.uint8
    assert torch.equal(unpacked_uint8, codes)

    # Unpack with custom target_dtype (e.g., int64, int32)
    unpacked_int64 = unpack_2bit(packed, target_dtype=torch.int64)
    assert unpacked_int64.shape == (n, d)
    assert unpacked_int64.dtype == torch.int64
    assert torch.equal(unpacked_int64, codes.to(torch.int64))

    # 2. 1D Tensor [768]
    codes_1d = torch.randint(0, 4, (d,), dtype=torch.uint8)
    packed_1d = pack_2bit(codes_1d)
    assert packed_1d.shape == (192,)
    assert packed_1d.dtype == torch.uint8

    unpacked_1d = unpack_2bit(packed_1d)
    assert unpacked_1d.shape == (d,)
    assert unpacked_1d.dtype == torch.uint8
    assert torch.equal(unpacked_1d, codes_1d)


def test_round_trip_invariance() -> None:
    """🔄 Verify unpack_2bit(pack_2bit(codes)) == codes for random inputs with values {0, 1, 2, 3}."""
    torch.manual_seed(123)

    test_shapes = [
        (1, 4),
        (2, 8),
        (5, 16),
        (16, 64),
        (32, 128),
        (64, 768),
        (10, 1024),
    ]

    for shape in test_shapes:
        for dtype in [torch.uint8, torch.int32, torch.int64]:
            codes = torch.randint(0, 4, shape, dtype=dtype)
            packed = pack_2bit(codes)
            unpacked = unpack_2bit(packed, target_dtype=dtype)

            assert torch.equal(unpacked, codes), (
                f"Round-trip invariance failed for shape {shape} and dtype {dtype}."
            )


def test_boundary_patterns() -> None:
    """🎯 Verify correctness on edge values: all 0s, all 3s, alternating 0 and 3, and individual bit positions."""
    # 1. All 0s: byte value must be 0
    zeros = torch.zeros((4, 16), dtype=torch.uint8)
    packed_zeros = pack_2bit(zeros)
    assert torch.all(packed_zeros == 0)
    assert torch.equal(unpack_2bit(packed_zeros), zeros)

    # 2. All 3s: byte value must be (3 << 6) | (3 << 4) | (3 << 2) | 3 = 192 + 48 + 12 + 3 = 255
    threes = torch.full((4, 16), 3, dtype=torch.uint8)
    packed_threes = pack_2bit(threes)
    assert torch.all(packed_threes == 255)
    assert torch.equal(unpack_2bit(packed_threes), threes)

    # 3. All 1s: byte value must be (1 << 6) | (1 << 4) | (1 << 2) | 1 = 64 + 16 + 4 + 1 = 85
    ones = torch.full((2, 8), 1, dtype=torch.uint8)
    packed_ones = pack_2bit(ones)
    assert torch.all(packed_ones == 85)
    assert torch.equal(unpack_2bit(packed_ones), ones)

    # 4. All 2s: byte value must be (2 << 6) | (2 << 4) | (2 << 2) | 2 = 128 + 32 + 8 + 2 = 170
    twos = torch.full((2, 8), 2, dtype=torch.uint8)
    packed_twos = pack_2bit(twos)
    assert torch.all(packed_twos == 170)
    assert torch.equal(unpack_2bit(packed_twos), twos)

    # 5. Alternating 0 and 3: [0, 3, 0, 3] -> (0 << 6) | (3 << 4) | (0 << 2) | 3 = 48 + 3 = 51
    alt_0_3 = torch.tensor([[0, 3, 0, 3] * 4], dtype=torch.uint8)
    packed_0_3 = pack_2bit(alt_0_3)
    assert torch.all(packed_0_3 == 51)
    assert torch.equal(unpack_2bit(packed_0_3), alt_0_3)

    # 6. Alternating 3 and 0: [3, 0, 3, 0] -> (3 << 6) | (0 << 4) | (3 << 2) | 0 = 192 + 12 = 204
    alt_3_0 = torch.tensor([[3, 0, 3, 0] * 4], dtype=torch.uint8)
    packed_3_0 = pack_2bit(alt_3_0)
    assert torch.all(packed_3_0 == 204)
    assert torch.equal(unpack_2bit(packed_3_0), alt_3_0)

    # 7. Positional bit tests: [b0, b1, b2, b3]
    pos_cases = [
        ([1, 0, 0, 0], 64),
        ([0, 1, 0, 0], 16),
        ([0, 0, 1, 0], 4),
        ([0, 0, 0, 1], 1),
        ([0, 1, 2, 3], 27),
        ([3, 2, 1, 0], 228),
    ]
    for pattern, expected_byte in pos_cases:
        t = torch.tensor([pattern], dtype=torch.uint8)
        packed_t = pack_2bit(t)
        assert packed_t[0, 0].item() == expected_byte
        assert torch.equal(unpack_2bit(packed_t), t)


def test_memory_footprint() -> None:
    """💾 Verify packed size is exactly 192 bytes per 768-dim vector (16x reduction vs float32)."""
    torch.manual_seed(999)
    dim = 768

    # Single vector of length 768
    codes_single = torch.randint(0, 4, (1, dim), dtype=torch.uint8)
    packed_single = pack_2bit(codes_single)

    # Memory in bytes: element_size() * nelement()
    packed_bytes = packed_single.element_size() * packed_single.nelement()
    assert packed_bytes == 192, (
        f"Expected exactly 192 bytes for packed 768-dim vector, got {packed_bytes}"
    )

    # Float32 uncompressed equivalent: 768 * 4 = 3072 bytes
    fp32_vec = torch.randn(1, dim, dtype=torch.float32)
    fp32_bytes = fp32_vec.element_size() * fp32_vec.nelement()
    assert fp32_bytes == 3072, f"Expected 3072 bytes for float32 vector, got {fp32_bytes}"

    # Exact 16x memory footprint reduction
    compression_ratio = fp32_bytes / packed_bytes
    assert compression_ratio == 16.0, (
        f"Expected 16.0x memory reduction, got {compression_ratio}x"
    )

    # Batch of 100 vectors of length 768
    num_vectors = 100
    codes_batch = torch.randint(0, 4, (num_vectors, dim), dtype=torch.uint8)
    packed_batch = pack_2bit(codes_batch)

    batch_packed_bytes = packed_batch.element_size() * packed_batch.nelement()
    expected_batch_bytes = num_vectors * 192
    assert batch_packed_bytes == expected_batch_bytes

    batch_fp32_bytes = num_vectors * 3072
    assert batch_fp32_bytes / batch_packed_bytes == 16.0


def test_validation_and_errors() -> None:
    """Verify input validation and informative errors."""
    # Last dimension not divisible by 4
    with pytest.raises(ValueError, match="divisible by 4"):
        pack_2bit(torch.randint(0, 4, (10, 767), dtype=torch.uint8))

    with pytest.raises(ValueError, match="divisible by 4"):
        pack_2bit(torch.randint(0, 4, (5,), dtype=torch.uint8))

    # Scalar tensor
    with pytest.raises(ValueError, match="scalar tensor"):
        pack_2bit(torch.tensor(2, dtype=torch.uint8))

    with pytest.raises(ValueError, match="scalar tensor"):
        unpack_2bit(torch.tensor(255, dtype=torch.uint8))

    # Unpack non-uint8 tensor
    with pytest.raises(TypeError, match="torch.uint8"):
        unpack_2bit(torch.tensor([[10, 20]], dtype=torch.int32))

    with pytest.raises(TypeError, match="torch.uint8"):
        unpack_2bit(torch.randn(5, 10, dtype=torch.float32))
