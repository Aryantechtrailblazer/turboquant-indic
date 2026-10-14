"""Physical 2-bit tensor packing and unpacking using vectorized bitwise operations."""

from __future__ import annotations

from typing import Union
import torch


def pack_2bit(codes: torch.Tensor) -> torch.Tensor:
    """Pack an integer tensor containing 2-bit values into a uint8 tensor.

    Takes an integer tensor of shape [N, D] (or [D]) where values are in {0, 1, 2, 3}
    and D is divisible by 4. Packs 4 consecutive 2-bit values along the last dimension
    into a single 8-bit byte.

    Slices across dimensions in strides of 4:
        b0 = (codes[..., 0::4] & 3) << 6
        b1 = (codes[..., 1::4] & 3) << 4
        b2 = (codes[..., 2::4] & 3) << 2
        b3 = (codes[..., 3::4] & 3)
    Combined with bitwise OR: (b0 | b1 | b2 | b3).to(torch.uint8).

    Args:
        codes: Integer tensor of shape [N, D] or [D] with values in {0, 1, 2, 3}.
               The last dimension D must be divisible by 4.

    Returns:
        torch.Tensor: Packed tensor of shape [N, D // 4] (or [D // 4]) of dtype torch.uint8.

    Raises:
        ValueError: If the last dimension of codes is not divisible by 4,
                    or if codes tensor has unsupported dimensionality.
    """
    if codes.ndim == 0:
        raise ValueError("Cannot pack a scalar tensor; expected at least 1D tensor.")

    d = codes.shape[-1]
    if d % 4 != 0:
        raise ValueError(
            f"The last dimension of codes must be divisible by 4, got {d}."
        )

    # 4-stride slicing and bit-shifting
    b0 = (codes[..., 0::4] & 3) << 6
    b1 = (codes[..., 1::4] & 3) << 4
    b2 = (codes[..., 2::4] & 3) << 2
    b3 = codes[..., 3::4] & 3

    packed = (b0 | b1 | b2 | b3).to(torch.uint8)
    return packed


def unpack_2bit(
    packed: torch.Tensor,
    target_dtype: torch.dtype = torch.uint8,
) -> torch.Tensor:
    """Unpack a packed uint8 tensor into original 2-bit elements.

    Takes a torch.uint8 tensor of shape [N, D // 4] (or [D // 4]) and unpacks
    each byte into four 2-bit values.

    Extracts four 2-bit elements per byte using right shifts and mask & 3:
        c0 = (packed >> 6) & 3
        c1 = (packed >> 4) & 3
        c2 = (packed >> 2) & 3
        c3 = packed & 3
    Interleaves the slices back into shape [N, D] in the original order [c0, c1, c2, c3],
    and casts to target_dtype.

    Args:
        packed: Packed tensor of dtype torch.uint8 and shape [N, D // 4] (or [D // 4]).
        target_dtype: Target data type for the returned tensor (default: torch.uint8).

    Returns:
        torch.Tensor: Unpacked tensor of shape [N, D] (or [D]) and dtype target_dtype.

    Raises:
        TypeError: If packed tensor is not of dtype torch.uint8.
    """
    if packed.dtype != torch.uint8:
        raise TypeError(
            f"Expected packed tensor of dtype torch.uint8, got {packed.dtype}."
        )

    if packed.ndim == 0:
        raise ValueError("Cannot unpack a scalar tensor; expected at least 1D tensor.")

    c0 = (packed >> 6) & 3
    c1 = (packed >> 4) & 3
    c2 = (packed >> 2) & 3
    c3 = packed & 3

    # Interleave slices back into original order [c0, c1, c2, c3]
    *batch_dims, packed_d = packed.shape
    orig_d = packed_d * 4

    unpacked = torch.empty(
        (*batch_dims, orig_d),
        dtype=target_dtype,
        device=packed.device,
    )
    unpacked[..., 0::4] = c0.to(target_dtype)
    unpacked[..., 1::4] = c1.to(target_dtype)
    unpacked[..., 2::4] = c2.to(target_dtype)
    unpacked[..., 3::4] = c3.to(target_dtype)

    return unpacked


__all__ = ["pack_2bit", "unpack_2bit"]
