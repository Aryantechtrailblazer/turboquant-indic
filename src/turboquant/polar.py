"""Polar quantization primitives for TurboQuant."""

import math
from typing import Optional, Tuple
import torch


def cartesian_to_polar(
    x1: torch.Tensor,
    x2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert Cartesian coordinates (x1, x2) to polar coordinates (r, theta).

    Args:
        x1: First Cartesian coordinate tensor.
        x2: Second Cartesian coordinate tensor.

    Returns:
        Tuple of (r, theta):
            - r: Radius tensor, r = sqrt(x1^2 + x2^2).
            - theta: Angle tensor in radians, theta = atan2(x2, x1).
    """
    r = torch.sqrt(x1**2 + x2**2)
    theta = torch.atan2(x2, x1)
    return r, theta


def quantize_angles(theta: torch.Tensor, bits: int) -> torch.Tensor:
    """Map continuous angles to discrete bin indices in [0, 2^bits - 1].

    The angle is mapped to [0, 2*pi) and uniformly partitioned into 2^bits bins.

    Args:
        theta: Continuous angle tensor in radians.
        bits: Number of quantization bits (>= 1).

    Returns:
        Tensor of integer bin indices in range [0, 2^bits - 1].
    """
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")

    num_bins = 1 << bits
    two_pi = 2.0 * math.pi

    # Normalize theta to [0, 2*pi)
    theta_mod = torch.remainder(theta, two_pi)
    bin_width = two_pi / num_bins

    # Compute bin indices and clamp within [0, 2^bits - 1]
    indices = torch.floor(theta_mod / bin_width).to(torch.long)
    return torch.clamp(indices, 0, num_bins - 1)


def dequantize_angles(
    indices: torch.Tensor,
    bits: int,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Map discrete bin indices back to continuous representative angles (bin centers).

    Args:
        indices: Tensor of integer bin indices in [0, 2^bits - 1].
        bits: Number of quantization bits (>= 1).
        dtype: Floating point dtype for the reconstructed angles (default: torch.float32).

    Returns:
        Tensor of reconstructed angles in radians (bin midpoints).
    """
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")

    if dtype is None:
        dtype = torch.float32

    num_bins = 1 << bits
    bin_width = (2.0 * math.pi) / num_bins

    # Center of the bin: (index + 0.5) * bin_width
    return (indices.to(dtype=dtype) + 0.5) * bin_width


def polar_to_cartesian(
    r: torch.Tensor,
    theta_quant: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct Cartesian coordinates from polar coordinates (r, theta_quant).

    Args:
        r: Radius tensor.
        theta_quant: Quantized angle tensor in radians.

    Returns:
        Tuple of (x1_hat, x2_hat):
            - x1_hat = r * cos(theta_quant)
            - x2_hat = r * sin(theta_quant)
    """
    x1_hat = r * torch.cos(theta_quant)
    x2_hat = r * torch.sin(theta_quant)
    return x1_hat, x2_hat
