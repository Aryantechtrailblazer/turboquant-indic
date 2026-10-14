"""Rotation primitives for TurboQuant."""

from typing import Optional, Union
import torch


def random_orthogonal_matrix(
    dim: int,
    device: Optional[Union[torch.device, str]] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a random orthogonal matrix distributed according to the Haar measure.

    Args:
        dim: Dimension of the square orthogonal matrix.
        device: Device on which to allocate the tensor (e.g., 'cpu', 'cuda').
        dtype: Desired data type of the returned tensor (default: torch.float32).

    Returns:
        A (dim, dim) orthogonal tensor Q satisfying Q.T @ Q == I.
    """
    if dim <= 0:
        raise ValueError(f"Dimension must be a positive integer, got {dim}")

    # 1. Sample a matrix from standard Gaussian noise: A ~ N(0, 1)
    a = torch.randn(dim, dim, device=device, dtype=dtype)

    # 2. Compute QR decomposition: Q, R = torch.linalg.qr(A)
    q, r = torch.linalg.qr(a)

    # 3. Adjust column signs using the diagonal of R to ensure a uniform distribution (Haar measure)
    d = torch.diagonal(r, dim1=-2, dim2=-1)
    ph = d.sign()
    ph[ph == 0] = 1.0
    q = q * ph.unsqueeze(-2)

    return q
