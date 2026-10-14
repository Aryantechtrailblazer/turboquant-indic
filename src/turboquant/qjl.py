"""1-Bit QJL Error Correction primitives for TurboQuant."""

import torch


def compute_residual(v: torch.Tensor, v_hat: torch.Tensor) -> torch.Tensor:
    """Compute the residual error vector between original and quantized vectors.

    Args:
        v: Original vector tensor.
        v_hat: Quantized or approximated vector tensor.

    Returns:
        Residual error vector e = v - v_hat.
    """
    return v - v_hat


def extract_qjl_signs(residual: torch.Tensor) -> torch.Tensor:
    """Extract 1-bit signs from the residual error vector.

    Extracts signs using torch.sign(residual). Ensure zeros are mapped to +1
    so values are strictly in {-1, +1}.

    Args:
        residual: Residual error vector tensor.

    Returns:
        Tensor of signs where each element is strictly in {-1, +1}.
    """
    signs = torch.sign(residual)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return signs


def estimate_dot_product(
    query: torch.Tensor,
    v_hat: torch.Tensor,
    signs: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Calculate the QJL-corrected inner product estimate.

    Calculates the corrected inner product:
        dot(query, v_hat) + scale * dot(query, signs)

    Args:
        query: Query vector tensor.
        v_hat: Quantized or approximated vector tensor.
        signs: 1-bit signs tensor in {-1, +1}.
        scale: Scaling factor for the 1-bit residual correction.

    Returns:
        Tensor representing the corrected inner product estimate.
    """
    base_dot = torch.sum(query * v_hat, dim=-1)
    signs_cast = signs.to(dtype=query.dtype)
    residual_dot = torch.sum(query * signs_cast, dim=-1)
    return base_dot + scale * residual_dot
