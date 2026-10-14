"""Tests for 1-Bit QJL Error Correction primitives in TurboQuant."""

import pytest
import torch

from turboquant.polar import (
    cartesian_to_polar,
    dequantize_angles,
    polar_to_cartesian,
    quantize_angles,
)
from turboquant.qjl import (
    compute_residual,
    estimate_dot_product,
    extract_qjl_signs,
)


def test_compute_residual_basic() -> None:
    """Verify compute_residual correctly computes e = v - v_hat."""
    v = torch.tensor([1.0, 2.5, -3.0, 0.0])
    v_hat = torch.tensor([0.8, 2.0, -3.2, 0.5])

    residual = compute_residual(v, v_hat)
    expected = torch.tensor([0.2, 0.5, 0.2, -0.5])

    assert torch.allclose(residual, expected, atol=1e-6)
    assert residual.shape == v.shape


def test_compute_residual_batched() -> None:
    """Verify compute_residual works on batched multi-dimensional tensors."""
    v = torch.randn(16, 64)
    v_hat = torch.randn(16, 64)

    residual = compute_residual(v, v_hat)
    assert torch.allclose(residual, v - v_hat, atol=1e-6)
    assert residual.shape == (16, 64)


def test_extract_qjl_signs_strictly_plus_minus_one() -> None:
    """Verify that extract_qjl_signs values are strictly in {-1, +1}, mapping 0 to +1."""
    # Test tensor containing positive, negative, zero, and boundary values
    residual = torch.tensor([
        2.5,
        -1.3,
        0.0,
        -0.0,
        1e-8,
        -1e-8,
        0.0,
        -5.0,
        10.0,
    ])

    signs = extract_qjl_signs(residual)

    # Check zero values mapped specifically to +1
    assert signs[2].item() == 1.0
    assert signs[3].item() == 1.0
    assert signs[6].item() == 1.0

    # Check positive mapped to +1, negative to -1
    assert signs[0].item() == 1.0
    assert signs[1].item() == -1.0
    assert signs[7].item() == -1.0
    assert signs[8].item() == 1.0

    # Check all elements strictly belong to {-1, +1}
    is_valid = (signs == 1.0) | (signs == -1.0)
    assert torch.all(is_valid), f"Found invalid sign values: {signs[~is_valid]}"

    unique_vals = set(torch.unique(signs).tolist())
    assert unique_vals.issubset({-1.0, 1.0})


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_extract_qjl_signs_random_distribution(dtype: torch.dtype) -> None:
    """Verify sign extraction on large random tensors with zeros."""
    torch.manual_seed(42)
    residual = torch.randn(100, 128, dtype=dtype)
    # Inject exact zeros to test zero handling at scale
    residual[residual.abs() < 0.2] = 0.0

    signs = extract_qjl_signs(residual)

    assert signs.shape == residual.shape
    assert signs.dtype == dtype
    assert torch.all((signs == 1.0) | (signs == -1.0))


def test_estimate_dot_product_known_values() -> None:
    """Verify estimate_dot_product against manually computed inner product."""
    query = torch.tensor([1.0, 2.0, -1.0])
    v_hat = torch.tensor([3.0, 1.0, 2.0])
    signs = torch.tensor([1.0, -1.0, 1.0])
    scale = 0.5

    # dot(query, v_hat) = 1*3 + 2*1 + (-1)*2 = 3 + 2 - 2 = 3.0
    # dot(query, signs) = 1*1 + 2*(-1) + (-1)*1 = 1 - 2 - 1 = -2.0
    # expected = 3.0 + 0.5 * (-2.0) = 2.0
    est = estimate_dot_product(query, v_hat, signs, scale)

    assert torch.allclose(est, torch.tensor(2.0), atol=1e-6)


@pytest.mark.parametrize("dim", [32, 64, 128])
def test_corrected_inner_product_lower_mse(dim: int) -> None:
    """Verify the corrected inner product has lower MSE against exact dot product than v_hat alone."""
    torch.manual_seed(42)
    n_samples = 2000

    # Generate random original vectors v and random queries
    v = torch.randn(n_samples, dim, dtype=torch.float32)
    queries = torch.randn(n_samples, dim, dtype=torch.float32)

    # Quantize v using polar quantization on coordinate pairs (dim must be even)
    x1 = v[:, 0::2]
    x2 = v[:, 1::2]
    r, theta = cartesian_to_polar(x1, x2)
    bits = 2
    indices = quantize_angles(theta, bits=bits)
    theta_hat = dequantize_angles(indices, bits=bits, dtype=v.dtype)
    x1_hat, x2_hat = polar_to_cartesian(r, theta_hat)

    v_hat = torch.empty_like(v)
    v_hat[:, 0::2] = x1_hat
    v_hat[:, 1::2] = x2_hat

    # Compute residual and QJL 1-bit signs
    residual = compute_residual(v, v_hat)
    signs = extract_qjl_signs(residual)

    # Scale factor: optimal 1-bit projection scale E[|residual|]
    scale = torch.mean(torch.abs(residual)).item()

    # Exact dot products
    exact_dot = torch.sum(queries * v, dim=-1)

    # Uncorrected dot product using v_hat alone
    uncorrected_dot = torch.sum(queries * v_hat, dim=-1)

    # Corrected dot product with 1-bit QJL error correction
    corrected_dot = estimate_dot_product(queries, v_hat, signs, scale)

    # Compute Mean Squared Errors
    mse_uncorrected = torch.mean((exact_dot - uncorrected_dot) ** 2).item()
    mse_corrected = torch.mean((exact_dot - corrected_dot) ** 2).item()

    # Verify that QJL correction strictly reduces MSE
    assert mse_corrected < mse_uncorrected, (
        f"Expected mse_corrected ({mse_corrected}) < "
        f"mse_uncorrected ({mse_uncorrected}) for dim={dim}"
    )

    # Check substantial error reduction (typically > 30% reduction)
    assert mse_corrected < 0.75 * mse_uncorrected, (
        f"Expected significant MSE reduction: corrected={mse_corrected}, "
        f"uncorrected={mse_uncorrected}"
    )
