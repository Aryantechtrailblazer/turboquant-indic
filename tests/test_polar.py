"""Tests for polar quantization primitives in TurboQuant."""

import math
import pytest
import torch

from turboquant.polar import (
    cartesian_to_polar,
    dequantize_angles,
    polar_to_cartesian,
    quantize_angles,
)


def test_cartesian_to_polar_known_values() -> None:
    """Verify cartesian_to_polar against known trigonometric points."""
    x1 = torch.tensor([1.0, 0.0, -1.0, 0.0, 0.0, 3.0])
    x2 = torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0, 4.0])

    r, theta = cartesian_to_polar(x1, x2)

    expected_r = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 5.0])
    expected_theta = torch.tensor([
        0.0,
        math.pi / 2,
        math.pi,
        -math.pi / 2,
        0.0,
        math.atan2(4.0, 3.0),
    ])

    assert torch.allclose(r, expected_r, atol=1e-6)
    assert torch.allclose(theta, expected_theta, atol=1e-6)


@pytest.mark.parametrize("bits", [1, 2, 3, 4, 8])
def test_angle_binning_range(bits: int) -> None:
    """Verify that angle binning maps strictly within [0, 2^bits - 1]."""
    num_bins = 1 << bits

    # Include random angles and boundary points
    random_angles = (torch.rand(1000) * 4.0 - 2.0) * math.pi
    boundary_angles = torch.tensor([
        0.0,
        -math.pi,
        math.pi,
        2 * math.pi,
        -2 * math.pi,
        1e-7,
        -1e-7,
        2 * math.pi - 1e-7,
    ])
    test_angles = torch.cat([random_angles, boundary_angles])

    indices = quantize_angles(test_angles, bits)

    assert indices.dtype == torch.long
    assert (indices >= 0).all()
    assert (indices <= num_bins - 1).all()


def test_reconstruction_without_quantization() -> None:
    """Verify that polar_to_cartesian is the exact inverse of cartesian_to_polar."""
    x1 = torch.randn(500, dtype=torch.float32)
    x2 = torch.randn(500, dtype=torch.float32)

    r, theta = cartesian_to_polar(x1, x2)
    x1_hat, x2_hat = polar_to_cartesian(r, theta)

    assert torch.allclose(x1, x1_hat, atol=1e-5)
    assert torch.allclose(x2, x2_hat, atol=1e-5)


@pytest.mark.parametrize(
    "bit_budgets",
    [
        (2, 4),
        (2, 3, 4, 6),
    ],
)
def test_reconstruction_error_decreases_with_bits(bit_budgets: tuple) -> None:
    """Verify reconstruction error decreases monotonically as bit budget increases."""
    torch.manual_seed(42)
    n_samples = 10000

    x1 = torch.randn(n_samples, dtype=torch.float32)
    x2 = torch.randn(n_samples, dtype=torch.float32)

    r, theta = cartesian_to_polar(x1, x2)

    mses = []
    for bits in bit_budgets:
        indices = quantize_angles(theta, bits)
        theta_quant = dequantize_angles(indices, bits, dtype=theta.dtype)
        x1_hat, x2_hat = polar_to_cartesian(r, theta_quant)

        mse = torch.mean((x1 - x1_hat) ** 2 + (x2 - x2_hat) ** 2).item()
        mses.append(mse)

    # Check strictly monotonic decrease
    for i in range(len(mses) - 1):
        assert mses[i] > mses[i + 1], (
            f"Expected MSE({bit_budgets[i]}-bit)={mses[i]} > "
            f"MSE({bit_budgets[i+1]}-bit)={mses[i+1]}"
        )

    # 4-bit MSE should be significantly lower than 2-bit MSE (at least ~8x lower)
    if 2 in bit_budgets and 4 in bit_budgets:
        mse_2bit = mses[bit_budgets.index(2)]
        mse_4bit = mses[bit_budgets.index(4)]
        assert mse_2bit > 8.0 * mse_4bit


def test_invalid_bits() -> None:
    """Verify that bits < 1 raises ValueError."""
    theta = torch.tensor([0.0, 1.0])
    with pytest.raises(ValueError):
        quantize_angles(theta, 0)
    with pytest.raises(ValueError):
        quantize_angles(theta, -1)
    with pytest.raises(ValueError):
        dequantize_angles(torch.tensor([0, 1]), 0)
