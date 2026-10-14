"""End-to-end integration tests for the full TurboQuant pipeline."""

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
from turboquant.rotation import random_orthogonal_matrix


def test_turboquant_pipeline() -> None:
    """Verify the end-to-end TurboQuant pipeline (Rotation + PolarQuant + QJL)."""
    torch.manual_seed(42)
    batch_size = 100
    dim = 128

    # 1. Generate random vectors for queries and keys of dimension d=128 across 100 samples
    q = torch.randn(batch_size, dim, dtype=torch.float32)
    k = torch.randn(batch_size, dim, dtype=torch.float32)

    # 2. Compute exact ground-truth dot products: dot(q, k)
    exact_dot = torch.sum(q * k, dim=-1)

    # 3. Apply the pipeline to keys and queries:
    # a. Rotate k using random_orthogonal_matrix(d)
    rot = random_orthogonal_matrix(dim=dim, dtype=torch.float32)
    k_rot = k @ rot.T

    # b. Rotate q using the same orthogonal matrix
    q_rot = q @ rot.T

    # Verify that rotation preserves inner products: dot(q_rot, k_rot) == dot(q, k)
    assert torch.allclose(torch.sum(q_rot * k_rot, dim=-1), exact_dot, atol=1e-4)

    # c. Compress and dequantize k using polar quantization (2-bit)
    x1 = k_rot[:, 0::2]
    x2 = k_rot[:, 1::2]
    r, theta = cartesian_to_polar(x1, x2)
    bits = 2
    indices = quantize_angles(theta, bits=bits)
    theta_hat = dequantize_angles(indices, bits=bits, dtype=k_rot.dtype)
    x1_hat, x2_hat = polar_to_cartesian(r, theta_hat)

    k_hat = torch.empty_like(k_rot)
    k_hat[:, 0::2] = x1_hat
    k_hat[:, 1::2] = x2_hat

    # d. Compute residual error (k_rot - k_hat) and extract signs via extract_qjl_signs
    residual = compute_residual(k_rot, k_hat)
    signs = extract_qjl_signs(residual)

    # Scale factor for 1-bit QJL correction
    scale = torch.mean(torch.abs(residual)).item()

    # 4. Calculate two approximations:
    # - Baseline: dot(q_rot, k_hat)
    baseline_dot = torch.sum(q_rot * k_hat, dim=-1)

    # - Corrected: estimate_dot_product(q_rot, k_hat, signs, scale)
    corrected_dot = estimate_dot_product(q_rot, k_hat, signs, scale)

    # 5. Compute the Mean Squared Error (MSE) for both approximations against exact dot product
    mse_baseline = torch.mean((exact_dot - baseline_dot) ** 2).item()
    mse_corrected = torch.mean((exact_dot - corrected_dot) ** 2).item()

    # 6. Assert that MSE_corrected < MSE_baseline
    assert mse_corrected < mse_baseline, (
        f"Expected MSE_corrected ({mse_corrected:.4f}) < MSE_baseline ({mse_baseline:.4f})"
    )
