"""Tests for rotation primitives in TurboQuant."""

import pytest
import torch

from turboquant.rotation import random_orthogonal_matrix


@pytest.mark.parametrize("dim", [2, 8, 32, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_orthogonality(dim: int, dtype: torch.dtype) -> None:
    """Verify that Q.T @ Q and Q @ Q.T are approximately the identity matrix."""
    q = random_orthogonal_matrix(dim=dim, dtype=dtype)

    assert q.shape == (dim, dim)
    assert q.dtype == dtype

    identity = torch.eye(dim, dtype=dtype)
    atol = 1e-5 if dtype == torch.float32 else 1e-10

    # Check Q.T @ Q == I
    assert torch.allclose(q.T @ q, identity, atol=atol)
    # Check Q @ Q.T == I
    assert torch.allclose(q @ q.T, identity, atol=atol)


@pytest.mark.parametrize("dim", [2, 16, 64])
def test_norm_preservation(dim: int) -> None:
    """Verify norm preservation: ||Q @ v|| == ||v|| for random test vectors."""
    q = random_orthogonal_matrix(dim=dim, dtype=torch.float32)

    # 1D vector
    v = torch.randn(dim, dtype=torch.float32)
    rotated_v = q @ v

    orig_norm = torch.linalg.vector_norm(v)
    rotated_norm = torch.linalg.vector_norm(rotated_v)

    assert torch.allclose(orig_norm, rotated_norm, atol=1e-5)

    # Batched vectors (batch_size, dim)
    batch_v = torch.randn(10, dim, dtype=torch.float32)
    # Applying rotation to each vector in batch: (batch_v @ Q.T) or (Q @ batch_v.T).T
    rotated_batch = batch_v @ q.T

    batch_orig_norms = torch.linalg.vector_norm(batch_v, dim=-1)
    batch_rotated_norms = torch.linalg.vector_norm(rotated_batch, dim=-1)

    assert torch.allclose(batch_orig_norms, batch_rotated_norms, atol=1e-5)


def test_invalid_dimension() -> None:
    """Ensure ValueError is raised for non-positive dimensions."""
    with pytest.raises(ValueError):
        random_orthogonal_matrix(0)
    with pytest.raises(ValueError):
        random_orthogonal_matrix(-5)
