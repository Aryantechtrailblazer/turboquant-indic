"""Unit tests for TurboQuant indexing and two-stage retrieval module."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from turboquant.index import SearchResult, TurboQuantIndex
from turboquant.rotation import random_orthogonal_matrix


def _normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize vectors along the last dimension."""
    return F.normalize(x, p=2, dim=-1)


def test_init_defaults_and_attributes() -> None:
    """Verify initialization defaults, attribute types, and error handling."""
    index = TurboQuantIndex()
    assert index.d == 768
    assert index.k_candidates == 50
    assert index.use_qjl is False
    assert index.C is None
    assert index.C_hat is None
    assert index.raw_corpus is None
    assert index.compressed_corpus is None
    assert index.packed_corpus is None
    assert len(index) == 0

    with pytest.raises(ValueError, match="Dimension d must be positive"):
        TurboQuantIndex(d=0)

    with pytest.raises(ValueError, match="k_candidates must be positive"):
        TurboQuantIndex(k_candidates=-5)


def test_search_output_shapes() -> None:
    """📐 Verify correct output shapes for search(): [1, top_k] for both scores and indices."""
    torch.manual_seed(42)
    d = 768
    num_docs = 100
    top_k = 10

    index = TurboQuantIndex(d=d, k_candidates=50)

    # Normalized corpus embeddings of shape [N, d]
    corpus = _normalize(torch.randn(num_docs, d, dtype=torch.float32))
    index.add(corpus)

    assert len(index) == num_docs
    assert index.C is not None and index.C.shape == (num_docs, d)
    assert index.C_hat is not None and index.C_hat.shape == (num_docs, d)

    # 1. 2D query of shape [1, d]
    query_2d = _normalize(torch.randn(1, d, dtype=torch.float32))
    scores, indices = index.search(query_2d, top_k=top_k)

    assert isinstance(scores, torch.Tensor)
    assert isinstance(indices, torch.Tensor)
    assert scores.shape == (1, top_k), f"Expected scores shape (1, {top_k}), got {scores.shape}"
    assert indices.shape == (1, top_k), f"Expected indices shape (1, {top_k}), got {indices.shape}"
    assert scores.dtype == torch.float32
    assert indices.dtype == torch.long

    # Verify SearchResult attribute access
    result = index.search(query_2d, top_k=top_k)
    assert isinstance(result, tuple)
    assert isinstance(result, SearchResult)
    assert torch.equal(result.scores, scores)
    assert torch.equal(result.indices, indices)
    assert torch.equal(result.values, scores)
    assert torch.equal(result.doc_ids, indices)

    # 2. 1D query of shape [d]
    query_1d = _normalize(torch.randn(d, dtype=torch.float32))
    scores_1d, indices_1d = index.search(query_1d, top_k=top_k)
    assert scores_1d.shape == (1, top_k)
    assert indices_1d.shape == (1, top_k)

    # 3. Different top_k values
    for k in [1, 5, 25]:
        s, idxs = index.search(query_2d, top_k=k)
        assert s.shape == (1, k)
        assert idxs.shape == (1, k)


def test_candidate_pool_containment() -> None:
    """🎯 Verify Stage 1 retrieves candidate indices within valid bounds [0, N-1]."""
    torch.manual_seed(123)
    d = 128
    num_docs = 200
    k_candidates = 50

    index = TurboQuantIndex(d=d, k_candidates=k_candidates)
    corpus = _normalize(torch.randn(num_docs, d, dtype=torch.float32))
    index.add(corpus)

    for _ in range(5):
        query = _normalize(torch.randn(1, d, dtype=torch.float32))

        # Directly inspect Stage 1 coarse filter candidates
        candidate_scores, candidate_indices = index.coarse_filter(query, k=k_candidates)

        assert candidate_indices.shape == (1, k_candidates)
        assert candidate_scores.shape == (1, k_candidates)

        # Confirm all Stage 1 candidate indices are within [0, N-1]
        assert (candidate_indices >= 0).all(), "Candidate indices contain negative values"
        assert (candidate_indices < num_docs).all(), f"Candidate indices exceed corpus size {num_docs}"

        # Verify no duplicate candidate indices within the retrieved pool
        unique_candidates = torch.unique(candidate_indices[0])
        assert unique_candidates.numel() == k_candidates, "Candidate indices must be unique"

        # Confirm Stage 2 final indices are within valid bounds and contained in candidate pool
        top_k = 10
        _, final_indices = index.search(query, top_k=top_k)
        assert (final_indices >= 0).all()
        assert (final_indices < num_docs).all()
        assert torch.isin(final_indices, candidate_indices).all(), (
            "Final indices must be contained in Stage 1 candidate pool"
        )


def test_candidate_pool_containment_small_corpus() -> None:
    """🎯 Verify Stage 1 and search handle corpus size smaller than k_candidates."""
    torch.manual_seed(42)
    d = 64
    num_docs = 15
    k_candidates = 50

    index = TurboQuantIndex(d=d, k_candidates=k_candidates)
    corpus = _normalize(torch.randn(num_docs, d, dtype=torch.float32))
    index.add(corpus)

    query = _normalize(torch.randn(1, d, dtype=torch.float32))

    # Stage 1 should retrieve at most num_docs candidates
    _, candidate_indices = index.coarse_filter(query)
    assert candidate_indices.shape == (1, num_docs)
    assert (candidate_indices >= 0).all()
    assert (candidate_indices < num_docs).all()

    # Stage 2 search with top_k = 10
    top_k = 10
    scores, indices = index.search(query, top_k=top_k)
    assert scores.shape == (1, top_k)
    assert indices.shape == (1, top_k)
    assert (indices >= 0).all()
    assert (indices < num_docs).all()


def test_ranking_consistency_orthogonal_vectors() -> None:
    """📈 Test with synthetic orthogonal vectors to verify true matching vector achieves rank #1."""
    torch.manual_seed(999)
    dim = 64
    k_candidates = 25
    top_k = 5

    # Generate synthetic orthogonal matrix: all rows are unit vectors and mutually orthogonal
    orthogonal_corpus = random_orthogonal_matrix(dim=dim, dtype=torch.float32)

    index = TurboQuantIndex(d=dim, k_candidates=k_candidates)
    index.add(orthogonal_corpus)

    # Test multiple target vectors from the orthogonal set
    target_indices = [0, 12, 35, 50, 63]
    for target_id in target_indices:
        query = orthogonal_corpus[target_id : target_id + 1]

        scores, indices = index.search(query, top_k=top_k)

        # 1. The true matching vector must achieve rank #1 (index 0)
        assert indices[0, 0].item() == target_id, (
            f"Expected rank #1 index {target_id}, got {indices[0, 0].item()}"
        )

        # 2. Perfect match dot product must equal 1.0 (float32 tolerance)
        assert torch.isclose(
            scores[0, 0], torch.tensor(1.0, dtype=torch.float32), atol=1e-5
        ), f"Expected top score 1.0, got {scores[0, 0].item()}"

        # 3. All non-matching orthogonal candidates must have exact dot product 0.0
        for rank in range(1, top_k):
            assert torch.isclose(
                scores[0, rank], torch.tensor(0.0, dtype=torch.float32), atol=1e-5
            ), f"Expected orthogonal non-match score 0.0 at rank {rank}, got {scores[0, rank].item()}"


def test_ranking_consistency_standard_basis() -> None:
    """📈 Test ranking consistency with canonical standard basis vectors."""
    dim = 32
    corpus = torch.eye(dim, dtype=torch.float32)

    index = TurboQuantIndex(d=dim, k_candidates=16)
    index.add(corpus)

    for target_id in [0, 7, 15, 31]:
        query = corpus[target_id : target_id + 1]
        scores, indices = index.search(query, top_k=3)

        assert indices[0, 0].item() == target_id
        assert torch.isclose(scores[0, 0], torch.tensor(1.0, dtype=torch.float32), atol=1e-5)
        for rank in range(1, 3):
            assert torch.isclose(scores[0, rank], torch.tensor(0.0, dtype=torch.float32), atol=1e-5)


def test_ranking_consistency_with_qjl() -> None:
    """Verify ranking consistency with optional QJL error correction enabled."""
    torch.manual_seed(777)
    dim = 64
    orthogonal_corpus = random_orthogonal_matrix(dim=dim, dtype=torch.float32)

    index = TurboQuantIndex(d=dim, k_candidates=20, use_qjl=True)
    index.add(orthogonal_corpus)

    for target_id in [1, 20, 45]:
        query = orthogonal_corpus[target_id : target_id + 1]
        scores, indices = index.search(query, top_k=5)

        assert indices[0, 0].item() == target_id
        assert torch.isclose(scores[0, 0], torch.tensor(1.0, dtype=torch.float32), atol=1e-5)


def test_incremental_add() -> None:
    """Verify that multiple add() calls correctly accumulate raw and compressed corpora."""
    d = 64
    index = TurboQuantIndex(d=d, k_candidates=10)

    part1 = _normalize(torch.randn(30, d, dtype=torch.float32))
    part2 = _normalize(torch.randn(20, d, dtype=torch.float32))

    index.add(part1)
    assert len(index) == 30

    index.add(part2)
    assert len(index) == 50
    assert index.C is not None and index.C.shape == (50, d)
    assert index.C_hat is not None and index.C_hat.shape == (50, d)

    query = _normalize(torch.randn(1, d, dtype=torch.float32))
    scores, indices = index.search(query, top_k=5)
    assert scores.shape == (1, 5)
    assert indices.shape == (1, 5)


def test_input_validation_and_errors() -> None:
    """Verify input validation across add, coarse_filter, and search."""
    d = 64
    index = TurboQuantIndex(d=d, k_candidates=10)

    # Search on empty index
    query = torch.randn(1, d)
    with pytest.raises(RuntimeError, match="Index is empty"):
        index.search(query)

    with pytest.raises(RuntimeError, match="Index is empty"):
        index.coarse_filter(query)

    # Add wrong dimension
    with pytest.raises(ValueError, match="Embeddings dimension 65 does not match"):
        index.add(torch.randn(10, d + 1))

    # Add 3D tensor
    with pytest.raises(ValueError, match="Expected 1D or 2D embeddings tensor"):
        index.add(torch.randn(2, 5, d))

    # Add valid embeddings
    index.add(torch.randn(20, d))

    # Search wrong dimension
    with pytest.raises(ValueError, match="Query dimension 63 does not match"):
        index.search(torch.randn(1, d - 1))

    # Search 3D tensor
    with pytest.raises(ValueError, match="Expected 1D or 2D query tensor"):
        index.search(torch.randn(2, 3, d))

    # Search invalid top_k
    with pytest.raises(ValueError, match="top_k must be positive"):
        index.search(torch.randn(1, d), top_k=0)

    # Dimension not divisible by 4
    with pytest.raises(ValueError, match="divisible by 4"):
        TurboQuantIndex(d=65)


def test_packed_corpus_storage() -> None:
    """💾 Verify index.packed_corpus.dtype == torch.uint8 and shape is [N, d // 4] (16x memory reduction)."""
    torch.manual_seed(42)
    d = 768
    num_docs = 100

    index = TurboQuantIndex(d=d, k_candidates=50)
    assert index.packed_corpus is None

    corpus = _normalize(torch.randn(num_docs, d, dtype=torch.float32))
    index.add(corpus)

    assert index.packed_corpus is not None
    assert index.packed_corpus.dtype == torch.uint8
    assert index.packed_corpus.shape == (num_docs, d // 4)
    assert index.packed_corpus.shape == (num_docs, 192)

    # Physical memory footprint: exactly 192 bytes per 768-dim vector
    packed_bytes = index.packed_corpus.element_size() * index.packed_corpus.nelement()
    assert packed_bytes == num_docs * 192

    raw_bytes = index.C.element_size() * index.C.nelement()
    assert raw_bytes == num_docs * 768 * 4
    assert raw_bytes / packed_bytes == 16.0


def test_round_trip_search_consistency() -> None:
    """🧪 Verify round-trip search consistency: Stage 1 candidate filtering and Stage 2 re-ranking."""
    torch.manual_seed(12345)
    d = 128
    num_docs = 150
    k_candidates = 30
    top_k = 10

    index = TurboQuantIndex(d=d, k_candidates=k_candidates)
    corpus = _normalize(torch.randn(num_docs, d, dtype=torch.float32))
    index.add(corpus)

    for _ in range(5):
        query = _normalize(torch.randn(1, d, dtype=torch.float32))

        # Stage 1 coarse filter
        candidate_scores, candidate_indices = index.coarse_filter(query, k=k_candidates)
        assert candidate_indices.shape == (1, k_candidates)
        assert candidate_scores.shape == (1, k_candidates)

        # Stage 2 search
        search_res = index.search(query, top_k=top_k)
        final_scores = search_res.scores
        final_indices = search_res.indices

        assert final_scores.shape == (1, top_k)
        assert final_indices.shape == (1, top_k)

        # Verify all final indices were selected from Stage 1 candidates
        assert torch.isin(final_indices, candidate_indices).all()

        # Verify final scores match exact dot products with raw corpus C
        expected_exact_scores = torch.sum(query * index.C[final_indices[0]], dim=-1, keepdim=True).T
        assert torch.allclose(final_scores, expected_exact_scores, atol=1e-5)

        # Verify scores are sorted in descending order
        diffs = final_scores[0, :-1] - final_scores[0, 1:]
        assert (diffs >= -1e-6).all(), "Scores must be in descending order"


def test_raw_vs_packed_memory_footprint() -> None:
    """Verify TurboQuantIndex achieves the theoretical 16x memory reduction in physical RAM."""
    torch.manual_seed(42)
    d = 768
    num_vectors = 1000

    # 1. Initialize TurboQuantIndex
    index = TurboQuantIndex(d=d)

    # 2. Generate synthetic corpus of N = 1000 normalized vectors
    embeddings = _normalize(torch.randn(num_vectors, d, dtype=torch.float32))

    # 3. Add vectors to index
    index.add(embeddings)

    # 4. Measure physical bytes
    assert index.raw_corpus is not None
    assert index.packed_corpus is not None

    raw_bytes = index.raw_corpus.numel() * index.raw_corpus.element_size()
    packed_bytes = index.packed_corpus.numel() * index.packed_corpus.element_size()
    compression_ratio = raw_bytes / packed_bytes

    # 5. Assertions
    assert raw_bytes == 1000 * 768 * 4, f"Expected 3,072,000 bytes, got {raw_bytes}"
    assert packed_bytes == 1000 * 192 * 1, f"Expected 192,000 bytes, got {packed_bytes}"
    assert compression_ratio == 16.0, f"Expected 16.0x compression ratio, got {compression_ratio}"

    # 6. Formatted summary print
    print("\n" + "=" * 60)
    print(" TurboQuant Physical Memory Footprint Verification")
    print("=" * 60)
    print(f" Corpus vectors        : {num_vectors}")
    print(f" Vector dimension (d)  : {d}")
    print(f" Raw FP32 Memory       : {raw_bytes:,} bytes ({raw_bytes / 1024:.2f} KB / {raw_bytes / (1024**2):.2f} MB)")
    print(f" Packed 2-bit Memory   : {packed_bytes:,} bytes ({packed_bytes / 1024:.2f} KB / {packed_bytes / (1024**2):.2f} MB)")
    print(f" Compression Ratio     : {compression_ratio:.1f}x reduction")
    print("=" * 60)


def test_lut_vs_unpacked_scoring_equivalence() -> None:
    """🧪 Verify that LUT-based coarse dot products match dot products via unpack_2bit within 1e-5."""
    from turboquant.packing import unpack_2bit

    torch.manual_seed(42)
    d = 768
    num_docs = 150
    k_candidates = 50

    for use_qjl in [False, True]:
        index = TurboQuantIndex(d=d, k_candidates=k_candidates, use_qjl=use_qjl)
        corpus = _normalize(torch.randn(num_docs, d, dtype=torch.float32))
        index.add(corpus)

        for _ in range(5):
            query = _normalize(torch.randn(1, d, dtype=torch.float32))

            # 1. Unpack-based ground truth dot products
            unpacked_codes = unpack_2bit(index.packed_corpus, target_dtype=torch.long)
            coarse_corpus = index.centroids[unpacked_codes]
            if use_qjl and index.qjl_correction is not None:
                coarse_corpus = coarse_corpus + index.qjl_correction
            expected_scores = torch.matmul(query, coarse_corpus.T)

            # 2. LUT-based coarse scores from index.coarse_filter()
            lut_topk_scores, lut_topk_indices = index.coarse_filter(query, k=num_docs)

            # Map expected scores to the retrieved ranks
            expected_sorted_scores = torch.gather(expected_scores, dim=-1, index=lut_topk_indices)

            # Assert numerical equivalence within 1e-5
            diff = torch.max(torch.abs(lut_topk_scores - expected_sorted_scores)).item()
            assert diff < 1e-5, f"Expected difference < 1e-5, got max_diff = {diff} (use_qjl={use_qjl})"
            assert torch.allclose(lut_topk_scores, expected_sorted_scores, atol=1e-5)
