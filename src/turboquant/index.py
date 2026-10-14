"""Indexing and two-stage retrieval module for TurboQuant with Lookup Table (LUT) scoring."""

from __future__ import annotations

from typing import Optional
import torch

from turboquant.packing import pack_2bit, unpack_2bit
from turboquant.polar import (
    cartesian_to_polar,
    dequantize_angles,
    polar_to_cartesian,
    quantize_angles,
)
from turboquant.qjl import (
    compute_residual,
    extract_qjl_signs,
)

# Standard 2-bit quantization centroids for range [-1.0, 1.0]
DEFAULT_CENTROIDS = torch.tensor([-0.75, -0.25, 0.25, 0.75], dtype=torch.float32)


def _build_byte_to_centroids_table(centroids: torch.Tensor) -> torch.Tensor:
    """Construct byte-to-centroid mapping table of shape [256, 4].

    Maps each byte value b in [0, 255] to its four corresponding centroid values:
        codes = [(b >> 6) & 3, (b >> 4) & 3, (b >> 2) & 3, b & 3]
        BYTE_TO_CENTROIDS[b] = centroids[codes]
    """
    bytes_range = torch.arange(256, dtype=torch.long)
    c0 = (bytes_range >> 6) & 3
    c1 = (bytes_range >> 4) & 3
    c2 = (bytes_range >> 2) & 3
    c3 = bytes_range & 3
    codes_all = torch.stack([c0, c1, c2, c3], dim=-1)  # [256, 4]
    return centroids[codes_all].to(dtype=torch.float32)


# Precomputed static mapping table for default centroids
BYTE_TO_CENTROIDS = _build_byte_to_centroids_table(DEFAULT_CENTROIDS)


class SearchResult(tuple):
    """Container for search results supporting tuple unpacking and attribute access.

    Unpacks as ``(scores, indices)``, matching PyTorch ``torch.topk``.

    Attributes:
        scores: Relevance / dot product scores tensor of shape [B, top_k].
        indices: Document IDs tensor of shape [B, top_k].
    """

    def __new__(cls, scores: torch.Tensor, indices: torch.Tensor) -> SearchResult:
        return super().__new__(cls, (scores, indices))

    @property
    def scores(self) -> torch.Tensor:
        """Similarity / dot product scores tensor."""
        return self[0]

    @property
    def values(self) -> torch.Tensor:
        """Alias for scores, matching torch.topk values."""
        return self[0]

    @property
    def indices(self) -> torch.Tensor:
        """Document ID indices tensor."""
        return self[1]

    @property
    def doc_ids(self) -> torch.Tensor:
        """Alias for document ID indices."""
        return self[1]

    def __repr__(self) -> str:
        return f"SearchResult(scores={self.scores}, indices={self.indices})"


def quantize_embeddings_2bit(
    embeddings: torch.Tensor,
    use_qjl: bool = False,
) -> torch.Tensor:
    """Quantize normalized embeddings using 2-bit polar centroid quantization.

    Args:
        embeddings: Tensor of shape [N, d] with normalized embeddings.
        use_qjl: Whether to apply 1-bit QJL residual error correction.

    Returns:
        Tensor of shape [N, d] containing reconstructed quantized embeddings.
    """
    if embeddings.shape[0] == 0:
        return torch.empty_like(embeddings)

    d = embeddings.shape[-1]
    if d % 2 == 0:
        x1 = embeddings[:, 0::2]
        x2 = embeddings[:, 1::2]
        r, theta = cartesian_to_polar(x1, x2)
        indices = quantize_angles(theta, bits=2)
        theta_hat = dequantize_angles(indices, bits=2, dtype=embeddings.dtype)
        x1_hat, x2_hat = polar_to_cartesian(r, theta_hat)

        c_hat = torch.empty_like(embeddings)
        c_hat[:, 0::2] = x1_hat
        c_hat[:, 1::2] = x2_hat
    else:
        even_dim = d - 1
        x1 = embeddings[:, :even_dim:2]
        x2 = embeddings[:, 1:even_dim:2]
        r, theta = cartesian_to_polar(x1, x2)
        indices = quantize_angles(theta, bits=2)
        theta_hat = dequantize_angles(indices, bits=2, dtype=embeddings.dtype)
        x1_hat, x2_hat = polar_to_cartesian(r, theta_hat)

        c_hat = torch.empty_like(embeddings)
        c_hat[:, :even_dim:2] = x1_hat
        c_hat[:, 1:even_dim:2] = x2_hat
        c_hat[:, -1] = embeddings[:, -1]

    if use_qjl:
        residual = compute_residual(embeddings, c_hat)
        signs = extract_qjl_signs(residual)
        scale = torch.mean(torch.abs(residual)).item()
        c_hat = c_hat + scale * signs

    return c_hat


class TurboQuantIndex:
    """Two-stage vector search index using physical 2-bit packed storage and LUT scoring.

    Holds internal buffers:
        - self.packed_corpus: torch.uint8 tensor of shape [N, d // 4] storing
          physically packed 2-bit centroid indices (16x memory reduction).
        - self.C: raw full-precision corpus (float32) of shape [N, d] for Stage 2 re-ranking.
        - self.byte_to_centroids: precomputed mapping table of shape [256, 4].

    Search operates in two stages:
        1. Stage 1 (Coarse Filter): Computes a precomputed query Lookup Table (LUT) of shape [M, 256]
           and scores packed documents directly via vectorized gather without on-the-fly unpacking.
        2. Stage 2 (Exact Re-ranking): Gathers full-precision candidates from raw corpus self.C,
           computes exact dot products against query, and sorts to return top-k results.
    """

    def __init__(
        self,
        d: int = 768,
        k_candidates: int = 50,
        use_qjl: bool = False,
        centroids: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize the TurboQuant index.

        Args:
            d: Dimension of the vector embeddings (default: 768, must be divisible by 4).
            k_candidates: Number of candidates K to retrieve in Stage 1 coarse filter (default: 50).
            use_qjl: Whether to include 1-bit QJL error correction in coarse filtering (default: False).
            centroids: Optional custom 4-element 1D tensor of centroid values (default: [-0.75, -0.25, 0.25, 0.75]).
        """
        if d <= 0:
            raise ValueError(f"Dimension d must be positive, got {d}")
        if d % 4 != 0:
            raise ValueError(f"Dimension d must be divisible by 4 for 2-bit packing, got {d}")
        if k_candidates <= 0:
            raise ValueError(f"k_candidates must be positive, got {k_candidates}")

        self.d = d
        self.k_candidates = k_candidates
        self.use_qjl = use_qjl

        # Centroid lookup values and byte-to-centroid table
        if centroids is None:
            self.centroids = DEFAULT_CENTROIDS.clone()
            self.byte_to_centroids = BYTE_TO_CENTROIDS.clone()
        else:
            self.centroids = centroids.to(dtype=torch.float32).clone()
            self.byte_to_centroids = _build_byte_to_centroids_table(self.centroids)
        self.boundaries = (self.centroids[:-1] + self.centroids[1:]) / 2.0

        # Internal buffers:
        # self.packed_corpus: [N, d // 4] uint8 physically packed 2-bit tensor
        # self.C: [N, d] float32 raw uncompressed corpus
        self.packed_corpus: Optional[torch.Tensor] = None
        self.C: Optional[torch.Tensor] = None
        self.qjl_correction: Optional[torch.Tensor] = None

    @property
    def raw_corpus(self) -> Optional[torch.Tensor]:
        """Raw full-precision corpus C."""
        return self.C

    @property
    def compressed_corpus(self) -> Optional[torch.Tensor]:
        """Compressed packed corpus representations."""
        return self.packed_corpus

    @property
    def C_hat(self) -> Optional[torch.Tensor]:
        """Reconstructed coarse corpus representations (derived on-the-fly from packed_corpus)."""
        if self.packed_corpus is None:
            return None
        unpacked_codes = unpack_2bit(self.packed_corpus, target_dtype=torch.long)
        centroids_dev = self.centroids.to(self.packed_corpus.device)
        coarse = centroids_dev[unpacked_codes]
        if self.use_qjl and self.qjl_correction is not None:
            coarse = coarse + self.qjl_correction
        return coarse

    def __len__(self) -> int:
        """Return the number of vectors stored in the index."""
        return 0 if self.packed_corpus is None else self.packed_corpus.shape[0]

    def _build_query_lut(self, query: torch.Tensor) -> torch.Tensor:
        """Build precomputed query Lookup Table (LUT) for fast Asymmetric Distance Computation.

        For a query vector of shape [d] (or [1, d]), reshapes to [M, 4] where M = d // 4.
        Computes LUT = torch.matmul(q_reshaped, self.byte_to_centroids.T) of shape [M, 256].

        Args:
            query: Query tensor of shape [d] or [1, d].

        Returns:
            torch.Tensor: LUT of shape [M, 256] and dtype torch.float32.
        """
        q_flat = query.reshape(-1).to(dtype=torch.float32)
        m = self.d // 4
        q_reshaped = q_flat.view(m, 4)
        byte_table = self.byte_to_centroids.to(query.device)
        # LUT shape: [M, 256]
        lut = torch.matmul(q_reshaped, byte_table.T)
        return lut

    def add(self, embeddings: torch.Tensor, use_qjl: Optional[bool] = None) -> None:
        """Quantize input vectors to discrete centroid indices, pack into uint8, and store.

        Retains full-precision vectors in raw_corpus (float32) for Stage 2 re-ranking.

        Args:
            embeddings: Normalized embeddings of shape [N, d] or [d].
            use_qjl: Optional override for QJL error correction.
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        elif embeddings.ndim != 2:
            raise ValueError(f"Expected 1D or 2D embeddings tensor, got {embeddings.ndim}D")

        if embeddings.shape[-1] != self.d:
            raise ValueError(
                f"Embeddings dimension {embeddings.shape[-1]} does not match index dimension {self.d}"
            )

        embeddings_fp32 = embeddings.to(dtype=torch.float32)
        qjl_flag = self.use_qjl if use_qjl is None else use_qjl

        # 1. Quantize input vectors to discrete centroid indices {0, 1, 2, 3}
        boundaries = self.boundaries.to(embeddings_fp32.device)
        codes = torch.bucketize(embeddings_fp32.contiguous(), boundaries)

        # 2. Pack indices into physical uint8 tensor of shape [N, d // 4]
        packed_codes = pack_2bit(codes)

        # Optional QJL error correction residual
        qjl_corr: Optional[torch.Tensor] = None
        if qjl_flag:
            unpacked_codes = unpack_2bit(packed_codes, target_dtype=torch.long)
            centroids_dev = self.centroids.to(embeddings_fp32.device)
            coarse_repr = centroids_dev[unpacked_codes]
            residual = compute_residual(embeddings_fp32, coarse_repr)
            signs = extract_qjl_signs(residual)
            scale = torch.mean(torch.abs(residual)).item()
            qjl_corr = scale * signs

        # 3. Store raw C and packed_corpus
        if self.C is None or self.packed_corpus is None:
            self.C = embeddings_fp32
            self.packed_corpus = packed_codes
            self.qjl_correction = qjl_corr
        else:
            self.C = torch.cat([self.C, embeddings_fp32], dim=0)
            self.packed_corpus = torch.cat([self.packed_corpus, packed_codes], dim=0)
            if qjl_corr is not None:
                self.qjl_correction = (
                    qjl_corr
                    if self.qjl_correction is None
                    else torch.cat([self.qjl_correction, qjl_corr], dim=0)
                )

    def coarse_filter(
        self,
        query: torch.Tensor,
        k: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage 1: Coarse filter scoring directly from packed bytes via precomputed query LUT.

        Computes query LUT of shape [M, 256], gathers 4-element sub-dot products directly from
        self.packed_corpus, sums along M chunks to form coarse scores, and selects top-K candidates.

        Args:
            query: Query tensor of shape [1, d] or [d] (or [B, d]).
            k: Candidate pool size K. Defaults to self.k_candidates.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (candidate_scores, candidate_indices).
        """
        if self.C is None or self.packed_corpus is None or self.packed_corpus.shape[0] == 0:
            raise RuntimeError("Index is empty. Call add() before searching.")

        if query.ndim == 1:
            query = query.unsqueeze(0)
        elif query.ndim != 2:
            raise ValueError(f"Expected 1D or 2D query tensor, got {query.ndim}D")

        if query.shape[-1] != self.d:
            raise ValueError(
                f"Query dimension {query.shape[-1]} does not match index dimension {self.d}"
            )

        num_docs = self.packed_corpus.shape[0]
        target_k = self.k_candidates if k is None else k
        if target_k <= 0:
            raise ValueError(f"k must be positive, got {target_k}")

        k_candidates = min(target_k, num_docs)
        query_fp32 = query.to(dtype=torch.float32)
        b = query_fp32.shape[0]
        m = self.d // 4

        # Packed indices: [N, M] cast to torch.long for gather
        packed_idx = self.packed_corpus.to(dtype=torch.long, device=query_fp32.device)

        if b == 1:
            # Build query LUT: [M, 256]
            lut = self._build_query_lut(query_fp32[0])
            # Direct vectorized gather from packed bytes without unpacking:
            # lut: [1, M, 256] -> expand to [N, M, 256]
            # packed_idx: [N, M, 1]
            chunk_scores = torch.gather(
                lut.unsqueeze(0).expand(num_docs, -1, -1),
                dim=2,
                index=packed_idx.unsqueeze(-1),
            ).squeeze(-1)  # [N, M]
            coarse_scores = chunk_scores.sum(dim=-1).unsqueeze(0)  # [1, N]
        else:
            # Batched LUT computation across B queries
            q_reshaped = query_fp32.view(b, m, 4)
            byte_table = self.byte_to_centroids.to(query_fp32.device)
            lut = torch.matmul(q_reshaped, byte_table.T)  # [B, M, 256]
            lut_exp = lut.unsqueeze(1).expand(b, num_docs, m, 256)
            packed_exp = packed_idx.unsqueeze(0).unsqueeze(-1).expand(b, num_docs, m, 1)
            chunk_scores = torch.gather(lut_exp, dim=3, index=packed_exp).squeeze(-1)  # [B, N, M]
            coarse_scores = chunk_scores.sum(dim=-1)  # [B, N]

        if self.use_qjl and self.qjl_correction is not None:
            qjl_corr = self.qjl_correction.to(query_fp32.device)
            coarse_scores = coarse_scores + torch.matmul(query_fp32, qjl_corr.T)

        stage1_topk = torch.topk(coarse_scores, k=k_candidates, dim=-1)
        return stage1_topk.values, stage1_topk.indices

    def search(
        self,
        query: torch.Tensor,
        top_k: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Two-stage retrieval: Stage 1 coarse filter followed by Stage 2 exact re-ranking.

        Stage 1 (Coarse Filter): Scores packed bytes directly via precomputed query LUT to select
        top-K candidate indices without creating unpacked full-corpus float32 buffers.

        Stage 2 (Exact Re-ranking): Retrieves full-precision candidate vectors from raw_corpus
        using Stage 1 candidate indices, computes exact dot products against query, and sorts.

        Args:
            query: Query tensor of shape [1, d] or [d] (or [B, d]).
            top_k: Number of nearest documents to return (default: 10).

        Returns:
            SearchResult: (scores, indices) each of shape [1, top_k] (or [B, top_k]).
        """
        if self.C is None or self.packed_corpus is None or self.packed_corpus.shape[0] == 0:
            raise RuntimeError("Index is empty. Call add() before searching.")

        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        if query.ndim == 1:
            query = query.unsqueeze(0)
        elif query.ndim != 2:
            raise ValueError(f"Expected 1D or 2D query tensor, got {query.ndim}D")

        if query.shape[-1] != self.d:
            raise ValueError(
                f"Query dimension {query.shape[-1]} does not match index dimension {self.d}"
            )

        num_docs = self.packed_corpus.shape[0]
        k_stage1 = min(max(self.k_candidates, top_k), num_docs)

        # Stage 1: Coarse filter via LUT
        _, candidate_indices = self.coarse_filter(query, k=k_stage1)

        # Stage 2: Exact re-ranking using raw_corpus
        query_fp32 = query.to(dtype=torch.float32)
        # Gather full-precision candidates C_candidates: [B, k_stage1, d]
        C_candidates = self.C[candidate_indices]

        # Exact dot products against query: [B, k_stage1]
        exact_scores = torch.sum(query_fp32.unsqueeze(1) * C_candidates, dim=-1)

        k_stage2 = min(top_k, k_stage1)
        stage2_topk = torch.topk(exact_scores, k=k_stage2, dim=-1)

        # Map candidate ranks back to original corpus document IDs
        final_indices = torch.gather(candidate_indices, dim=-1, index=stage2_topk.indices)
        final_scores = stage2_topk.values

        return SearchResult(final_scores, final_indices)


__all__ = [
    "BYTE_TO_CENTROIDS",
    "DEFAULT_CENTROIDS",
    "SearchResult",
    "TurboQuantIndex",
    "quantize_embeddings_2bit",
]
