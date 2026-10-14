"""Latency profiling and comparison benchmark: Lookup Table (LUT) scoring vs. Unpack baseline.

Profiles TurboQuantIndex across corpus sizes N in [1_000, 10_000, 50_000] with dimension d = 768:
- Compares on-the-fly unpacking against Precomputed Query LUT Asymmetric Distance Computation.
- Measures latency per step, total latency, Queries Per Second (QPS), and peak temporary memory.
"""

from __future__ import annotations

import gc
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from turboquant.index import TurboQuantIndex
from turboquant.packing import unpack_2bit


def get_unpack_peak_memory_bytes(
    n: int,
    d: int,
) -> int:
    """Calculate the peak temporary buffer size in bytes during the unpacking step."""
    # unpacked_codes (long: 8 bytes) + coarse_corpus (float32: 4 bytes)
    return n * d * 8 + n * d * 4


def get_lut_peak_memory_bytes(
    d: int,
) -> int:
    """Calculate the peak temporary buffer size in bytes during the query LUT step."""
    m = d // 4
    # Query LUT table of shape [M, 256] in float32: 192 * 256 * 4 bytes
    return m * 256 * 4


def profile_scale_comparison(
    n: int,
    d: int = 768,
    k_candidates: int = 25,
    top_k: int = 10,
    num_warmup: int = 10,
    num_timed: int = 100,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Profile and compare Unpack vs LUT search step latencies for a given corpus size N."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_cuda = device.type == "cuda"
    print(f"\n---> Profiling Scale N = {n:,} (d = {d}, K = {k_candidates}, top_k = {top_k}) on {device.type.upper()}...")

    torch.manual_seed(42)
    corpus = F.normalize(torch.randn(n, d, device=device, dtype=torch.float32), p=2, dim=-1)

    index = TurboQuantIndex(d=d, k_candidates=k_candidates)
    index.add(corpus)

    total_queries = num_warmup + num_timed
    queries = F.normalize(torch.randn(total_queries, d, device=device, dtype=torch.float32), p=2, dim=-1)

    # =========================================================================
    # 1. Warmup
    # =========================================================================
    print(f"     Running {num_warmup} warmup queries...")
    for i in range(num_warmup):
        q = queries[i : i + 1]
        _ = index.search(q, top_k=top_k)

    # =========================================================================
    # 2. Benchmark Baseline Unpack-Based Search
    # =========================================================================
    print(f"     [A] Profiling Baseline Unpack-Based Search ({num_timed} queries)...", flush=True)
    t_unpack_records: List[float] = []
    t_unpack_matmul_records: List[float] = []
    t_unpack_total_records: List[float] = []

    # For N=50k, sample 25 queries for unpack baseline to avoid excessive CPU wait
    unpack_timed_count = min(num_timed, 25) if n >= 50000 else num_timed

    for i in range(num_warmup, num_warmup + unpack_timed_count):
        q = queries[i : i + 1]

        if is_cuda:
            torch.cuda.synchronize()

        # Step 1: Unpack
        t0 = time.perf_counter()
        unpacked_codes = unpack_2bit(index.packed_corpus, target_dtype=torch.long)
        centroids_dev = index.centroids.to(index.packed_corpus.device)
        coarse_corpus = centroids_dev[unpacked_codes]
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Step 2: Matmul
        coarse_scores = torch.matmul(q, coarse_corpus.T)
        if is_cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        # Step 3: Top-K + Rerank
        stage1_topk = torch.topk(coarse_scores, k=min(k_candidates, n), dim=-1)
        candidate_indices = stage1_topk.indices
        C_candidates = index.C[candidate_indices]
        exact_scores = torch.sum(q.unsqueeze(1) * C_candidates, dim=-1)
        stage2_topk = torch.topk(exact_scores, k=min(top_k, candidate_indices.shape[-1]), dim=-1)
        _ = torch.gather(candidate_indices, dim=-1, index=stage2_topk.indices)
        if is_cuda:
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        t_unpack_records.append((t1 - t0) * 1000.0)
        t_unpack_matmul_records.append((t2 - t1) * 1000.0)
        t_unpack_total_records.append((t3 - t0) * 1000.0)

    # =========================================================================
    # 3. Benchmark LUT-Based Search (Optimized)
    # =========================================================================
    print(f"     [B] Profiling Query LUT-Based Search ({num_timed} queries)...", flush=True)
    t_lut_build_records: List[float] = []
    t_lut_gather_records: List[float] = []
    t_lut_topk_records: List[float] = []
    t_lut_rerank_records: List[float] = []
    t_lut_total_records: List[float] = []

    num_docs = index.packed_corpus.shape[0]

    for i in range(num_warmup, total_queries):
        q = queries[i : i + 1]

        if is_cuda:
            torch.cuda.synchronize()

        # Step 1: Build LUT
        t0 = time.perf_counter()
        lut = index._build_query_lut(q[0])
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Step 2: Vectorized gather and sum
        packed_idx = index.packed_corpus.to(dtype=torch.long, device=q.device)
        chunk_scores = torch.gather(
            lut.unsqueeze(0).expand(num_docs, -1, -1),
            dim=2,
            index=packed_idx.unsqueeze(-1),
        ).squeeze(-1)
        coarse_scores = chunk_scores.sum(dim=-1).unsqueeze(0)
        if is_cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        # Step 3: Top-K
        stage1_topk = torch.topk(coarse_scores, k=min(k_candidates, n), dim=-1)
        candidate_indices = stage1_topk.indices
        if is_cuda:
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        # Step 4: Re-rank
        C_candidates = index.C[candidate_indices]
        exact_scores = torch.sum(q.unsqueeze(1) * C_candidates, dim=-1)
        stage2_topk = torch.topk(exact_scores, k=min(top_k, candidate_indices.shape[-1]), dim=-1)
        _ = torch.gather(candidate_indices, dim=-1, index=stage2_topk.indices)
        if is_cuda:
            torch.cuda.synchronize()
        t4 = time.perf_counter()

        # Also measure end-to-end index.search() directly
        t_search_start = time.perf_counter()
        _ = index.search(q, top_k=top_k)
        if is_cuda:
            torch.cuda.synchronize()
        t_search_end = time.perf_counter()

        t_lut_build_records.append((t1 - t0) * 1000.0)
        t_lut_gather_records.append((t2 - t1) * 1000.0)
        t_lut_topk_records.append((t3 - t2) * 1000.0)
        t_lut_rerank_records.append((t4 - t3) * 1000.0)
        t_lut_total_records.append((t_search_end - t_search_start) * 1000.0)

    # Compute averages
    mean_unpack_step = sum(t_unpack_records) / len(t_unpack_records)
    mean_unpack_total = sum(t_unpack_total_records) / len(t_unpack_total_records)
    unpack_qps = 1000.0 / mean_unpack_total

    mean_lut_build = sum(t_lut_build_records) / len(t_lut_build_records)
    mean_lut_gather = sum(t_lut_gather_records) / len(t_lut_gather_records)
    mean_lut_topk = sum(t_lut_topk_records) / len(t_lut_topk_records)
    mean_lut_rerank = sum(t_lut_rerank_records) / len(t_lut_rerank_records)
    mean_lut_total = sum(t_lut_total_records) / len(t_lut_total_records)
    lut_qps = 1000.0 / mean_lut_total

    speedup = mean_unpack_total / mean_lut_total if mean_lut_total > 0 else 1.0

    unpack_peak_bytes = get_unpack_peak_memory_bytes(n, d)
    lut_peak_bytes = get_lut_peak_memory_bytes(d)
    mem_reduction = unpack_peak_bytes / lut_peak_bytes

    return {
        "n": n,
        "d": d,
        "k_candidates": k_candidates,
        # Baseline Unpack
        "unpack_step_ms": mean_unpack_step,
        "unpack_total_ms": mean_unpack_total,
        "unpack_qps": unpack_qps,
        "unpack_peak_bytes": unpack_peak_bytes,
        "unpack_peak_mb": unpack_peak_bytes / (1024 * 1024),
        # Optimized LUT
        "lut_build_ms": mean_lut_build,
        "lut_gather_ms": mean_lut_gather,
        "lut_topk_ms": mean_lut_topk,
        "lut_rerank_ms": mean_lut_rerank,
        "lut_total_ms": mean_lut_total,
        "lut_qps": lut_qps,
        "lut_peak_bytes": lut_peak_bytes,
        "lut_peak_kb": lut_peak_bytes / 1024,
        # Comparison
        "speedup": speedup,
        "mem_reduction": mem_reduction,
    }


def format_head_to_head_table(results: List[Dict[str, Any]]) -> str:
    """Format comparative summary table between Unpack baseline and Query LUT."""
    headers = [
        "Corpus Size (N)",
        "Unpack Latency (ms)",
        "LUT Latency (ms)",
        "Speedup (x)",
        "Unpack QPS",
        "LUT QPS",
        "Unpack Temp Mem",
        "LUT Temp Mem",
        "Mem Reduction",
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for res in results:
        row = [
            f"{res['n']:,}",
            f"{res['unpack_total_ms']:.2f} ms",
            f"{res['lut_total_ms']:.2f} ms",
            f"**{res['speedup']:.2f}x**",
            f"{res['unpack_qps']:.1f} q/s",
            f"**{res['lut_qps']:.1f} q/s**",
            f"{res['unpack_peak_mb']:.2f} MB",
            f"{res['lut_peak_kb']:.1f} KB",
            f"**{res['mem_reduction']:.0f}x**",
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def format_lut_breakdown_table(results: List[Dict[str, Any]]) -> str:
    """Format detailed breakdown table of LUT pipeline stages."""
    headers = [
        "Corpus Size (N)",
        "Build LUT (ms)",
        "Gather & Sum (ms)",
        "Stage 1 Top-K (ms)",
        "Stage 2 Re-rank (ms)",
        "Total LUT Search (ms)",
        "Throughput (QPS)",
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for res in results:
        row = [
            f"{res['n']:,}",
            f"{res['lut_build_ms']:.3f} ms",
            f"{res['lut_gather_ms']:.3f} ms",
            f"{res['lut_topk_ms']:.3f} ms",
            f"{res['lut_rerank_ms']:.3f} ms",
            f"**{res['lut_total_ms']:.3f} ms**",
            f"**{res['lut_qps']:.1f} q/s**",
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def run_benchmark() -> None:
    """Run full comparison benchmark across corpus scales."""
    print("=" * 90)
    print(" TurboQuant Performance Benchmark: Precomputed Query LUT vs. Unpack Baseline")
    print("=" * 90)

    scales = [1_000, 10_000, 50_000]
    d = 768
    k_candidates = 25
    top_k = 10
    num_warmup = 10
    num_timed = 100

    results: List[Dict[str, Any]] = []

    for n in scales:
        res = profile_scale_comparison(
            n=n,
            d=d,
            k_candidates=k_candidates,
            top_k=top_k,
            num_warmup=num_warmup,
            num_timed=num_timed,
        )
        results.append(res)
        gc.collect()

    print("\n" + "=" * 90)
    print(" HEAD-TO-HEAD COMPARISON: UNPACK BASELINE vs. QUERY LUT SCORING")
    print("=" * 90 + "\n")
    print(format_head_to_head_table(results))

    print("\n" + "=" * 90)
    print(" DETAILED LUT PIPELINE LATENCY BREAKDOWN")
    print("=" * 90 + "\n")
    print(format_lut_breakdown_table(results))
    print("\n" + "=" * 90)


if __name__ == "__main__":
    run_benchmark()
