#!/usr/bin/env python3
"""Inspect model configuration of sarvamai/sarvam-1 and calculate KV cache memory footprints.

Fetches model configuration using Hugging Face AutoConfig (without downloading weights)
and computes key GQA dimensions and baseline vs TurboQuant KV cache footprints.
"""

from __future__ import annotations

import os
import sys

# Prevent OpenBLAS thread allocation memory errors on Windows
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from typing import Any, Dict
from transformers import AutoConfig


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def format_bytes(num_bytes: float) -> str:
    """Format bytes into human-readable binary unit string (B, KiB, MiB, GiB)."""
    if num_bytes < 1024:
        return f"{num_bytes:.0f} B"
    elif num_bytes < 1024**2:
        return f"{num_bytes / 1024:.2f} KiB"
    elif num_bytes < 1024**3:
        return f"{num_bytes / (1024**2):.2f} MiB"
    else:
        return f"{num_bytes / (1024**3):.2f} GiB"


def inspect_sarvam_config(model_id: str = "sarvamai/sarvam-1") -> Dict[str, Any]:
    """Load config and calculate architectural & KV cache memory metrics."""
    print(f"Loading configuration for '{model_id}' via AutoConfig...")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # 1. Extract Key Hyperparameters
    n_layers = getattr(config, "num_hidden_layers", None)
    h_q = getattr(config, "num_attention_heads", None)
    h_kv = getattr(config, "num_key_value_heads", h_q)
    hidden_size = getattr(config, "hidden_size", None)
    vocab_size = getattr(config, "vocab_size", None)
    max_position_embeddings = getattr(config, "max_position_embeddings", None)
    dtype_val = getattr(config, "dtype", getattr(config, "torch_dtype", "bfloat16"))

    # Compute or retrieve per-head dimension d_h
    if hasattr(config, "head_dim") and config.head_dim is not None:
        d_h = config.head_dim
    else:
        d_h = hidden_size // h_q

    # Compute GQA sharing ratio (H_q / H_kv)
    gqa_ratio = h_q / h_kv if h_kv else 1.0

    # 2. KV Footprint Calculations
    # Factor of 2 accounts for Key and Value tensors
    elements_per_token_per_layer = 2 * h_kv * d_h
    elements_per_token_all_layers = 2 * n_layers * h_kv * d_h

    # Memory per token in standard bfloat16 (2 bytes / element)
    bytes_per_token_bf16 = elements_per_token_all_layers * 2

    # Memory per token in TurboQuant 2-bit (0.25 bytes / element pure codes)
    bytes_per_token_tq_pure = elements_per_token_all_layers * 0.25

    # Memory per token in TurboQuant 2-bit + 1-bit QJL error correction:
    # 2 bits (0.25 bytes/elem) + 1 bit QJL sign (0.125 bytes/elem) = 0.375 bytes/elem
    # + bfloat16 scale per KV head per layer: 2 tensors (K, V) * n_layers * h_kv * 2 bytes
    qjl_sign_bytes_per_token = elements_per_token_all_layers * 0.125
    metadata_scale_bytes_per_token = 2 * n_layers * h_kv * 2  # 2 bytes per fp16/bf16 scale
    bytes_per_token_tq_qjl = (
        bytes_per_token_tq_pure + qjl_sign_bytes_per_token + metadata_scale_bytes_per_token
    )

    # Sequence lengths for cache scaling
    seq_lengths = [2048, 8192, 32768]
    scaling_results = []
    for L in seq_lengths:
        bf16_total = L * bytes_per_token_bf16
        tq_pure_total = L * bytes_per_token_tq_pure
        tq_qjl_total = L * bytes_per_token_tq_qjl
        scaling_results.append(
            {
                "seq_len": L,
                "bf16_bytes": bf16_total,
                "tq_pure_bytes": tq_pure_total,
                "tq_qjl_bytes": tq_qjl_total,
                "tq_pure_ratio": bf16_total / tq_pure_total,
                "tq_qjl_ratio": bf16_total / tq_qjl_total,
            }
        )

    return {
        "model_id": model_id,
        "model_type": getattr(config, "model_type", "llama"),
        "torch_dtype": dtype_val,
        "vocab_size": vocab_size,
        "max_position_embeddings": max_position_embeddings,
        "n_layers": n_layers,
        "h_q": h_q,
        "h_kv": h_kv,
        "hidden_size": hidden_size,
        "d_h": d_h,
        "gqa_ratio": gqa_ratio,
        "elements_per_token_per_layer": elements_per_token_per_layer,
        "elements_per_token_all_layers": elements_per_token_all_layers,
        "bytes_per_token_bf16": bytes_per_token_bf16,
        "bytes_per_token_tq_pure": bytes_per_token_tq_pure,
        "bytes_per_token_tq_qjl": bytes_per_token_tq_qjl,
        "metadata_scale_bytes_per_token": metadata_scale_bytes_per_token,
        "scaling_results": scaling_results,
    }


def print_summary(data: Dict[str, Any]) -> None:
    """Print structured tables of model architecture and cache footprints."""
    sep = "=" * 80
    subsep = "-" * 80

    print(f"\n{sep}")
    print(f" MODEL ARCHITECTURE & GQA CONFIGURATION: {data['model_id']}")
    print(sep)
    print(f" Model Type                : {data['model_type']}")
    print(f" Native Dtype              : {data['torch_dtype']}")
    print(f" Vocab Size                : {data['vocab_size']:,}")
    print(f" Max Position Embeddings   : {data['max_position_embeddings']:,} tokens")
    print(f" Hidden Size (d)           : {data['hidden_size']}")
    print(f" Number of Layers (n_layers): {data['n_layers']}")
    print(f" Query Attention Heads (H_q): {data['h_q']}")
    print(f" KV Attention Heads (H_kv) : {data['h_kv']}")
    print(f" Per-Head Dimension (d_h)  : {data['d_h']}")
    print(
        f" GQA Sharing Ratio (H_q/H_kv): {data['gqa_ratio']:.1f}:1 "
        f"({int(data['gqa_ratio'])} query heads share 1 KV head)"
    )

    print(f"\n{sep}")
    print(" KV CACHE PER-TOKEN ELEMENT & MEMORY FOOTPRINT")
    print(sep)
    print(
        f" Elements / Token / Layer  (2 x H_kv x d_h)        : {data['elements_per_token_per_layer']:,} elements"
    )
    print(
        f" Elements / Token Across All Layers (2 x n_layers x H_kv x d_h) : {data['elements_per_token_all_layers']:,} elements"
    )
    print(subsep)
    print(
        f" Standard bfloat16 Footprint (2.00 bytes/element) : {data['bytes_per_token_bf16']:,} bytes/token "
        f"({format_bytes(data['bytes_per_token_bf16'])}/token)"
    )
    print(
        f" TurboQuant 2-Bit Pure Codes (0.25 bytes/element) : {int(data['bytes_per_token_tq_pure']):,} bytes/token "
        f"({format_bytes(data['bytes_per_token_tq_pure'])}/token) [8.00x reduction]"
    )
    print(
        f" TurboQuant 2-Bit + 1-Bit QJL + Scale Metadata     : {int(data['bytes_per_token_tq_qjl']):,} bytes/token "
        f"({format_bytes(data['bytes_per_token_tq_qjl'])}/token) [{data['bytes_per_token_bf16'] / data['bytes_per_token_tq_qjl']:.2f}x reduction]"
    )
    print(
        f"   |- 2-bit codes: {int(data['bytes_per_token_tq_pure']):,} B | QJL signs: {int(data['elements_per_token_all_layers'] * 0.125):,} B | Scales: {data['metadata_scale_bytes_per_token']:,} B"
    )

    print(f"\n{sep}")
    print(" PROJECTED KV CACHE MEMORY SCALING ACROSS CONTEXT WINDOWS")
    print(sep)
    header = (
        f"{'Seq Length (L)':<16} | {'bfloat16 Baseline':<18} | "
        f"{'TurboQuant 2-Bit':<18} | {'TQ 2-Bit + QJL + Meta':<21}"
    )
    print(header)
    print(subsep)

    for res in data["scaling_results"]:
        l_str = f"{res['seq_len']:,} tokens"
        bf16_str = f"{format_bytes(res['bf16_bytes'])}"
        tq_str = f"{format_bytes(res['tq_pure_bytes'])} (8.00x)"
        tq_qjl_str = f"{format_bytes(res['tq_qjl_bytes'])} ({res['tq_qjl_ratio']:.2f}x)"
        print(f"{l_str:<16} | {bf16_str:<18} | {tq_str:<18} | {tq_qjl_str:<21}")
    print(sep + "\n")


def main() -> None:
    try:
        data = inspect_sarvam_config()
        print_summary(data)
    except Exception as e:
        print(f"Error during inspection: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
