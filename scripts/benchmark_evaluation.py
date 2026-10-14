#!/usr/bin/env python3
"""End-to-End Evaluation & Benchmarking Harness for TurboQuant.

Benchmarks TurboQuant 2-Bit KV Cache against native bfloat16 on CPU across:
1. Synthetic KV Cache Memory Footprint & Scaling (L in [512, 1024, 2048, 4096])
2. Single-Token Autoregressive CPU Decoding Latency & Throughput (L in [512, 1024, 2048])
3. Generation Fidelity on Indic Prompts (Hindi, Bengali, Telugu) with Sarvam-1
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Tuple

# Prevent OpenBLAS / OMP thread contention on Windows
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from turboquant.attention import compressed_attention_decode
from turboquant.cache import TurboQuantCache
from turboquant.patch import turboquant_context
from turboquant.quantizer import DynamicKVQuantizer


def format_bytes(num_bytes: float) -> str:
    """Format bytes into binary unit string (B, KiB, MiB, GiB)."""
    if num_bytes < 1024:
        return f"{num_bytes:.0f} B"
    elif num_bytes < 1024**2:
        return f"{num_bytes / 1024:.2f} KiB"
    elif num_bytes < 1024**3:
        return f"{num_bytes / (1024**2):.2f} MiB"
    else:
        return f"{num_bytes / (1024**3):.2f} GiB"


def get_process_rss() -> int:
    """Return resident memory (RSS) in bytes for current process."""
    if psutil is not None:
        return psutil.Process().memory_info().rss
    return 0


# ==============================================================================
# Task 1: Synthetic KV Cache Scaling Benchmark
# ==============================================================================


def benchmark_kv_scaling(
    seq_lengths: Optional[List[int]] = None,
    n_layers: int = 28,
    h_q: int = 16,
    h_kv: int = 8,
    d_h: int = 128,
    use_qjl: bool = False,
) -> List[Dict[str, Any]]:
    """Benchmark KV cache memory footprint across context lengths without full model weights.

    Args:
        seq_lengths: Sequence lengths to evaluate (default: [512, 1024, 2048, 4096]).
        n_layers: Number of transformer layers (Sarvam-1 = 28).
        h_q: Number of query attention heads (Sarvam-1 = 16).
        h_kv: Number of key-value heads (Sarvam-1 = 8).
        d_h: Dimension per head (Sarvam-1 = 128).
        use_qjl: Whether to include 1-bit QJL error correction.

    Returns:
        List of benchmark metric dictionaries.
    """
    if seq_lengths is None:
        seq_lengths = [512, 1024, 2048, 4096]

    results: List[Dict[str, Any]] = []
    quantizer = DynamicKVQuantizer(head_dim=d_h, use_qjl=use_qjl)

    for seq_len in seq_lengths:
        # --- 1. Theoretical Sizes ---
        # Factor of 2 accounts for Keys and Values
        total_elements = 2 * n_layers * h_kv * seq_len * d_h

        # Baseline: bfloat16 is 2 bytes per element
        bf16_theo_bytes = total_elements * 2

        # TurboQuant:
        # 2-bit packed codes: 0.25 bytes per element (uint8 storing 4 codes)
        tq_codes_bytes = int(total_elements * 0.25)
        # Scale metadata: 1 scale per head per token per tensor (K, V)
        # Using float32 scales (4 bytes)
        tq_scales_bytes = 2 * n_layers * h_kv * seq_len * 4
        # Optional QJL: 1 bit per coordinate (0.125 B) + residual scale
        tq_qjl_bytes = (
            int(total_elements * 0.125 + 2 * n_layers * h_kv * seq_len * 4)
            if use_qjl
            else 0
        )
        tq_theo_bytes = tq_codes_bytes + tq_scales_bytes + tq_qjl_bytes
        theo_ratio = bf16_theo_bytes / tq_theo_bytes if tq_theo_bytes > 0 else 0.0

        # --- 2. Empirical Allocation (tracemalloc + RSS) ---
        # Measure Baseline DynamicCache allocation
        gc.collect()
        rss_before_base = get_process_rss()
        tracemalloc.start()

        base_cache = DynamicCache()
        for layer_idx in range(n_layers):
            k_tensor = torch.randn(1, h_kv, seq_len, d_h, dtype=torch.bfloat16)
            v_tensor = torch.randn(1, h_kv, seq_len, d_h, dtype=torch.bfloat16)
            base_cache.update(k_tensor, v_tensor, layer_idx=layer_idx)

        _, peak_base_trace = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after_base = get_process_rss()
        emp_base_rss = max(0, rss_after_base - rss_before_base)

        # Free baseline cache
        del base_cache
        gc.collect()

        # Measure TurboQuantCache allocation
        rss_before_tq = get_process_rss()
        tracemalloc.start()

        tq_cache = TurboQuantCache(quantizer=quantizer, use_qjl=use_qjl)
        for layer_idx in range(n_layers):
            k_tensor = torch.randn(1, h_kv, seq_len, d_h, dtype=torch.float32)
            v_tensor = torch.randn(1, h_kv, seq_len, d_h, dtype=torch.float32)
            tq_cache.update(k_tensor, v_tensor, layer_idx=layer_idx)

        _, peak_tq_trace = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after_tq = get_process_rss()
        emp_tq_rss = max(0, rss_after_tq - rss_before_tq)

        # Actual tensor storage bytes held in TurboQuantCache
        tq_actual_tensor_bytes = 0
        for i in range(len(tq_cache.k_packed)):
            if tq_cache.k_packed[i] is not None:
                tq_actual_tensor_bytes += tq_cache.k_packed[i].numel() * tq_cache.k_packed[i].element_size()
            if tq_cache.v_packed[i] is not None:
                tq_actual_tensor_bytes += tq_cache.v_packed[i].numel() * tq_cache.v_packed[i].element_size()
            if tq_cache.k_scales[i] is not None:
                tq_actual_tensor_bytes += tq_cache.k_scales[i].numel() * tq_cache.k_scales[i].element_size()
            if tq_cache.v_scales[i] is not None:
                tq_actual_tensor_bytes += tq_cache.v_scales[i].numel() * tq_cache.v_scales[i].element_size()

        del tq_cache
        gc.collect()

        # Compute empirical compression ratio (using actual tensor bytes or RSS)
        emp_ratio = (
            bf16_theo_bytes / tq_actual_tensor_bytes
            if tq_actual_tensor_bytes > 0
            else theo_ratio
        )

        results.append(
            {
                "seq_len": seq_len,
                "elements_total": total_elements,
                "bf16_theo_bytes": bf16_theo_bytes,
                "tq_theo_bytes": tq_theo_bytes,
                "theo_compression_ratio": theo_ratio,
                "tq_actual_tensor_bytes": tq_actual_tensor_bytes,
                "emp_base_rss": emp_base_rss,
                "emp_tq_rss": emp_tq_rss,
                "emp_compression_ratio": emp_ratio,
            }
        )

    return results


# ==============================================================================
# Task 2: CPU Decoding Latency Benchmark
# ==============================================================================


def benchmark_decode_latency(
    seq_lengths: Optional[List[int]] = None,
    h_q: int = 16,
    h_kv: int = 8,
    d_h: int = 128,
    num_warmup: int = 5,
    num_steps: int = 50,
) -> List[Dict[str, Any]]:
    """Measure single-token autoregressive decode steps (S=1) latency on CPU.

    Args:
        seq_lengths: Context lengths to evaluate (default: [512, 1024, 2048]).
        h_q: Query attention heads (16).
        h_kv: Key-Value attention heads (8).
        d_h: Per-head vector dimension (128).
        num_warmup: Warmup iterations before timing (5).
        num_steps: Timed decoding iterations (50).

    Returns:
        List of latency & throughput metrics across context lengths.
    """
    if seq_lengths is None:
        seq_lengths = [512, 1024, 2048]

    results: List[Dict[str, Any]] = []
    quantizer = DynamicKVQuantizer(head_dim=d_h, use_qjl=False)
    gqa_ratio = h_q // h_kv

    for seq_len in seq_lengths:
        torch.manual_seed(42)
        # Single-token query during autoregressive decode step: [1, H_q, 1, d_h]
        query = torch.randn(1, h_q, 1, d_h)

        # Baseline KV tensors: [1, H_kv, seq_len, d_h]
        k_base = torch.randn(1, h_kv, seq_len, d_h)
        v_base = torch.randn(1, h_kv, seq_len, d_h)

        # Repeat KV heads for GQA alignment in PyTorch SDPA
        k_rep = torch.repeat_interleave(k_base, gqa_ratio, dim=1)
        v_rep = torch.repeat_interleave(v_base, gqa_ratio, dim=1)

        # Compressed KV tensors for TurboQuant
        k_quant = quantizer.quantize(k_base)
        v_quant = quantizer.quantize(v_base)

        # --- 1. Baseline Latency Measurement ---
        for _ in range(num_warmup):
            _ = F.scaled_dot_product_attention(query, k_rep, v_rep)

        t0 = time.perf_counter()
        for _ in range(num_steps):
            _ = F.scaled_dot_product_attention(query, k_rep, v_rep)
        t1 = time.perf_counter()
        base_avg_ms = ((t1 - t0) / num_steps) * 1000.0
        base_tok_per_sec = 1000.0 / base_avg_ms if base_avg_ms > 0 else 0.0

        # --- 2. TurboQuant Latency Measurement ---
        for _ in range(num_warmup):
            _ = compressed_attention_decode(
                query=query,
                k_cache=k_quant,
                v_cache=v_quant,
                quantizer=quantizer,
            )

        t0 = time.perf_counter()
        for _ in range(num_steps):
            _ = compressed_attention_decode(
                query=query,
                k_cache=k_quant,
                v_cache=v_quant,
                quantizer=quantizer,
            )
        t1 = time.perf_counter()
        tq_avg_ms = ((t1 - t0) / num_steps) * 1000.0
        tq_tok_per_sec = 1000.0 / tq_avg_ms if tq_avg_ms > 0 else 0.0

        results.append(
            {
                "seq_len": seq_len,
                "base_latency_ms": base_avg_ms,
                "base_tokens_sec": base_tok_per_sec,
                "tq_latency_ms": tq_avg_ms,
                "tq_tokens_sec": tq_tok_per_sec,
                "relative_factor": tq_avg_ms / base_avg_ms if base_avg_ms > 0 else 1.0,
            }
        )

    return results


# ==============================================================================
# Task 3: Generation Fidelity on Indic Prompts
# ==============================================================================

INDIC_BENCHMARK_PROMPTS = [
    {
        "language": "Hindi",
        "prompt": "भारत की राजधानी नई दिल्ली है। यहाँ का मुख्य आकर्षण ऐतिहासिक लाल किला और इंडिया गेट हैं।",
    },
    {
        "language": "Bengali",
        "prompt": "কলকাতা পশ্চিমবঙ্গের রাজধানী এবং ভারতের একটি প্রধান সাংস্কৃতিক কেন্দ্র। এই শহরের ঐতিহ্যবাহী ট্রাম",
    },
    {
        "language": "Telugu",
        "prompt": "హైదరాబాద్ భారతదేశంలోని తెలంగాణ రాష్ట్ర రాజధాని. ఈ నగరం చారిత్రక చార్మినార్ మరియు హైటెక్",
    },
]


def load_or_init_sarvam_model(
    model_id: str = "sarvamai/sarvam-1",
    force_download: bool = False,
) -> Tuple[Any, Any, str]:
    """Load sarvamai/sarvam-1 or initialize architecture model if memory is constrained.

    Returns:
        Tuple of (model, tokenizer, model_status_description).
    """
    print(f"Loading tokenizer for '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Check available system RAM
    mem_avail_gb = 8.0
    if psutil is not None:
        mem_avail_gb = psutil.virtual_memory().available / (1024**3)

    model = None
    status = ""

    # If user didn't force download and free RAM is below 4.5 GB, initialize architecture model
    # to prevent system OOM freezes
    if not force_download and mem_avail_gb < 4.5:
        print(
            f"Note: Available RAM ({mem_avail_gb:.2f} GB) is below 4.5 GB required for full weights."
        )
        print("Initializing lightweight Sarvam-1 architecture model on CPU in bfloat16...")
        torch.set_default_dtype(torch.bfloat16)
        config = AutoConfig.from_pretrained(model_id)
        config.num_hidden_layers = 2  # 2 layers for fast, high-fidelity KV testing
        config.vocab_size = len(tokenizer)
        model = LlamaForCausalLM(config)
        torch.set_default_dtype(torch.float32)
        model.eval()
        status = f"Sarvam-1 Architecture (2 Layers, bfloat16, {config.hidden_size} hidden, {config.num_key_value_heads} KV heads)"
        return model, tokenizer, status

    try:
        from transformers import AutoModelForCausalLM

        print(f"Loading weights for '{model_id}' via AutoModelForCausalLM (bfloat16)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        model.eval()
        status = f"Full Weights ({model_id})"
    except Exception as e:
        print(f"Notice: Could not load full pretrained weights ({e}).")
        print("Falling back to Sarvam-1 architecture model...")
        torch.set_default_dtype(torch.bfloat16)
        config = AutoConfig.from_pretrained(model_id)
        config.num_hidden_layers = 2
        config.vocab_size = len(tokenizer)
        model = LlamaForCausalLM(config)
        torch.set_default_dtype(torch.float32)
        model.eval()
        status = "Sarvam-1 Architecture Fallback (bfloat16)"

    return model, tokenizer, status


def eval_indic_fidelity(
    model: Any,
    tokenizer: Any,
    prompts: Optional[List[Dict[str, str]]] = None,
    max_new_tokens: int = 32,
) -> List[Dict[str, Any]]:
    """Compare greedy generation fidelity between DynamicCache and TurboQuantCache.

    Computes:
    - Step-by-step Logits Cosine Similarity
    - Greedy Token Match Rate
    - Side-by-side decoded strings

    Args:
        model: HuggingFace model.
        tokenizer: Model tokenizer.
        prompts: Indic prompt dataset.
        max_new_tokens: Number of greedy decode steps (32).

    Returns:
        List of prompt fidelity results.
    """
    if prompts is None:
        prompts = INDIC_BENCHMARK_PROMPTS

    head_dim = getattr(
        model.config,
        "head_dim",
        model.config.hidden_size // model.config.num_attention_heads,
    )
    quantizer = DynamicKVQuantizer(head_dim=head_dim, use_qjl=False)
    results: List[Dict[str, Any]] = []

    for item in prompts:
        lang = item["language"]
        prompt_text = item["prompt"]
        inputs = tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs.input_ids

        # --- 1. Baseline Generation (Standard DynamicCache) ---
        base_cache = DynamicCache()
        base_tokens: List[int] = []
        base_step_logits: List[torch.Tensor] = []

        with torch.no_grad():
            # Prefill
            out_base = model(input_ids, past_key_values=base_cache, use_cache=True)
            next_logits = out_base.logits[:, -1, :].float()
            next_tok = torch.argmax(next_logits, dim=-1, keepdim=True)
            base_tokens.append(next_tok.item())
            base_step_logits.append(next_logits.squeeze(0))

            curr_input = next_tok
            for _ in range(max_new_tokens - 1):
                out_base = model(
                    curr_input, past_key_values=base_cache, use_cache=True
                )
                next_logits = out_base.logits[:, -1, :].float()
                next_tok = torch.argmax(next_logits, dim=-1, keepdim=True)
                base_tokens.append(next_tok.item())
                base_step_logits.append(next_logits.squeeze(0))
                curr_input = next_tok

        # --- 2. TurboQuant Generation (TurboQuantCache + monkey patch) ---
        tq_cache = TurboQuantCache(quantizer=quantizer)
        tq_tokens: List[int] = []
        tq_step_logits: List[torch.Tensor] = []

        with torch.no_grad(), turboquant_context(model, quantizer=quantizer):
            # Prefill
            out_tq = model(input_ids, past_key_values=tq_cache, use_cache=True)
            next_logits = out_tq.logits[:, -1, :].float()
            next_tok = torch.argmax(next_logits, dim=-1, keepdim=True)
            tq_tokens.append(next_tok.item())
            tq_step_logits.append(next_logits.squeeze(0))

            curr_input = next_tok
            for _ in range(max_new_tokens - 1):
                out_tq = model(
                    curr_input, past_key_values=tq_cache, use_cache=True
                )
                next_logits = out_tq.logits[:, -1, :].float()
                next_tok = torch.argmax(next_logits, dim=-1, keepdim=True)
                tq_tokens.append(next_tok.item())
                tq_step_logits.append(next_logits.squeeze(0))
                curr_input = next_tok

        # --- 3. Compute Fidelity Metrics ---
        cosine_sims: List[float] = []
        for l_base, l_tq in zip(base_step_logits, tq_step_logits):
            sim = F.cosine_similarity(
                l_base.unsqueeze(0), l_tq.unsqueeze(0), dim=-1
            ).item()
            cosine_sims.append(sim)

        avg_cosine = sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0.0

        matches = sum(b == t for b, t in zip(base_tokens, tq_tokens))
        match_rate = (matches / len(base_tokens)) * 100.0 if base_tokens else 0.0

        base_decoded = tokenizer.decode(base_tokens, skip_special_tokens=True)
        tq_decoded = tokenizer.decode(tq_tokens, skip_special_tokens=True)

        results.append(
            {
                "language": lang,
                "prompt": prompt_text,
                "avg_cosine_similarity": avg_cosine,
                "token_match_rate": match_rate,
                "num_tokens": len(base_tokens),
                "base_decoded": base_decoded,
                "tq_decoded": tq_decoded,
            }
        )

    return results


# ==============================================================================
# Task 4: Terminal Reporting in Markdown
# ==============================================================================


def print_markdown_reports(
    scaling_data: List[Dict[str, Any]],
    latency_data: List[Dict[str, Any]],
    fidelity_data: List[Dict[str, Any]],
    model_status: str,
) -> None:
    """Format and print all benchmark sections into clean markdown tables."""
    sep = "=" * 80

    print("\n" + sep)
    print(" 🚀 TURBOQUANT END-TO-END BENCHMARK & EVALUATION REPORT")
    print(sep + "\n")

    # Table 1: KV Cache Scaling & Memory Footprint
    print("### 1. 💾 Synthetic KV Cache Memory Scaling (Sarvam-1: 28 Layers, 8 KV Heads, Dim 128)\n")
    print(
        "| Context Length (Tokens) | bfloat16 Baseline | TurboQuant 2-Bit | Theoretical Ratio | Actual TQ Buffers | Empirical Ratio |"
    )
    print(
        "|:-----------------------:|:-----------------:|:----------------:|:-----------------:|:-----------------:|:---------------:|"
    )
    for row in scaling_data:
        l_str = f"{row['seq_len']:,}"
        bf16_str = format_bytes(row['bf16_theo_bytes'])
        tq_str = format_bytes(row['tq_theo_bytes'])
        tq_actual = format_bytes(row['tq_actual_tensor_bytes'])
        t_ratio = f"{row['theo_compression_ratio']:.2f}x"
        e_ratio = f"{row['emp_compression_ratio']:.2f}x"
        print(f"| {l_str:<23} | {bf16_str:<17} | {tq_str:<16} | {t_ratio:<17} | {tq_actual:<17} | {e_ratio:<15} |")

    # Table 2: CPU Decoding Latency & Throughput
    print("\n### 2. ⏱️ Single-Token CPU Decoding Latency & Throughput (S = 1)\n")
    print(
        "| Context Length (Tokens) | Baseline Latency | Baseline Throughput | TurboQuant Latency | TurboQuant Throughput | Relative Factor |"
    )
    print(
        "|:-----------------------:|:----------------:|:-------------------:|:------------------:|:---------------------:|:---------------:|"
    )
    for row in latency_data:
        l_str = f"{row['seq_len']:,}"
        b_lat = f"{row['base_latency_ms']:.3f} ms"
        b_tps = f"{row['base_tokens_sec']:.1f} tok/s"
        t_lat = f"{row['tq_latency_ms']:.3f} ms"
        t_tps = f"{row['tq_tokens_sec']:.1f} tok/s"
        rel = f"{row['relative_factor']:.2f}x"
        print(f"| {l_str:<23} | {b_lat:<16} | {b_tps:<19} | {t_lat:<18} | {t_tps:<21} | {rel:<15} |")

    # Table 3: Indic Generation Fidelity
    print(f"\n### 3. 🎯 Generation Fidelity on Indic Prompts ({model_status})\n")
    print(
        "| Language | Next-Token Logits Cosine Sim | Greedy Token Match Rate | Evaluated Tokens |"
    )
    print(
        "|:--------:|:----------------------------:|:-----------------------:|:----------------:|"
    )
    for row in fidelity_data:
        lang = row['language']
        sim = f"{row['avg_cosine_similarity']:.4f}"
        match = f"{row['token_match_rate']:.1f}%"
        num_tok = row['num_tokens']
        print(f"| {lang:<8} | {sim:<28} | {match:<23} | {num_tok:<16} |")

    # Table 4: Side-by-Side Decoded Text
    print("\n### 4. 📝 Side-by-Side Text Output Comparison (Greedy Decode, 32 Tokens)\n")
    for row in fidelity_data:
        lang = row['language']
        prompt = row['prompt']
        base_out = repr(row['base_decoded'])
        tq_out = repr(row['tq_decoded'])
        print(f"**[{lang}] Prompt:** *\"{prompt}\"*")
        print(f"- **Baseline Output   :** {base_out}")
        print(f"- **TurboQuant Output :** {tq_out}\n")

    print(sep)
    print(" ✅ All benchmark benchmarks completed successfully.")
    print(sep + "\n")


# ==============================================================================
# CLI Entrypoint
# ==============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TurboQuant End-to-End Evaluation & Benchmarking Harness"
    )
    parser.add_argument(
        "--context-sweep",
        type=str,
        default="512,1024,2048,4096",
        help="Comma-separated sequence lengths for KV cache scaling benchmark.",
    )
    parser.add_argument(
        "--latency-sweep",
        type=str,
        default="512,1024,2048",
        help="Comma-separated sequence lengths for latency benchmark.",
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=50,
        help="Number of timed decode steps for latency measurement.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Max generated tokens for generation fidelity.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="sarvamai/sarvam-1",
        help="Model ID or checkpoint path.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force downloading full 4GB model weights even on low-RAM systems.",
    )
    parser.add_argument(
        "--use-qjl",
        action="store_true",
        help="Enable 1-bit QJL error correction for KV cache scaling.",
    )

    args = parser.parse_args()

    scaling_lengths = [int(x.strip()) for x in args.context_sweep.split(",") if x.strip()]
    latency_lengths = [int(x.strip()) for x in args.latency_sweep.split(",") if x.strip()]

    print("Running 1. Synthetic KV Cache Scaling Benchmark...")
    scaling_data = benchmark_kv_scaling(
        seq_lengths=scaling_lengths,
        use_qjl=args.use_qjl,
    )

    print("Running 2. CPU Decoding Latency Benchmark...")
    latency_data = benchmark_decode_latency(
        seq_lengths=latency_lengths,
        num_steps=args.decode_steps,
    )

    print("Running 3. Generation Fidelity on Indic Prompts...")
    model, tokenizer, model_status = load_or_init_sarvam_model(
        model_id=args.model_id,
        force_download=args.force_download,
    )
    fidelity_data = eval_indic_fidelity(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
    )

    print_markdown_reports(
        scaling_data=scaling_data,
        latency_data=latency_data,
        fidelity_data=fidelity_data,
        model_status=model_status,
    )


if __name__ == "__main__":
    main()
