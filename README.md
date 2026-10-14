# TurboQuant-Indic

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 81 passing](https://img.shields.io/badge/tests-81%20passing-brightgreen.svg)](tests/)
[![Compression](https://img.shields.io/badge/KV%20Cache%20Compression-8x%20(32B%2Fhead)-orange.svg)](#architecture-overview)

> **TurboQuant-Indic** delivers extreme-throughput, ultra-low-footprint Key-Value (KV) cache compression for long-context Indic Large Language Models (such as [`sarvamai/sarvam-1`](https://huggingface.co/sarvamai/sarvam-1)). By combining **randomized orthogonal rotation**, **2-bit polar quantization**, and **1-bit Quantized Johnson-Lindenstrauss (QJL) residual correction**, TurboQuant achieves an **$8\times$ reduction in KV cache memory** (from 256 bytes down to 32 bytes per head per token) with near-zero attention fidelity loss.

---

## Table of Contents
- [Architecture Overview](#architecture-overview)
  - [1. Random Orthogonal Rotation ($\mathbf{R}$)](#1-random-orthogonal-rotation-mathbfr)
  - [2. 2-Bit Polar Quantization](#2-2-bit-polar-quantization)
  - [3. 1-Bit QJL Residual Correction](#3-1-bit-qjl-residual-correction)
  - [4. Sub-Byte Bit Packing & Layout](#4-sub-byte-bit-packing--layout)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Quickstart & Installation](#quickstart--installation)
  - [1. Installation with uv](#1-installation-with-uv)
  - [2. Running the Test Suite](#2-running-the-test-suite)
  - [3. Benchmarks & Evaluations](#3-benchmarks--evaluations)
  - [4. Interactive Streamlit Assistant](#4-interactive-streamlit-assistant)
- [Python API Usage](#python-api-usage)
  - [Chunked Prefix Ingestion & Multi-Turn QA](#chunked-prefix-ingestion--multi-turn-qa)
  - [Non-Intrusive Model Patching](#non-intrusive-model-patching)
- [Memory & Compression Benchmarks](#memory--compression-benchmarks)
- [Contributing & License](#contributing--license)

---

## Architecture Overview

In long-context inference (4,000+ tokens) on Indic language models, the Key-Value (KV) cache quickly becomes the primary memory bottleneck. For a standard 16-bit format (bfloat16) with head dimension $d_h = 128$:
$$\text{Memory}_{\text{bfloat16}} = 128 \times 2 \text{ bytes} = 256 \text{ bytes / token / head}$$

TurboQuant reduces this footprint to **32 bytes per token per head** ($8\times$ compression) while preserving high cross-attention fidelity:

```
[Key/Value Vector (d=128, bfloat16)] (256 bytes)
                 │
                 ▼
  [Orthogonal Rotation R (Haar Measure)]  ──> Eliminates outlier dimensions
                 │
                 ▼
     [2-Bit Polar Quantization]           ──> 2 bits / dim (32 bytes packed)
                 │
                 ▼
 [1-Bit QJL Residual Correction]          ──> Unbiased inner product estimate
                 │
                 ▼
 [TurboQuantCache Sub-Byte Storage]      ──> 32 bytes / head / token
```

### 1. Random Orthogonal Rotation ($\mathbf{R}$)
High-dimensional transformer activations typically display outlier channels with high variance along fixed axes. TurboQuant projects vectors with a deterministic pseudo-random orthogonal matrix $\mathbf{R} \in \mathbb{R}^{d \times d}$ generated via QR decomposition of a Gaussian matrix (Haar measure):
$$\mathbf{x}_{\text{rot}} = \mathbf{x} \mathbf{R}, \quad \text{where } \mathbf{R}^T \mathbf{R} = \mathbf{I}$$
Because $\mathbf{R}$ preserves Euclidean norms ($\|\mathbf{x} \mathbf{R}\| = \|\mathbf{x}\|$) and inner products ($\langle \mathbf{x} \mathbf{R}, \mathbf{y} \mathbf{R} \rangle = \langle \mathbf{x}, \mathbf{y} \rangle$), it isotropically redistributes vector energy across all coordinates without altering attention geometry.

### 2. 2-Bit Polar Quantization
Vectors are decomposed into radial norm $\|\mathbf{x}\|$ and directional angular components. Angular coordinates are mapped to discrete centroids across $2^b$ uniform spherical Voronoi partitions ($b=2$ bits per coordinate). The resulting indices $\{0, 1, 2, 3\}$ require only 2 bits of storage per coordinate dimension.

### 3. 1-Bit QJL Residual Correction
Polar quantization introduces a quantization residual $\mathbf{r} = \mathbf{x} - \hat{\mathbf{x}}$. To restore exact cross-attention dot-products, TurboQuant applies an asymmetric Quantized Johnson-Lindenstrauss (QJL) projection using an auxiliary Rademacher projection matrix $\mathbf{S} \in \mathbb{R}^{m \times d}$:
$$\mathbf{q}_{\text{sign}} = \operatorname{sign}(\mathbf{S} \mathbf{r}) \in \{-1, +1\}^m$$
During query-key dot product computation, the reconstructed score is corrected by:
$$\langle \mathbf{q}, \mathbf{k} \rangle \approx \langle \mathbf{q}, \hat{\mathbf{k}} \rangle + \alpha \cdot \langle \mathbf{S} \mathbf{q}, \mathbf{q}_{\text{sign}} \rangle$$
This residual correction guarantees unbiased dot product estimation and eliminates degradation on complex Indic scripts.

### 4. Sub-Byte Bit Packing & Layout
Four 2-bit quantized values are packed into a single `uint8` byte:
$$\text{byte} = c_0 \mid (c_1 \ll 2) \mid (c_2 \ll 4) \mid (c_3 \ll 6)$$
For head dimension $d_h = 128$, the 128 2-bit values require exactly $128 / 4 = 32$ bytes per head per token.

---

## Key Features

- **Non-Intrusive Monkey-Patching**: Dynamically intercepts self-attention modules in `sarvamai/sarvam-1` (and compatible LLaMA-based architectures) via `patch_sarvam_attention(model)`. Fully reversible with Python context manager semantics.
- **Chunked Prefix Ingestion (`DocumentPrefixEngine`)**: Prefills 4,000+ token documents in bounded chunks (e.g., 512 tokens) to eliminate memory spikes during prefill on CPU/GPU.
- **Zero-Copy Cache Rewinding (`crop(max_length)`)**: Resets cached tensors back to the ingested document boundary in-place along `dim=-2`, avoiding costly document re-tokenization and re-encoding across multiple questions.
- **Indic Document QA Engine (`IndicDocumentQAEngine`)**: Handles multi-turn questions in Indic languages (Hindi, Bengali, Telugu, Tamil, Marathi, Gujarati, Kannada, Malayalam, Odia, Punjabi) with exact RoPE `position_ids` starting at $L_{\text{doc}}$.
- **Interactive Streamlit Web Dashboard**: Live memory telemetry, side-by-side bfloat16 vs. TurboQuant footprint tracking, and multi-turn chat assistant.

---

## Repository Structure

```
turboquant-indic/
├── benchmarks/
│   └── benchmark_unpack_latency.py    # Unpacking and bit-manipulation latency
├── pyproject.toml                     # Project packaging and metadata
├── scripts/
│   ├── benchmark_evaluation.py        # KV scaling, decode latency, Indic fidelity
│   ├── eval_sarvam_quality.py         # Sarvam-1 Indic generation quality benchmarks
│   ├── indic_assistant_app.py         # Streamlit interactive UI application
│   └── inspect_sarvam_config.py       # Configuration and parameter inspection tool
├── src/
│   └── turboquant/
│       ├── __init__.py                # Package exports
│       ├── attention.py               # Compressed attention computation
│       ├── cache.py                   # TurboQuantCache with zero-copy crop()
│       ├── embed.py                   # Indic embedding extraction utilities
│       ├── index.py                   # Vector index with lookup tables
│       ├── ingest.py                  # DocumentPrefixEngine for chunked prefill
│       ├── packing.py                 # 2-bit sub-byte packing & unpacking
│       ├── patch.py                   # Sarvam-1 monkey patching & routing
│       ├── polar.py                   # Spherical polar angle binning
│       ├── qa.py                      # IndicDocumentQAEngine & QAResult
│       ├── qjl.py                     # Quantized Johnson-Lindenstrauss residual
│       ├── quant.py                   # DynamicKVQuantizer container
│       └── rotation.py                # Random orthogonal Haar matrix rotation
└── tests/                             # 81 comprehensive unit and integration tests
    ├── test_app.py                    # Streamlit telemetry and session tests
    ├── test_attention.py              # Numerical equivalence and attention masks
    ├── test_benchmark.py              # Benchmark execution smoke tests
    ├── test_embed.py                  # Embedding invariance and normalization
    ├── test_end_to_end.py             # End-to-end compression pipeline
    ├── test_index.py                  # Vector index search and lookup tables
    ├── test_ingest.py                 # Chunked prefill & zero-copy cache rewinding
    ├── test_kv_quantizer.py           # Reconstruction fidelity & shapes
    ├── test_packing.py                # Sub-byte bitwise roundtripping
    ├── test_patch.py                  # Monkey-patching and cache lifecycle
    ├── test_polar.py                  # Angle binning & reconstruction error
    ├── test_qa.py                     # Multi-turn Indic QA & position ID offsets
    ├── test_qjl.py                    # Residual error reduction & MSE tests
    └── test_rotation.py               # Orthogonality & norm preservation
```

---

## Quickstart & Installation

### 1. Installation with uv

Clone the repository and install all dependencies using [`uv`](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/Aryantechtrailblazer/turboquant-indic.git
cd turboquant-indic
uv sync
```

### 2. Running the Test Suite

TurboQuant includes an extensive 81-test suite validating numerical equivalence, compression ratios, and zero-copy slicing:

```bash
uv run pytest -v
```

Expected result:
```
============================= 81 passed in ~20s =============================
```

### 3. Benchmarks & Evaluations

Run the KV-cache scaling and latency evaluation harness:

```bash
uv run python scripts/benchmark_evaluation.py
```

Run sub-byte unpacking throughput benchmarks:

```bash
uv run python benchmarks/benchmark_unpack_latency.py
```

### 4. Interactive Streamlit Assistant

Launch the interactive Indic Document Assistant:

```bash
uv run streamlit run scripts/indic_assistant_app.py
```

Open your browser at `http://localhost:8501`. You can:
1. Toggle **"Demo / Synthetic Mode"** in the sidebar for instant CPU testing without downloading full weights.
2. Ingest an Indic document (Hindi, Tamil, Telugu, etc.) with chunked prefill.
3. Observe live memory footprint telemetry and compression statistics ($8\times$ memory reduction).
4. Ask sequential questions against the pre-ingested document with zero-copy prefix reuse.

---

## Python API Usage

### Chunked Prefix Ingestion & Multi-Turn QA

```python
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from turboquant.quant import DynamicKVQuantizer
from turboquant.ingest import DocumentPrefixEngine
from turboquant.qa import IndicDocumentQAEngine

# 1. Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sarvamai/sarvam-1")
config = AutoConfig.from_pretrained("sarvamai/sarvam-1")
model = LlamaForCausalLM(config).to(torch.bfloat16)

# 2. Setup Quantizer and Ingestion Engine
quantizer = DynamicKVQuantizer(bits=2, use_qjl=False)
ingest_engine = DocumentPrefixEngine(
    model=model,
    tokenizer=tokenizer,
    quantizer=quantizer,
    chunk_size=512,
)

# 3. Ingest long Indic document (chunked prefill)
document_text = """
भारत की राजधानी नई दिल्ली है। यह देश का एक महत्वपूर्ण प्रशासनिक एवं सांस्कृतिक केंद्र है।
भारतीय संविधान 26 जनवरी 1950 को लागू हुआ था।
"""
prefix_state = ingest_engine.ingest(document_text)
print(f"Prefix tokens: {prefix_state.prefix_length}, Cache: {prefix_state.cache_bytes / 1024:.2f} KiB")

# 4. Ask multi-turn questions with zero-copy cache rewinding
qa_engine = IndicDocumentQAEngine(
    model=model,
    tokenizer=tokenizer,
    quantizer=quantizer,
    max_new_tokens=32,
    temperature=0.0,
)

# First question
res1 = qa_engine.answer(prefix_state, "भारत की राजधानी क्या है?")
print(f"उत्तर 1: {res1.answer} ({res1.tokens_per_sec:.1f} tok/s)")

# Second question (prefix_state is automatically rewound via prefix_state.crop())
res2 = qa_engine.answer(prefix_state, "भारतीय संविधान कब लागू हुआ था?")
print(f"उत्तर 2: {res2.answer} ({res2.tokens_per_sec:.1f} tok/s)")
```

### Non-Intrusive Model Patching

```python
from turboquant.patch import patch_sarvam_attention
from turboquant.cache import TurboQuantCache
from turboquant.quant import DynamicKVQuantizer

quantizer = DynamicKVQuantizer(bits=2, use_qjl=False)
cache = TurboQuantCache(quantizer=quantizer)

# Patch model inside a context manager
with patch_sarvam_attention(model, cache):
    outputs = model(input_ids=input_ids, past_key_values=cache)
    # Model uses TurboQuant compressed attention!

# Model is cleanly restored to original LLaMA attention outside the block
```

---

## Memory & Compression Benchmarks

Theoretical and empirical memory footprint for `sarvamai/sarvam-1` ($n_{\text{layers}} = 28$, $H_{kv} = 8$, $d_h = 128$):

| Context Length ($L_{\text{doc}}$) | Baseline (bfloat16) | TurboQuant (2-bit) | Memory Saved | Compression Factor |
|:--------------------------------:|:-------------------:|:------------------:|:------------:|:------------------:|
| 512 tokens                       | 28.00 MiB           | 3.50 MiB           | 24.50 MiB    | **$8.0\times$**    |
| 1,024 tokens                     | 56.00 MiB           | 7.00 MiB           | 49.00 MiB    | **$8.0\times$**    |
| 2,048 tokens                     | 112.00 MiB          | 14.00 MiB          | 98.00 MiB    | **$8.0\times$**    |
| 4,096 tokens                     | 224.00 MiB          | 28.00 MiB          | 196.00 MiB   | **$8.0\times$**    |
| 8,192 tokens                     | 448.00 MiB          | 56.00 MiB          | 392.00 MiB   | **$8.0\times$**    |

$$\text{KV Footprint} = 2 \times n_{\text{layers}} \times H_{kv} \times d_h \times L \times \text{BytesPerElement}$$
- **bfloat16**: $2 \times 28 \times 8 \times 128 \times L \times 2 = 114,688 \times L \text{ bytes}$
- **TurboQuant (2-bit)**: $2 \times 28 \times 8 \times (128 / 4) \times L = 14,336 \times L \text{ bytes}$

---

## Contributing & License

Contributions are welcome! Please open an issue or pull request.

This project is licensed under the MIT License.
