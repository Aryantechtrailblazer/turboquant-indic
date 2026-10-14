"""Pytest configuration and environment setup for TurboQuant test suite."""

import os
from pathlib import Path
import sys

# Ensure repository root is on sys.path for importing scripts and packages
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Limit OpenBLAS and OMP thread allocations on Windows to prevent buffer memory errors
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
