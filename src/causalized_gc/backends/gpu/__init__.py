"""NVIDIA CUDA runtime backend for c-GC and c-GC*."""

from __future__ import annotations

from .backend import CUML_AVAILABLE, GPU_AVAILABLE
from .estimator import GcStar

GpuGcStar = GcStar

__all__ = ["GcStar", "GpuGcStar", "GPU_AVAILABLE", "CUML_AVAILABLE"]
