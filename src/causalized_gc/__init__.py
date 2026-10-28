"""Unified CPU and NVIDIA GPU runtime backends for c-GC and c-GC*."""

from .capabilities import (
    BackendDecision,
    BackendUnavailableError,
    GpuProbe,
    probe_gpu,
)
from .estimator import GcStar

__all__ = [
    "BackendDecision",
    "BackendUnavailableError",
    "GcStar",
    "GpuProbe",
    "probe_gpu",
]

__version__ = "0.1.0"
