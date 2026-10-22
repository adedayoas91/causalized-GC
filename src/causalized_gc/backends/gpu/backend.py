"""Optional CuPy and cuML imports for the GPU c-GC backend.

Importing this module is safe on machines without a RAPIDS installation.
Missing optional dependencies are reported only when GPU execution or the
cuML regression implementation is requested.
"""

from __future__ import annotations

from types import ModuleType

try:  # pragma: no cover - exercised only on a CUDA-capable host
    import cupy as _cupy

    cp: ModuleType | None = _cupy
except Exception:  # noqa: BLE001 - any import failure means no usable CuPy
    cp = None

try:  # pragma: no cover - exercised only on a RAPIDS-capable host
    from cuml.linear_model import LinearRegression as _LinearRegression
    from cuml.linear_model import Ridge as _Ridge

    LinearRegression: type | None = _LinearRegression
    Ridge: type | None = _Ridge
except Exception:  # noqa: BLE001 - cuML is an optional dependency
    LinearRegression = None
    Ridge = None


# These flags describe import availability only. Device/runtime compatibility
# is checked by the package-level capability probe before this backend is used.
GPU_AVAILABLE: bool = cp is not None
CUML_AVAILABLE: bool = LinearRegression is not None


_INSTALL_HINT = (
    "The GPU c-GC backend requires a working CuPy/CUDA installation.\n"
    "Install the optional dependency matching your CUDA line, for example:\n"
    '    uv pip install -e ".[gpu-cu12]"   # CUDA 12\n'
    '    uv pip install -e ".[gpu-cu13]"   # CUDA 13\n'
    "RAPIDS requires a compatible NVIDIA GPU on Linux or WSL2.\n"
    "Use backend='cpu' when no compatible GPU is available."
)


def require_cupy() -> ModuleType:
    """Return CuPy or raise an actionable error for GPU execution."""

    if cp is None:
        raise ImportError(_INSTALL_HINT)
    return cp


def require_cuml() -> type:
    """Return cuML ``LinearRegression`` or raise an actionable error."""

    if LinearRegression is None:
        raise ImportError(
            "regression_backend='cuml' requires cuML "
            '(install e.g. `uv pip install -e ".[gpu-cu12]"`).'
        )
    return LinearRegression


__all__ = [
    "cp",
    "LinearRegression",
    "Ridge",
    "GPU_AVAILABLE",
    "CUML_AVAILABLE",
    "require_cupy",
    "require_cuml",
]
