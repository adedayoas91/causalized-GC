"""Runtime capability checks for the optional NVIDIA backend."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import platform
from typing import Literal


BackendName = Literal["auto", "cpu", "gpu"]


class BackendUnavailableError(RuntimeError):
    """Raised when an explicitly requested execution backend is unavailable."""


@dataclass(frozen=True)
class GpuProbe:
    """Result of checking one CUDA device and the optional cuML runtime."""

    available: bool
    reason: str
    device_id: int
    device_count: int = 0
    device_name: str | None = None
    compute_capability: tuple[int, int] | None = None
    cupy_version: str | None = None
    cuml_available: bool | None = None
    platform: str = platform.platform()


@dataclass(frozen=True)
class BackendDecision:
    """Public record of how an estimator backend was selected."""

    requested: BackendName
    selected: Literal["cpu", "gpu"]
    reason: str
    gpu: GpuProbe | None = None


def _decode_name(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


@lru_cache(maxsize=None)
def probe_gpu(
    *,
    device_id: int = 0,
    require_cuml: bool = False,
    minimum_compute_capability: tuple[int, int] = (7, 0),
) -> GpuProbe:
    """Check whether a concrete NVIDIA device is usable by the GPU backend.

    The probe is deliberately lazy: importing :mod:`causalized_gc` never imports
    CuPy or initializes CUDA. Driver/runtime failures are converted to structured
    diagnostics so ``backend="auto"`` can choose the CPU implementation.
    """

    if device_id < 0:
        return GpuProbe(
            available=False,
            reason="device_id must be non-negative",
            device_id=device_id,
        )

    try:
        import cupy as cp
    except Exception as exc:  # optional dependency and driver-link failures
        return GpuProbe(
            available=False,
            reason=f"CuPy is not importable: {exc}",
            device_id=device_id,
        )

    cupy_version = getattr(cp, "__version__", "unknown")
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        return GpuProbe(
            available=False,
            reason=f"CUDA runtime initialization failed: {exc}",
            device_id=device_id,
            cupy_version=cupy_version,
        )

    if device_count == 0:
        return GpuProbe(
            available=False,
            reason="CUDA reported no NVIDIA devices",
            device_id=device_id,
            cupy_version=cupy_version,
        )
    if device_id >= device_count:
        return GpuProbe(
            available=False,
            reason=f"CUDA device {device_id} does not exist (found {device_count})",
            device_id=device_id,
            device_count=device_count,
            cupy_version=cupy_version,
        )

    try:
        properties = cp.cuda.runtime.getDeviceProperties(device_id)
        name = _decode_name(properties.get("name", f"CUDA device {device_id}"))
        capability = (int(properties["major"]), int(properties["minor"]))
        with cp.cuda.Device(device_id):
            cp.zeros(1).sum().item()
    except Exception as exc:
        return GpuProbe(
            available=False,
            reason=f"CUDA device {device_id} is not usable: {exc}",
            device_id=device_id,
            device_count=device_count,
            cupy_version=cupy_version,
        )

    if capability < minimum_compute_capability:
        required = ".".join(map(str, minimum_compute_capability))
        actual = ".".join(map(str, capability))
        return GpuProbe(
            available=False,
            reason=(
                f"{name} has compute capability {actual}; "
                f"the GPU backend requires at least {required}"
            ),
            device_id=device_id,
            device_count=device_count,
            device_name=name,
            compute_capability=capability,
            cupy_version=cupy_version,
        )

    cuml_available: bool | None = None
    if require_cuml:
        try:
            import cuml  # noqa: F401
        except Exception as exc:
            return GpuProbe(
                available=False,
                reason=f"cuML was requested but is not importable: {exc}",
                device_id=device_id,
                device_count=device_count,
                device_name=name,
                compute_capability=capability,
                cupy_version=cupy_version,
                cuml_available=False,
            )
        cuml_available = True

    return GpuProbe(
        available=True,
        reason=f"CUDA device {device_id} is compatible",
        device_id=device_id,
        device_count=device_count,
        device_name=name,
        compute_capability=capability,
        cupy_version=cupy_version,
        cuml_available=cuml_available,
    )


def select_backend(
    requested: BackendName,
    *,
    device_id: int = 0,
    require_cuml: bool = False,
) -> BackendDecision:
    """Resolve a backend request according to the public fallback policy."""

    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError("backend must be one of 'auto', 'cpu', or 'gpu'")
    if requested == "cpu":
        return BackendDecision(
            requested="cpu",
            selected="cpu",
            reason="CPU execution was explicitly requested",
        )

    probe = probe_gpu(device_id=device_id, require_cuml=require_cuml)
    if probe.available:
        return BackendDecision(
            requested=requested,
            selected="gpu",
            reason=probe.reason,
            gpu=probe,
        )
    if requested == "gpu":
        raise BackendUnavailableError(
            f"GPU execution was requested but is unavailable: {probe.reason}. "
            "Install the optional CUDA extra matching the machine, or use "
            "backend='auto'/'cpu'."
        )
    return BackendDecision(
        requested="auto",
        selected="cpu",
        reason=f"GPU unavailable; using CPU: {probe.reason}",
        gpu=probe,
    )


def reset_gpu_probe_cache() -> None:
    """Clear cached probe results (primarily for tests and changed runtimes)."""

    probe_gpu.cache_clear()


__all__ = [
    "BackendDecision",
    "BackendName",
    "BackendUnavailableError",
    "GpuProbe",
    "probe_gpu",
    "reset_gpu_probe_cache",
    "select_backend",
]
