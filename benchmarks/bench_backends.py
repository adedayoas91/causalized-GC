"""Compare conventional CPU and NVIDIA GPU c-GC implementations.

Run this on the target NVIDIA machine after installing a matching GPU extra.
The JSON artifact records timings and runtime metadata; it does not assume the
GPU is faster for every problem size.

Example::

    python benchmarks/bench_backends.py --out benchmark.json \
        --n-variables 5 10 --n-timepoints 256 1024 --n-perm 32 128 \
        --repeats 3 --regression-backend cuml
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import statistics
import time
from typing import Any

import numpy as np

from causalized_gc import GcStar, probe_gpu


@dataclass(frozen=True)
class TimingSummary:
    backend: str
    samples_s: list[float]
    median_s: float
    minimum_s: float
    maximum_s: float


def make_data(n_variables: int, n_timepoints: int, seed: int) -> np.ndarray:
    """Create repeatable data with a small number of lagged couplings."""

    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n_variables, n_timepoints))
    if n_variables > 1 and n_timepoints > 2:
        data[1, 1:] += 0.4 * data[0, :-1]
    if n_variables > 2 and n_timepoints > 3:
        data[2, 2:] -= 0.3 * data[1, :-2]
    return data


def synchronize_gpu() -> None:
    """Wait for queued GPU work without making CuPy a base dependency."""

    try:
        import cupy as cp
    except Exception:
        return
    cp.cuda.runtime.deviceSynchronize()


def timed_fit(
    data: np.ndarray,
    *,
    backend: str,
    n_perm: int,
    n_pasts: int,
    n_lags: int,
    method: str,
    seed: int,
    repeats: int,
    warmups: int,
    regression_backend: str,
    batch_size: int | None,
    async_passes: bool,
    device_id: int,
) -> tuple[TimingSummary, list[int]]:
    """Time complete public ``fit`` and connectivity calls."""

    durations: list[float] = []
    connectivity_shape: list[int] = []
    for run in range(warmups + repeats):
        model = GcStar(
            n_perm=n_perm,
            n_pasts=n_pasts,
            n_lags=n_lags,
            method=method,  # type: ignore[arg-type]
            backend=backend,  # type: ignore[arg-type]
            device_id=device_id,
            dtype="float32",
            random_state=seed,
            batch_size=batch_size,
            regression_backend=regression_backend,  # type: ignore[arg-type]
            async_passes=async_passes,
        )
        synchronize_gpu()
        started = time.perf_counter()
        model.fit(data, verbose=0)
        connectivity = model.get_connectivity_matrix()
        synchronize_gpu()
        elapsed = time.perf_counter() - started
        connectivity_shape = list(connectivity.shape)
        if run >= warmups:
            durations.append(elapsed)

    return (
        TimingSummary(
            backend=backend,
            samples_s=durations,
            median_s=statistics.median(durations),
            minimum_s=min(durations),
            maximum_s=max(durations),
        ),
        connectivity_shape,
    )


def runtime_context(device_id: int, seed: int) -> dict[str, Any]:
    context: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "seed": seed,
        "gpu_probe": asdict(probe_gpu(device_id=device_id)),
    }
    try:
        import cupy as cp

        context["cupy"] = cp.__version__
        context["cuda_runtime"] = int(cp.cuda.runtime.runtimeGetVersion())
    except Exception as exc:
        context["cupy_error"] = str(exc)
    try:
        import cuml

        context["cuml"] = cuml.__version__
    except Exception as exc:
        context["cuml_error"] = str(exc)
    return context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("benchmark.json"))
    parser.add_argument("--n-variables", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--n-timepoints", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--n-perm", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--n-pasts", type=int, default=1)
    parser.add_argument("--n-lags", type=int, default=1)
    parser.add_argument("--method", choices=["cgc", "fcgc"], default="cgc")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--regression-backend",
        choices=["auto", "cupy_lstsq", "cuml"],
        default="auto",
    )
    parser.add_argument("--async-passes", action="store_true")
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="Skip CPU timing when it would be prohibitively slow.",
    )
    args = parser.parse_args()

    if args.repeats < 1 or args.warmups < 0:
        parser.error("repeats must be positive and warmups must be non-negative")
    probe = probe_gpu(
        device_id=args.device_id,
        require_cuml=args.regression_backend == "cuml",
    )
    if not probe.available:
        raise SystemExit(f"GPU benchmark unavailable: {probe.reason}")

    results: list[dict[str, Any]] = []
    for n_variables in args.n_variables:
        for n_timepoints in args.n_timepoints:
            for n_perm in args.n_perm:
                data = make_data(n_variables, n_timepoints, args.seed)
                case: dict[str, Any] = {
                    "n_variables": n_variables,
                    "n_timepoints": n_timepoints,
                    "n_perm": n_perm,
                    "n_pasts": args.n_pasts,
                    "n_lags": args.n_lags,
                    "method": args.method,
                    "regression_backend": args.regression_backend,
                    "async_passes": args.async_passes,
                }
                if not args.gpu_only:
                    cpu, cpu_shape = timed_fit(
                        data,
                        backend="cpu",
                        n_perm=n_perm,
                        n_pasts=args.n_pasts,
                        n_lags=args.n_lags,
                        method=args.method,
                        seed=args.seed,
                        repeats=args.repeats,
                        warmups=args.warmups,
                        regression_backend=args.regression_backend,
                        batch_size=args.batch_size,
                        async_passes=args.async_passes,
                        device_id=args.device_id,
                    )
                    case["cpu"] = asdict(cpu)
                    case["cpu_connectivity_shape"] = cpu_shape

                gpu, gpu_shape = timed_fit(
                    data,
                    backend="gpu",
                    n_perm=n_perm,
                    n_pasts=args.n_pasts,
                    n_lags=args.n_lags,
                    method=args.method,
                    seed=args.seed,
                    repeats=args.repeats,
                    warmups=args.warmups,
                    regression_backend=args.regression_backend,
                    batch_size=args.batch_size,
                    async_passes=args.async_passes,
                    device_id=args.device_id,
                )
                case["gpu"] = asdict(gpu)
                case["gpu_connectivity_shape"] = gpu_shape
                if not args.gpu_only:
                    case["speedup_cpu_over_gpu"] = cpu.median_s / gpu.median_s
                results.append(case)
                speedup = case.get("speedup_cpu_over_gpu")
                suffix = "" if speedup is None else f", speedup={speedup:.2f}x"
                print(
                    f"variables={n_variables}, T={n_timepoints}, perms={n_perm}: "
                    f"gpu={gpu.median_s:.4f}s{suffix}"
                )

    payload = {"context": runtime_context(args.device_id, args.seed), "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} benchmark cases to {args.out}")


if __name__ == "__main__":
    main()
