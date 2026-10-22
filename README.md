# causalized-GC

This project implements causalized Granger causality (c-GC), including checks
for spurious links induced by latent confounders. It provides one public
estimator that can run the conventional CPU implementation or an NVIDIA GPU
implementation derived from fastCGC.

## Installation

For development and CPU use:

```bash
uv sync
```

The GPU backend requires a working NVIDIA CUDA environment plus CuPy. cuML is
optional and enables the cuML regression path. GPU dependencies are
platform-sensitive: choose exactly one extra that matches the CUDA major
version installed on the target machine.

```bash
# CUDA 12
uv sync --extra gpu-cu12
# or: uv pip install --extra-index-url https://pypi.nvidia.com \
#       -e '.[gpu-cu12]'

# CUDA 13
uv sync --extra gpu-cu13
# or: uv pip install --extra-index-url https://pypi.nvidia.com \
#       -e '.[gpu-cu13]'
```

Install CUDA-specific CuPy/RAPIDS wheels according to their upstream
compatibility matrices; do not mix wheels built for a different CUDA major
version.

## Unified CPU/GPU API

Import `GcStar` from the public package:

```python
from causalized_gc import GcStar

model = GcStar(
    n_perm=1000,
    n_pasts=3,
    n_lags=1,
    method="cgc",
    backend="auto",
)
model.fit(data)

print(model.selected_backend_)      # "cpu" or "gpu"
print(model.backend_diagnostics_)   # selection details and fallback reason
```

The `backend` option controls execution:

- `backend="auto"` uses a compatible NVIDIA GPU when the required runtime is
  available, otherwise it falls back to the conventional CPU implementation.
- `backend="cpu"` always uses the conventional implementation.
- `backend="gpu"` requires the NVIDIA backend and raises a clear error when it
  is unavailable; it does not silently fall back to CPU.

GPU users can also choose `regression_backend="auto"`, `"cupy_lstsq"`, or
`"cuml"`. The selected execution backend is exposed as `selected_backend_`,
while `backend_diagnostics_` records why that backend was selected. These are
useful in notebooks and benchmark reports:

```python
model = GcStar(backend="auto", regression_backend="auto")
print(model.selected_backend_)
print(model.backend_diagnostics_)
```

## GPU validation status

The integration and CPU fallback can be exercised on non-GPU machines, but the
NVIDIA implementation has not yet been validated on representative GPU
hardware. Numerical parity, memory use, and end-to-end performance benchmarks
against the CPU implementation are still pending. Until those checks are
complete, record `selected_backend_` and `backend_diagnostics_` with every GPU
result and avoid treating speed-up or parity as established.

Run the comparison benchmark on the target GPU machine and retain its JSON
output with the hardware and CUDA details:

```bash
python benchmarks/bench_backends.py --out benchmark.json \
  --n-variables 5 10 --n-timepoints 256 1024 --n-perm 32 128 \
  --repeats 3 --regression-backend auto
```

Example notebooks are available under `examples/`. Their `GcStar` imports use
the unified estimator and default to `backend="auto"`, so the same notebook can
run on CPU-only and compatible NVIDIA GPU machines.
