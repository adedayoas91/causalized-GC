"""GPU residual regression primitives for conditional dependence."""

from __future__ import annotations

from typing import Any

from .backend import require_cuml

ArrayModule = Any
Array = Any

_VALID_BACKENDS = {"auto", "cupy_lstsq", "cuml"}


def resolve_backend(regression_backend: str) -> str:
    """Validate and resolve the GPU regression implementation."""

    if regression_backend not in _VALID_BACKENDS:
        raise ValueError(
            f"regression_backend must be one of {sorted(_VALID_BACKENDS)}, "
            f"got {regression_backend!r}."
        )
    if regression_backend == "auto":
        return "cupy_lstsq"
    return regression_backend


def residual_pair(
    xp: ArrayModule,
    x: Array,
    y: Array,
    z: Array,
    *,
    backend: str = "cupy_lstsq",
) -> tuple[Array, Array]:
    """Regress ``z`` out of ``x`` and ``y`` and return their residuals."""

    if z.shape[0] == 0:
        return x - x.mean(), y - y.mean()

    design = z.T
    targets = xp.stack([x, y], axis=1)

    if backend == "cuml":
        return _residual_cuml(design, targets)

    # The conventional implementation uses sklearn LinearRegression with an
    # intercept. Add the intercept column explicitly for the CuPy least-squares
    # path so CPU and GPU residuals represent the same model.
    intercept = xp.ones((design.shape[0], 1), dtype=design.dtype)
    augmented = xp.concatenate([design, intercept], axis=1)
    coef, *_ = xp.linalg.lstsq(augmented, targets, rcond=None)
    residuals = targets - augmented @ coef
    return residuals[:, 0], residuals[:, 1]


def _residual_cuml(design: Array, targets: Array) -> tuple[Array, Array]:
    """Compute residuals with cuML ``LinearRegression``."""

    linear_regression = require_cuml()
    residuals: list[Array] = []
    for col in range(targets.shape[1]):
        model = linear_regression(fit_intercept=True, output_type="cupy")
        model.fit(design, targets[:, col])
        residuals.append(targets[:, col] - model.predict(design))
    return residuals[0], residuals[1]


__all__ = ["resolve_backend", "residual_pair"]
