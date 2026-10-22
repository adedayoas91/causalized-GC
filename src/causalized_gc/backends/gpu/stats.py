"""Array-module-agnostic statistical primitives for the GPU backend.

The helpers accept either CuPy for device execution or NumPy for CPU-only
contract tests. Arrays have shape ``(n_rows, n_timepoints)``.
"""

from __future__ import annotations

from typing import Any

ArrayModule = Any
Array = Any


def center_rows(xp: ArrayModule, data: Array) -> Array:
    """Subtract the mean of each row along the time axis."""

    return data - data.mean(axis=1, keepdims=True)


def row_norms(xp: ArrayModule, centered: Array) -> Array:
    """Return the L2 norm of each centered row as ``(n_rows, 1)``."""

    return xp.sqrt((centered * centered).sum(axis=1, keepdims=True))


def safe_divide(xp: ArrayModule, numerator: Array, denominator: Array) -> Array:
    """Divide while replacing zero denominators with one."""

    safe = xp.where(denominator > 0, denominator, xp.ones_like(denominator))
    return numerator / safe


def normalize_rows(xp: ArrayModule, data: Array) -> tuple[Array, Array]:
    """Center and L2-normalize rows, returning values and a validity mask."""

    centered = center_rows(xp, data)
    norms = row_norms(xp, centered)
    unit = safe_divide(xp, centered, norms)
    valid = norms[:, 0] > 0
    return unit, valid


def abs_correlation_matrix(
    xp: ArrayModule, data: Array
) -> tuple[Array, Array]:
    """Return absolute row correlations and the non-constant-row mask."""

    unit, valid = normalize_rows(xp, data)
    corr = xp.abs(unit @ unit.T)
    return corr, valid


def pairwise_abs_correlation(
    xp: ArrayModule, x_rows: Array, y_rows: Array
) -> tuple[Array, Array]:
    """Return absolute correlations of aligned row pairs and validity."""

    x_unit, x_valid = normalize_rows(xp, x_rows)
    y_unit, y_valid = normalize_rows(xp, y_rows)
    corr = xp.abs((x_unit * y_unit).sum(axis=1))
    valid = x_valid & y_valid
    return corr, valid


__all__ = [
    "center_rows",
    "row_norms",
    "safe_divide",
    "normalize_rows",
    "abs_correlation_matrix",
    "pairwise_abs_correlation",
]
