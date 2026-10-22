"""Batched circular-shift permutation p-values for the GPU backend."""

from __future__ import annotations

from typing import Any

import numpy as np

from .stats import normalize_rows

ArrayModule = Any
Array = Any


def draw_shifts(
    n_perm: int,
    n_timepoints: int,
    random_state: int | None,
    minimum_shift: int = 30,
) -> np.ndarray:
    """Draw a reproducible host-side circular-shift schedule.

    The conventional implementation draws from ``[30, T)`` to keep surrogate
    samples away from near-zero temporal shifts. For shorter inputs, clamp the
    lower bound to ``T - 1`` instead of raising the opaque NumPy range error.
    """

    if n_perm <= 0 or n_timepoints <= 1:
        return np.empty(0, dtype=np.int64)
    lower = min(max(minimum_shift, 1), n_timepoints - 1)
    rng = np.random.default_rng(random_state)
    return rng.integers(lower, n_timepoints, size=n_perm, dtype=np.int64)


def _shift_index(xp: ArrayModule, n_timepoints: int, shifts: Array) -> Array:
    """Build ``(n_shifts, n_timepoints)`` circular gather indices."""

    base = xp.arange(n_timepoints)
    return (base[None, :] + shifts[:, None]) % n_timepoints


def marginal_pvalues(
    xp: ArrayModule,
    shifted: Array,
    n_targets: int,
    n_perm: int,
    shifts: Array,
    batch_size: int,
) -> Array:
    """Compute permutation p-values for every row/target marginal pair."""

    n_rows, n_timepoints = shifted.shape
    if n_timepoints <= 1 or n_perm <= 0:
        return xp.ones((n_rows, n_targets))

    unit, valid = normalize_rows(xp, shifted)
    targets = unit[:n_targets]
    observed = xp.abs(unit @ targets.T)

    counts = xp.zeros((n_rows, n_targets))
    index = _shift_index(xp, n_timepoints, shifts)
    for start in range(0, n_perm, batch_size):
        batch = index[start : start + batch_size]
        rolled = unit[:, batch]
        perm = xp.abs(xp.einsum("rbt,jt->rbj", rolled, targets))
        counts += (perm >= observed[:, None, :]).sum(axis=1)

    pvals = counts / n_perm
    pvals[~valid, :] = 1.0
    pvals[:, ~valid[:n_targets]] = 1.0
    return pvals


def aligned_pvalues(
    xp: ArrayModule,
    x_rows: Array,
    y_rows: Array,
    n_perm: int,
    shifts: Array,
    batch_size: int,
) -> Array:
    """Compute permutation p-values for aligned ``x``/``y`` row pairs."""

    n_pairs, n_timepoints = x_rows.shape
    if n_timepoints <= 1 or n_perm <= 0:
        return xp.ones(n_pairs)

    x_unit, x_valid = normalize_rows(xp, x_rows)
    y_unit, y_valid = normalize_rows(xp, y_rows)
    valid = x_valid & y_valid
    observed = xp.abs((x_unit * y_unit).sum(axis=1))

    counts = xp.zeros(n_pairs)
    index = _shift_index(xp, n_timepoints, shifts)
    for start in range(0, n_perm, batch_size):
        batch = index[start : start + batch_size]
        rolled = x_unit[:, batch]
        perm = xp.abs(xp.einsum("pbt,pt->pb", rolled, y_unit))
        counts += (perm >= observed[:, None]).sum(axis=1)

    pvals = counts / n_perm
    pvals[~valid] = 1.0
    return pvals


__all__ = ["draw_shifts", "marginal_pvalues", "aligned_pvalues"]
