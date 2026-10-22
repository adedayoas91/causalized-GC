"""CPU-safe numerical contract tests for GPU backend primitives."""

from __future__ import annotations

import numpy as np
import pytest

from causalized_gc import metrics
from causalized_gc.backends.gpu import permutation, regression, stats


def _unit(row: np.ndarray) -> tuple[np.ndarray, bool]:
    centered = row - row.mean()
    norm = np.linalg.norm(centered)
    if norm == 0:
        return np.zeros_like(centered), False
    return centered / norm, True


def _reference_marginal(
    shifted: np.ndarray, n_targets: int, shifts: np.ndarray
) -> np.ndarray:
    n_rows, n_timepoints = shifted.shape
    normalized = [_unit(row) for row in shifted]
    units = np.asarray([unit for unit, _valid in normalized])
    valid = np.asarray([is_valid for _unit_row, is_valid in normalized])
    pvalues = np.zeros((n_rows, n_targets))

    for row_index in range(n_rows):
        for target_index in range(n_targets):
            observed = abs(units[row_index] @ units[target_index])
            count = 0
            for shift in shifts:
                index = (np.arange(n_timepoints) + shift) % n_timepoints
                if abs(units[row_index][index] @ units[target_index]) >= observed:
                    count += 1
            pvalues[row_index, target_index] = count / len(shifts)

    pvalues[~valid, :] = 1.0
    pvalues[:, ~valid[:n_targets]] = 1.0
    return pvalues


def _reference_aligned(
    x_rows: np.ndarray, y_rows: np.ndarray, shifts: np.ndarray
) -> np.ndarray:
    n_pairs, n_timepoints = x_rows.shape
    pvalues = np.zeros(n_pairs)
    for pair_index in range(n_pairs):
        x_unit, x_valid = _unit(x_rows[pair_index])
        y_unit, y_valid = _unit(y_rows[pair_index])
        if not (x_valid and y_valid):
            pvalues[pair_index] = 1.0
            continue

        observed = abs(x_unit @ y_unit)
        count = 0
        for shift in shifts:
            index = (np.arange(n_timepoints) + shift) % n_timepoints
            if abs(x_unit[index] @ y_unit) >= observed:
                count += 1
        pvalues[pair_index] = count / len(shifts)
    return pvalues


def test_abs_correlation_matrix_matches_numpy() -> None:
    data = np.random.default_rng(1).normal(size=(6, 40))

    correlation, valid = stats.abs_correlation_matrix(np, data)

    np.testing.assert_allclose(correlation, np.abs(np.corrcoef(data)), atol=1e-12)
    assert valid.all()


def test_constant_rows_are_zero_and_marked_invalid() -> None:
    data = np.random.default_rng(0).normal(size=(3, 20))
    data[1] = 5.0

    normalized, valid = stats.normalize_rows(np, data)

    assert valid[0] and not valid[1] and valid[2]
    np.testing.assert_array_equal(normalized[1], 0.0)


def test_marginal_pvalues_match_reference() -> None:
    shifted = np.random.default_rng(2).normal(size=(5, 13))
    shifts = np.array([1, 2, 3, 5, 7, 9, 11], dtype=np.int64)

    actual = permutation.marginal_pvalues(
        np,
        shifted,
        n_targets=3,
        n_perm=len(shifts),
        shifts=shifts,
        batch_size=3,
    )

    np.testing.assert_allclose(
        actual, _reference_marginal(shifted, 3, shifts), atol=1e-12
    )


def test_aligned_pvalues_match_reference() -> None:
    rng = np.random.default_rng(3)
    x_rows = rng.normal(size=(4, 15))
    y_rows = rng.normal(size=(4, 15))
    shifts = np.array([1, 4, 6, 9, 13], dtype=np.int64)

    actual = permutation.aligned_pvalues(
        np,
        x_rows,
        y_rows,
        n_perm=len(shifts),
        shifts=shifts,
        batch_size=2,
    )

    np.testing.assert_allclose(
        actual, _reference_aligned(x_rows, y_rows, shifts), atol=1e-12
    )


def test_marginal_pvalues_are_one_for_constant_rows_and_targets() -> None:
    data = np.random.default_rng(4).normal(size=(3, 16))
    data[1] = 2.0
    shifts = np.array([1, 3, 5, 7], dtype=np.int64)

    pvalues = permutation.marginal_pvalues(
        np,
        data,
        n_targets=3,
        n_perm=len(shifts),
        shifts=shifts,
        batch_size=2,
    )

    np.testing.assert_array_equal(pvalues[1, :], 1.0)
    np.testing.assert_array_equal(pvalues[:, 1], 1.0)


def test_degenerate_series_short_circuits_to_pvalue_one() -> None:
    data = np.random.default_rng(5).normal(size=(2, 1))

    pvalues = permutation.marginal_pvalues(
        np,
        data,
        n_targets=2,
        n_perm=0,
        shifts=np.empty(0, dtype=np.int64),
        batch_size=1,
    )

    np.testing.assert_array_equal(pvalues, 1.0)


def test_residual_pair_matches_intercept_lstsq() -> None:
    rng = np.random.default_rng(6)
    x = rng.normal(size=30)
    y = rng.normal(size=30)
    conditioning = rng.normal(size=(3, 30))

    x_residual, y_residual = regression.residual_pair(
        np, x, y, conditioning, backend="cupy_lstsq"
    )

    design = conditioning.T
    targets = np.column_stack([x, y])
    augmented = np.column_stack([design, np.ones(design.shape[0])])
    coefficients, *_ = np.linalg.lstsq(augmented, targets, rcond=None)
    expected = targets - augmented @ coefficients
    np.testing.assert_allclose(x_residual, expected[:, 0], atol=1e-10)
    np.testing.assert_allclose(y_residual, expected[:, 1], atol=1e-10)


def test_empty_conditioning_set_centers_targets() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=12)
    y = rng.normal(size=12)
    conditioning = np.empty((0, 12))

    x_residual, y_residual = regression.residual_pair(np, x, y, conditioning)

    np.testing.assert_allclose(x_residual, x - x.mean(), atol=1e-12)
    np.testing.assert_allclose(y_residual, y - y.mean(), atol=1e-12)


def test_regression_backend_resolution() -> None:
    assert regression.resolve_backend("auto") == "cupy_lstsq"
    assert regression.resolve_backend("cupy_lstsq") == "cupy_lstsq"
    assert regression.resolve_backend("cuml") == "cuml"
    with pytest.raises(ValueError, match="regression_backend"):
        regression.resolve_backend("does-not-exist")


def test_draw_shifts_are_in_range_and_reproducible() -> None:
    first = permutation.draw_shifts(50, 80, random_state=7)
    second = permutation.draw_shifts(50, 80, random_state=7)

    np.testing.assert_array_equal(first, second)
    assert first.min() >= 30
    assert first.max() < 80
    assert permutation.draw_shifts(0, 80, None).size == 0
    assert permutation.draw_shifts(10, 1, None).size == 0


def test_draw_shifts_clamps_minimum_for_short_series() -> None:
    shifts = permutation.draw_shifts(5, 10, random_state=8)
    np.testing.assert_array_equal(shifts, 9)


def test_connectivity_matrix_selects_lag_block_and_reweights() -> None:
    shape = (4, 2)
    corr = np.zeros(shape)
    inv_corr = np.zeros(shape)
    pval_corr = np.ones(shape)
    pval_inv_corr = np.ones(shape)
    corr[0, 0] = 0.9
    corr[2, 0] = 0.7
    inv_corr[2, 0] = 0.5
    pval_corr[2, 0] = 1e-3
    pval_inv_corr[2, 0] = 1e-4

    actual = metrics.connectivity_matrix(
        np,
        corr,
        pval_corr,
        inv_corr,
        pval_inv_corr,
        n_pasts=1,
        n_lags=1,
        simulation=True,
        alpha=0.01,
        beta=0.001,
    )

    np.testing.assert_allclose(actual, np.array([[0.9, 0.0], [0.0, 0.0]]))


def test_ground_truth_overlay_classifies_edges() -> None:
    truth = np.eye(2)
    inferred = np.array([[1.0, 0.9], [0.0, 0.0]])

    image = metrics.compare_with_ground_truth(truth, inferred, simulation=True)

    np.testing.assert_allclose(image[0, 0], (0.0, 0.6, 0.0))
    np.testing.assert_allclose(image[0, 1], (0.8, 0.0, 0.0))
    np.testing.assert_allclose(image[1, 0], (1.0, 1.0, 1.0))
    np.testing.assert_allclose(image[1, 1], (0.0, 0.0, 0.8))
