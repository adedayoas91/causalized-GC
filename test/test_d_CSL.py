#!/usr/bin/env python
"""Regression tests for the conventional CPU implementation."""

import numpy as np
import pytest

from src import d_CSL


def make_estimator(*, n_pasts: int = 2, n_lags: int = 2, method: str = "cgc"):
    return d_CSL.GcStar(
        n_perm=10,
        n_pasts=n_pasts,
        n_lags=n_lags,
        temporal=True,
        method=method,
    )


def test_perm_test_counts_all_equivalent_circular_shifts():
    alternating = np.tile(np.array([-1.0, 1.0]), 20)

    assert d_CSL.perm_test(alternating, alternating, 20) == 1.0


@pytest.mark.parametrize("n_pasts", [0, 1, 2])
def test_prep_data_matches_explicit_lag_stacking(n_pasts):
    data = np.arange(24).reshape(3, 8)
    expected = np.concatenate(
        [data[:, n_pasts - lag : data.shape[1] - lag] for lag in range(n_pasts + 1)],
        axis=0,
    )

    np.testing.assert_array_equal(d_CSL.prep_data(data, n_pasts), expected)


def test_compare_with_ground_truth_uses_expected_cell_codes():
    truth = np.array([[0, 1], [0, 0]])
    inferred = np.array([[0, 1], [1, 0]])

    expected = np.array([[10, 40], [30, 10]])
    np.testing.assert_array_equal(
        d_CSL.compare_with_GT(truth, inferred, simulation=False), expected
    )


def test_compute_confusion_matrix_preserves_matrix_layout():
    estimator = make_estimator()
    estimator.conn_mat = np.array([[0.0, 0.5], [0.0, 0.0]])
    truth = np.array([[0, 1], [1, 0]])

    expected = np.array([[1, 0], [1, 2]])
    np.testing.assert_array_equal(
        estimator.compute_confusion_matrix(truth, simulation=False), expected
    )
