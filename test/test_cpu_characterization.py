"""Behavioral characterization of the conventional CPU c-GC implementation."""

import builtins

import numpy as np
import pytest

from causalized_gc.backends import cpu as cpu_backend
from src import d_CSL


def make_estimator(*, n_pasts: int = 2, n_lags: int = 2, method: str = "cgc"):
    return d_CSL.GcStar(
        n_perm=5,
        n_pasts=n_pasts,
        n_lags=n_lags,
        temporal=True,
        method=method,
    )


@pytest.mark.parametrize("n_pasts", [0, 1, 3])
def test_shift_data_stacks_current_then_successive_pasts(n_pasts):
    data = np.arange(30).reshape(3, 10)
    estimator = make_estimator(n_pasts=n_pasts, n_lags=max(1, n_pasts))
    expected = np.concatenate(
        [data[:, n_pasts - lag : data.shape[1] - lag] for lag in range(n_pasts + 1)],
        axis=0,
    )

    shifted = estimator.shift_data(data)

    np.testing.assert_array_equal(shifted, expected)
    assert shifted.dtype == data.dtype
    assert estimator.n_neur == data.shape[0]


def test_residual_removes_intercept_and_linear_predictors():
    first = np.linspace(-2.0, 2.0, 25)
    second = first**2
    predictors = np.vstack([first, second])
    response = 4.5 + 2.0 * first - 0.75 * second

    actual = d_CSL.residual(response, predictors)

    np.testing.assert_allclose(actual, np.zeros_like(response), atol=1e-12)


def test_residual_rejects_an_empty_conditioning_set():
    with pytest.raises(ValueError, match="0 feature"):
        d_CSL.residual(np.arange(5.0), np.empty((0, 5)))


def test_cgc_conditioning_set_excludes_effect_cause_and_cause_futures():
    data = np.arange(30).reshape(3, 10)
    estimator = make_estimator(n_pasts=2)
    shifted = estimator.shift_data(data)

    actual = estimator.get_conditioning_set(data, i=6, j=2)

    np.testing.assert_array_equal(actual, shifted[[1, 4, 5, 7, 8]])


def test_connectivity_aggregates_simulated_lags_and_uses_current_weights():
    estimator = make_estimator(n_pasts=2, n_lags=2)
    estimator.corr_ = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.0, 0.8],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.7, 0.0],
        ]
    )
    estimator.inv_corr_ = estimator.corr_.copy()
    estimator.pVal_corr_ = np.where(estimator.corr_ != 0, 0.0, 1.0)
    estimator.pVal_inv_corr_ = estimator.pVal_corr_.copy()

    actual = estimator.get_connectivity_matrix(
        simulation=True, alpha=0.01, beta=0.001
    )

    np.testing.assert_allclose(actual, np.array([[0.0, 0.2], [0.3, 0.0]]))


def test_fit_characterizes_both_conditioning_methods(monkeypatch):
    rng = np.random.default_rng(7)
    data = rng.normal(size=(3, 40))
    monkeypatch.setattr(cpu_backend, "perm_test", lambda _x, _y, _n_perm: 0.25)

    for method in ("cgc", "cgc_star"):
        estimator = make_estimator(n_pasts=1, n_lags=1, method=method).fit(
            data, verbose=0
        )

        assert estimator.corr_.shape == (6, 3)
        assert estimator.inv_corr_.shape == (6, 3)
        np.testing.assert_array_equal(estimator.pVal_corr_, np.full((6, 3), 0.25))
        np.testing.assert_array_equal(
            estimator.pVal_inv_corr_, np.full((6, 3), 0.25)
        )
        assert np.isfinite(estimator.corr_).all()
        assert np.isfinite(estimator.inv_corr_).all()


def test_fit_rising_passes_data_to_conditioning_set(monkeypatch):
    rng = np.random.default_rng(11)
    data = rng.normal(size=(2, 40))
    common_indices = [np.arange(40), np.arange(40)]
    estimator = make_estimator(n_pasts=1, n_lags=1)
    monkeypatch.setattr(cpu_backend, "tqdm", lambda values: values)
    monkeypatch.setattr(estimator, "_GcStar__perm_test_", lambda _x, _y: 0.5)

    fitted = estimator.fit_rising(data, common_indices, verbose=0)

    assert fitted is estimator
    assert estimator.corr_.shape == (2, 2)
    assert estimator.inv_corr_.shape == (2, 2)


def test_shd_dependency_is_imported_only_when_metric_is_requested(monkeypatch):
    estimator = make_estimator()
    real_import = builtins.__import__

    def reject_cdt(
        name, global_context=None, local_context=None, fromlist=(), level=0
    ):
        if name == "cdt.metrics":
            raise ImportError("cdt is intentionally unavailable")
        return real_import(name, global_context, local_context, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_cdt)

    with pytest.raises(ImportError, match="cdt.metrics is required"):
        estimator.compute_shd_sid(np.eye(2), np.eye(2), simulation=False)
