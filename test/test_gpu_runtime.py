"""End-to-end tests intended for a configured NVIDIA machine."""

from __future__ import annotations

import numpy as np
import pytest

from causalized_gc import GcStar, probe_gpu
from causalized_gc.backends.gpu import permutation


GPU = probe_gpu()
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not GPU.available, reason=GPU.reason),
]


def toy_data(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(3, 96))
    data[1, 1:] += 0.5 * data[0, :-1]
    data[2, 2:] -= 0.4 * data[1, :-2]
    return data


@pytest.mark.parametrize("method", ["cgc", "cgc_star"])
def test_gpu_fit_through_public_api(method: str) -> None:
    model = GcStar(
        n_perm=32,
        n_pasts=2,
        n_lags=2,
        method=method,  # type: ignore[arg-type]
        backend="gpu",
        random_state=0,
        batch_size=8,
    ).fit(toy_data())

    assert model.selected_backend_ == "gpu"
    assert model.corr_.shape == (9, 3)
    connectivity = model.get_connectivity_matrix()
    assert isinstance(connectivity, np.ndarray)
    assert connectivity.shape == (3, 3)
    assert np.all(np.isfinite(connectivity))


def test_gpu_can_return_device_array() -> None:
    import cupy as cp

    model = GcStar(
        n_perm=16,
        n_pasts=1,
        backend="gpu",
        return_gpu=True,
        random_state=1,
    ).fit(toy_data(1))
    assert isinstance(model.get_connectivity_matrix(), cp.ndarray)


def test_cuml_regression_path_when_available() -> None:
    cuml_probe = probe_gpu(require_cuml=True)
    if not cuml_probe.available:
        pytest.skip(cuml_probe.reason)

    model = GcStar(
        n_perm=16,
        n_pasts=1,
        backend="gpu",
        regression_backend="cuml",
        random_state=2,
    ).fit(toy_data(2))
    assert model.selected_backend_ == "gpu"
    assert model.get_connectivity_matrix().shape == (3, 3)


def test_gpu_confusion_matrix_and_metrics_roundtrip() -> None:
    model = GcStar(
        n_perm=16,
        n_pasts=1,
        backend="gpu",
        random_state=3,
    ).fit(toy_data(3))
    model.get_connectivity_matrix()

    confusion = model.compute_confusion_matrix(np.eye(3))
    scores = model.compute_metrics()

    assert confusion.shape == (2, 2)
    assert scores.shape == (6,)
    assert np.all((scores >= 0) & (scores <= 1))


def test_gpu_float64_dtype_runs() -> None:
    import cupy as cp

    model = GcStar(
        n_perm=16,
        n_pasts=1,
        backend="gpu",
        dtype="float64",
        random_state=4,
    ).fit(toy_data(4))

    assert model.corr_.dtype == cp.float64


def test_gpu_async_passes_match_sequential_outputs() -> None:
    import cupy as cp

    options = {
        "n_perm": 16,
        "n_pasts": 1,
        "n_lags": 1,
        "method": "cgc",
        "backend": "gpu",
        "dtype": "float64",
        "random_state": 5,
        "batch_size": 8,
    }
    data = toy_data(5)
    sequential = GcStar(**options).fit(data)  # type: ignore[arg-type]
    asynchronous = GcStar(**options, async_passes=True).fit(data)  # type: ignore[arg-type]

    for async_value, sequential_value in (
        (asynchronous.corr_, sequential.corr_),
        (asynchronous.pVal_corr_, sequential.pVal_corr_),
        (asynchronous.inv_corr_, sequential.inv_corr_),
        (asynchronous.pVal_inv_corr_, sequential.pVal_inv_corr_),
    ):
        np.testing.assert_allclose(
            cp.asnumpy(async_value), cp.asnumpy(sequential_value)
        )


def test_cupy_permutation_kernel_matches_numpy_reference() -> None:
    import cupy as cp

    shifted = np.random.default_rng(6).normal(size=(5, 48))
    shifts = np.array([30, 31, 33, 37, 41, 47], dtype=np.int64)
    expected = permutation.marginal_pvalues(
        np,
        shifted,
        n_targets=3,
        n_perm=len(shifts),
        shifts=shifts,
        batch_size=2,
    )
    actual = permutation.marginal_pvalues(
        cp,
        cp.asarray(shifted),
        n_targets=3,
        n_perm=len(shifts),
        shifts=cp.asarray(shifts),
        batch_size=2,
    )

    np.testing.assert_allclose(cp.asnumpy(actual), expected, atol=1e-6)
