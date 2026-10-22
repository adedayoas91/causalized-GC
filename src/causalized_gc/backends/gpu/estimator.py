"""GPU implementation of the causalised Granger-causality estimator.

The computation remains on the selected CUDA device throughout ``fit`` and
crosses back to the host only at public boundaries unless ``return_gpu=True``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
from typing import Any

import numpy as np

from ... import metrics as _metrics
from .backend import cp, require_cupy
from .permutation import aligned_pvalues, draw_shifts, marginal_pvalues
from .regression import residual_pair, resolve_backend
from .stats import abs_correlation_matrix, pairwise_abs_correlation

Array = Any

_DTYPES = {"float32": np.float32, "float64": np.float64}


@dataclass
class GcStar:
    """GPU c-GC/fcGC estimator.

    Parameters
    ----------
    n_perm:
        Number of circular-shift permutations for p-value estimation.
    n_pasts:
        Conditioning depth used to build the shifted state.
    n_lags:
        Number of lag blocks merged into the final connectivity matrix.
    temporal:
        Compatibility flag retained from the conventional implementation.
    method:
        Either ``"cgc"`` or ``"fcgc"``.
    dtype:
        Device computation dtype: ``"float32"`` or ``"float64"``.
    random_state:
        Seed for the circular-shift schedule.
    batch_size:
        Maximum number of shifts materialized per permutation batch.
    regression_backend:
        ``"auto"``, ``"cupy_lstsq"``, or ``"cuml"``.
    async_passes:
        Enqueue marginal and conditional passes on separate CUDA streams.
    return_gpu:
        Return CuPy arrays from public numerical getters when true.
    device_id:
        CUDA device selected by the public backend dispatcher.
    """

    n_perm: int = 1000
    n_pasts: int = 1
    n_lags: int = 1
    temporal: bool = True
    method: str = "cgc"
    dtype: str = "float32"
    random_state: int | None = None
    batch_size: int | None = None
    regression_backend: str = "auto"
    async_passes: bool = False
    return_gpu: bool = False
    device_id: int = 0

    corr_: Array | None = field(default=None, repr=False)
    pVal_corr_: Array | None = field(default=None, repr=False)
    inv_corr_: Array | None = field(default=None, repr=False)
    pVal_inv_corr_: Array | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.method not in {"cgc", "fcgc"}:
            raise ValueError("method must be 'cgc' or 'fcgc'.")
        if self.n_pasts < 0:
            raise ValueError("n_pasts must be non-negative.")
        if self.n_lags < 1:
            raise ValueError("n_lags must be at least 1.")
        if self.dtype not in _DTYPES:
            raise ValueError("dtype must be 'float32' or 'float64'.")
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("batch_size must be positive when set.")
        if self.device_id < 0:
            raise ValueError("device_id must be non-negative.")
        # Validate the name eagerly. GPU availability is enforced only by fit.
        resolve_backend(self.regression_backend)

        self.logger = logging.getLogger(__name__)
        self._xp: Any = None
        self._resolved_backend: str | None = None
        self._shifts: Array | None = None
        self._shifts_timepoints: int | None = None
        self.n_neur: int = 0

    def _configure_logging(self, verbose: int) -> None:
        level = logging.WARNING
        if verbose == 1:
            level = logging.INFO
        elif verbose >= 2:
            level = logging.DEBUG
        logging.basicConfig(level=level, force=True)

    def _cp_dtype(self) -> Any:
        return _DTYPES[self.dtype]

    def _batch(self) -> int:
        if self.batch_size is None:
            return max(self.n_perm, 1)
        return self.batch_size

    def _get_shifts(self, n_timepoints: int) -> Array:
        """Draw once on the host and cache the device-resident shifts."""

        if self._shifts is None or self._shifts_timepoints != n_timepoints:
            host = draw_shifts(self.n_perm, n_timepoints, self.random_state)
            self._shifts = self._xp.asarray(host)
            self._shifts_timepoints = n_timepoints
        return self._shifts

    def _to_numpy(self, arr: Array) -> np.ndarray:
        if cp is not None and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return np.asarray(arr)

    def shift_data(self, arr: Array) -> Array:
        """Create stacked lagged views of ``arr``."""

        xp = self._xp
        self.n_neur = arr.shape[0]
        if self.n_pasts == 0:
            return arr.copy()

        trimmed = arr[:, self.n_pasts :]
        for index in range(self.n_pasts):
            start = self.n_pasts - 1 - index
            stop = -index - 1
            trimmed = xp.concatenate([trimmed, arr[:, start:stop]], axis=0)
        return trimmed

    def _delete_rows(self, arr: Array, excluded: list[int]) -> Array:
        xp = self._xp
        mask = xp.ones(arr.shape[0], dtype=bool)
        if excluded:
            mask[xp.asarray(np.asarray(excluded, dtype=np.int64))] = False
        return arr[mask]

    def get_conditioning_set(self, shifted: Array, i: int, j: int) -> Array:
        """Build the c-GC conditioning set for one directed pair."""

        source_index = i % self.n_neur
        excluded_history = [
            source_index + lag * self.n_neur for lag in range(i // self.n_neur)
        ]
        excluded = excluded_history + [i, j]
        return self._delete_rows(shifted, excluded)

    def _conditioning_set(self, shifted: Array, i: int, j: int) -> Array:
        if self.method == "fcgc":
            return self._delete_rows(shifted, [i, j])
        return self.get_conditioning_set(shifted, i, j)

    def _correlation_from_shifted(
        self, shifted: Array, shifts: Array
    ) -> tuple[Array, Array]:
        """Compute unconditional effects and p-values on shifted data."""

        xp = self._xp
        corr, _valid = abs_correlation_matrix(xp, shifted)
        pvals = marginal_pvalues(
            xp,
            shifted,
            self.n_neur,
            self.n_perm,
            shifts,
            self._batch(),
        )
        return corr[:, : self.n_neur], pvals

    def correlation_func(self, data: Array) -> tuple[Array, Array]:
        """Compute unconditional dependence effects and p-values."""

        self.n_neur = data.shape[0]
        shifted = self.shift_data(data)
        return self._correlation_from_shifted(
            shifted, self._get_shifts(shifted.shape[1])
        )

    def _inv_correlation_from_shifted(
        self, shifted: Array, shifts: Array
    ) -> tuple[Array, Array]:
        """Compute conditional effects and p-values on shifted data."""

        xp = self._xp
        n_rows = shifted.shape[0]

        x_res_rows: list[Array] = []
        y_res_rows: list[Array] = []
        for i in range(n_rows):
            for j in range(self.n_neur):
                z = self._conditioning_set(shifted, i, j)
                x_res, y_res = residual_pair(
                    xp,
                    shifted[i],
                    shifted[j],
                    z,
                    backend=self._resolved_backend,
                )
                x_res_rows.append(x_res)
                y_res_rows.append(y_res)

        x_res_all = xp.stack(x_res_rows)
        y_res_all = xp.stack(y_res_rows)

        effect, valid = pairwise_abs_correlation(xp, x_res_all, y_res_all)
        effect = xp.where(valid, effect, xp.zeros_like(effect))
        pvals = aligned_pvalues(
            xp,
            x_res_all,
            y_res_all,
            self.n_perm,
            shifts,
            self._batch(),
        )

        inv_corr = effect.reshape(n_rows, self.n_neur)
        pvals = pvals.reshape(n_rows, self.n_neur)
        return inv_corr, pvals

    def inv_correlation_func(self, data: Array) -> tuple[Array, Array]:
        """Compute conditional dependence through residual correlations."""

        self.n_neur = data.shape[0]
        shifted = self.shift_data(data)
        return self._inv_correlation_from_shifted(
            shifted, self._get_shifts(shifted.shape[1])
        )

    def _dependence_passes_sequential(
        self, shifted: Array, shifts: Array
    ) -> tuple[Array, Array, Array, Array]:
        corr, pval_corr = self._correlation_from_shifted(shifted, shifts)
        inv_corr, pval_inv_corr = self._inv_correlation_from_shifted(
            shifted, shifts
        )
        return corr, pval_corr, inv_corr, pval_inv_corr

    def _enqueue_dependence_pass(
        self,
        stream: Array,
        ready_event: Array,
        func: Any,
        shifted: Array,
        shifts: Array,
    ) -> tuple[tuple[Array, Array], Array]:
        """Queue one dependence pass and return its result and done event."""

        xp = self._xp
        # CUDA's current device is thread-local, so worker threads must select
        # the configured device independently of the outer fit context.
        with xp.cuda.Device(self.device_id):
            with stream:
                stream.wait_event(ready_event)
                result = func(shifted, shifts)
                done_event = xp.cuda.Event()
                done_event.record()
        return result, done_event

    def _dependence_passes_async(
        self, shifted: Array, shifts: Array
    ) -> tuple[Array, Array, Array, Array]:
        """Run marginal and conditional passes on separate CUDA streams."""

        xp = self._xp
        ready_event = xp.cuda.Event()
        ready_event.record()

        corr_stream = xp.cuda.Stream(non_blocking=True)
        inv_stream = xp.cuda.Stream(non_blocking=True)

        with ThreadPoolExecutor(max_workers=2) as executor:
            corr_future = executor.submit(
                self._enqueue_dependence_pass,
                corr_stream,
                ready_event,
                self._correlation_from_shifted,
                shifted,
                shifts,
            )
            inv_future = executor.submit(
                self._enqueue_dependence_pass,
                inv_stream,
                ready_event,
                self._inv_correlation_from_shifted,
                shifted,
                shifts,
            )

            (corr, pval_corr), corr_done = corr_future.result()
            (inv_corr, pval_inv_corr), inv_done = inv_future.result()

        corr_done.synchronize()
        inv_done.synchronize()
        return corr, pval_corr, inv_corr, pval_inv_corr

    def fit(self, data: np.ndarray, verbose: int = 0) -> "GcStar":
        """Fit data shaped ``(n_variables, n_timepoints)`` on the GPU."""

        self._xp = require_cupy()
        self._resolved_backend = resolve_backend(self.regression_backend)
        self._configure_logging(verbose)

        host = np.asarray(data)
        if host.ndim != 2:
            raise ValueError("data must be 2-D, shaped (n_variables, n_timepoints).")
        if self.n_pasts >= host.shape[1]:
            raise ValueError("n_pasts must be smaller than the number of timepoints.")

        # Keep all allocation, kernels, streams, and synchronization scoped to
        # the exact device selected by the capability probe/dispatcher.
        with self._xp.cuda.Device(self.device_id):
            self.data = self._xp.asarray(host, dtype=self._cp_dtype())
            self.shifted_data = self.shift_data(self.data)
            shifts = self._get_shifts(self.shifted_data.shape[1])

            if self.async_passes:
                (
                    self.corr_,
                    self.pVal_corr_,
                    self.inv_corr_,
                    self.pVal_inv_corr_,
                ) = self._dependence_passes_async(self.shifted_data, shifts)
            else:
                (
                    self.corr_,
                    self.pVal_corr_,
                    self.inv_corr_,
                    self.pVal_inv_corr_,
                ) = self._dependence_passes_sequential(self.shifted_data, shifts)
        return self

    def get_connectivity_matrix(
        self,
        *,
        simulation: bool = True,
        alpha: float = 0.01,
        beta: float = 0.001,
    ) -> np.ndarray:
        """Build a weighted connectivity matrix from significance masks."""

        if self.corr_ is None or self.inv_corr_ is None:
            raise RuntimeError("fit must be called before get_connectivity_matrix.")

        with self._xp.cuda.Device(self.device_id):
            conn = _metrics.connectivity_matrix(
                self._xp,
                self.corr_,
                self.pVal_corr_,
                self.inv_corr_,
                self.pVal_inv_corr_,
                n_pasts=self.n_pasts,
                n_lags=self.n_lags,
                simulation=simulation,
                alpha=alpha,
                beta=beta,
            )
        self.conn_mat = conn
        return conn if self.return_gpu else self._to_numpy(conn)

    def compute_confusion_matrix(
        self,
        truth: np.ndarray,
        *,
        simulation: bool = True,
    ) -> np.ndarray:
        """Return the confusion matrix against ground-truth adjacency."""

        if not hasattr(self, "conn_mat"):
            raise RuntimeError("get_connectivity_matrix must be called first.")

        conn = self._to_numpy(self.conn_mat)
        self.confusion_matrix = _metrics.confusion_matrix(
            conn, np.asarray(truth), simulation=simulation
        )
        return self.confusion_matrix

    def compute_metrics(self) -> np.ndarray:
        """Return accuracy, precision, recall, FPR, balanced accuracy, and F1."""

        if not hasattr(self, "confusion_matrix"):
            raise RuntimeError("compute_confusion_matrix must be called first.")
        return _metrics.classification_metrics(self.confusion_matrix)

    def compute_shd_sid(
        self,
        truth: np.ndarray,
        inferred: np.ndarray,
        *,
        simulation: bool = True,
    ) -> float:
        """Return Structural Hamming Distance using the optional metric."""

        self.shd_ = _metrics.shd(
            np.asarray(truth), np.asarray(inferred), simulation=simulation
        )
        return self.shd_


__all__ = ["GcStar"]
