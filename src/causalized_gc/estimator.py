"""Unified CPU/GPU façade for causalised Granger causality."""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np

from .capabilities import BackendDecision, BackendName, select_backend


LOGGER = logging.getLogger(__name__)


class GcStar:
    """Run c-GC on the conventional CPU or accelerated NVIDIA backend.

    ``backend="auto"`` selects a usable NVIDIA backend and otherwise records a
    CPU fallback reason in :attr:`backend_diagnostics_`. An explicit GPU request
    never falls back silently.
    """

    def __init__(
        self,
        n_perm: int = 1000,
        n_pasts: int = 1,
        n_lags: int = 1,
        temporal: bool = True,
        method: Literal["cgc", "fcgc"] = "cgc",
        *,
        backend: BackendName = "auto",
        device_id: int = 0,
        dtype: Literal["float32", "float64"] = "float32",
        random_state: int | None = None,
        batch_size: int | None = None,
        regression_backend: Literal["auto", "cupy_lstsq", "cuml"] = "auto",
        async_passes: bool = False,
        return_gpu: bool = False,
    ) -> None:
        if method not in {"cgc", "fcgc"}:
            raise ValueError("method must be 'cgc' or 'fcgc'")
        if regression_backend not in {"auto", "cupy_lstsq", "cuml"}:
            raise ValueError(
                "regression_backend must be 'auto', 'cupy_lstsq', or 'cuml'"
            )

        self.backend_diagnostics_: BackendDecision = select_backend(
            backend,
            device_id=device_id,
            require_cuml=regression_backend == "cuml",
        )
        self.selected_backend_ = self.backend_diagnostics_.selected
        self.device_id = device_id
        self._configuration = {
            "n_perm": n_perm,
            "n_pasts": n_pasts,
            "n_lags": n_lags,
            "temporal": temporal,
            "method": method,
        }
        self._gpu_configuration = {
            "dtype": dtype,
            "random_state": random_state,
            "batch_size": batch_size,
            "regression_backend": regression_backend,
            "async_passes": async_passes,
            "return_gpu": return_gpu,
            "device_id": device_id,
        }
        self._impl = self._make_implementation()
        if backend == "auto" and self.selected_backend_ == "cpu":
            LOGGER.info(self.backend_diagnostics_.reason)

    def _make_implementation(self) -> Any:
        if self.selected_backend_ == "gpu":
            from .backends.gpu import GpuGcStar

            return GpuGcStar(**self._configuration, **self._gpu_configuration)

        from .backends.cpu import GcStar as CpuGcStar

        return CpuGcStar(**self._configuration)

    def _call_cpu_only(self, name: str, *args: Any, **kwargs: Any) -> Any:
        method = getattr(self._impl, name, None)
        if not callable(method):
            raise NotImplementedError(
                f"{name} is currently CPU-only; construct GcStar with "
                "backend='cpu' for this workflow"
            )
        return method(*args, **kwargs)  # pylint: disable=not-callable

    @property
    def implementation_(self) -> Any:
        """Concrete backend estimator, exposed for advanced diagnostics."""

        return self._impl

    @property
    def corr_(self) -> Any:
        return self._impl.corr_

    @property
    def pVal_corr_(self) -> Any:
        return self._impl.pVal_corr_

    @property
    def inv_corr_(self) -> Any:
        return self._impl.inv_corr_

    @property
    def pVal_inv_corr_(self) -> Any:
        return self._impl.pVal_inv_corr_

    @property
    def conn_mat(self) -> Any:
        if not hasattr(self._impl, "conn_mat"):
            raise RuntimeError("get_connectivity_matrix must be called first")
        return self._impl.conn_mat

    def fit(self, data: np.ndarray, verbose: int = 0) -> "GcStar":
        """Fit the selected implementation and return this façade."""

        self._impl.fit(data, verbose=verbose)
        return self

    def get_connectivity_matrix(
        self,
        *,
        simulation: bool = True,
        alpha: float = 0.01,
        beta: float = 0.001,
    ) -> np.ndarray:
        return self._impl.get_connectivity_matrix(
            simulation=simulation,
            alpha=alpha,
            beta=beta,
        )

    def compute_confusion_matrix(
        self,
        truth: np.ndarray,
        *,
        simulation: bool = True,
    ) -> np.ndarray:
        return self._impl.compute_confusion_matrix(truth, simulation=simulation)

    def compute_metrics(self) -> np.ndarray:
        method = getattr(self._impl, "compute_metrics", None)
        if method is not None:
            return method()
        return self._impl.all_metrics()

    def all_metrics(self) -> np.ndarray:
        """Compatibility alias for notebooks using the conventional API."""

        return self.compute_metrics()

    def compute_shd_sid(
        self,
        truth: np.ndarray,
        inferred: np.ndarray,
        *,
        simulation: bool = True,
    ) -> float:
        return self._impl.compute_shd_sid(
            truth,
            inferred,
            simulation=simulation,
        )

    def get_rising_flanks(
        self,
        array: np.ndarray,
        f_c: float,
        f_s: float,
        length: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the legacy CPU-only rising-flank preprocessing helper."""

        return self._call_cpu_only("get_rising_flanks", array, f_c, f_s, length)

    def fit_rising(
        self,
        data: np.ndarray,
        indices: np.ndarray,
        verbose: int = 1,
    ) -> "GcStar":
        """Run the legacy CPU-only rising-flank estimator."""

        self._call_cpu_only("fit_rising", data, indices, verbose=verbose)
        return self

    def plot_extended_connectivity_matrix(
        self,
        alpha: float = 0.01,
        beta: float = 0.001,
    ) -> Any:
        """Plot lag blocks through the legacy CPU visualization helper."""

        return self._call_cpu_only(
            "plot_extended_connectivity_matrix", alpha=alpha, beta=beta
        )


__all__ = ["GcStar"]
