"""Backend-selection contract for the unified estimator."""

from __future__ import annotations

import pytest

from causalized_gc import GcStar
from causalized_gc.capabilities import (
    BackendUnavailableError,
    GpuProbe,
    select_backend,
)


def _unavailable(reason: str = "test GPU unavailable") -> GpuProbe:
    return GpuProbe(available=False, reason=reason, device_id=0)


def _available() -> GpuProbe:
    return GpuProbe(
        available=True,
        reason="test GPU compatible",
        device_id=0,
        device_count=1,
        device_name="Test GPU",
        compute_capability=(8, 0),
    )


def test_forced_cpu_does_not_probe_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_probe(**_: object) -> GpuProbe:
        raise AssertionError("the CPU backend must not probe or import CUDA")

    monkeypatch.setattr(
        "causalized_gc.capabilities.probe_gpu",
        unexpected_probe,
    )
    decision = select_backend("cpu")
    assert decision.selected == "cpu"
    assert decision.gpu is None


def test_auto_falls_back_with_structured_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "causalized_gc.capabilities.probe_gpu",
        lambda **_: _unavailable("driver mismatch"),
    )
    decision = select_backend("auto")
    assert decision.selected == "cpu"
    assert "driver mismatch" in decision.reason
    assert decision.gpu == _unavailable("driver mismatch")


def test_explicit_gpu_never_silently_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "causalized_gc.capabilities.probe_gpu",
        lambda **_: _unavailable("no device"),
    )
    with pytest.raises(BackendUnavailableError, match="no device"):
        select_backend("gpu")


@pytest.mark.parametrize("requested", ["auto", "gpu"])
def test_compatible_gpu_is_selected(
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
) -> None:
    monkeypatch.setattr(
        "causalized_gc.capabilities.probe_gpu",
        lambda **_: _available(),
    )
    decision = select_backend(requested)  # type: ignore[arg-type]
    assert decision.selected == "gpu"
    assert decision.gpu is not None
    assert decision.gpu.device_name == "Test GPU"


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="backend"):
        select_backend("tpu")  # type: ignore[arg-type]


@pytest.mark.parametrize("method", ["cgc_star", "fcgc"])
def test_cgc_star_name_and_legacy_alias_share_one_algorithm(method: str) -> None:
    estimator = GcStar(
        n_perm=1,
        method=method,  # type: ignore[arg-type]
        backend="cpu",
    )

    assert estimator.implementation_.method == "cgc_star"
