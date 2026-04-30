from __future__ import annotations

import pytest

pytest.importorskip("torch")

from ml4t.models._internal.torch_runtime import import_torch, resolve_device, seed_torch


def test_resolve_device_cpu() -> None:
    torch = import_torch()
    assert resolve_device(torch, "cpu").type == "cpu"


def test_resolve_device_mps_or_cpu_fallback() -> None:
    torch = import_torch()
    d = resolve_device(torch, "mps")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        assert d.type == "mps"
    else:
        assert d.type == "cpu"


def test_seed_torch_mps_does_not_raise() -> None:
    torch = import_torch()
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None or not mps_backend.is_available():
        pytest.skip("MPS not available")
    device = torch.device("mps")
    seed_torch(torch, 123, device)
