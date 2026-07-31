from __future__ import annotations

from dataclasses import dataclass

import pytest

from ml4t.models._internal.torch_runtime import resolve_device, seed_torch


@dataclass(frozen=True)
class _FakeDevice:
    type: str


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self.available = available
        self.seeds: list[int] = []

    def is_available(self) -> bool:
        return self.available

    def manual_seed_all(self, seed: int) -> None:
        self.seeds.append(seed)

    def device_count(self) -> int:
        return 1 if self.available else 0


class _FakeMPSBackend:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available


class _FakeMPS:
    def __init__(self) -> None:
        self.seeds: list[int] = []

    def manual_seed(self, seed: int) -> None:
        self.seeds.append(seed)


class _FakeTorch:
    def __init__(self, *, cuda_available: bool = False, mps_available: bool = False) -> None:
        self.cuda = _FakeCuda(cuda_available)
        self.backends = type("Backends", (), {"mps": _FakeMPSBackend(mps_available)})()
        self.mps = _FakeMPS()
        self.seeds: list[int] = []

    def device(self, requested: str) -> _FakeDevice:
        if requested != requested.strip().lower():
            raise ValueError("fake torch expects normalized device strings")
        return _FakeDevice(requested.split(":", maxsplit=1)[0])

    def manual_seed(self, seed: int) -> None:
        self.seeds.append(seed)


def test_resolve_device_cpu() -> None:
    torch = _FakeTorch()

    assert resolve_device(torch, "cpu").type == "cpu"


def test_resolve_device_cuda_when_available() -> None:
    torch = _FakeTorch(cuda_available=True)

    assert resolve_device(torch, "cuda:0").type == "cuda"
    assert resolve_device(torch, "cuda").type == "cuda"


def test_resolve_device_normalizes_cuda_request() -> None:
    torch = _FakeTorch(cuda_available=True)

    assert resolve_device(torch, " CUDA:0 ").type == "cuda"


def test_resolve_device_rejects_unavailable_cuda() -> None:
    torch = _FakeTorch(cuda_available=False)

    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device(torch, "cuda:0")


def test_resolve_device_rejects_out_of_range_cuda_index() -> None:
    torch = _FakeTorch(cuda_available=True)

    with pytest.raises(RuntimeError, match="CUDA index 1"):
        resolve_device(torch, "cuda:1")


def test_resolve_device_mps_when_available() -> None:
    torch = _FakeTorch(mps_available=True)

    assert resolve_device(torch, "mps").type == "mps"


def test_resolve_device_rejects_unavailable_mps() -> None:
    torch = _FakeTorch(mps_available=False)

    with pytest.raises(RuntimeError, match="MPS"):
        resolve_device(torch, "mps")


def test_resolve_device_rejects_unknown_device() -> None:
    with pytest.raises(ValueError, match="requested device"):
        resolve_device(_FakeTorch(), "tpu")


def test_seed_torch_dispatches_by_device() -> None:
    torch = _FakeTorch(cuda_available=True, mps_available=True)

    seed_torch(torch, 11, _FakeDevice("cpu"))
    seed_torch(torch, 13, _FakeDevice("cuda"))
    seed_torch(torch, 17, _FakeDevice("mps"))

    assert torch.seeds == [11, 13, 17]
    assert torch.cuda.seeds == [13]
    assert torch.mps.seeds == [17]


def test_seed_torch_allows_mps_without_manual_seed() -> None:
    torch = _FakeTorch(mps_available=True)
    torch.mps = object()

    seed_torch(torch, 19, _FakeDevice("mps"))

    assert torch.seeds == [19]
