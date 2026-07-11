from __future__ import annotations

from dataclasses import dataclass

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


def test_resolve_device_normalizes_cuda_request() -> None:
    torch = _FakeTorch(cuda_available=True)

    assert resolve_device(torch, " CUDA:0 ").type == "cuda"


def test_resolve_device_cuda_falls_back_to_cpu() -> None:
    torch = _FakeTorch(cuda_available=False)

    assert resolve_device(torch, "cuda:0").type == "cpu"


def test_resolve_device_mps_when_available() -> None:
    torch = _FakeTorch(mps_available=True)

    assert resolve_device(torch, "mps").type == "mps"


def test_resolve_device_mps_falls_back_to_cpu() -> None:
    torch = _FakeTorch(mps_available=False)

    assert resolve_device(torch, "mps").type == "cpu"


def test_seed_torch_dispatches_by_device() -> None:
    torch = _FakeTorch(cuda_available=True, mps_available=True)

    seed_torch(torch, 11, _FakeDevice("cpu"))
    seed_torch(torch, 13, _FakeDevice("cuda"))
    seed_torch(torch, 17, _FakeDevice("mps"))

    assert torch.seeds == [11, 13, 17]
    assert torch.cuda.seeds == [13]
    assert torch.mps.seeds == [17]
