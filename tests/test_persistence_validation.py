from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest
import torch

from ml4t.models import PCAConfig, PCAModel, PersistentPanelBatch
from ml4t.models._internal import persistence


def _artifact(tmp_path: Path) -> Path:
    return persistence.save_artifact(
        tmp_path / "model.ml4t",
        model_type="test.Model",
        config=PCAConfig(n_factors=1),
        state={"asset_ids": ("A", "B"), "fitted": True},
        arrays={"weights": np.arange(4, dtype=np.float64).reshape(2, 2)},
    )


def _payloads(path: Path) -> tuple[dict[str, object], bytes]:
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        arrays = archive.read("arrays.npz")
    return manifest, arrays


def _rewrite(path: Path, manifest: dict[str, object] | bytes, arrays: bytes) -> None:
    manifest_payload = (
        manifest
        if isinstance(manifest, bytes)
        else json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
    )
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", manifest_payload)
        archive.writestr("arrays.npz", arrays)


def test_artifact_loader_rejects_invalid_archive_and_member_inventory(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.ml4t"
    invalid.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="invalid ml4t-models artifact"):
        persistence.load_artifact(invalid, expected_model_type="test.Model")

    wrong_members = tmp_path / "members.ml4t"
    with ZipFile(wrong_members, "w") as archive:
        archive.writestr("manifest.json", b"{}")
    with pytest.raises(ValueError, match="artifact members mismatch"):
        persistence.load_artifact(wrong_members, expected_model_type="test.Model")


def test_artifact_loader_rejects_invalid_manifest_and_identity(tmp_path: Path) -> None:
    path = _artifact(tmp_path)
    manifest, arrays = _payloads(path)

    _rewrite(path, b"not-json", arrays)
    with pytest.raises(ValueError, match="not valid UTF-8 JSON"):
        persistence.load_artifact(path, expected_model_type="test.Model")

    manifest["schema_version"] = 2
    _rewrite(path, manifest, arrays)
    with pytest.raises(ValueError, match="unsupported artifact schema_version"):
        persistence.load_artifact(path, expected_model_type="test.Model")

    manifest["schema_version"] = 1
    manifest["model_type"] = "other.Model"
    _rewrite(path, manifest, arrays)
    with pytest.raises(ValueError, match="artifact model_type mismatch"):
        persistence.load_artifact(path, expected_model_type="test.Model")


def test_artifact_loader_rejects_payload_and_manifest_tampering(tmp_path: Path) -> None:
    path = _artifact(tmp_path)
    manifest, arrays = _payloads(path)

    manifest["arrays_sha256"] = "0" * 64
    _rewrite(path, manifest, arrays)
    with pytest.raises(ValueError, match="checksum mismatch"):
        persistence.load_artifact(path, expected_model_type="test.Model")

    manifest["arrays_sha256"] = sha256(arrays).hexdigest()
    manifest["arrays"] = {}
    _rewrite(path, manifest, arrays)
    with pytest.raises(ValueError, match="array inventory"):
        persistence.load_artifact(path, expected_model_type="test.Model")

    manifest["arrays"] = {"weights": {"dtype": "<f8", "shape": [4]}}
    _rewrite(path, manifest, arrays)
    with pytest.raises(ValueError, match="array metadata mismatch"):
        persistence.load_artifact(path, expected_model_type="test.Model")


@pytest.mark.parametrize(("field", "value"), [("config", []), ("state", [])])
def test_artifact_loader_requires_mapping_config_and_state(
    tmp_path: Path, field: str, value: object
) -> None:
    path = _artifact(tmp_path)
    manifest, arrays = _payloads(path)
    manifest[field] = value
    _rewrite(path, manifest, arrays)

    with pytest.raises(ValueError, match="config and state must be mappings"):
        persistence.load_artifact(path, expected_model_type="test.Model")


def test_artifact_loader_rejects_invalid_array_payload(tmp_path: Path) -> None:
    path = _artifact(tmp_path)
    manifest, _ = _payloads(path)
    arrays = b"not-an-npz"
    manifest["arrays_sha256"] = sha256(arrays).hexdigest()
    _rewrite(path, manifest, arrays)

    with pytest.raises(ValueError, match="array payload is invalid"):
        persistence.load_artifact(path, expected_model_type="test.Model")


def test_artifact_array_requirements_report_inventory_and_dimensions(tmp_path: Path) -> None:
    artifact = persistence.load_artifact(_artifact(tmp_path), expected_model_type="test.Model")

    np.testing.assert_array_equal(
        persistence.require_array(artifact, "weights", ndim=2),
        np.arange(4).reshape(2, 2),
    )
    persistence.require_array_names(artifact, {"weights"})
    with pytest.raises(ValueError, match="missing required array"):
        persistence.require_array(artifact, "missing", ndim=1)
    with pytest.raises(ValueError, match="must be 1D"):
        persistence.require_array(artifact, "weights", ndim=1)
    with pytest.raises(ValueError, match="artifact arrays mismatch"):
        persistence.require_array_names(artifact, {"other"})


def test_load_config_supports_device_override_and_rejects_invalid_values(tmp_path: Path) -> None:
    artifact = persistence.load_artifact(_artifact(tmp_path), expected_model_type="test.Model")

    config = persistence.load_config(artifact, PCAConfig, device="cpu")
    assert config.device == "cpu"

    invalid = persistence.LoadedArtifact(
        config={**artifact.config, "n_factors": 0},
        state=artifact.state,
        arrays=artifact.arrays,
    )
    with pytest.raises(ValueError, match="invalid PCAConfig config"):
        persistence.load_config(invalid, PCAConfig, device=None)


def test_tensor_tree_round_trip_supports_all_container_types() -> None:
    value = {
        "list": [torch.tensor([1.0]), torch.tensor([2.0])],
        3: (torch.tensor([3.0]),),
        ("phase", 1): torch.tensor([4.0]),
    }

    tree, arrays = persistence.pack_tensor_tree(value)
    recovered = persistence.unpack_tensor_tree(tree, arrays, torch=torch)

    torch.testing.assert_close(recovered["list"][0], torch.tensor([1.0]))
    torch.testing.assert_close(recovered[3][0], torch.tensor([3.0]))
    torch.testing.assert_close(recovered[("phase", 1)], torch.tensor([4.0]))


def test_pack_tensor_tree_rejects_non_tensor_leaf() -> None:
    with pytest.raises(TypeError, match="leaves must provide"):
        persistence.pack_tensor_tree({"bad": object()})


@pytest.mark.parametrize(
    ("tree", "arrays", "message"),
    [
        ([], {}, "node is invalid"),
        ({"kind": "tensor", "array": "missing"}, {}, "references a missing array"),
        (
            {
                "kind": "list",
                "items": [
                    {"kind": "tensor", "array": "a"},
                    {"kind": "tensor", "array": "a"},
                ],
            },
            {"a": np.ones(1)},
            "reuses array",
        ),
        ({"kind": "list", "items": None}, {}, "must contain items"),
        ({"kind": "dict", "items": [["only-one"]]}, {}, "dict entry is invalid"),
        (
            {
                "kind": "dict",
                "items": [[["unsupported"], {"kind": "list", "items": []}]],
            },
            {},
            "unsupported type",
        ),
        (
            {
                "kind": "dict",
                "items": [
                    ["key", {"kind": "list", "items": []}],
                    ["key", {"kind": "list", "items": []}],
                ],
            },
            {},
            "duplicate key",
        ),
        ({"kind": "unknown", "items": []}, {}, "kind is unsupported"),
        ({"kind": "list", "items": []}, {"unused": np.ones(1)}, "does not reference every"),
    ],
)
def test_unpack_tensor_tree_rejects_invalid_structures(
    tree: object, arrays: dict[str, np.ndarray], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        persistence.unpack_tensor_tree(tree, arrays, torch=torch)


@pytest.mark.parametrize("name", ["", 3])
def test_prepare_arrays_rejects_invalid_names(name: object) -> None:
    with pytest.raises(ValueError, match="names must be non-empty strings"):
        persistence._prepare_arrays({name: np.ones(1)})


def test_prepare_arrays_rejects_object_dtype() -> None:
    with pytest.raises(TypeError, match="must not use object dtype"):
        persistence._prepare_arrays({"objects": np.array([object()], dtype=object)})


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (np.nan, ValueError, "non-finite"),
        ({1: "value"}, TypeError, "mapping keys must be strings"),
        ({"value": object()}, TypeError, "unsupported type object"),
    ],
)
def test_json_encoder_rejects_unsafe_metadata(
    value: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        persistence._encode_json(value)


def test_json_encoder_and_decoder_round_trip_supported_values() -> None:
    value = {
        "none": None,
        "bool": True,
        "int": 1,
        "string": "value",
        "float": 1.5,
        "tuple": (1, "two"),
        "list": [3.0],
    }

    assert persistence._decode_json(persistence._encode_json(value)) == value
    assert persistence._decode_json((1, [2])) == (1, [2])


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"__tuple__": "not-a-list"}, "tuple encoding must contain a list"),
        (float("inf"), "manifest contains non-finite"),
        (object(), "unsupported JSON value"),
    ],
)
def test_json_decoder_rejects_invalid_values(value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        persistence._decode_json(value)


def test_failed_atomic_save_preserves_prior_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _artifact(tmp_path)
    original = path.read_bytes()
    real_replace = persistence.os.replace

    def fail_replace(source: object, destination: object) -> None:
        if Path(destination) == path:
            raise OSError("injected replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(persistence.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        persistence.save_artifact(
            path,
            model_type="test.Model",
            config=PCAConfig(n_factors=1),
            state={},
            arrays={"weights": np.ones((1, 1))},
        )

    assert path.read_bytes() == original


def test_directory_fsync_is_skipped_when_unsupported_on_windows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(persistence.os, "name", "nt")

    def fail_open(*_: object, **__: object) -> int:
        raise AssertionError("directory file descriptors are not supported on Windows")

    monkeypatch.setattr(persistence.os, "open", fail_open)

    persistence._fsync_directory(tmp_path)
    assert not tuple(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (None, "missing a valid fit run record"),
        ({"schema_version": 2}, "invalid fit run record"),
    ],
)
def test_model_load_rejects_missing_or_invalid_fit_run_record(
    tmp_path: Path,
    replacement: dict[str, object] | None,
    message: str,
) -> None:
    model = PCAModel(PCAConfig(n_factors=1))
    model.fit(
        PersistentPanelBatch(
            returns=np.array([[1.0, 2.0], [2.0, 3.0]]),
            asset_ids=("A", "B"),
        )
    )
    path = model.save(tmp_path / "pca.ml4t")
    manifest, arrays = _payloads(path)
    state = manifest["state"]
    assert isinstance(state, dict)
    if replacement is None:
        state.pop("fit_run_record")
    else:
        state["fit_run_record"] = replacement
    _rewrite(path, manifest, arrays)

    with pytest.raises(ValueError, match=message):
        PCAModel.load(path)
