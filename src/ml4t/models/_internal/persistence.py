"""Safe, versioned model artifact persistence."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, cast
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

import numpy as np

_SCHEMA_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_ARRAYS_NAME = "arrays.npz"


@dataclass(frozen=True, slots=True)
class LoadedArtifact:
    config: dict[str, Any]
    state: dict[str, Any]
    arrays: dict[str, np.ndarray]


def save_artifact(
    path: str | Path,
    *,
    model_type: str,
    config: Any,
    state: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> Path:
    output_path = Path(path)
    prepared_arrays = _prepare_arrays(arrays)
    array_buffer = BytesIO()
    savez_compressed = cast(Any, np.savez_compressed)
    savez_compressed(array_buffer, **prepared_arrays)
    array_payload = array_buffer.getvalue()
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "model_type": model_type,
        "config": _encode_json(asdict(config)),
        "state": _encode_json(state),
        "arrays_sha256": sha256(array_payload).hexdigest(),
        "arrays": {
            name: {"dtype": array.dtype.str, "shape": list(array.shape)}
            for name, array in prepared_arrays.items()
        },
    }
    manifest_payload = json.dumps(
        manifest,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        with ZipFile(temporary_path, mode="w", compression=ZIP_DEFLATED) as archive:
            archive.writestr(_MANIFEST_NAME, manifest_payload)
            archive.writestr(_ARRAYS_NAME, array_payload)
        with temporary_path.open("rb+") as artifact_file:
            os.fsync(artifact_file.fileno())
        os.replace(temporary_path, output_path)
        _fsync_directory(output_path.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output_path


def load_artifact(
    path: str | Path,
    *,
    expected_model_type: str,
) -> LoadedArtifact:
    artifact_path = Path(path)
    try:
        with ZipFile(artifact_path, mode="r") as archive:
            members = set(archive.namelist())
            expected_members = {_MANIFEST_NAME, _ARRAYS_NAME}
            if members != expected_members:
                raise ValueError(
                    f"artifact members mismatch: expected {sorted(expected_members)}, "
                    f"got {sorted(members)}"
                )
            manifest_payload = archive.read(_MANIFEST_NAME)
            array_payload = archive.read(_ARRAYS_NAME)
    except (BadZipFile, KeyError) as exc:
        raise ValueError(f"invalid ml4t-models artifact at {artifact_path}") from exc

    try:
        manifest = json.loads(manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("artifact manifest is not valid UTF-8 JSON") from exc
    if manifest.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(
            f"unsupported artifact schema_version: expected {_SCHEMA_VERSION}, "
            f"got {manifest.get('schema_version')!r}"
        )
    if manifest.get("model_type") != expected_model_type:
        raise ValueError(
            f"artifact model_type mismatch: expected {expected_model_type!r}, "
            f"got {manifest.get('model_type')!r}"
        )
    actual_hash = sha256(array_payload).hexdigest()
    if manifest.get("arrays_sha256") != actual_hash:
        raise ValueError("artifact array payload checksum mismatch")

    arrays = _load_arrays(array_payload)
    expected_arrays = manifest.get("arrays")
    if not isinstance(expected_arrays, dict) or set(arrays) != set(expected_arrays):
        raise ValueError("artifact array inventory does not match its manifest")
    for name, array in arrays.items():
        expected = expected_arrays[name]
        if expected != {"dtype": array.dtype.str, "shape": list(array.shape)}:
            raise ValueError(f"artifact array metadata mismatch for {name!r}")

    config = _decode_json(manifest.get("config"))
    state = _decode_json(manifest.get("state"))
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError("artifact config and state must be mappings")
    return LoadedArtifact(config=config, state=state, arrays=arrays)


def require_array(
    artifact: LoadedArtifact,
    name: str,
    *,
    ndim: int,
) -> np.ndarray:
    try:
        array = artifact.arrays[name]
    except KeyError as exc:
        raise ValueError(f"artifact is missing required array {name!r}") from exc
    if array.ndim != ndim:
        raise ValueError(f"artifact array {name!r} must be {ndim}D; got shape {array.shape}")
    return array.copy()


def require_array_names(artifact: LoadedArtifact, expected: set[str]) -> None:
    if set(artifact.arrays) != expected:
        raise ValueError(
            f"artifact arrays mismatch: expected {sorted(expected)}, got {sorted(artifact.arrays)}"
        )


def load_config(
    artifact: LoadedArtifact,
    config_type: Any,
    *,
    device: str | None,
) -> Any:
    config_values = dict(artifact.config)
    if device is not None:
        config_values["device"] = device
    try:
        return config_type(**config_values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"artifact contains an invalid {config_type.__name__} config") from exc


def pack_tensor_tree(value: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}

    def pack(item: Any) -> dict[str, Any]:
        if isinstance(item, dict):
            return {
                "kind": "dict",
                "items": [[_encode_json(key), pack(child)] for key, child in item.items()],
            }
        if isinstance(item, list):
            return {"kind": "list", "items": [pack(child) for child in item]}
        if isinstance(item, tuple):
            return {"kind": "tuple", "items": [pack(child) for child in item]}
        detach = getattr(item, "detach", None)
        if not callable(detach):
            raise TypeError(
                f"tensor tree leaves must provide detach().cpu().numpy(); got {type(item).__name__}"
            )
        array_name = f"tensor_{len(arrays):08d}"
        arrays[array_name] = np.asarray(detach().cpu().numpy())
        return {"kind": "tensor", "array": array_name}

    return pack(value), arrays


def unpack_tensor_tree(
    tree: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    torch: Any,
) -> Any:
    used_arrays: set[str] = set()

    def unpack(node: Any) -> Any:
        if not isinstance(node, dict) or not isinstance(node.get("kind"), str):
            raise ValueError("artifact tensor tree node is invalid")
        kind = node["kind"]
        if kind == "tensor":
            array_name = node.get("array")
            if not isinstance(array_name, str) or array_name not in arrays:
                raise ValueError("artifact tensor tree references a missing array")
            if array_name in used_arrays:
                raise ValueError(f"artifact tensor tree reuses array {array_name!r}")
            used_arrays.add(array_name)
            return torch.from_numpy(arrays[array_name].copy())
        items = node.get("items")
        if not isinstance(items, list):
            raise ValueError(f"artifact tensor tree {kind!r} node must contain items")
        if kind == "list":
            return [unpack(child) for child in items]
        if kind == "tuple":
            return tuple(unpack(child) for child in items)
        if kind == "dict":
            result: dict[Any, Any] = {}
            for entry in items:
                if not isinstance(entry, list) or len(entry) != 2:
                    raise ValueError("artifact tensor tree dict entry is invalid")
                key = _decode_json(entry[0])
                if not isinstance(key, str | int | tuple):
                    raise ValueError("artifact tensor tree dict key has an unsupported type")
                if key in result:
                    raise ValueError(f"artifact tensor tree contains duplicate key {key!r}")
                result[key] = unpack(entry[1])
            return result
        raise ValueError(f"artifact tensor tree kind is unsupported: {kind!r}")

    result = unpack(tree)
    if used_arrays != set(arrays):
        raise ValueError("artifact tensor tree does not reference every stored array")
    return result


def _prepare_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    prepared: dict[str, np.ndarray] = {}
    for name, values in arrays.items():
        if not name or not isinstance(name, str):
            raise ValueError("artifact array names must be non-empty strings")
        array = np.asarray(values)
        if array.dtype.hasobject:
            raise TypeError(f"artifact array {name!r} must not use object dtype")
        prepared[name] = np.ascontiguousarray(array)
    return prepared


def _load_arrays(payload: bytes) -> dict[str, np.ndarray]:
    try:
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError) as exc:
        raise ValueError("artifact array payload is invalid") from exc
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError("artifact array payload contains object dtype")
    return arrays


def _encode_json(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"artifact metadata contains non-finite value {value!r}")
        return value
    if isinstance(value, tuple):
        return {"__tuple__": [_encode_json(item) for item in value]}
    if isinstance(value, list):
        return [_encode_json(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("artifact metadata mapping keys must be strings")
        return {key: _encode_json(item) for key, item in value.items()}
    raise TypeError(f"artifact metadata contains unsupported type {type(value).__name__}")


def _decode_json(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(_decode_json(item) for item in value)
    if isinstance(value, dict):
        if set(value) == {"__tuple__"}:
            items = value["__tuple__"]
            if not isinstance(items, list):
                raise ValueError("artifact tuple encoding must contain a list")
            return tuple(_decode_json(item) for item in items)
        return {key: _decode_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_json(item) for item in value]
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"artifact manifest contains non-finite value {value!r}")
        return value
    if value is None or isinstance(value, bool | int | str):
        return value
    raise ValueError(f"artifact manifest contains unsupported JSON value {value!r}")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
