"""Admission tests for the online smoothing attachment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from vllm_ascend_quant_ext.attachment import (
    MANIFEST_FILENAME,
    SCHEMA_VERSION,
    TENSOR_FILENAME,
    AttachmentError,
    validate_attachment,
)

QKV = "model.layers.0.self_attn.qkv_proj"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_attachment(tmp_path: Path, *, verified: bool = True, tamper: bool = False) -> Path:
    tensors = {
        f"{QKV}.col_scale": np.arange(1.0, 5.0, dtype=np.float32),
        f"{QKV}.row_scale": np.arange(1.0, 9.0, dtype=np.float32),
        "norm:model.layers.0.input_layernorm": np.full(4, 0.5, dtype=np.float32),
    }
    directory = tmp_path / "attachment"
    directory.mkdir()
    tensor_path = directory / TENSOR_FILENAME
    save_file(dict(sorted(tensors.items())), str(tensor_path))
    file_record = {
        "name": TENSOR_FILENAME,
        "size": tensor_path.stat().st_size,
        "sha256": _sha256(tensor_path),
    }
    if tamper:
        payload = bytearray(tensor_path.read_bytes())
        payload[-1] ^= 0xFF
        tensor_path.write_bytes(bytes(payload))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "iter_smooth_replay",
        "attachment_id": "deadbeef",
        "verified": verified,
        "consumer": {"runtime_quant_type": "ASCEND_QUANT_W8A8_ONLINE"},
        "packed_aliases": [{"alias": QKV, "shards": ["q_proj", "k_proj", "v_proj"], "shard_rows": [4, 2, 2]}],
        "tensors": [
            {"name": name, "dtype": str(array.dtype), "shape": [int(size) for size in array.shape]}
            for name, array in sorted(tensors.items())
        ],
        "files": [file_record],
    }
    (directory / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def test_admits_a_verified_attachment(tmp_path: Path):
    attachment = validate_attachment(_write_attachment(tmp_path))

    assert attachment.attachment_id == "deadbeef"
    assert attachment.runtime_quant_type == "ASCEND_QUANT_W8A8_ONLINE"
    assert attachment.packed_geometry[QKV] == [4, 2, 2]
    assert attachment.has_prefix(QKV) is True
    assert attachment.has_prefix("model.layers.0.self_attn.o_proj") is False
    assert attachment.norm_modules() == ["model.layers.0.input_layernorm"]
    assert attachment.open_tensor(f"{QKV}.col_scale").shape == (4,)


def test_unverified_attachment_fails_closed(tmp_path: Path):
    with pytest.raises(AttachmentError, match="did not verify"):
        validate_attachment(_write_attachment(tmp_path, verified=False))


def test_tampered_attachment_fails_closed(tmp_path: Path):
    with pytest.raises(AttachmentError, match="sha256 mismatch"):
        validate_attachment(_write_attachment(tmp_path, tamper=True))


def test_missing_manifest_fails_closed(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(AttachmentError, match="missing"):
        validate_attachment(empty)


def test_undeclared_tensor_fails_closed(tmp_path: Path):
    directory = _write_attachment(tmp_path)
    manifest_path = directory / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tensors"] = [
        record for record in manifest["tensors"] if not record["name"].startswith("norm:")
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AttachmentError, match="undeclared tensors"):
        validate_attachment(directory)


def test_non_float_factor_fails_closed(tmp_path: Path):
    directory = _write_attachment(tmp_path)
    manifest_path = directory / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tensors"][0]["dtype"] = "int8"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AttachmentError, match="dtype mismatch"):
        validate_attachment(directory)
