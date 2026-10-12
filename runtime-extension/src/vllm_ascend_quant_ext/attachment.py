"""Fail-closed admission and reading of the online smoothing attachment.

The attachment is produced offline by ``ascend-quant-toolkit build-smoothing``
and is the only extra input the online W8A8 path needs: it carries the
``iter_smooth`` factors that let the runtime reproduce the artifact's
per-channel INT8 weights from a BF16 checkpoint.

Nothing in this module imports torch, vLLM or vLLM-Ascend.  Admission runs
before any device is touched and reads only headers, sizes and digests, so an
unknown, tampered with or unverified attachment fails closed at register time.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "ascend_quant_smoothing.json"
TENSOR_FILENAME = "ascend_quant_smoothing.safetensors"
SCHEMA_VERSION = "1.0.0"
KIND = "iter_smooth_replay"

COL_SUFFIX = "col_scale"
ROW_SUFFIX = "row_scale"
NORM_PREFIX = "norm:"

MAX_SAFETENSORS_HEADER_BYTES = 128 * 1024 * 1024
# Factors are floating point; anything else is a producer bug.  The attachment
# manifest may spell dtypes either way, so both notations are canonicalized.
FLOAT_DTYPES = frozenset({"F32", "F16", "BF16"})
DTYPE_ALIASES = {
    "FLOAT32": "F32",
    "FLOAT16": "F16",
    "BFLOAT16": "BF16",
    "DOUBLE": "F64",
    "FLOAT64": "F64",
    "INT8": "I8",
    "INT32": "I32",
    "INT64": "I64",
}


def canonical_dtype(value: object) -> str:
    """Return the safetensors spelling of a dtype written by either producer."""

    text = str(value).upper()
    return DTYPE_ALIASES.get(text, text)


class AttachmentError(ValueError):
    """The attachment is missing, unverified, or inconsistent."""


@dataclass(frozen=True)
class SmoothingAttachment:
    """An admitted attachment directory."""

    directory: Path
    manifest: dict[str, Any]
    tensors: dict[str, dict[str, Any]]
    packed_geometry: dict[str, list[int]]

    @property
    def attachment_id(self) -> str:
        return str(self.manifest.get("attachment_id", ""))

    @property
    def runtime_quant_type(self) -> str:
        return str(self.manifest.get("consumer", {}).get("runtime_quant_type", ""))

    @property
    def tensor_path(self) -> Path:
        return self.directory / TENSOR_FILENAME

    def has_prefix(self, prefix: str) -> bool:
        return f"{prefix}.{COL_SUFFIX}" in self.tensors and f"{prefix}.{ROW_SUFFIX}" in self.tensors

    def norm_modules(self) -> list[str]:
        return sorted(
            name[len(NORM_PREFIX) :] for name in self.tensors if name.startswith(NORM_PREFIX)
        )

    def open_tensor(self, name: str) -> Any:
        """Read one admitted tensor as a CPU tensor (imports safetensors lazily)."""

        if name not in self.tensors:
            raise AttachmentError(f"{name} is not part of the admitted attachment")
        from safetensors import safe_open

        with safe_open(str(self.tensor_path), framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    """Return the raw safetensors header without loading any tensor payload."""

    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            raw_length = stream.read(8)
            if len(raw_length) != 8:
                raise AttachmentError(f"truncated safetensors header: {path.name}")
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length <= 0 or header_length > MAX_SAFETENSORS_HEADER_BYTES:
                raise AttachmentError(f"unsafe safetensors header length: {path.name}")
            if 8 + header_length > size:
                raise AttachmentError(f"safetensors header exceeds the file size: {path.name}")
            header = json.loads(stream.read(header_length))
    except (OSError, json.JSONDecodeError) as exc:
        raise AttachmentError(f"invalid safetensors header: {path.name}: {exc}") from exc
    if not isinstance(header, dict):
        raise AttachmentError(f"safetensors header must be an object: {path.name}")
    return header


def validate_attachment(attachment: str | Path) -> SmoothingAttachment:
    """Admit one attachment directory, or raise :class:`AttachmentError`."""

    directory = Path(attachment).resolve()
    try:
        manifest = json.loads((directory / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AttachmentError(f"missing {MANIFEST_FILENAME} in {directory}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise AttachmentError(f"invalid {MANIFEST_FILENAME}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise AttachmentError(f"{MANIFEST_FILENAME} must be an object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise AttachmentError(f"unsupported attachment schema: {manifest.get('schema_version')}")
    if manifest.get("kind") != KIND:
        raise AttachmentError(f"unsupported attachment kind: {manifest.get('kind')}")
    if manifest.get("verified") is not True:
        raise AttachmentError("the producer did not verify this attachment against its artifact")

    declared_files = manifest.get("files")
    if not isinstance(declared_files, list) or not declared_files:
        raise AttachmentError("the attachment declares no files")
    declared_names: set[str] = set()
    for record in declared_files:
        name = str(record.get("name", "")) if isinstance(record, dict) else ""
        if not name or Path(name).name != name:
            raise AttachmentError(f"unsafe attachment file name: {name!r}")
        path = directory / name
        if not path.is_file():
            raise AttachmentError(f"missing attachment file: {name}")
        if path.stat().st_size != record.get("size"):
            raise AttachmentError(f"size mismatch for {name}")
        if _sha256(path) != record.get("sha256"):
            raise AttachmentError(f"sha256 mismatch for {name}")
        declared_names.add(name)
    if TENSOR_FILENAME not in declared_names:
        raise AttachmentError(f"{TENSOR_FILENAME} must be declared in the attachment manifest")

    declared_tensors = manifest.get("tensors")
    if not isinstance(declared_tensors, list) or not declared_tensors:
        raise AttachmentError("the attachment declares no tensors")
    inventory: dict[str, dict[str, Any]] = {}
    for record in declared_tensors:
        name = record.get("name") if isinstance(record, dict) else None
        if not isinstance(name, str) or not name:
            raise AttachmentError("every tensor record needs a name")
        inventory[name] = record

    header = _read_safetensors_header(directory / TENSOR_FILENAME)
    present = {name for name in header if name != "__metadata__"}
    missing = sorted(set(inventory) - present)
    if missing:
        raise AttachmentError(f"declared tensors are absent from the attachment: {missing[:5]}")
    unexpected = sorted(present - set(inventory))
    if unexpected:
        raise AttachmentError(f"undeclared tensors in the attachment: {unexpected[:5]}")
    for name, record in inventory.items():
        entry = header[name]
        entry_dtype = canonical_dtype(entry.get("dtype"))
        if entry_dtype != canonical_dtype(record.get("dtype")):
            raise AttachmentError(f"dtype mismatch for {name}")
        if entry_dtype not in FLOAT_DTYPES:
            raise AttachmentError(f"{name} must be a floating point factor, got {entry_dtype}")
        entry_shape = [int(size) for size in entry.get("shape", [])]
        record_shape = [int(size) for size in record.get("shape", [])]
        if entry_shape != record_shape:
            raise AttachmentError(f"shape mismatch for {name}")

    packed_geometry: dict[str, list[int]] = {}
    for entry in manifest.get("packed_aliases") or []:
        if isinstance(entry, dict) and isinstance(entry.get("alias"), str):
            packed_geometry[str(entry["alias"])] = [int(size) for size in entry.get("shard_rows", [])]
    return SmoothingAttachment(
        directory=directory,
        manifest=manifest,
        tensors=inventory,
        packed_geometry=packed_geometry,
    )
