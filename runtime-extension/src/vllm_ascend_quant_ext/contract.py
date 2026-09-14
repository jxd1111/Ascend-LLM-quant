"""Closed, read-only validation of Ascend quantized model artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from dataclasses import dataclass
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

CONTRACT_FILENAME = "ascend_quant_artifact.json"
SCHEMA_VERSION = "1.1.0"
MAX_SAFETENSORS_HEADER_BYTES = 128 * 1024 * 1024
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


class ContractError(ValueError):
    """An artifact is incomplete, unknown, or incompatible."""


class ArtifactContractValidator:
    """Import-only carrier exposed to the Extension Manager.

    Loading this object has no torch, vLLM, vLLM-Ascend, device, or model
    side effects. Runtime activation remains blocked until the host publishes
    the versioned loader and operator-selection protocols declared by the
    Bundle manifest.
    """

    status = "import_only"

    @staticmethod
    def validate(model_path: str | Path, *, check_software: bool = True) -> dict[str, Any]:
        return validate_artifact(Path(model_path), check_software=check_software)


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    file: str


def _exact_keys(value: Any, required: set[str], optional: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{where} must be an object")
    keys = set(value)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise ContractError(f"{where} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ContractError(f"{where} contains unknown fields: {', '.join(sorted(unknown))}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, name: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"missing {name}: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid {name}: {exc}") from exc


def _read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            raw_length = stream.read(8)
            if len(raw_length) != 8:
                raise ContractError(f"truncated safetensors header: {path.name}")
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length <= 0 or header_length > MAX_SAFETENSORS_HEADER_BYTES:
                raise ContractError(
                    f"unsafe safetensors header length {header_length}: {path.name}"
                )
            if 8 + header_length > size:
                raise ContractError(f"safetensors header exceeds file size: {path.name}")
            header = json.loads(stream.read(header_length))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid safetensors file {path.name}: {exc}") from exc
    if not isinstance(header, dict):
        raise ContractError(f"safetensors header is not an object: {path.name}")
    return header, size - 8 - header_length


def _collect_tensors(model_path: Path, filenames: list[str]) -> dict[str, TensorInfo]:
    tensors: dict[str, TensorInfo] = {}
    for filename in filenames:
        if Path(filename).name != filename:
            raise ContractError(f"weight filename must not contain a path: {filename}")
        path = model_path / filename
        if not path.is_file():
            raise ContractError(f"missing weight shard: {filename}")
        header, payload_size = _read_safetensors_header(path)
        intervals: list[tuple[int, int, str]] = []
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ContractError(f"duplicate tensor across weight shards: {name}")
            meta = _exact_keys(
                metadata, {"dtype", "shape", "data_offsets"}, set(), f"tensor {name}"
            )
            shape = meta["shape"]
            offsets = meta["data_offsets"]
            if (
                not isinstance(shape, list)
                or any(not isinstance(dim, int) or dim < 0 for dim in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(not isinstance(offset, int) or offset < 0 for offset in offsets)
                or offsets[0] > offsets[1]
            ):
                raise ContractError(f"invalid tensor metadata: {name}")
            dtype = str(meta["dtype"])
            width = SAFETENSORS_DTYPE_BYTES.get(dtype)
            if width is None:
                raise ContractError(f"unsupported safetensors dtype {dtype}: {name}")
            elements = 1
            for dim in shape:
                elements *= dim
            expected_size = elements * width
            start, end = offsets
            if end > payload_size:
                raise ContractError(f"tensor payload exceeds file size: {name}")
            if end - start != expected_size:
                raise ContractError(f"tensor payload size does not match dtype/shape: {name}")
            intervals.append((start, end, name))
            tensors[name] = TensorInfo(dtype, tuple(shape), filename)
        cursor = 0
        for start, end, name in sorted(intervals):
            if start != cursor:
                raise ContractError(f"non-contiguous or overlapping tensor payload: {name}")
            cursor = end
        if cursor != payload_size:
            raise ContractError(f"unclaimed bytes in safetensors payload: {filename}")
    return tensors


def _validate_file_record(value: Any, expected_name: str | None, where: str) -> dict[str, Any]:
    record = _exact_keys(value, {"name", "size", "sha256"}, set(), where)
    name = record["name"]
    if not isinstance(name, str) or Path(name).name != name or not name:
        raise ContractError(f"{where}.name must be a plain filename")
    if expected_name is not None and name != expected_name:
        raise ContractError(f"{where}.name must be {expected_name}")
    if (
        not isinstance(record["size"], int)
        or isinstance(record["size"], bool)
        or record["size"] < 0
    ):
        raise ContractError(f"{where}.size must be a non-negative integer")
    if not isinstance(record["sha256"], str) or not SHA256_PATTERN.fullmatch(record["sha256"]):
        raise ContractError(f"{where}.sha256 must be 64 lowercase hexadecimal characters")
    return record


def _artifact_content_digest(files: dict[str, Any]) -> str:
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _validate_evidence_record(value: Any, where: str) -> dict[str, Any]:
    record = _exact_keys(value, {"name", "size", "sha256"}, set(), where)
    name = record["name"]
    if not isinstance(name, str):
        raise ContractError(f"{where}.name must be a safe relative path")
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ContractError(f"{where}.name must be a safe relative path")
    if not isinstance(record["size"], int) or isinstance(record["size"], bool) or record["size"] < 0:
        raise ContractError(f"{where}.size must be a non-negative integer")
    if not isinstance(record["sha256"], str) or not SHA256_PATTERN.fullmatch(record["sha256"]):
        raise ContractError(f"{where}.sha256 must be 64 lowercase hexadecimal characters")
    return record


def _validate_evidence_claims(
    model_path: Path,
    contract: dict[str, Any],
    artifact_files_digest: str,
) -> None:
    declared = contract["evidence"]
    profiles: set[str] = set()
    required_root = {
        "schema_version",
        "run_id",
        "profile",
        "artifact",
        "software",
        "hardware",
        "workload",
        "results",
        "raw_logs",
    }
    for record in declared["results"]:
        evidence = _exact_keys(
            _read_json(model_path / record["name"], "evidence result"),
            required_root,
            set(),
            f"evidence result {record['name']}",
        )
        if evidence["schema_version"] != "1.1.0":
            raise ContractError(f"unsupported evidence schema: {record['name']}")
        profile = evidence["profile"]
        if not isinstance(profile, str) or profile not in declared["verified_profiles"]:
            raise ContractError(f"undeclared evidence profile: {record['name']}")
        artifact = _exact_keys(
            evidence["artifact"],
            {"model", "artifact_id", "artifact_files_sha256"},
            set(),
            f"evidence artifact {record['name']}",
        )
        if profile != "BF16" and (
            artifact["artifact_id"] != contract["artifact_id"]
            or artifact["artifact_files_sha256"] != artifact_files_digest
        ):
            raise ContractError(f"evidence artifact identity mismatch: {record['name']}")
        profiles.add(profile)
    if profiles != set(declared["verified_profiles"]):
        raise ContractError("every verified profile must have a matching evidence result")
    if declared["level"] == "matched_benchmark" and not {
        "BF16",
        contract["quantization"]["scheme"],
    } <= profiles:
        raise ContractError("matched_benchmark requires BF16 and quantized evidence records")


def _installed_version(aliases: tuple[str, ...]) -> tuple[str, str] | None:
    """Resolve one distribution without trusting arbitrary metadata order.

    ``importlib.metadata.version`` returns the first matching ``dist-info``
    directory. Editable environments can retain several versions of the same
    distribution. Reject conflicting metadata instead of admitting an
    artifact against a stale version record.
    """

    for name in aliases:
        canonical_name = canonicalize_name(name)
        versions = {
            str(distribution.version)
            for distribution in distributions()
            if distribution.metadata.get("Name")
            and canonicalize_name(distribution.metadata["Name"]) == canonical_name
        }
        if not versions:
            continue
        if len(versions) != 1:
            found = ", ".join(sorted(versions, key=Version))
            raise ContractError(
                f"ambiguous installed package metadata for {name}: {found}"
            )
        return name, versions.pop()
    return None


def _cann_version() -> str | None:
    candidates: list[Path] = []
    toolkit_home = os.environ.get("ASCEND_TOOLKIT_HOME")
    if toolkit_home:
        candidates.append(Path(toolkit_home) / "aarch64-linux/ascend_toolkit_install.info")
    candidates.extend(
        sorted(Path("/usr/local/Ascend").glob("cann-*/aarch64-linux/ascend_toolkit_install.info"))
    )
    for path in candidates:
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("version="):
                    return line.split("=", 1)[1].strip()
        except OSError:
            continue
    return None


def _check_version(label: str, installed: str | None, requirement: str) -> str:
    if installed is None:
        raise ContractError(f"cannot determine installed {label} version")
    try:
        accepted = Version(installed) in SpecifierSet(requirement)
    except (InvalidSpecifier, InvalidVersion) as exc:
        raise ContractError(f"invalid {label} compatibility declaration: {requirement}") from exc
    if not accepted:
        raise ContractError(f"incompatible {label}: installed {installed}, required {requirement}")
    return installed


def _validate_structure(contract: Any) -> dict[str, Any]:
    root = _exact_keys(
        contract,
        {
            "schema_version",
            "artifact_id",
            "format",
            "model",
            "quantization",
            "runtime",
            "software",
            "files",
            "evidence",
        },
        set(),
        "contract",
    )
    if root["schema_version"] != SCHEMA_VERSION:
        raise ContractError(f"unsupported schema_version: {root['schema_version']!r}")
    if root["format"] != "modelslim-ascend-v1":
        raise ContractError(f"unsupported artifact format: {root['format']!r}")
    if not isinstance(root["artifact_id"], str) or not root["artifact_id"]:
        raise ContractError("artifact_id must be a non-empty string")

    model = _exact_keys(
        root["model"], {"model_type", "architectures", "supported_shapes"}, set(), "model"
    )
    shapes = _exact_keys(
        model["supported_shapes"],
        {"weight_rank", "input_multiple", "output_multiple"},
        set(),
        "model.supported_shapes",
    )
    if shapes["weight_rank"] != 2:
        raise ContractError("only 2D weight tensors are supported")
    for key in ("input_multiple", "output_multiple"):
        if not isinstance(shapes[key], int) or shapes[key] < 1:
            raise ContractError(f"model.supported_shapes.{key} must be positive")
    if (
        not isinstance(model["architectures"], list)
        or not model["architectures"]
        or not all(isinstance(x, str) for x in model["architectures"])
    ):
        raise ContractError("model.architectures must be a non-empty string array")

    quant = _exact_keys(
        root["quantization"],
        {
            "scheme",
            "producer_quant_type",
            "runtime_quant_type",
            "weight",
            "activation",
            "scale",
            "zero_point",
        },
        set(),
        "quantization",
    )
    if quant["scheme"] not in {"W8A8", "W4A4", "W4A8"}:
        raise ContractError(f"unsupported quantization scheme: {quant['scheme']!r}")
    if not isinstance(quant["producer_quant_type"], str) or not quant["producer_quant_type"]:
        raise ContractError("producer_quant_type must be a non-empty string")
    weight = _exact_keys(
        quant["weight"],
        {
            "bits",
            "storage_dtype",
            "signed",
            "packing",
            "layout",
            "granularity",
            "axis",
            "group_size",
        },
        set(),
        "quantization.weight",
    )
    activation = _exact_keys(
        quant["activation"],
        {"bits", "dtype", "granularity", "dynamic"},
        set(),
        "quantization.activation",
    )
    scale = _exact_keys(
        quant["scale"],
        {"dtype", "weight_shape", "activation_shape", "scale_bias"},
        set(),
        "quantization.scale",
    )
    zero = _exact_keys(
        quant["zero_point"],
        {"weight", "activation", "dtype", "semantics"},
        set(),
        "quantization.zero_point",
    )
    expected_bits = {"W8A8": (8, 8), "W4A4": (4, 4), "W4A8": (4, 8)}[quant["scheme"]]
    if (weight["bits"], activation["bits"]) != expected_bits:
        raise ContractError(f"{quant['scheme']} bit-width declaration is inconsistent")
    if weight["storage_dtype"] != "int8" or weight["signed"] is not True or weight["axis"] != 0:
        raise ContractError("only signed int8 storage on output axis 0 is supported")
    if weight["bits"] == 8 and (
        weight["packing"] != "none" or weight["layout"] != "logical_out_in"
    ):
        raise ContractError("W8 weights must use unpacked logical_out_in layout")
    if weight["bits"] == 4 and weight["packing"] != "signed_int4_nibble_low_high":
        raise ContractError("W4 weights must use signed low-high nibble packing")
    if weight["layout"] not in {"logical_out_in", "packed_out_in"}:
        raise ContractError("unsupported weight layout")
    if (weight["packing"] == "none") != (weight["layout"] == "logical_out_in"):
        raise ContractError("weight packing and layout disagree")
    if weight["granularity"] not in {"per_channel", "per_group"}:
        raise ContractError("unsupported weight granularity")
    if weight["granularity"] == "per_group":
        if not isinstance(weight["group_size"], int) or weight["group_size"] < 1:
            raise ContractError("per_group weight requires a positive group_size")
    elif weight["granularity"] == "per_channel" and weight["group_size"] is not None:
        raise ContractError("per_channel weight requires group_size=null")
    expected_activation_dtype = {4: "int4", 8: "int8"}[activation["bits"]]
    if activation["dtype"] != expected_activation_dtype:
        raise ContractError("activation dtype and bit width disagree")
    if activation["granularity"] not in {"per_token", "per_tensor", "pd_mix"}:
        raise ContractError("unsupported activation granularity")
    if not isinstance(activation["dynamic"], bool):
        raise ContractError("activation.dynamic must be Boolean")
    if scale["dtype"] not in {"float32", "bfloat16", "float16"}:
        raise ContractError("unsupported scale dtype")
    if scale["weight_shape"] not in {"out_1", "out_groups"} or scale["activation_shape"] not in {
        "one",
        "runtime_per_token",
    }:
        raise ContractError("unsupported scale shape semantics")
    if scale["scale_bias"] not in {"forbidden", "required_out_1_or_16"}:
        raise ContractError("unsupported scale_bias semantics")
    if quant["scheme"] == "W4A8" and weight["granularity"] == "per_group":
        raise ContractError(
            "W4A8 per-group hierarchical scale/offset is not admitted by contract v1"
        )
    if quant["scheme"] == "W4A8" and scale["scale_bias"] != "required_out_1_or_16":
        raise ContractError("packed W4A8 v1 requires scale_bias")
    if quant["scheme"] != "W4A8" and scale["scale_bias"] != "forbidden":
        raise ContractError("scale_bias is only admitted for W4A8")
    if zero["weight"] not in {"required", "forbidden"} or zero["activation"] not in {
        "required",
        "forbidden",
        "runtime",
    }:
        raise ContractError("unsupported zero-point presence semantics")
    if zero["dtype"] not in {"float32", "int32"}:
        raise ContractError("unsupported zero-point dtype")
    if zero["semantics"] != "additive_offset_before_scale":
        raise ContractError("unsupported zero-point semantics")
    if quant["scheme"] == "W8A8":
        if weight["granularity"] != "per_channel":
            raise ContractError("W8A8 v1.1 requires per-channel weights")
        if activation != {
            "bits": 8,
            "dtype": "int8",
            "granularity": "pd_mix",
            "dynamic": True,
        }:
            raise ContractError("W8A8 activation semantics do not match the v1.1 profile")
        if scale != {
            "dtype": "float32",
            "weight_shape": "out_1",
            "activation_shape": "one",
            "scale_bias": "forbidden",
        }:
            raise ContractError("W8A8 scale semantics do not match the v1.1 profile")
        if zero != {
            "weight": "required",
            "activation": "required",
            "dtype": "float32",
            "semantics": "additive_offset_before_scale",
        }:
            raise ContractError("W8A8 zero-point semantics do not match the v1.1 profile")
    elif quant["scheme"] == "W4A4":
        if activation != {
            "bits": 4,
            "dtype": "int4",
            "granularity": "per_token",
            "dynamic": True,
        }:
            raise ContractError("W4A4 activation semantics do not match the v1.1 profile")
        expected_weight_shape = (
            "out_groups" if weight["granularity"] == "per_group" else "out_1"
        )
        if scale != {
            "dtype": "float32",
            "weight_shape": expected_weight_shape,
            "activation_shape": "runtime_per_token",
            "scale_bias": "forbidden",
        }:
            raise ContractError("W4A4 scale semantics do not match the v1.1 profile")
        if zero != {
            "weight": "required",
            "activation": "runtime",
            "dtype": "float32",
            "semantics": "additive_offset_before_scale",
        }:
            raise ContractError("W4A4 zero-point semantics do not match the v1.1 profile")
    else:
        if weight["granularity"] != "per_channel":
            raise ContractError("W4A8 v1.1 requires per-channel weights")
        if activation != {
            "bits": 8,
            "dtype": "int8",
            "granularity": "per_token",
            "dynamic": True,
        }:
            raise ContractError("W4A8 activation semantics do not match the v1.1 profile")
        if scale != {
            "dtype": "float32",
            "weight_shape": "out_1",
            "activation_shape": "runtime_per_token",
            "scale_bias": "required_out_1_or_16",
        }:
            raise ContractError("W4A8 scale semantics do not match the v1.1 profile")
        if zero != {
            "weight": "required",
            "activation": "runtime",
            "dtype": "float32",
            "semantics": "additive_offset_before_scale",
        }:
            raise ContractError("W4A8 zero-point semantics do not match the v1.1 profile")
    allowed_types = {
        "W8A8": {"ASCEND_QUANT_W8A8"},
        "W4A4": {"W4A4_DYNAMIC"},
        "W4A8": {"W4A8_DYNAMIC"},
    }
    if quant["runtime_quant_type"] not in allowed_types[quant["scheme"]]:
        raise ContractError("runtime quant type is not admitted for the declared scheme")

    runtime = _exact_keys(
        root["runtime"], {"host", "loader", "scheme_provider", "operators"}, set(), "runtime"
    )
    if (
        runtime["host"] != "vllm-ascend"
        or runtime["loader"] != "modelslim"
        or runtime["scheme_provider"] != "vllm-ascend-quant-ext"
    ):
        raise ContractError("runtime host/loader/provider is not supported")
    allowed_operators = {
        "torch_npu.npu_quant_matmul",
        "torch_npu.npu_dynamic_quant",
        "torch_npu.npu_convert_weight_to_int4pack",
    }
    if (
        not isinstance(runtime["operators"], list)
        or not runtime["operators"]
        or not all(
            isinstance(operator, str) and operator in allowed_operators
            for operator in runtime["operators"]
        )
    ):
        raise ContractError("runtime contains an unknown or empty operator list")
    expected_operators = {
        "W8A8": {
            "torch_npu.npu_dynamic_quant",
            "torch_npu.npu_quant_matmul",
        },
        "W4A4": allowed_operators,
        "W4A8": allowed_operators,
    }[quant["scheme"]]
    if len(runtime["operators"]) != len(set(runtime["operators"])) or set(
        runtime["operators"]
    ) != expected_operators:
        raise ContractError("runtime operator set does not match the declared scheme")

    software = _exact_keys(
        root["software"], {"cann", "torch_npu", "vllm", "vllm_ascend"}, set(), "software"
    )
    if not all(isinstance(value, str) and value for value in software.values()):
        raise ContractError("all software compatibility ranges must be non-empty strings")
    files = _exact_keys(
        root["files"],
        {"config", "description", "indexes", "weights"},
        set(),
        "files",
    )
    _validate_file_record(files["config"], "config.json", "files.config")
    _validate_file_record(
        files["description"],
        "quant_model_description.json",
        "files.description",
    )
    if not isinstance(files["weights"], list) or not files["weights"]:
        raise ContractError("files.weights must be a non-empty array")
    if not isinstance(files["indexes"], list):
        raise ContractError("files.indexes must be an array")
    index_names: set[str] = set()
    for index, value in enumerate(files["indexes"]):
        record = _validate_file_record(value, None, f"files.indexes[{index}]")
        if not record["name"].endswith(".safetensors.index.json"):
            raise ContractError("index filenames must end with .safetensors.index.json")
        if record["name"] in index_names:
            raise ContractError(f"duplicate index file record: {record['name']}")
        index_names.add(record["name"])
    weight_names: set[str] = set()
    for index, value in enumerate(files["weights"]):
        record = _validate_file_record(value, None, f"files.weights[{index}]")
        if not record["name"].endswith(".safetensors"):
            raise ContractError("weight filenames must end with .safetensors")
        if record["name"] in weight_names:
            raise ContractError(f"duplicate weight file record: {record['name']}")
        weight_names.add(record["name"])
    evidence = _exact_keys(
        root["evidence"], {"level", "verified_profiles", "results"}, set(), "evidence"
    )
    if evidence["level"] not in {"schema_only", "correctness", "npu_e2e", "matched_benchmark"}:
        raise ContractError("unknown evidence level")
    if not isinstance(evidence["verified_profiles"], list) or not isinstance(
        evidence["results"], list
    ):
        raise ContractError("invalid evidence arrays")
    if evidence["level"] == "schema_only":
        if evidence["verified_profiles"] or evidence["results"]:
            raise ContractError("schema_only evidence must not declare profiles or results")
    elif quant["scheme"] not in evidence["verified_profiles"] or not evidence["results"]:
        raise ContractError(
            f"{evidence['level']} evidence requires the {quant['scheme']} profile and results"
        )
    if not all(isinstance(profile, str) for profile in evidence["verified_profiles"]):
        raise ContractError("evidence profiles must be strings")
    if len(evidence["verified_profiles"]) != len(set(evidence["verified_profiles"])):
        raise ContractError("evidence.verified_profiles contains duplicates")
    if not set(evidence["verified_profiles"]) <= {"BF16", "W8A8", "W4A4", "W4A8"}:
        raise ContractError("evidence contains an unknown profile")
    result_names: set[str] = set()
    for index, result in enumerate(evidence["results"]):
        record = _validate_evidence_record(result, f"evidence.results[{index}]")
        if record["name"] in result_names:
            raise ContractError(f"duplicate evidence result: {record['name']}")
        result_names.add(record["name"])
    return root


def _validate_tensor_contract(
    model_path: Path, contract: dict[str, Any], description: dict[str, Any]
) -> dict[str, TensorInfo]:
    filenames = [record["name"] for record in contract["files"]["weights"]]
    actual_files = sorted(path.name for path in model_path.glob("*.safetensors"))
    if sorted(filenames) != actual_files:
        raise ContractError("declared weight shards do not exactly match artifact files")
    tensors = _collect_tensors(model_path, filenames)
    index_names = [record["name"] for record in contract["files"]["indexes"]]
    if len(filenames) > 1 and len(index_names) != 1:
        raise ContractError("multi-shard artifacts require exactly one safetensors index")
    if len(index_names) > 1:
        raise ContractError("artifacts may declare at most one safetensors index")
    if index_names:
        index_value = _read_json(model_path / index_names[0], "safetensors index")
        index = _exact_keys(index_value, {"weight_map"}, {"metadata"}, "safetensors index")
        weight_map = index["weight_map"]
        if (
            not isinstance(weight_map, dict)
            or not all(
                isinstance(name, str) and isinstance(filename, str)
                for name, filename in weight_map.items()
            )
        ):
            raise ContractError("safetensors index weight_map must map strings to strings")
        if set(weight_map) != set(tensors):
            raise ContractError("safetensors index tensor names do not match weight shards")
        for name, filename in weight_map.items():
            if filename not in filenames or tensors[name].file != filename:
                raise ContractError(f"safetensors index maps {name} to the wrong shard")
    runtime_type = contract["quantization"]["runtime_quant_type"]
    producer_type = contract["quantization"]["producer_quant_type"]
    reserved_names = {
        "version",
        "model_quant_type",
        "metadata",
        "group_size",
        "optional",
        "w4a4_weight_packed",
    }
    unknown_metadata = {
        name for name in description if "." not in name and name not in reserved_names
    }
    if unknown_metadata:
        raise ContractError(
            f"description contains unknown metadata fields: {sorted(unknown_metadata)}"
        )
    if description.get("version") != "1.0.0":
        raise ContractError("description version must be 1.0.0")
    if description.get("model_quant_type") != runtime_type:
        raise ContractError("description model_quant_type does not match the contract")
    if not isinstance(description.get("metadata", {}), dict) or not isinstance(
        description.get("optional", {}), dict
    ):
        raise ContractError("description metadata/optional must be objects")
    tensor_description = {
        name: value for name, value in description.items() if name not in reserved_names
    }
    unknown_types = {
        value
        for value in tensor_description.values()
        if isinstance(value, str) and value not in {runtime_type, "FLOAT"}
    }
    if unknown_types:
        raise ContractError(
            f"description contains unknown quantization types: {sorted(unknown_types)}"
        )
    if (
        any(value == producer_type for value in tensor_description.values())
        and producer_type != runtime_type
    ):
        raise ContractError(
            f"artifact still contains producer type {producer_type}; runtime artifact is incomplete"
        )
    if any(not isinstance(value, str) for value in tensor_description.values()):
        raise ContractError("description tensor mappings must be strings")
    declared_quant_tensors = {
        name for name, value in tensor_description.items() if value == runtime_type
    }
    if not declared_quant_tensors:
        raise ContractError(f"description contains no {runtime_type} tensors")
    missing = declared_quant_tensors - set(tensors)
    if missing:
        raise ContractError(f"description references missing tensors: {sorted(missing)[:3]}")

    quant = contract["quantization"]
    shape_contract = contract["model"]["supported_shapes"]
    scale_dtype = {"float32": "F32", "float16": "F16", "bfloat16": "BF16"}[quant["scale"]["dtype"]]
    packed_factor = 2 if quant["weight"]["packing"] == "signed_int4_nibble_low_high" else 1
    for name in sorted(declared_quant_tensors):
        if not name.endswith(".weight"):
            continue
        info = tensors[name]
        if info.dtype != "I8" or len(info.shape) != shape_contract["weight_rank"]:
            raise ContractError(f"{name} must be a rank-2 I8 tensor")
        stored_out, input_size = info.shape
        output_size = stored_out * packed_factor
        if input_size % shape_contract["input_multiple"]:
            raise ContractError(f"{name} input dimension violates input_multiple")
        if output_size % shape_contract["output_multiple"]:
            raise ContractError(f"{name} logical output dimension violates output_multiple")
        prefix = name[: -len("weight")]
        scale_name = prefix + "weight_scale"
        offset_name = prefix + "weight_offset"
        if scale_name not in tensors:
            raise ContractError(f"missing weight scale for {name}")
        scale_info = tensors[scale_name]
        groups = 1
        if quant["weight"]["granularity"] == "per_group":
            group_size = quant["weight"]["group_size"]
            if input_size % group_size:
                raise ContractError(f"{name} input dimension is not divisible by group_size")
            groups = input_size // group_size
        if scale_info.dtype != scale_dtype or scale_info.shape != (output_size, groups):
            raise ContractError(
                f"{scale_name} has {scale_info.dtype}{scale_info.shape}, expected {scale_dtype}{(output_size, groups)}"
            )
        if quant["zero_point"]["weight"] == "required":
            offset_info = tensors.get(offset_name)
            expected_offset_dtype = {"float32": "F32", "int32": "I32"}[quant["zero_point"]["dtype"]]
            if (
                offset_info is None
                or offset_info.dtype != expected_offset_dtype
                or offset_info.shape != (output_size, groups)
            ):
                raise ContractError(f"invalid or missing weight offset for {name}")
        elif offset_name in tensors:
            raise ContractError(f"forbidden weight offset is present for {name}")
        if quant["scheme"] == "W8A8":
            required_aux = {
                prefix + "input_scale": ("F32", (1,)),
                prefix + "input_offset": ("F32", (1,)),
                prefix + "deq_scale": ("F32", (output_size,)),
                prefix + "quant_bias": ("I32", (output_size,)),
            }
            for aux_name, expected in required_aux.items():
                aux = tensors.get(aux_name)
                if aux is None or (aux.dtype, aux.shape) != expected:
                    raise ContractError(
                        f"{aux_name} is missing or incompatible; expected {expected[0]}{expected[1]}"
                    )
        scale_bias_name = prefix + "scale_bias"
        if quant["scale"]["scale_bias"] == "required_out_1_or_16":
            scale_bias = tensors.get(scale_bias_name)
            if (
                scale_bias is None
                or scale_bias.dtype != "F32"
                or scale_bias.shape not in {(output_size, 1), (output_size, 16)}
            ):
                raise ContractError(f"{scale_bias_name} is missing or incompatible for W4A8")
        elif scale_bias_name in tensors:
            raise ContractError(f"forbidden scale_bias is present for {name}")
    return tensors


def validate_artifact(model_path: Path, *, check_software: bool = True) -> dict[str, Any]:
    """Validate without importing torch, vLLM, torch-npu, or vLLM-Ascend."""

    model_path = model_path.resolve()
    if not model_path.is_dir():
        raise ContractError(f"model directory does not exist: {model_path}")
    contract = _validate_structure(_read_json(model_path / CONTRACT_FILENAME, "artifact contract"))
    files = contract["files"]
    file_records = [
        files["config"],
        files["description"],
        *files["indexes"],
        *files["weights"],
    ]
    for record in file_records:
        path = model_path / record["name"]
        if not path.is_file():
            raise ContractError(f"missing declared artifact file: {record['name']}")
        if path.stat().st_size != record["size"]:
            raise ContractError(f"artifact file size mismatch: {record['name']}")
        if _sha256(path) != record["sha256"]:
            raise ContractError(f"artifact file hash mismatch: {record['name']}")
    declared_indexes = sorted(record["name"] for record in files["indexes"])
    actual_indexes = sorted(path.name for path in model_path.glob("*.safetensors.index.json"))
    if declared_indexes != actual_indexes:
        raise ContractError("declared index files do not exactly match artifact files")
    expected_artifact_id = (
        f"{contract['model']['model_type']}:{contract['quantization']['scheme']}:"
        f"{_artifact_content_digest(files)[:16]}"
    )
    if contract["artifact_id"] != expected_artifact_id:
        raise ContractError("artifact_id does not match the declared file inventory")

    config_path = model_path / files["config"]["name"]
    config_value = _read_json(config_path, "model config")
    if not isinstance(config_value, dict):
        raise ContractError("config.json must be an object")
    config = _exact_keys(
        config_value,
        {"model_type"},
        set(config_value.keys()) - {"model_type"},
        "config.json",
    )
    model = contract["model"]
    if config["model_type"] != model["model_type"]:
        raise ContractError("contract model_type does not match config.json")
    config_architectures = config.get("architectures", [])
    if config_architectures != model["architectures"]:
        raise ContractError("contract architectures do not match config.json")

    description_path = model_path / files["description"]["name"]
    description = _read_json(description_path, "quantization description")
    if not isinstance(description, dict):
        raise ContractError("quantization description must be an object")
    tensors = _validate_tensor_contract(model_path, contract, description)

    for result in contract["evidence"]["results"]:
        path = model_path / result["name"]
        if not path.is_file():
            raise ContractError(f"missing evidence result: {result['name']}")
        if path.stat().st_size != result["size"]:
            raise ContractError(f"evidence result size mismatch: {result['name']}")
        if _sha256(path) != result["sha256"]:
            raise ContractError(f"evidence result hash mismatch: {result['name']}")
    _validate_evidence_claims(
        model_path,
        contract,
        _artifact_content_digest(files),
    )

    installed: dict[str, str] = {}
    if check_software:
        software = contract["software"]
        installed["cann"] = _check_version("CANN", _cann_version(), software["cann"])
        package_aliases = {
            "torch_npu": ("torch-npu", "torch_npu"),
            "vllm": ("vllm-hust", "vllm"),
            "vllm_ascend": ("vllm-ascend-hust", "vllm-ascend"),
        }
        for label, aliases in package_aliases.items():
            found = _installed_version(aliases)
            installed[label] = _check_version(label, found[1] if found else None, software[label])
    return {
        "valid": True,
        "model": str(model_path),
        "artifact_id": contract["artifact_id"],
        "schema_version": contract["schema_version"],
        "scheme": contract["quantization"]["scheme"],
        "runtime_quant_type": contract["quantization"]["runtime_quant_type"],
        "tensor_count": len(tensors),
        "installed": installed,
        "evidence": contract["evidence"],
    }
