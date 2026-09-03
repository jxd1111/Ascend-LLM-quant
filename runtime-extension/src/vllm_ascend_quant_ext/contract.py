"""Closed, read-only validation of Ascend quantized model artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

CONTRACT_FILENAME = "ascend_quant_artifact.json"
SCHEMA_VERSION = "1.0.0"
MAX_SAFETENSORS_HEADER_BYTES = 128 * 1024 * 1024


class ContractError(ValueError):
    """An artifact is incomplete, unknown, or incompatible."""


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


def _read_safetensors_header(path: Path) -> dict[str, Any]:
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
    return header


def _collect_tensors(model_path: Path, filenames: list[str]) -> dict[str, TensorInfo]:
    tensors: dict[str, TensorInfo] = {}
    for filename in filenames:
        if Path(filename).name != filename:
            raise ContractError(f"weight filename must not contain a path: {filename}")
        path = model_path / filename
        if not path.is_file():
            raise ContractError(f"missing weight shard: {filename}")
        for name, metadata in _read_safetensors_header(path).items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ContractError(f"duplicate tensor across weight shards: {name}")
            meta = _exact_keys(metadata, {"dtype", "shape", "data_offsets"}, set(), f"tensor {name}")
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
            tensors[name] = TensorInfo(str(meta["dtype"]), tuple(shape), filename)
    return tensors


def _installed_version(aliases: tuple[str, ...]) -> tuple[str, str] | None:
    for name in aliases:
        try:
            return name, version(name)
        except PackageNotFoundError:
            continue
    return None


def _cann_version() -> str | None:
    candidates: list[Path] = []
    toolkit_home = os.environ.get("ASCEND_TOOLKIT_HOME")
    if toolkit_home:
        candidates.append(Path(toolkit_home) / "aarch64-linux/ascend_toolkit_install.info")
    candidates.extend(sorted(Path("/usr/local/Ascend").glob("cann-*/aarch64-linux/ascend_toolkit_install.info")))
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
        {"schema_version", "artifact_id", "format", "model", "quantization", "runtime", "software", "files", "evidence"},
        set(),
        "contract",
    )
    if root["schema_version"] != SCHEMA_VERSION:
        raise ContractError(f"unsupported schema_version: {root['schema_version']!r}")
    if root["format"] != "modelslim-ascend-v1":
        raise ContractError(f"unsupported artifact format: {root['format']!r}")
    if not isinstance(root["artifact_id"], str) or not root["artifact_id"]:
        raise ContractError("artifact_id must be a non-empty string")

    model = _exact_keys(root["model"], {"model_type", "architectures", "supported_shapes"}, set(), "model")
    shapes = _exact_keys(model["supported_shapes"], {"weight_rank", "input_multiple", "output_multiple"}, set(), "model.supported_shapes")
    if shapes["weight_rank"] != 2:
        raise ContractError("only 2D weight tensors are supported")
    for key in ("input_multiple", "output_multiple"):
        if not isinstance(shapes[key], int) or shapes[key] < 1:
            raise ContractError(f"model.supported_shapes.{key} must be positive")
    if not isinstance(model["architectures"], list) or not model["architectures"] or not all(isinstance(x, str) for x in model["architectures"]):
        raise ContractError("model.architectures must be a non-empty string array")

    quant = _exact_keys(root["quantization"], {"scheme", "producer_quant_type", "runtime_quant_type", "weight", "activation", "scale", "zero_point"}, set(), "quantization")
    if quant["scheme"] not in {"W8A8", "W4A4", "W4A8"}:
        raise ContractError(f"unsupported quantization scheme: {quant['scheme']!r}")
    weight = _exact_keys(quant["weight"], {"bits", "storage_dtype", "signed", "packing", "layout", "granularity", "axis", "group_size"}, set(), "quantization.weight")
    activation = _exact_keys(quant["activation"], {"bits", "dtype", "granularity", "dynamic"}, set(), "quantization.activation")
    scale = _exact_keys(quant["scale"], {"dtype", "weight_shape", "activation_shape", "scale_bias"}, set(), "quantization.scale")
    zero = _exact_keys(quant["zero_point"], {"weight", "activation", "dtype", "semantics"}, set(), "quantization.zero_point")
    expected_bits = {"W8A8": (8, 8), "W4A4": (4, 4), "W4A8": (4, 8)}[quant["scheme"]]
    if (weight["bits"], activation["bits"]) != expected_bits:
        raise ContractError(f"{quant['scheme']} bit-width declaration is inconsistent")
    if weight["storage_dtype"] != "int8" or weight["signed"] is not True or weight["axis"] != 0:
        raise ContractError("only signed int8 storage on output axis 0 is supported")
    if weight["bits"] == 8 and (weight["packing"] != "none" or weight["layout"] != "logical_out_in"):
        raise ContractError("W8 weights must use unpacked logical_out_in layout")
    if weight["bits"] == 4 and weight["packing"] not in {"none", "signed_int4_nibble_low_high"}:
        raise ContractError("unsupported int4 packing")
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
    if scale["weight_shape"] not in {"out_1", "out_groups"} or scale["activation_shape"] not in {"one", "runtime_per_token"}:
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
    if zero["weight"] not in {"required", "forbidden"} or zero["activation"] not in {"required", "forbidden", "runtime"}:
        raise ContractError("unsupported zero-point presence semantics")
    if zero["dtype"] not in {"float32", "int32"}:
        raise ContractError("unsupported zero-point dtype")
    if zero["semantics"] != "additive_offset_before_scale":
        raise ContractError("unsupported zero-point semantics")
    allowed_types = {
        "W8A8": {"JXD_W8A8_PDMIX"},
        "W4A4": {"W4A4_DYNAMIC"},
        "W4A8": {"W4A8_DYNAMIC"},
    }
    if quant["runtime_quant_type"] not in allowed_types[quant["scheme"]]:
        raise ContractError("runtime quant type is not admitted for the declared scheme")

    runtime = _exact_keys(root["runtime"], {"host", "loader", "scheme_provider", "operators"}, set(), "runtime")
    if runtime["host"] != "vllm-ascend" or runtime["loader"] != "modelslim":
        raise ContractError("runtime host/loader is not supported")
    allowed_operators = {
        "torch_npu.npu_quant_matmul",
        "torch_npu.npu_dynamic_quant",
        "torch_npu.npu_convert_weight_to_int4pack",
    }
    if not isinstance(runtime["operators"], list) or not runtime["operators"] or not set(runtime["operators"]) <= allowed_operators:
        raise ContractError("runtime contains an unknown or empty operator list")

    software = _exact_keys(root["software"], {"cann", "torch_npu", "vllm", "vllm_ascend"}, set(), "software")
    if not all(isinstance(value, str) and value for value in software.values()):
        raise ContractError("all software compatibility ranges must be non-empty strings")
    files = _exact_keys(root["files"], {"config", "description", "description_sha256", "weights"}, set(), "files")
    if files["config"] != "config.json" or files["description"] != "quant_model_description.json":
        raise ContractError("non-standard config or description path")
    if not isinstance(files["weights"], list) or not files["weights"]:
        raise ContractError("files.weights must be a non-empty array")
    evidence = _exact_keys(root["evidence"], {"level", "verified_profiles", "results"}, set(), "evidence")
    if evidence["level"] not in {"schema_only", "correctness", "npu_e2e", "matched_benchmark"}:
        raise ContractError("unknown evidence level")
    if not isinstance(evidence["verified_profiles"], list) or not isinstance(evidence["results"], list):
        raise ContractError("invalid evidence arrays")
    return root


def _validate_tensor_contract(model_path: Path, contract: dict[str, Any], description: dict[str, Any]) -> dict[str, TensorInfo]:
    filenames = contract["files"]["weights"]
    actual_files = sorted(path.name for path in model_path.glob("*.safetensors"))
    if sorted(filenames) != actual_files:
        raise ContractError("declared weight shards do not exactly match artifact files")
    tensors = _collect_tensors(model_path, filenames)
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
    unknown_metadata = {name for name in description if "." not in name and name not in reserved_names}
    if unknown_metadata:
        raise ContractError(f"description contains unknown metadata fields: {sorted(unknown_metadata)}")
    if description.get("version") != "1.0.0":
        raise ContractError("description version must be 1.0.0")
    if description.get("model_quant_type") != runtime_type:
        raise ContractError("description model_quant_type does not match the contract")
    if not isinstance(description.get("metadata", {}), dict) or not isinstance(description.get("optional", {}), dict):
        raise ContractError("description metadata/optional must be objects")
    tensor_description = {name: value for name, value in description.items() if name not in reserved_names}
    unknown_types = {
        value for value in tensor_description.values()
        if isinstance(value, str) and value not in {runtime_type, "FLOAT"}
    }
    if unknown_types:
        raise ContractError(f"description contains unknown quantization types: {sorted(unknown_types)}")
    if any(value == producer_type for value in tensor_description.values()) and producer_type != runtime_type:
        raise ContractError(f"artifact still contains producer type {producer_type}; runtime artifact is incomplete")
    if any(not isinstance(value, str) for value in tensor_description.values()):
        raise ContractError("description tensor mappings must be strings")
    declared_quant_tensors = {name for name, value in tensor_description.items() if value == runtime_type}
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
        prefix = name[:-len("weight")]
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
            if offset_info is None or offset_info.dtype != expected_offset_dtype or offset_info.shape != (output_size, groups):
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
                raise ContractError(
                    f"{scale_bias_name} is missing or incompatible for W4A8"
                )
        elif scale_bias_name in tensors:
            raise ContractError(f"forbidden scale_bias is present for {name}")
    return tensors


def validate_artifact(model_path: Path, *, check_software: bool = True) -> dict[str, Any]:
    """Validate without importing torch, vLLM, torch-npu, or vLLM-Ascend."""

    model_path = model_path.resolve()
    if not model_path.is_dir():
        raise ContractError(f"model directory does not exist: {model_path}")
    contract = _validate_structure(_read_json(model_path / CONTRACT_FILENAME, "artifact contract"))
    config = _exact_keys(
        _read_json(model_path / contract["files"]["config"], "model config"),
        {"model_type"},
        set(_read_json(model_path / contract["files"]["config"], "model config").keys()) - {"model_type"},
        "config.json",
    )
    model = contract["model"]
    if config["model_type"] != model["model_type"]:
        raise ContractError("contract model_type does not match config.json")
    config_architectures = config.get("architectures", [])
    if config_architectures != model["architectures"]:
        raise ContractError("contract architectures do not match config.json")

    description_path = model_path / contract["files"]["description"]
    description = _read_json(description_path, "quantization description")
    if not isinstance(description, dict):
        raise ContractError("quantization description must be an object")
    if _sha256(description_path) != contract["files"]["description_sha256"]:
        raise ContractError("quantization description hash mismatch")
    tensors = _validate_tensor_contract(model_path, contract, description)

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
