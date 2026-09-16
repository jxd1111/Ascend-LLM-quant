"""Validation and provenance manifest support for quantized model artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from . import __version__
from .config import Recipe
from .evidence import load_evidence

MANIFEST_FILENAME = "ascend_quant_manifest.json"
RUNTIME_CONTRACT_FILENAME = "ascend_quant_artifact.json"


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    """Return immutable identity metadata for one artifact file."""

    return {
        "name": path.name,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _relative_file_record(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    return {
        "name": relative,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _artifact_content_digest(files: dict[str, Any]) -> str:
    """Hash the canonical file inventory used to derive ``artifact_id``."""

    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _quant_type_counts(description: Any) -> Counter[str]:
    counts: Counter[str] = Counter()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)
        elif isinstance(value, str) and value.startswith(("W", "INT", "FAK", "ASCEND_QUANT_")):
            counts[value] += 1

    visit(description)
    return counts


def inspect_artifact(model_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    config_path = model_path / "config.json"
    description_path = model_path / "quant_model_description.json"

    config: dict[str, Any] = {}
    description: Any = {}
    if not config_path.is_file():
        errors.append("missing config.json")
    else:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid config.json: {exc}")

    if not description_path.is_file():
        errors.append("missing quant_model_description.json")
    else:
        try:
            description = json.loads(description_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid quant_model_description.json: {exc}")

    weight_files = sorted(model_path.glob("*.safetensors"))
    if not weight_files:
        errors.append("no .safetensors weight files found")
    index_files = sorted(model_path.glob("*.safetensors.index.json"))
    if len(weight_files) > 1 and not index_files:
        warnings.append("multiple weight shards found without a safetensors index")

    quant_counts = _quant_type_counts(description)
    if description_path.is_file() and not quant_counts:
        warnings.append("no recognized quantization type found in model description")

    return {
        "valid": not errors,
        "path": str(model_path.resolve()),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures", []),
        "weight_files": [path.name for path in weight_files],
        "index_files": [path.name for path in index_files],
        "weight_bytes": sum(path.stat().st_size for path in weight_files),
        "quant_type_counts": dict(sorted(quant_counts.items())),
        "errors": errors,
        "warnings": warnings,
    }


def _replace_exact_strings(value: Any, source: str, target: str) -> tuple[Any, int]:
    if isinstance(value, dict):
        replaced: dict[Any, Any] = {}
        changes = 0
        for key, nested in value.items():
            replaced_value, nested_changes = _replace_exact_strings(nested, source, target)
            replaced[key] = replaced_value
            changes += nested_changes
        return replaced, changes
    if isinstance(value, list):
        replaced_list = []
        changes = 0
        for nested in value:
            replaced_value, nested_changes = _replace_exact_strings(nested, source, target)
            replaced_list.append(replaced_value)
            changes += nested_changes
        return replaced_list, changes
    if value == source:
        return target, 1
    return value, 0


def prepare_runtime_metadata(model_path: Path, recipe: Recipe) -> dict[str, Any]:
    """Atomically convert producer metadata to the plugin-owned runtime type.

    The original ModelSlim description is retained beside the active file so
    the operation is reversible. Re-running this function is safe.
    """

    description_path = model_path / "quant_model_description.json"
    backup_path = model_path / "quant_model_description.modelslim.json"
    if not description_path.is_file():
        raise ValueError(f"Missing quantization description: {description_path}")

    description = json.loads(description_path.read_text(encoding="utf-8"))
    current_counts = _quant_type_counts(description)
    source = recipe.producer_quant_type
    target = recipe.runtime_quant_type

    if current_counts[target] and not current_counts[source]:
        return {
            "changed": False,
            "replacements": 0,
            "backup": str(backup_path) if backup_path.exists() else None,
            "runtime_quant_type": target,
        }
    if current_counts[target] and current_counts[source]:
        raise ValueError(
            f"Artifact contains both {source} and {target}; refusing a partial migration"
        )
    if not current_counts[source]:
        raise ValueError(f"Artifact does not contain producer quantization type {source}")

    migrated, replacements = _replace_exact_strings(description, source, target)
    if replacements != current_counts[source]:
        raise ValueError(f"Expected {current_counts[source]} replacements, produced {replacements}")

    if not backup_path.exists():
        shutil.copy2(description_path, backup_path)

    serialized = json.dumps(migrated, ensure_ascii=False, indent=2) + "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{description_path.name}.",
        suffix=".tmp",
        dir=model_path,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, description_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    verified = json.loads(description_path.read_text(encoding="utf-8"))
    verified_counts = _quant_type_counts(verified)
    if verified_counts[source] or verified_counts[target] != replacements:
        raise ValueError("Runtime metadata verification failed after atomic replacement")

    return {
        "changed": True,
        "replacements": replacements,
        "backup": str(backup_path),
        "runtime_quant_type": target,
    }


def restore_modelslim_metadata(model_path: Path, recipe: Recipe) -> dict[str, Any]:
    """Restore the preserved ModelSlim description with an atomic replacement."""

    description_path = model_path / "quant_model_description.json"
    backup_path = model_path / "quant_model_description.modelslim.json"
    if not description_path.is_file():
        raise ValueError(f"Missing quantization description: {description_path}")
    if not backup_path.is_file():
        raise ValueError(f"ModelSlim metadata backup does not exist: {backup_path}")

    active = json.loads(description_path.read_text(encoding="utf-8"))
    backup_text = backup_path.read_text(encoding="utf-8")
    backup = json.loads(backup_text)
    active_counts = _quant_type_counts(active)
    backup_counts = _quant_type_counts(backup)
    if not backup_counts[recipe.producer_quant_type]:
        raise ValueError(f"Backup does not contain {recipe.producer_quant_type}: {backup_path}")
    if backup_counts[recipe.runtime_quant_type]:
        raise ValueError(f"Backup unexpectedly contains {recipe.runtime_quant_type}: {backup_path}")

    if active_counts[recipe.producer_quant_type] and not active_counts[recipe.runtime_quant_type]:
        return {
            "changed": False,
            "restored": False,
            "active_quant_type": recipe.producer_quant_type,
            "source": str(backup_path),
        }
    if active_counts[recipe.producer_quant_type] and active_counts[recipe.runtime_quant_type]:
        raise ValueError("Active metadata contains a partial quantization-type migration")
    if not active_counts[recipe.runtime_quant_type]:
        raise ValueError(
            f"Active metadata does not contain {recipe.runtime_quant_type}; "
            "refusing to overwrite it"
        )

    expected_active, replacements = _replace_exact_strings(
        backup, recipe.producer_quant_type, recipe.runtime_quant_type
    )
    if replacements != backup_counts[recipe.producer_quant_type] or active != expected_active:
        raise ValueError(
            "Active metadata differs from the preserved ModelSlim migration; "
            "refusing to overwrite later changes"
        )

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{description_path.name}.",
        suffix=".restore.tmp",
        dir=model_path,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            stream.write(backup_text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, description_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    return {
        "changed": True,
        "restored": True,
        "active_quant_type": recipe.producer_quant_type,
        "source": str(backup_path),
    }


def build_manifest(
    model_path: Path,
    recipe: Recipe,
    source_model: Path | None = None,
) -> dict[str, Any]:
    inspection = inspect_artifact(model_path)
    return {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "producer": {
            "name": "ascend-quant-toolkit",
            "version": __version__,
            "python": platform.python_version(),
            "modelslim": _package_version("msmodelslim"),
        },
        "recipe": {
            **recipe.as_dict(),
            "config_sha256": sha256_file(recipe.config_path()),
        },
        "source_model": str(source_model.resolve()) if source_model else None,
        "artifact": inspection,
    }


def write_manifest(
    model_path: Path,
    recipe: Recipe,
    source_model: Path | None = None,
) -> Path:
    destination = model_path / MANIFEST_FILENAME
    if source_model is None and destination.is_file():
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
            existing_source = existing.get("source_model")
            if existing_source:
                source_model = Path(existing_source)
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    manifest = build_manifest(model_path, recipe, source_model)
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def build_runtime_contract(
    model_path: Path,
    recipe: Recipe,
    *,
    evidence_level: str = "schema_only",
    verified_profiles: list[str] | None = None,
    evidence_results: list[str] | None = None,
) -> dict[str, Any]:
    """Build the immutable Toolkit-to-Runtime artifact contract.

    This function is part of offline artifact generation. Runtime code only
    reads the resulting file.
    """

    inspection = inspect_artifact(model_path)
    if not inspection["valid"]:
        raise ValueError("invalid artifact: " + "; ".join(inspection["errors"]))
    config_path = model_path / "config.json"
    description_path = model_path / "quant_model_description.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        raise ValueError("config.json must declare at least one architecture")

    profiles: dict[str, dict[str, Any]] = {
        "W8A8": {
            "weight": {
                "bits": 8,
                "storage_dtype": "int8",
                "signed": True,
                "packing": "none",
                "layout": "logical_out_in",
                "granularity": "per_channel",
                "axis": 0,
                "group_size": None,
            },
            "activation": {
                "bits": 8,
                "dtype": "int8",
                "granularity": "pd_mix",
                "dynamic": True,
            },
            "scale": {
                "dtype": "float32",
                "weight_shape": "out_1",
                "activation_shape": "one",
                "scale_bias": "forbidden",
            },
            "zero_point": {
                "weight": "required",
                "activation": "required",
                "dtype": "float32",
                "semantics": "additive_offset_before_scale",
            },
            "operators": [
                "torch_npu.npu_dynamic_quant",
                "torch_npu.npu_quant_matmul",
            ],
        },
        "W4A4": {
            "weight": {
                "bits": 4,
                "storage_dtype": "int8",
                "signed": True,
                "packing": "signed_int4_nibble_low_high",
                "layout": "packed_out_in",
                "granularity": "per_channel",
                "axis": 0,
                "group_size": None,
            },
            "activation": {
                "bits": 4,
                "dtype": "int4",
                "granularity": "per_token",
                "dynamic": True,
            },
            "scale": {
                "dtype": "float32",
                "weight_shape": "out_1",
                "activation_shape": "runtime_per_token",
                "scale_bias": "forbidden",
            },
            "zero_point": {
                "weight": "required",
                "activation": "runtime",
                "dtype": "float32",
                "semantics": "additive_offset_before_scale",
            },
            "operators": [
                "torch_npu.npu_dynamic_quant",
                "torch_npu.npu_convert_weight_to_int4pack",
                "torch_npu.npu_quant_matmul",
            ],
        },
        "W4A8": {
            "weight": {
                "bits": 4,
                "storage_dtype": "int8",
                "signed": True,
                "packing": "signed_int4_nibble_low_high",
                "layout": "packed_out_in",
                "granularity": "per_channel",
                "axis": 0,
                "group_size": None,
            },
            "activation": {"bits": 8, "dtype": "int8", "granularity": "per_token", "dynamic": True},
            "scale": {
                "dtype": "float32",
                "weight_shape": "out_1",
                "activation_shape": "runtime_per_token",
                "scale_bias": "required_out_1_or_16",
            },
            "zero_point": {
                "weight": "required",
                "activation": "runtime",
                "dtype": "float32",
                "semantics": "additive_offset_before_scale",
            },
            "operators": [
                "torch_npu.npu_dynamic_quant",
                "torch_npu.npu_convert_weight_to_int4pack",
                "torch_npu.npu_quant_matmul",
            ],
        },
    }
    if recipe.quant_scheme not in profiles:
        raise ValueError(f"unsupported contract profile: {recipe.quant_scheme}")
    profile = profiles[recipe.quant_scheme]
    file_contract = {
        "config": _file_record(config_path),
        "description": _file_record(description_path),
        "indexes": [_file_record(model_path / filename) for filename in inspection["index_files"]],
        "weights": [_file_record(model_path / filename) for filename in inspection["weight_files"]],
    }
    content_digest = _artifact_content_digest(file_contract)
    artifact_id = f"{config['model_type']}:{recipe.quant_scheme}:{content_digest[:16]}"
    profiles_evidence = verified_profiles or []
    results_evidence = evidence_results or []
    if evidence_level == "schema_only":
        if profiles_evidence or results_evidence:
            raise ValueError("schema_only evidence must not declare profiles or results")
    elif recipe.quant_scheme not in profiles_evidence or not results_evidence:
        raise ValueError(
            f"{evidence_level} evidence requires the {recipe.quant_scheme} profile "
            "and at least one result"
        )
    result_records = []
    result_profiles: set[str] = set()
    for result in results_evidence:
        result_path = Path(result)
        if result_path.is_absolute() or ".." in result_path.parts or not result_path.parts:
            raise ValueError(f"evidence result must be a safe relative path: {result}")
        if not (model_path / result_path).is_file():
            raise ValueError(f"evidence result does not exist: {result}")
        evidence = load_evidence(model_path / result_path)
        if evidence["profile"] not in profiles_evidence:
            raise ValueError(
                f"evidence result profile {evidence['profile']} is not verified: {result}"
            )
        if evidence["profile"] != "BF16":
            if evidence["artifact"]["artifact_id"] != artifact_id:
                raise ValueError(f"evidence result artifact_id does not match: {result}")
            if evidence["artifact"]["artifact_files_sha256"] != content_digest:
                raise ValueError(f"evidence result artifact file digest does not match: {result}")
        result_profiles.add(evidence["profile"])
        result_records.append(_relative_file_record(model_path, result))
    if evidence_level != "schema_only" and set(profiles_evidence) != result_profiles:
        raise ValueError("every verified profile must have a matching evidence result")
    if evidence_level == "matched_benchmark" and not {
        "BF16",
        recipe.quant_scheme,
    } <= result_profiles:
        raise ValueError("matched_benchmark requires BF16 and quantized evidence records")
    return {
        "schema_version": "1.1.0",
        "artifact_id": artifact_id,
        "format": "modelslim-ascend-v1",
        "model": {
            "model_type": config["model_type"],
            "architectures": architectures,
            "supported_shapes": {
                "weight_rank": 2,
                "input_multiple": 16,
                "output_multiple": 16,
            },
        },
        "quantization": {
            "scheme": recipe.quant_scheme,
            "producer_quant_type": recipe.producer_quant_type,
            "runtime_quant_type": recipe.runtime_quant_type,
            "weight": profile["weight"],
            "activation": profile["activation"],
            "scale": profile["scale"],
            "zero_point": profile["zero_point"],
        },
        "runtime": {
            "host": "vllm-ascend",
            "loader": "modelslim",
            "scheme_provider": "vllm-ascend-quant-ext",
            "operators": profile["operators"],
        },
        "software": {
            "cann": ">=9.1,<9.2",
            "torch_npu": ">=2.13.0rc1,<2.14",
            "vllm": ">=0.28.1rc0,<0.29",
            "vllm_ascend": ">=0.25.1rc1,<0.29",
        },
        "files": file_contract,
        "evidence": {
            "level": evidence_level,
            "verified_profiles": profiles_evidence,
            "results": result_records,
        },
    }


def write_runtime_contract(
    model_path: Path,
    recipe: Recipe,
    *,
    evidence_level: str = "schema_only",
    verified_profiles: list[str] | None = None,
    evidence_results: list[str] | None = None,
) -> Path:
    contract = build_runtime_contract(
        model_path,
        recipe,
        evidence_level=evidence_level,
        verified_profiles=verified_profiles,
        evidence_results=evidence_results,
    )
    destination = model_path / RUNTIME_CONTRACT_FILENAME
    destination.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destination
