"""Pure-data Extension Manager boundary; no implementation imports."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from .contract import validate_artifact
from .plugin import ARTIFACT_ENV, ENABLE_ENV

MANAGER_ADAPTER_API_VERSION = "1.0"


def manifest_path() -> Path:
    return Path(str(files("vllm_ascend_quant_ext").joinpath("extension-manifest.json")))


def load_manifest() -> dict[str, Any]:
    return validate_manifest(json.loads(manifest_path().read_text(encoding="utf-8")))


def validate_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("extension manifest must be an object")
    required = {
        "schema_version",
        "extension_id",
        "extension_version",
        "kind",
        "host",
        "runtime",
        "lifecycle_owner",
        "protocols",
        "implementation",
        "permissions",
        "requires_services",
        "compatibility",
    }
    missing = required - set(value)
    unknown = set(value) - required
    if missing or unknown:
        raise ValueError(
            f"closed extension manifest mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if value["schema_version"] != "0.2-experimental":
        raise ValueError("unsupported extension manifest schema")
    if value["extension_id"] != "vllm-ascend-quant":
        raise ValueError("unexpected extension identity")
    if value["kind"] != "model_weight_quantization_runtime":
        raise ValueError("unexpected extension kind")
    if value["host"].get("provider") != "vllm-ascend":
        raise ValueError("unexpected extension host")
    if value["runtime"].get("isolation") != "trusted_in_process":
        raise ValueError("unsupported isolation claim")
    if value["permissions"] or value["requires_services"]:
        raise ValueError("this extension does not admit permissions or external services")
    if value["compatibility"].get("kv_cache_quantization") is not False:
        raise ValueError("model-weight and KV-cache quantization must remain separate")
    return value


def check(model_path: Path) -> dict[str, Any]:
    return {"extension": load_manifest(), "artifact": validate_artifact(model_path)}


def plan(model_path: Path) -> dict[str, Any]:
    report = validate_artifact(model_path)
    return {
        "extension_id": "vllm-ascend-quant",
        "action": "enable_for_next_vllm_start",
        "lifecycle_owner": "vllm-ascend",
        "mutates_model": False,
        "implementation_imported": False,
        "artifact": report,
    }


def render(model_path: Path) -> dict[str, Any]:
    report = validate_artifact(model_path)
    return {
        "extension_id": "vllm-ascend-quant",
        "environment": {
            ENABLE_ENV: "1",
            ARTIFACT_ENV: str(model_path.resolve()),
        },
        "vllm_plugins_add": ["vllm_ascend_quant"],
        "vllm_arguments": [],
        "artifact": report,
    }


def validate_manager_config(value: Any) -> dict[str, Any]:
    """Validate the proposed Manager-facing configuration without side effects.

    This is an extension-owned adapter contract.  The Extension Manager remains
    responsible for mapping its final, versioned configuration API to it.
    """

    if not isinstance(value, dict):
        raise ValueError("manager configuration must be an object")
    required = {"enabled", "model"}
    missing = required - set(value)
    unknown = set(value) - required
    if missing or unknown:
        raise ValueError(
            f"closed manager configuration mismatch: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if not isinstance(value["enabled"], bool):
        raise ValueError("manager configuration enabled must be Boolean")
    if not isinstance(value["model"], str) or not value["model"]:
        raise ValueError("manager configuration model must be a non-empty path")
    return value


class ManagerAdapter:
    """Pure-data hand-off boundary for a future typed Manager materializer."""

    api_version = MANAGER_ADAPTER_API_VERSION
    extension_id = "vllm-ascend-quant"
    kind = "model_weight_quantization_runtime"
    host = "vllm-ascend"

    def descriptor(self) -> dict[str, Any]:
        return {
            "api_version": self.api_version,
            "extension_id": self.extension_id,
            "kind": self.kind,
            "host": self.host,
            "manifest": load_manifest(),
            "implementation_imported": False,
        }

    def check(self, configuration: Any) -> dict[str, Any]:
        config = validate_manager_config(configuration)
        if not config["enabled"]:
            return {
                "extension_id": self.extension_id,
                "enabled": False,
                "admitted": True,
                "artifact_checked": False,
                "implementation_imported": False,
            }
        report = validate_artifact(Path(config["model"]))
        return {
            "extension_id": self.extension_id,
            "enabled": True,
            "admitted": True,
            "artifact_checked": True,
            "implementation_imported": False,
            "artifact": report,
        }

    def plan(self, configuration: Any) -> dict[str, Any]:
        config = validate_manager_config(configuration)
        if not config["enabled"]:
            return {
                "extension_id": self.extension_id,
                "action": "disable_for_next_vllm_start",
                "lifecycle_owner": "vllm-ascend",
                "mutates_model": False,
                "implementation_imported": False,
            }
        return plan(Path(config["model"]))

    def render(self, configuration: Any) -> dict[str, Any]:
        config = validate_manager_config(configuration)
        if not config["enabled"]:
            return {
                "extension_id": self.extension_id,
                "environment_set": {},
                "environment_unset": [ENABLE_ENV, ARTIFACT_ENV],
                "vllm_plugins_add": [],
                "vllm_plugins_remove": ["vllm_ascend_quant"],
                "vllm_arguments": [],
            }
        rendered = render(Path(config["model"]))
        return {
            "extension_id": self.extension_id,
            "environment_set": rendered["environment"],
            "environment_unset": [],
            "vllm_plugins_add": rendered["vllm_plugins_add"],
            "vllm_plugins_remove": [],
            "vllm_arguments": rendered["vllm_arguments"],
            "artifact": rendered["artifact"],
        }


# Stable symbol for the framework team's thin host-provider adapter.  It is not
# advertised as an official Manager entry point until that API is finalized.
provider = ManagerAdapter()
