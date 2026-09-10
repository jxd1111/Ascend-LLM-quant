"""Pure-data Extension Manager boundary; no implementation imports."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from .contract import validate_artifact
from .plugin import ARTIFACT_ENV, ENABLE_ENV

MANAGER_ADAPTER_API_VERSION = "1.0"
BUNDLE_ID = "org.vllm-hust.ascend-quant-runtime"
COMPONENT_NAME = "ascend-quant-artifact-validator"
COMPONENT_ID = f"{BUNDLE_ID}/{COMPONENT_NAME}"
IMPORT_ONLY_ERROR = (
    "extension is import_only: activation requires owner-approved "
    "vllm.ascend.quantized-artifact-loader and "
    "vllm.ascend.quantized-operator-selection host protocols"
)


def manifest_path() -> Path:
    return Path(
        str(
            files("vllm_ascend_quant_ext.manifests").joinpath(
                "vllm-hust-extension-v0.2.json"
            )
        )
    )


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
        "requires_services",
        "components",
        "activation",
    }
    missing = required - set(value)
    unknown = set(value) - required
    if missing or unknown:
        raise ValueError(
            f"closed extension manifest mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if value["schema_version"] != "0.2-experimental":
        raise ValueError("unsupported extension manifest schema")
    if value["extension_id"] != BUNDLE_ID:
        raise ValueError("unexpected extension identity")
    if value["kind"] != "in_process_plugin":
        raise ValueError("unexpected extension kind")
    if value["host"] != {
        "provider": "vllm",
        "name": "vllm-ascend",
        "version_range": ">=0",
    }:
        raise ValueError("unexpected extension host")
    if value["lifecycle_owner"] != "vllm":
        raise ValueError("unexpected lifecycle owner")
    if value["runtime"] != {
        "type": "python",
        "process_scope": "vllm-ascend-worker",
        "isolation": "trusted_in_process",
    }:
        raise ValueError("unsupported runtime declaration")
    if value["protocols"] != [
        {
            "name": "vllm.ascend.quantized-artifact-loader",
            "version_range": ">=1,<2",
        },
        {
            "name": "vllm.ascend.quantized-operator-selection",
            "version_range": ">=1,<2",
        },
    ]:
        raise ValueError("unexpected host protocol declaration")
    if value["requires_services"]:
        raise ValueError("this extension does not admit external services")
    implementation = value["implementation"]
    if not isinstance(implementation, list) or len(implementation) != 1:
        raise ValueError("the Bundle must declare one implementation carrier")
    carrier = implementation[0]
    if carrier != {
        "type": "python_module",
        "module": "vllm_ascend_quant_ext.contract",
        "object": "ArtifactContractValidator",
        "status": "import_only",
    }:
        raise ValueError("unexpected import-only implementation carrier")
    components = value["components"]
    if not isinstance(components, list) or len(components) != 1:
        raise ValueError("the Bundle must declare one typed component")
    component = components[0]
    if component.get("component_id") != COMPONENT_NAME:
        raise ValueError("unexpected component identity")
    if component.get("contracts") != [
        "vllm.ascend.quantized-artifact-loader.v1"
    ]:
        raise ValueError("unexpected component contract")
    if component.get("execution_planes") != ["worker", "device"]:
        raise ValueError("unexpected component execution plane")
    if component.get("implementation_ref") != (
        "vllm_ascend_quant_ext.contract:ArtifactContractValidator"
    ):
        raise ValueError("unexpected component implementation reference")
    if component.get("permissions") != ["filesystem_read", "device_access"]:
        raise ValueError("unexpected component permissions")
    activation = value["activation"]
    if activation != {"entry_points": [], "environment": {}, "additional_config": {}}:
        raise ValueError("typed component activation must not inject legacy runtime state")
    return value


def check(model_path: Path) -> dict[str, Any]:
    return {"extension": load_manifest(), "artifact": validate_artifact(model_path)}


def plan(model_path: Path) -> dict[str, Any]:
    """Render the explicit direct-diagnostic path, not Manager activation."""

    report = validate_artifact(model_path)
    return {
        "extension_id": BUNDLE_ID,
        "action": "enable_for_next_vllm_start",
        "lifecycle_owner": "vllm",
        "mutates_model": False,
        "implementation_imported": False,
        "artifact": report,
    }


def render(model_path: Path) -> dict[str, Any]:
    report = validate_artifact(model_path)
    return {
        "extension_id": BUNDLE_ID,
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
    """Import-only hand-off boundary for the experimental Manager schema."""

    api_version = MANAGER_ADAPTER_API_VERSION
    extension_id = BUNDLE_ID
    kind = "in_process_plugin"
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
            "admitted": False,
            "enable_allowed": False,
            "reason": IMPORT_ONLY_ERROR,
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
                "lifecycle_owner": "vllm",
                "mutates_model": False,
                "implementation_imported": False,
            }
        validate_artifact(Path(config["model"]))
        raise RuntimeError(IMPORT_ONLY_ERROR)

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
        validate_artifact(Path(config["model"]))
        raise RuntimeError(IMPORT_ONLY_ERROR)


# Stable symbol for the framework team's thin host-provider adapter.  It is not
# advertised as an official Manager entry point until that API is finalized.
provider = ManagerAdapter()
