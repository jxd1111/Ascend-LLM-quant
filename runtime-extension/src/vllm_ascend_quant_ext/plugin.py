"""Default-off vLLM entry point with pre-import artifact admission."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .contract import ContractError, validate_artifact
from .host_baseline import host_description

LOGGER = logging.getLogger(__name__)
ENABLE_ENV = "VLLM_ASCEND_QUANT_EXT_ENABLE"
ARTIFACT_ENV = "VLLM_ASCEND_QUANT_EXT_ARTIFACT"


def is_enabled() -> bool:
    return os.environ.get(ENABLE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def register_artifact(artifact: str | Path) -> list[str]:
    artifact_path = Path(artifact).resolve()
    # Admission deliberately happens before importing torch/vLLM/vLLM-Ascend.
    try:
        report = validate_artifact(artifact_path)
    except ContractError as exc:
        raise RuntimeError(f"Ascend quant artifact admission failed: {exc}") from exc

    runtime_type = report["runtime_quant_type"]
    if runtime_type == "ASCEND_QUANT_W8A8":
        from .schemes.w8a8 import register_schemes

        registered = register_schemes()
        LOGGER.info("Ascend quant runtime admitted %s; registered=%s", artifact_path, registered)
        return registered

    # W4 profiles use backend-owned typed schemes. They still require strict
    # artifact admission and an explicit scheme-presence check.
    from vllm_ascend.quantization.methods import get_scheme_class

    if get_scheme_class(runtime_type, "linear") is None:
        raise RuntimeError(f"vLLM-Ascend does not provide required scheme {runtime_type}/linear")
    LOGGER.info("Ascend quant runtime admitted %s using backend scheme %s", artifact_path, runtime_type)
    return []


def register() -> None:
    """vLLM ``vllm.general_plugins`` entry point.

    vLLM can import this function in every runtime process.  Keep it
    idempotent and default-off: installation and discovery alone must not
    register a scheme or touch a device.
    """

    if not is_enabled():
        return
    artifact = os.environ.get(ARTIFACT_ENV)
    if not artifact:
        raise RuntimeError(f"{ARTIFACT_ENV} is required when {ENABLE_ENV}=1")
    register_artifact(artifact)


def status() -> dict[str, object]:
    return {
        "enabled": is_enabled(),
        "enable_environment_variable": ENABLE_ENV,
        "artifact_environment_variable": ARTIFACT_ENV,
        "entry_point": "vllm.general_plugins/vllm_ascend_quant",
        "host": host_description(),
    }


def plan(artifact: str | Path) -> dict[str, object]:
    """Describe the next-process activation without mutating the artifact."""

    artifact_path = Path(artifact).resolve()
    report = validate_artifact(artifact_path)
    return {
        "action": "enable_for_next_vllm_start",
        "lifecycle_owner": "vllm",
        "mutates_model": False,
        "entry_point": "vllm.general_plugins/vllm_ascend_quant",
        "artifact": report,
    }


def render(artifact: str | Path) -> dict[str, object]:
    """Render the environment consumed by the native vLLM plugin loader."""

    artifact_path = Path(artifact).resolve()
    report = validate_artifact(artifact_path)
    return {
        "environment": {
            ENABLE_ENV: "1",
            ARTIFACT_ENV: str(artifact_path),
        },
        "vllm_arguments": [],
        "note": (
            "Do not narrow VLLM_PLUGINS unless every required frozen-host "
            "platform and general plugin is included."
        ),
        "artifact": report,
    }
