"""Default-off vLLM entry point with pre-import artifact admission."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from .contract import ContractError, validate_artifact

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
    if runtime_type == "JXD_W8A8_PDMIX":
        from .schemes.w8a8_pdmix import register_schemes

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
    """Legacy vLLM runtime entry point retained for explicit activation.

    Bundle discovery uses ``vllm_hust.extension_bundles``. This entry point is
    a direct diagnostic bridge and is deliberately not selected by the Bundle
    activation record while the typed host component seam is being finalized.
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
        "bundle_entry_point": "vllm_hust.extension_bundles/org.vllm-hust.ascend-quant",
    }
