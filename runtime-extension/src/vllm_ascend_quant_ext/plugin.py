"""Default-off vLLM entry point with pre-import artifact admission."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .attachment import AttachmentError, validate_attachment
from .contract import ContractError, validate_artifact
from .host_baseline import host_description

LOGGER = logging.getLogger(__name__)
ENABLE_ENV = "VLLM_ASCEND_QUANT_EXT_ENABLE"
ARTIFACT_ENV = "VLLM_ASCEND_QUANT_EXT_ARTIFACT"
SMOOTHING_ENV = "VLLM_ASCEND_QUANT_EXT_SMOOTHING"

_NORM_HOOK_INSTALLED = False


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


def register_smoothing(attachment: str | Path) -> list[str]:
    """Admit a smoothing attachment and register the online W8A8 scheme.

    The attachment is the only extra input of the online path.  Admission stays
    dependency free (no torch, no vLLM, no device) and fails closed.
    """

    attachment_path = Path(attachment).resolve()
    try:
        admitted = validate_attachment(attachment_path)
    except AttachmentError as exc:
        raise RuntimeError(f"Ascend quant smoothing admission failed: {exc}") from exc

    from .schemes.online_w8a8 import register_schemes

    registered = register_schemes(admitted)
    _install_norm_hook()
    LOGGER.info(
        "Ascend quant online runtime admitted %s (attachment_id=%s); registered=%s",
        attachment_path,
        admitted.attachment_id,
        registered,
    )
    return registered


def apply_norm_scales(model: object) -> int:
    """Multiply every admitted ``norm:<module>`` factor into that norm weight.

    A linear method never sees a norm module, so the activation side of a
    norm-linear subgraph is completed here: the artifact stores the norm weight
    already divided by the smoothing factor, and this rewrites the loaded BF16
    weight the same way so the replayed linear receives ``x / s``.
    """

    from .schemes.online_w8a8 import active_attachment

    attachment = active_attachment()
    if attachment is None:
        return 0
    modules = attachment.norm_modules()
    if not modules:
        return 0
    if not hasattr(model, "get_submodule"):
        raise RuntimeError("the loaded model cannot resolve admitted norm modules")

    unresolved: list[str] = []
    applied = 0
    for name in modules:
        try:
            module = model.get_submodule(name)
        except AttributeError:
            unresolved.append(name)
            continue
        weight = getattr(module, "weight", None)
        factor = attachment.open_tensor(f"norm:{name}")
        if weight is None or tuple(weight.shape) != tuple(factor.shape):
            raise RuntimeError(f"norm factor does not match the model module {name}")
        weight.data.mul_(factor.to(weight.dtype).to(weight.device))
        applied += 1
    if unresolved:
        raise RuntimeError(f"admitted norm modules are absent from the model: {unresolved[:5]}")
    return applied


def _install_norm_hook() -> None:
    """Apply the norm factors after every ``load_model`` call.

    ``load_model`` is defined on ``BaseModelLoader`` and not on
    ``DefaultModelLoader``, so patching the base class covers every load format.
    The hook runs once per load, after ``process_weights_after_loading`` has
    quantized the linears; norm weights stay floating point in both paths, so
    the two rewrites never overlap.
    """

    global _NORM_HOOK_INSTALLED
    if _NORM_HOOK_INSTALLED:
        return
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader

    original = BaseModelLoader.load_model

    def load_model(self, *args, **kwargs):
        model = original(self, *args, **kwargs)
        applied = apply_norm_scales(model)
        LOGGER.info("Ascend quant online norm factors applied to %d modules", applied)
        return model

    BaseModelLoader.load_model = load_model
    _NORM_HOOK_INSTALLED = True


def register() -> None:
    """vLLM ``vllm.general_plugins`` entry point.

    vLLM can import this function in every runtime process.  Keep it
    idempotent and default-off: installation and discovery alone must not
    register a scheme or touch a device.
    """

    if not is_enabled():
        return
    artifact = os.environ.get(ARTIFACT_ENV)
    smoothing = os.environ.get(SMOOTHING_ENV)
    if artifact and smoothing:
        raise RuntimeError(f"set either {ARTIFACT_ENV} or {SMOOTHING_ENV}, not both")
    if smoothing:
        register_smoothing(smoothing)
        return
    if not artifact:
        raise RuntimeError(f"{ARTIFACT_ENV} or {SMOOTHING_ENV} is required when {ENABLE_ENV}=1")
    register_artifact(artifact)


def status() -> dict[str, object]:
    return {
        "enabled": is_enabled(),
        "enable_environment_variable": ENABLE_ENV,
        "artifact_environment_variable": ARTIFACT_ENV,
        "smoothing_environment_variable": SMOOTHING_ENV,
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
