"""Experimental direct-diagnostic carrier; not advertised by the Bundle.

The module intentionally avoids importing torch, vLLM, or vLLM-Ascend at
import time.  The host calls ``from_vllm_config`` inside the model worker; only
``activate`` imports the scheme implementation after artifact admission.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...contract import validate_artifact

BUNDLE_ID = "org.vllm-hust.ascend-quant-runtime"
COMPONENT_ID = "w8a8-runtime-diagnostic"
FULL_COMPONENT_ID = f"{BUNDLE_ID}/{COMPONENT_ID}"
CONTRACT = "unversioned-direct-diagnostic"


def _model_path_from_config(vllm_config: Any) -> Path:
    model_config = getattr(vllm_config, "model_config", None)
    model = getattr(model_config, "model", None)
    if not isinstance(model, str) or not model:
        raise ValueError("vLLM config does not provide model_config.model")
    return Path(model).resolve()


class AscendQuantRuntimeComponent:
    """W8A8 carrier used only by explicit, non-Manager diagnostics."""

    vllm_ascend_quantization_scheme_api_version = 1
    component_id = FULL_COMPONENT_ID
    contract = CONTRACT

    def __init__(self, model_path: Path):
        self.model_path = model_path.resolve()
        self._report = validate_artifact(self.model_path)
        self._activated = False
        self._registered: list[str] = []

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> "AscendQuantRuntimeComponent":
        return cls(_model_path_from_config(vllm_config))

    def admission_report(self) -> dict[str, Any]:
        return dict(self._report)

    def activate(self) -> list[str]:
        """Re-admit the artifact and register its namespaced runtime schemes."""

        from ...plugin import register_artifact

        self._registered = register_artifact(self.model_path)
        self._activated = True
        return list(self._registered)

    def export_status(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "contract": self.contract,
            "artifact_id": self._report["artifact_id"],
            "runtime_quant_type": self._report["runtime_quant_type"],
            "activated": self._activated,
            "registered": list(self._registered),
        }
