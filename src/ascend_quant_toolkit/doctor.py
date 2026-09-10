"""Read-only environment checks for quantization and serving integration."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _version_aliases(*names: str) -> str | None:
    for name in names:
        found = _version(name)
        if found is not None:
            return found
    return None


def collect_checks(model_path: Path | None = None) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "python": sys.version.split()[0],
        "msmodelslim_executable": shutil.which("msmodelslim"),
        "msmodelslim_version": _version("msmodelslim"),
        "torch_version": _version("torch"),
        "torch_npu_version": _version("torch-npu"),
        "vllm_version": _version_aliases("vllm-hust", "vllm"),
        "vllm_ascend_version": _version_aliases("vllm-ascend-hust", "vllm-ascend"),
        "modules": {
            name: importlib.util.find_spec(name) is not None
            for name in ("msmodelslim", "torch", "torch_npu", "vllm", "vllm_ascend")
        },
    }
    target = model_path or Path.cwd()
    usage = shutil.disk_usage(target if target.exists() else target.parent)
    checks["storage"] = {
        "path": str(target),
        "free_bytes": usage.free,
        "total_bytes": usage.total,
    }
    checks["ready_for_modelslim"] = bool(
        checks["msmodelslim_executable"]
        and checks["modules"]["msmodelslim"]
        and checks["modules"]["torch"]
        and checks["modules"]["torch_npu"]
    )
    return checks
