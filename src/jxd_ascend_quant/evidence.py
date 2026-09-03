"""Closed validation for matched BF16/quantized evaluation evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROFILES = {"BF16", "W8A8", "W4A4", "W4A8"}


def _closed(value: Any, required: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    missing = required - set(value)
    unknown = set(value) - required
    if missing or unknown:
        raise ValueError(f"{where} mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}")
    return value


def validate_evidence(value: Any) -> dict[str, Any]:
    root = _closed(
        value,
        {"schema_version", "run_id", "profile", "artifact", "software", "hardware", "workload", "results", "raw_logs"},
        "evidence",
    )
    if root["schema_version"] != "1.0.0" or root["profile"] not in PROFILES:
        raise ValueError("unsupported evidence schema/profile")
    _closed(root["artifact"], {"model", "artifact_id", "contract_sha256"}, "artifact")
    _closed(root["software"], {"cann", "torch_npu", "vllm", "vllm_ascend", "extension"}, "software")
    _closed(root["hardware"], {"npu_model", "npu_count", "tensor_parallel", "device_ids"}, "hardware")
    _closed(root["workload"], {"dataset", "dataset_revision", "dataset_sha256", "prompts", "request_rate", "max_concurrency", "input_tokens", "output_tokens", "warmup_runs", "recorded_runs"}, "workload")
    results = _closed(root["results"], {"successful", "failed", "correctness", "quality", "throughput", "latency", "hbm"}, "results")
    _closed(results["correctness"], {"passed", "metric", "value", "tolerance"}, "results.correctness")
    _closed(results["quality"], {"metric", "value", "bf16_delta"}, "results.quality")
    _closed(results["throughput"], {"requests_per_second", "output_tokens_per_second", "total_tokens_per_second"}, "results.throughput")
    _closed(results["latency"], {"mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms"}, "results.latency")
    _closed(results["hbm"], {"idle_mib", "loaded_mib", "peak_mib"}, "results.hbm")
    if not isinstance(root["raw_logs"], list) or not root["raw_logs"]:
        raise ValueError("raw_logs must contain at least one path")
    if not isinstance(root["workload"]["prompts"], int) or root["workload"]["prompts"] < 1:
        raise ValueError("workload.prompts must be positive")
    if not isinstance(results["correctness"]["passed"], bool):
        raise ValueError("results.correctness.passed must be Boolean")
    return root


def load_evidence(path: Path) -> dict[str, Any]:
    return validate_evidence(json.loads(path.read_text(encoding="utf-8")))
