"""Closed validation for matched BF16/quantized evaluation evidence."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

PROFILES = {"BF16", "W8A8", "W4A4", "W4A8"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _closed(value: Any, required: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    missing = required - set(value)
    unknown = set(value) - required
    if missing or unknown:
        raise ValueError(f"{where} mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}")
    return value


def _non_empty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty string")
    return value


def _number(value: Any, where: str, *, minimum: float = 0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{where} must be finite and >= {minimum}")
    return result


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{where} must be an integer >= {minimum}")
    return value


def validate_evidence(value: Any) -> dict[str, Any]:
    root = _closed(
        value,
        {
            "schema_version",
            "run_id",
            "profile",
            "artifact",
            "software",
            "hardware",
            "workload",
            "results",
            "raw_logs",
        },
        "evidence",
    )
    if root["schema_version"] != "1.1.0" or root["profile"] not in PROFILES:
        raise ValueError("unsupported evidence schema/profile")
    _non_empty_string(root["run_id"], "run_id")
    artifact = _closed(
        root["artifact"],
        {"model", "artifact_id", "artifact_files_sha256"},
        "artifact",
    )
    software = _closed(
        root["software"], {"cann", "torch_npu", "vllm", "vllm_ascend", "extension"}, "software"
    )
    hardware = _closed(
        root["hardware"], {"npu_model", "npu_count", "tensor_parallel", "device_ids"}, "hardware"
    )
    workload = _closed(
        root["workload"],
        {
            "dataset",
            "dataset_revision",
            "dataset_sha256",
            "prompts",
            "request_rate",
            "max_concurrency",
            "input_tokens",
            "output_tokens",
            "warmup_runs",
            "recorded_runs",
        },
        "workload",
    )
    results = _closed(
        root["results"],
        {"successful", "failed", "correctness", "quality", "throughput", "latency", "hbm"},
        "results",
    )
    correctness = _closed(
        results["correctness"], {"passed", "metric", "value", "tolerance"}, "results.correctness"
    )
    quality = _closed(results["quality"], {"metric", "value", "bf16_delta"}, "results.quality")
    throughput = _closed(
        results["throughput"],
        {"requests_per_second", "output_tokens_per_second", "total_tokens_per_second"},
        "results.throughput",
    )
    latency = _closed(
        results["latency"],
        {"mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms"},
        "results.latency",
    )
    hbm = _closed(results["hbm"], {"idle_mib", "loaded_mib", "peak_mib"}, "results.hbm")

    for key in ("model", "artifact_id"):
        _non_empty_string(artifact[key], f"artifact.{key}")
    if not isinstance(artifact["artifact_files_sha256"], str) or not SHA256_PATTERN.fullmatch(
        artifact["artifact_files_sha256"]
    ):
        raise ValueError(
            "artifact.artifact_files_sha256 must be 64 lowercase hexadecimal characters"
        )
    for key, value in software.items():
        _non_empty_string(value, f"software.{key}")
    _non_empty_string(hardware["npu_model"], "hardware.npu_model")
    npu_count = _integer(hardware["npu_count"], "hardware.npu_count", minimum=1)
    tensor_parallel = _integer(hardware["tensor_parallel"], "hardware.tensor_parallel", minimum=1)
    if tensor_parallel > npu_count:
        raise ValueError("hardware.tensor_parallel must not exceed npu_count")
    if (
        not isinstance(hardware["device_ids"], list)
        or len(hardware["device_ids"]) != npu_count
        or len(set(hardware["device_ids"])) != npu_count
        or any(
            isinstance(device, bool) or not isinstance(device, int) or device < 0
            for device in hardware["device_ids"]
        )
    ):
        raise ValueError("hardware.device_ids must contain one unique non-negative ID per NPU")

    for key in ("dataset", "dataset_revision"):
        _non_empty_string(workload[key], f"workload.{key}")
    if not isinstance(workload["dataset_sha256"], str) or not SHA256_PATTERN.fullmatch(
        workload["dataset_sha256"]
    ):
        raise ValueError("workload.dataset_sha256 must be 64 lowercase hexadecimal characters")
    prompts = _integer(workload["prompts"], "workload.prompts", minimum=1)
    _number(workload["request_rate"], "workload.request_rate")
    _integer(workload["max_concurrency"], "workload.max_concurrency", minimum=1)
    _integer(workload["warmup_runs"], "workload.warmup_runs")
    _integer(workload["recorded_runs"], "workload.recorded_runs", minimum=1)
    if not isinstance(root["raw_logs"], list) or not root["raw_logs"]:
        raise ValueError("raw_logs must contain at least one path")
    for raw_log in root["raw_logs"]:
        _non_empty_string(raw_log, "raw_logs item")
        path = Path(raw_log)
        if not raw_log.startswith("https://") and (path.is_absolute() or ".." in path.parts):
            raise ValueError(f"raw_logs contains an unsafe path: {raw_log}")
    successful = _integer(results["successful"], "results.successful")
    failed = _integer(results["failed"], "results.failed")
    if successful + failed != prompts:
        raise ValueError("results.successful + results.failed must equal workload.prompts")
    if not isinstance(correctness["passed"], bool):
        raise ValueError("results.correctness.passed must be Boolean")
    _non_empty_string(correctness["metric"], "results.correctness.metric")
    _number(correctness["value"], "results.correctness.value")
    _number(correctness["tolerance"], "results.correctness.tolerance")
    _non_empty_string(quality["metric"], "results.quality.metric")
    _number(quality["value"], "results.quality.value")
    _number(quality["bf16_delta"], "results.quality.bf16_delta", minimum=-float("inf"))
    for key, value in throughput.items():
        _number(value, f"results.throughput.{key}")
    for key, value in latency.items():
        _number(value, f"results.latency.{key}")
    hbm_values = {key: _number(value, f"results.hbm.{key}") for key, value in hbm.items()}
    if not hbm_values["idle_mib"] <= hbm_values["loaded_mib"] <= hbm_values["peak_mib"]:
        raise ValueError("HBM values must satisfy idle_mib <= loaded_mib <= peak_mib")
    return root


def load_evidence(path: Path) -> dict[str, Any]:
    evidence = validate_evidence(json.loads(path.read_text(encoding="utf-8")))
    for raw_log in evidence["raw_logs"]:
        if not raw_log.startswith("https://") and not (path.parent / raw_log).is_file():
            raise ValueError(f"raw log does not exist: {raw_log}")
    return evidence
