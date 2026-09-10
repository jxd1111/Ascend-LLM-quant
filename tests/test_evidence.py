import copy

import pytest

from ascend_quant_toolkit.evidence import validate_evidence


def sample():
    return {
        "schema_version": "1.0.0",
        "run_id": "qwen25-w8a8-sharegpt-rps8",
        "profile": "W8A8",
        "artifact": {"model": "Qwen2.5-14B", "artifact_id": "qwen2:W8A8:x", "contract_sha256": "0" * 64},
        "software": {"cann": "8.5.0", "torch_npu": "2.9.0", "vllm": "0.17", "vllm_ascend": "0.1", "extension": "0.3.0"},
        "hardware": {"npu_model": "910B3", "npu_count": 1, "tensor_parallel": 1, "device_ids": [7]},
        "workload": {"dataset": "ShareGPT-V3", "dataset_revision": "local", "dataset_sha256": "0" * 64, "prompts": 200, "request_rate": 8.0, "max_concurrency": 80, "input_tokens": "dataset", "output_tokens": "dataset", "warmup_runs": 1, "recorded_runs": 3},
        "results": {
            "successful": 200,
            "failed": 0,
            "correctness": {"passed": True, "metric": "smoke", "value": 1.0, "tolerance": 0.0},
            "quality": {"metric": "ppl", "value": 7.3, "bf16_delta": 0.1},
            "throughput": {"requests_per_second": 3.37, "output_tokens_per_second": 668.13, "total_tokens_per_second": 1384.92},
            "latency": {"mean_ttft_ms": 159.7, "p99_ttft_ms": 268.67, "mean_tpot_ms": 57.62, "p99_tpot_ms": 65.57},
            "hbm": {"idle_mib": 0, "loaded_mib": 0, "peak_mib": 0},
        },
        "raw_logs": ["results/raw.log"],
    }


def test_complete_evidence_is_valid():
    assert validate_evidence(sample())["profile"] == "W8A8"


def test_incomplete_evidence_fails_closed():
    value = copy.deepcopy(sample())
    del value["results"]["hbm"]
    with pytest.raises(ValueError, match="missing"):
        validate_evidence(value)
