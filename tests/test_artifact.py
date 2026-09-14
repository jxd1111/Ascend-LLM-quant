import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from ascend_quant_toolkit.artifact import (
    build_runtime_contract,
    inspect_artifact,
    prepare_runtime_metadata,
    restore_modelslim_metadata,
    write_manifest,
    write_runtime_contract,
)
from ascend_quant_toolkit.registry import get_recipe


def test_inspect_artifact(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]})
    )
    (tmp_path / "quant_model_description.json").write_text(
        json.dumps({"model.layers.0.mlp.down_proj.weight": "W8A8_MIX"})
    )
    (tmp_path / "quant_model_weights.safetensors").write_bytes(b"weights")

    result = inspect_artifact(tmp_path)

    assert result["valid"] is True
    assert result["quant_type_counts"] == {"W8A8_MIX": 1}


def test_inspect_artifact_reports_missing_files(tmp_path: Path):
    result = inspect_artifact(tmp_path)
    assert result["valid"] is False
    assert "missing config.json" in result["errors"]


def test_write_manifest_records_recipe_and_source(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    (tmp_path / "quant_model_description.json").write_text(json.dumps({"layer.weight": "W8A8_MIX"}))
    (tmp_path / "weights.safetensors").write_bytes(b"weights")
    source_model = tmp_path.parent / "source-model"

    destination = write_manifest(
        tmp_path,
        get_recipe("qwen25-w8a8"),
        source_model=source_model,
    )
    manifest = json.loads(destination.read_text())

    assert destination.name == "ascend_quant_manifest.json"
    assert manifest["schema_version"] == 2
    assert manifest["producer"]["name"] == "ascend-quant-toolkit"
    assert manifest["producer"]["version"] == "0.4.1"
    assert manifest["recipe"]["producer_quant_type"] == "W8A8_MIX"
    assert manifest["recipe"]["runtime_quant_type"] == "ASCEND_QUANT_W8A8"
    assert manifest["source_model"] == str(source_model.resolve())


def test_prepare_runtime_metadata_is_atomic_and_idempotent(tmp_path: Path):
    description_path = tmp_path / "quant_model_description.json"
    original = {
        "layer.weight": "W8A8_MIX",
        "layer.input_scale": "W8A8_MIX",
        "lm_head.weight": "FLOAT",
        "version": {"format": "1.0.0"},
    }
    description_path.write_text(json.dumps(original))
    recipe = get_recipe("qwen25-w8a8")

    first = prepare_runtime_metadata(tmp_path, recipe)
    second = prepare_runtime_metadata(tmp_path, recipe)
    migrated = json.loads(description_path.read_text())
    backup = json.loads((tmp_path / "quant_model_description.modelslim.json").read_text())

    assert first["changed"] is True
    assert first["replacements"] == 2
    assert second["changed"] is False
    assert migrated["layer.weight"] == "ASCEND_QUANT_W8A8"
    assert migrated["lm_head.weight"] == "FLOAT"
    assert backup == original

    restored = restore_modelslim_metadata(tmp_path, recipe)
    assert restored["changed"] is True
    assert restored["active_quant_type"] == "W8A8_MIX"
    assert json.loads(description_path.read_text()) == original
    assert (
        description_path.read_bytes()
        == (tmp_path / "quant_model_description.modelslim.json").read_bytes()
    )

    second_restore = restore_modelslim_metadata(tmp_path, recipe)
    assert second_restore["changed"] is False


def test_restore_modelslim_refuses_to_overwrite_modified_runtime_metadata(tmp_path: Path):
    description_path = tmp_path / "quant_model_description.json"
    original = {"layer.weight": "W8A8_MIX", "lm_head.weight": "FLOAT"}
    description_path.write_text(json.dumps(original))
    recipe = get_recipe("qwen25-w8a8")
    prepare_runtime_metadata(tmp_path, recipe)

    modified = json.loads(description_path.read_text())
    modified["new.tensor"] = "FLOAT"
    description_path.write_text(json.dumps(modified))

    with pytest.raises(ValueError, match="refusing to overwrite later changes"):
        restore_modelslim_metadata(tmp_path, recipe)

    assert json.loads(description_path.read_text()) == modified


def test_runtime_contract_records_closed_w8a8_format(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]})
    )
    (tmp_path / "quant_model_description.json").write_text(
        json.dumps({"layer.weight": "ASCEND_QUANT_W8A8"})
    )
    (tmp_path / "weights.safetensors").write_bytes(b"weights")
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "raw.log").write_text("e2e passed")
    base_contract = build_runtime_contract(tmp_path, get_recipe("qwen25-w8a8"))
    inventory_digest = hashlib.sha256(
        json.dumps(
            base_contract["files"], sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    evidence = {
        "schema_version": "1.1.0",
        "run_id": "qwen25-w8a8-e2e",
        "profile": "W8A8",
        "artifact": {
            "model": "Qwen2.5-14B-Instruct-w8a8",
            "artifact_id": base_contract["artifact_id"],
            "artifact_files_sha256": inventory_digest,
        },
        "software": {
            "cann": "8.5.0",
            "torch_npu": "2.9.0",
            "vllm": "0.17.2",
            "vllm_ascend": "0.1.dev2790",
            "extension": "0.4.1",
        },
        "hardware": {
            "npu_model": "910B3",
            "npu_count": 1,
            "tensor_parallel": 1,
            "device_ids": [7],
        },
        "workload": {
            "dataset": "smoke",
            "dataset_revision": "fixture",
            "dataset_sha256": "0" * 64,
            "prompts": 1,
            "request_rate": 1.0,
            "max_concurrency": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "warmup_runs": 0,
            "recorded_runs": 1,
        },
        "results": {
            "successful": 1,
            "failed": 0,
            "correctness": {
                "passed": True,
                "metric": "smoke",
                "value": 1.0,
                "tolerance": 0.0,
            },
            "quality": {"metric": "smoke", "value": 1.0, "bf16_delta": 0.0},
            "throughput": {
                "requests_per_second": 1.0,
                "output_tokens_per_second": 1.0,
                "total_tokens_per_second": 2.0,
            },
            "latency": {
                "mean_ttft_ms": 1.0,
                "p99_ttft_ms": 1.0,
                "mean_tpot_ms": 1.0,
                "p99_tpot_ms": 1.0,
            },
            "hbm": {"idle_mib": 1, "loaded_mib": 2, "peak_mib": 3},
        },
        "raw_logs": ["raw.log"],
    }
    (results_dir / "qwen25-w8a8-e2e.json").write_text(json.dumps(evidence))

    destination = write_runtime_contract(
        tmp_path,
        get_recipe("qwen25-w8a8"),
        evidence_level="npu_e2e",
        verified_profiles=["W8A8"],
        evidence_results=["results/qwen25-w8a8-e2e.json"],
    )
    contract = json.loads(destination.read_text())

    assert contract["schema_version"] == "1.1.0"
    assert contract["quantization"]["weight"]["packing"] == "none"
    assert contract["quantization"]["zero_point"]["semantics"] == "additive_offset_before_scale"
    assert contract["software"]["cann"] == ">=8.5,<8.6"
    assert contract["evidence"]["verified_profiles"] == ["W8A8"]
    assert contract["files"]["config"]["name"] == "config.json"
    assert len(contract["files"]["config"]["sha256"]) == 64
    assert contract["files"]["weights"][0]["name"] == "weights.safetensors"
    assert len(contract["files"]["weights"][0]["sha256"]) == 64

    schema_path = (
        Path(__file__).parents[1]
        / "contracts"
        / "ascend-quant-artifact-v1.schema.json"
    )
    schema = json.loads(schema_path.read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(contract)


def test_runtime_contract_rejects_unsubstantiated_evidence(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]})
    )
    (tmp_path / "quant_model_description.json").write_text(
        json.dumps({"layer.weight": "ASCEND_QUANT_W8A8"})
    )
    (tmp_path / "weights.safetensors").write_bytes(b"weights")

    with pytest.raises(ValueError, match="requires the W8A8 profile"):
        write_runtime_contract(
            tmp_path,
            get_recipe("qwen25-w8a8"),
            evidence_level="npu_e2e",
        )
