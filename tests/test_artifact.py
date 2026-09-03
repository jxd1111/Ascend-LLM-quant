import json
from pathlib import Path

from jxd_ascend_quant.artifact import (
    inspect_artifact,
    prepare_runtime_metadata,
    restore_modelslim_metadata,
    write_manifest,
    write_runtime_contract,
)
from jxd_ascend_quant.registry import get_recipe


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
    (tmp_path / "quant_model_description.json").write_text(
        json.dumps({"layer.weight": "W8A8_MIX"})
    )
    (tmp_path / "weights.safetensors").write_bytes(b"weights")
    source_model = tmp_path.parent / "source-model"

    destination = write_manifest(
        tmp_path,
        get_recipe("qwen25-w8a8-pdmix"),
        source_model=source_model,
    )
    manifest = json.loads(destination.read_text())

    assert manifest["schema_version"] == 2
    assert manifest["recipe"]["producer_quant_type"] == "W8A8_MIX"
    assert manifest["recipe"]["runtime_quant_type"] == "JXD_W8A8_PDMIX"
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
    recipe = get_recipe("qwen25-w8a8-pdmix")

    first = prepare_runtime_metadata(tmp_path, recipe)
    second = prepare_runtime_metadata(tmp_path, recipe)
    migrated = json.loads(description_path.read_text())
    backup = json.loads(
        (tmp_path / "quant_model_description.modelslim.json").read_text()
    )

    assert first["changed"] is True
    assert first["replacements"] == 2
    assert second["changed"] is False
    assert migrated["layer.weight"] == "JXD_W8A8_PDMIX"
    assert migrated["lm_head.weight"] == "FLOAT"
    assert backup == original

    restored = restore_modelslim_metadata(tmp_path, recipe)
    assert restored["active_quant_type"] == "W8A8_MIX"
    assert json.loads(description_path.read_text()) == original


def test_runtime_contract_records_closed_w8a8_format(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]})
    )
    (tmp_path / "quant_model_description.json").write_text(
        json.dumps({"layer.weight": "JXD_W8A8_PDMIX"})
    )
    (tmp_path / "weights.safetensors").write_bytes(b"weights")

    destination = write_runtime_contract(
        tmp_path,
        get_recipe("qwen25-w8a8-pdmix"),
        evidence_level="npu_e2e",
        verified_profiles=["W8A8"],
        evidence_results=["results/qwen25-w8a8-e2e.json"],
    )
    contract = json.loads(destination.read_text())

    assert contract["schema_version"] == "1.0.0"
    assert contract["quantization"]["weight"]["packing"] == "none"
    assert contract["quantization"]["zero_point"]["semantics"] == "additive_offset_before_scale"
    assert contract["software"]["cann"] == ">=8.5,<8.6"
    assert contract["evidence"]["verified_profiles"] == ["W8A8"]
