import hashlib
import json
import struct
from pathlib import Path

import pytest

from vllm_ascend_quant_ext.contract import ContractError, validate_artifact
from vllm_ascend_quant_ext.manager import plan, provider, render


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    header = {}
    cursor = 0
    widths = {"I8": 1, "I32": 4, "F32": 4}
    for name, (dtype, shape) in tensors.items():
        size = widths[dtype]
        for dim in shape:
            size *= dim
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [cursor, cursor + size]}
        cursor += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(cursor))


def file_record(path: Path) -> dict:
    return {
        "name": path.name,
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def refresh_file_contract(path: Path, contract: dict) -> None:
    files = {
        "config": file_record(path / "config.json"),
        "description": file_record(path / "quant_model_description.json"),
        "indexes": [file_record(item) for item in sorted(path.glob("*.safetensors.index.json"))],
        "weights": [file_record(path / "weights.safetensors")],
    }
    contract["files"] = files
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    contract["artifact_id"] = (
        f"{contract['model']['model_type']}:{contract['quantization']['scheme']}:{digest[:16]}"
    )


def evidence_payload(contract: dict, profile: str = "W8A8") -> dict:
    digest = hashlib.sha256(
        json.dumps(contract["files"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": "1.1.0",
        "run_id": f"fixture-{profile.lower()}",
        "profile": profile,
        "artifact": {
            "model": "fixture",
            "artifact_id": contract["artifact_id"],
            "artifact_files_sha256": digest,
        },
        "software": {},
        "hardware": {},
        "workload": {},
        "results": {},
        "raw_logs": ["fixture.log"],
    }


def make_artifact(path: Path) -> dict:
    config = {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}
    description = {
        "layer.weight": "ASCEND_QUANT_W8A8",
        "layer.weight_scale": "ASCEND_QUANT_W8A8",
        "layer.weight_offset": "ASCEND_QUANT_W8A8",
        "layer.input_scale": "ASCEND_QUANT_W8A8",
        "layer.input_offset": "ASCEND_QUANT_W8A8",
        "layer.deq_scale": "ASCEND_QUANT_W8A8",
        "layer.quant_bias": "ASCEND_QUANT_W8A8",
        "version": "1.0.0",
        "model_quant_type": "ASCEND_QUANT_W8A8",
        "metadata": {},
        "group_size": 0,
        "optional": {},
    }
    (path / "config.json").write_text(json.dumps(config))
    description_path = path / "quant_model_description.json"
    description_path.write_text(json.dumps(description))
    tensors = {
        "layer.weight": ("I8", [16, 16]),
        "layer.weight_scale": ("F32", [16, 1]),
        "layer.weight_offset": ("F32", [16, 1]),
        "layer.input_scale": ("F32", [1]),
        "layer.input_offset": ("F32", [1]),
        "layer.deq_scale": ("F32", [16]),
        "layer.quant_bias": ("I32", [16]),
    }
    write_safetensors(path / "weights.safetensors", tensors)
    contract = {
        "schema_version": "1.1.0",
        "artifact_id": "pending",
        "format": "modelslim-ascend-v1",
        "model": {
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "supported_shapes": {"weight_rank": 2, "input_multiple": 16, "output_multiple": 16},
        },
        "quantization": {
            "scheme": "W8A8",
            "producer_quant_type": "W8A8_MIX",
            "runtime_quant_type": "ASCEND_QUANT_W8A8",
            "weight": {
                "bits": 8,
                "storage_dtype": "int8",
                "signed": True,
                "packing": "none",
                "layout": "logical_out_in",
                "granularity": "per_channel",
                "axis": 0,
                "group_size": None,
            },
            "activation": {"bits": 8, "dtype": "int8", "granularity": "pd_mix", "dynamic": True},
            "scale": {
                "dtype": "float32",
                "weight_shape": "out_1",
                "activation_shape": "one",
                "scale_bias": "forbidden",
            },
            "zero_point": {
                "weight": "required",
                "activation": "required",
                "dtype": "float32",
                "semantics": "additive_offset_before_scale",
            },
        },
        "runtime": {
            "host": "vllm-ascend",
            "loader": "modelslim",
            "scheme_provider": "vllm-ascend-quant-ext",
            "operators": ["torch_npu.npu_dynamic_quant", "torch_npu.npu_quant_matmul"],
        },
        "software": {
            "cann": ">=8.5,<8.6",
            "torch_npu": ">=2.9,<2.10",
            "vllm": ">=0.17,<0.18",
            "vllm_ascend": ">=0.1.dev2790,<0.2",
        },
        "files": {},
        "evidence": {"level": "schema_only", "verified_profiles": [], "results": []},
    }
    refresh_file_contract(path, contract)
    (path / "ascend_quant_artifact.json").write_text(json.dumps(contract))
    return contract


def convert_to_w4(path: Path, scheme: str) -> None:
    contract = json.loads((path / "ascend_quant_artifact.json").read_text())
    runtime_type = {"W4A4": "W4A4_DYNAMIC", "W4A8": "W4A8_DYNAMIC"}[scheme]
    activation_bits = 4 if scheme == "W4A4" else 8
    description = {
        "layer.weight": runtime_type,
        "layer.weight_scale": runtime_type,
        "layer.weight_offset": runtime_type,
        "version": "1.0.0",
        "model_quant_type": runtime_type,
        "metadata": {},
        "group_size": 0,
        "optional": {},
        "w4a4_weight_packed": True,
    }
    description_path = path / "quant_model_description.json"
    description_path.write_text(json.dumps(description))
    write_safetensors(
        path / "weights.safetensors",
        {
            "layer.weight": ("I8", [8, 16]),
            "layer.weight_scale": ("F32", [16, 1]),
            "layer.weight_offset": ("F32", [16, 1]),
            **({"layer.scale_bias": ("F32", [16, 1])} if scheme == "W4A8" else {}),
        },
    )
    contract["quantization"]["scheme"] = scheme
    contract["quantization"]["runtime_quant_type"] = runtime_type
    contract["quantization"]["weight"].update(
        {"bits": 4, "packing": "signed_int4_nibble_low_high", "layout": "packed_out_in"}
    )
    contract["quantization"]["activation"].update(
        {
            "bits": activation_bits,
            "dtype": "int4" if activation_bits == 4 else "int8",
            "granularity": "per_token",
            "dynamic": True,
        }
    )
    contract["quantization"]["scale"]["activation_shape"] = "runtime_per_token"
    contract["quantization"]["scale"]["scale_bias"] = (
        "required_out_1_or_16" if scheme == "W4A8" else "forbidden"
    )
    contract["quantization"]["zero_point"]["activation"] = "runtime"
    contract["runtime"]["operators"] = [
        "torch_npu.npu_dynamic_quant",
        "torch_npu.npu_convert_weight_to_int4pack",
        "torch_npu.npu_quant_matmul",
    ]
    refresh_file_contract(path, contract)
    (path / "ascend_quant_artifact.json").write_text(json.dumps(contract))


def test_valid_w8a8_contract(tmp_path: Path):
    make_artifact(tmp_path)
    report = validate_artifact(tmp_path, check_software=False)
    assert report["valid"] is True
    assert report["scheme"] == "W8A8"


@pytest.mark.parametrize("scheme", ["W4A4", "W4A8"])
def test_valid_packed_w4_contracts(tmp_path: Path, scheme: str):
    make_artifact(tmp_path)
    convert_to_w4(tmp_path, scheme)
    report = validate_artifact(tmp_path, check_software=False)
    assert report["scheme"] == scheme


def test_unknown_contract_field_fails_closed(tmp_path: Path):
    contract = make_artifact(tmp_path)
    contract["surprise"] = True
    (tmp_path / "ascend_quant_artifact.json").write_text(json.dumps(contract))
    with pytest.raises(ContractError, match="unknown fields"):
        validate_artifact(tmp_path, check_software=False)


def test_wrong_weight_dtype_fails_closed(tmp_path: Path):
    contract = make_artifact(tmp_path)
    tensors = {
        "layer.weight": ("F32", [16, 16]),
        "layer.weight_scale": ("F32", [16, 1]),
        "layer.weight_offset": ("F32", [16, 1]),
        "layer.input_scale": ("F32", [1]),
        "layer.input_offset": ("F32", [1]),
        "layer.deq_scale": ("F32", [16]),
        "layer.quant_bias": ("I32", [16]),
    }
    write_safetensors(tmp_path / "weights.safetensors", tensors)
    refresh_file_contract(tmp_path, contract)
    (tmp_path / "ascend_quant_artifact.json").write_text(json.dumps(contract))
    with pytest.raises(ContractError, match="rank-2 I8"):
        validate_artifact(tmp_path, check_software=False)


def test_incompatible_shape_fails_closed(tmp_path: Path):
    contract = make_artifact(tmp_path)
    tensors = {
        "layer.weight": ("I8", [15, 16]),
        "layer.weight_scale": ("F32", [15, 1]),
        "layer.weight_offset": ("F32", [15, 1]),
        "layer.input_scale": ("F32", [1]),
        "layer.input_offset": ("F32", [1]),
        "layer.deq_scale": ("F32", [15]),
        "layer.quant_bias": ("I32", [15]),
    }
    write_safetensors(tmp_path / "weights.safetensors", tensors)
    refresh_file_contract(tmp_path, contract)
    (tmp_path / "ascend_quant_artifact.json").write_text(json.dumps(contract))
    with pytest.raises(ContractError, match="output dimension"):
        validate_artifact(tmp_path, check_software=False)


def test_description_hash_mismatch_fails_closed(tmp_path: Path):
    make_artifact(tmp_path)
    (tmp_path / "quant_model_description.json").write_text("{}")
    with pytest.raises(ContractError, match="(size|hash) mismatch"):
        validate_artifact(tmp_path, check_software=False)


def test_weight_payload_hash_mismatch_fails_closed(tmp_path: Path):
    make_artifact(tmp_path)
    path = tmp_path / "weights.safetensors"
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 0xFF
    path.write_bytes(payload)
    with pytest.raises(ContractError, match="hash mismatch"):
        validate_artifact(tmp_path, check_software=False)


def test_config_hash_mismatch_fails_closed(tmp_path: Path):
    make_artifact(tmp_path)
    (tmp_path / "config.json").write_text('{"model_type":"changed"}')
    with pytest.raises(ContractError, match="(size|hash) mismatch"):
        validate_artifact(tmp_path, check_software=False)


def test_artifact_id_must_match_file_inventory(tmp_path: Path):
    contract = make_artifact(tmp_path)
    contract["artifact_id"] = "qwen2:W8A8:incorrect"
    (tmp_path / "ascend_quant_artifact.json").write_text(json.dumps(contract))
    with pytest.raises(ContractError, match="artifact_id"):
        validate_artifact(tmp_path, check_software=False)


def test_undeclared_safetensors_index_fails_closed(tmp_path: Path):
    make_artifact(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    with pytest.raises(ContractError, match="declared index files"):
        validate_artifact(tmp_path, check_software=False)


def test_safetensors_index_must_match_tensor_inventory(tmp_path: Path):
    contract = make_artifact(tmp_path)
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"metadata": {}, "weight_map": {"unknown.weight": "weights.safetensors"}})
    )
    refresh_file_contract(tmp_path, contract)
    (tmp_path / "ascend_quant_artifact.json").write_text(json.dumps(contract))
    with pytest.raises(ContractError, match="tensor names do not match"):
        validate_artifact(tmp_path, check_software=False)


def test_valid_safetensors_index_is_admitted(tmp_path: Path):
    contract = make_artifact(tmp_path)
    weight_path = tmp_path / "weights.safetensors"
    raw = weight_path.read_bytes()
    header_length = struct.unpack("<Q", raw[:8])[0]
    tensor_names = [
        name for name in json.loads(raw[8 : 8 + header_length]) if name != "__metadata__"
    ]
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": len(raw)},
                "weight_map": {name: "weights.safetensors" for name in tensor_names},
            }
        )
    )
    refresh_file_contract(tmp_path, contract)
    (tmp_path / "ascend_quant_artifact.json").write_text(json.dumps(contract))
    assert validate_artifact(tmp_path, check_software=False)["valid"] is True


def test_out_of_bounds_tensor_payload_fails_closed(tmp_path: Path):
    contract = make_artifact(tmp_path)
    path = tmp_path / "weights.safetensors"
    raw = path.read_bytes()
    header_length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_length])
    header["layer.weight"]["data_offsets"] = [0, 999999999]
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0")
    refresh_file_contract(tmp_path, contract)
    (tmp_path / "ascend_quant_artifact.json").write_text(json.dumps(contract))
    with pytest.raises(ContractError, match="payload exceeds file size"):
        validate_artifact(tmp_path, check_software=False)


def test_tensor_payload_span_must_match_shape(tmp_path: Path):
    contract = make_artifact(tmp_path)
    path = tmp_path / "weights.safetensors"
    raw = path.read_bytes()
    header_length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_length])
    header["layer.weight"]["shape"] = [16, 15]
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[8 + header_length :])
    refresh_file_contract(tmp_path, contract)
    (tmp_path / "ascend_quant_artifact.json").write_text(json.dumps(contract))
    with pytest.raises(ContractError, match="payload size does not match"):
        validate_artifact(tmp_path, check_software=False)


def test_overlapping_tensor_payload_fails_closed(tmp_path: Path):
    contract = make_artifact(tmp_path)
    path = tmp_path / "weights.safetensors"
    raw = path.read_bytes()
    header_length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_length])
    header["layer.weight_scale"]["data_offsets"] = [0, 64]
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[8 + header_length :])
    refresh_file_contract(tmp_path, contract)
    (tmp_path / "ascend_quant_artifact.json").write_text(json.dumps(contract))
    with pytest.raises(ContractError, match="overlapping tensor payload"):
        validate_artifact(tmp_path, check_software=False)


def test_invalid_scheme_provider_fails_closed(tmp_path: Path):
    contract = make_artifact(tmp_path)
    contract["runtime"]["scheme_provider"] = "unknown-provider"
    (tmp_path / "ascend_quant_artifact.json").write_text(json.dumps(contract))
    with pytest.raises(ContractError, match="host/loader/provider"):
        validate_artifact(tmp_path, check_software=False)


def test_empty_npu_e2e_claim_fails_closed(tmp_path: Path):
    contract = make_artifact(tmp_path)
    contract["evidence"] = {
        "level": "npu_e2e",
        "verified_profiles": [],
        "results": [],
    }
    (tmp_path / "ascend_quant_artifact.json").write_text(json.dumps(contract))
    with pytest.raises(ContractError, match="requires the W8A8 profile and results"):
        validate_artifact(tmp_path, check_software=False)


def test_evidence_result_hash_mismatch_fails_closed(tmp_path: Path):
    contract = make_artifact(tmp_path)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    result_path = results_dir / "e2e.json"
    result_path.write_text(json.dumps(evidence_payload(contract)))
    contract["evidence"] = {
        "level": "npu_e2e",
        "verified_profiles": ["W8A8"],
        "results": [
            {
                "name": "results/e2e.json",
                "size": result_path.stat().st_size,
                "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            }
        ],
    }
    (tmp_path / "ascend_quant_artifact.json").write_text(json.dumps(contract))
    result_path.write_text('{"changed":true}')
    with pytest.raises(ContractError, match="evidence result (size|hash) mismatch"):
        validate_artifact(tmp_path, check_software=False)


def test_valid_hashed_npu_e2e_evidence(tmp_path: Path):
    contract = make_artifact(tmp_path)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    result_path = results_dir / "e2e.json"
    result_path.write_text(json.dumps(evidence_payload(contract)))
    contract["evidence"] = {
        "level": "npu_e2e",
        "verified_profiles": ["W8A8"],
        "results": [
            {
                "name": "results/e2e.json",
                "size": result_path.stat().st_size,
                "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            }
        ],
    }
    (tmp_path / "ascend_quant_artifact.json").write_text(json.dumps(contract))
    assert validate_artifact(tmp_path, check_software=False)["valid"] is True


def test_unknown_operator_fails_closed(tmp_path: Path):
    contract = make_artifact(tmp_path)
    contract["runtime"]["operators"].append("unknown.kernel")
    (tmp_path / "ascend_quant_artifact.json").write_text(json.dumps(contract))
    with pytest.raises(ContractError, match="operator"):
        validate_artifact(tmp_path, check_software=False)


def test_missing_required_operator_fails_closed(tmp_path: Path):
    contract = make_artifact(tmp_path)
    contract["runtime"]["operators"] = ["torch_npu.npu_quant_matmul"]
    (tmp_path / "ascend_quant_artifact.json").write_text(json.dumps(contract))
    with pytest.raises(ContractError, match="operator set"):
        validate_artifact(tmp_path, check_software=False)


def test_w8a8_profile_semantics_fail_closed(tmp_path: Path):
    contract = make_artifact(tmp_path)
    contract["quantization"]["activation"]["granularity"] = "per_token"
    (tmp_path / "ascend_quant_artifact.json").write_text(json.dumps(contract))
    with pytest.raises(ContractError, match="activation semantics"):
        validate_artifact(tmp_path, check_software=False)


def test_w4_requires_packed_weights(tmp_path: Path):
    make_artifact(tmp_path)
    convert_to_w4(tmp_path, "W4A4")
    contract = json.loads((tmp_path / "ascend_quant_artifact.json").read_text())
    contract["quantization"]["weight"].update(
        {"packing": "none", "layout": "logical_out_in"}
    )
    (tmp_path / "ascend_quant_artifact.json").write_text(json.dumps(contract))
    with pytest.raises(ContractError, match="signed low-high nibble"):
        validate_artifact(tmp_path, check_software=False)


def test_model_mismatch_fails_closed(tmp_path: Path):
    contract = make_artifact(tmp_path)
    contract["model"]["model_type"] = "unsupported"
    refresh_file_contract(tmp_path, contract)
    (tmp_path / "ascend_quant_artifact.json").write_text(json.dumps(contract))
    with pytest.raises(ContractError, match="model_type"):
        validate_artifact(tmp_path, check_software=False)


def test_incompatible_software_fails_closed(tmp_path: Path, monkeypatch):
    contract = make_artifact(tmp_path)
    contract["software"]["cann"] = ">=99"
    (tmp_path / "ascend_quant_artifact.json").write_text(json.dumps(contract))
    monkeypatch.setattr(
        "vllm_ascend_quant_ext.contract._cann_version",
        lambda: "8.5.0",
    )
    with pytest.raises(ContractError, match="incompatible CANN"):
        validate_artifact(tmp_path, check_software=True)


def test_missing_cann_fails_closed(tmp_path: Path, monkeypatch):
    make_artifact(tmp_path)
    monkeypatch.setattr(
        "vllm_ascend_quant_ext.contract._cann_version",
        lambda: None,
    )
    with pytest.raises(
        ContractError,
        match="cannot determine installed CANN version",
    ):
        validate_artifact(tmp_path, check_software=True)


def test_plan_and_render_are_read_only(tmp_path: Path, monkeypatch):
    make_artifact(tmp_path)
    monkeypatch.setattr(
        "vllm_ascend_quant_ext.manager.validate_artifact", lambda path: {"valid": True}
    )
    before = {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()}
    assert plan(tmp_path)["mutates_model"] is False
    assert render(tmp_path)["environment"]["VLLM_ASCEND_QUANT_EXT_ENABLE"] == "1"
    after = {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()}
    assert before == after


def test_manager_adapter_disabled_render_removes_only_extension_state():
    rendered = provider.render({"enabled": False, "model": "/unused"})
    assert rendered["environment_set"] == {}
    assert set(rendered["environment_unset"]) == {
        "VLLM_ASCEND_QUANT_EXT_ENABLE",
        "VLLM_ASCEND_QUANT_EXT_ARTIFACT",
    }
    assert rendered["vllm_plugins_remove"] == ["vllm_ascend_quant"]
    assert "ascend" not in rendered["vllm_plugins_remove"]


def test_manager_adapter_enabled_check_is_import_only(tmp_path: Path, monkeypatch):
    make_artifact(tmp_path)
    monkeypatch.setattr(
        "vllm_ascend_quant_ext.manager.validate_artifact", lambda path: {"valid": True}
    )
    before = {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()}
    checked = provider.check({"enabled": True, "model": str(tmp_path)})
    assert checked["admitted"] is False
    assert checked["enable_allowed"] is False
    assert "import_only" in checked["reason"]
    assert {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()} == before


def test_manager_adapter_refuses_enabled_plan_and_render(tmp_path: Path, monkeypatch):
    make_artifact(tmp_path)
    monkeypatch.setattr(
        "vllm_ascend_quant_ext.manager.validate_artifact", lambda path: {"valid": True}
    )
    configuration = {"enabled": True, "model": str(tmp_path)}
    with pytest.raises(RuntimeError, match="import_only"):
        provider.plan(configuration)
    with pytest.raises(RuntimeError, match="import_only"):
        provider.render(configuration)
