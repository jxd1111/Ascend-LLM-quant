import pytest

from vllm_ascend_quant_ext import __version__
from vllm_ascend_quant_ext.manager import (
    BUNDLE_ID,
    COMPONENT_ID,
    load_manifest,
    provider,
    validate_manager_config,
    validate_manifest,
)


def test_manager_manifest_has_explicit_boundary():
    manifest = load_manifest()
    assert manifest["extension_id"] == BUNDLE_ID
    assert manifest["extension_version"] == __version__
    assert manifest["kind"] == "model_weight_quantization_runtime"
    assert manifest["host"]["provider"] == "vllm-ascend"
    assert manifest["lifecycle_owner"] == "host"
    assert manifest["requires_services"] == []
    assert manifest["runtime"] == {
        "type": "python",
        "process_scope": "model_worker",
        "isolation": "trusted_in_process",
    }
    assert manifest["implementation"][0]["status"] == "active"
    assert manifest["components"][0]["component_id"] == "w8a8-runtime"
    assert manifest["components"][0]["contracts"] == [
        "vllm-ascend.quantization.scheme.v1"
    ]
    assert manifest["components"][0]["execution_planes"] == ["model_worker"]
    assert manifest["components"][0]["permissions"] == []
    assert manifest["activation"] == {
        "entry_points": [],
        "environment": {},
        "additional_config": {},
    }


def test_manager_manifest_unknown_fields_fail_closed():
    manifest = load_manifest()
    manifest["unknown"] = True
    with pytest.raises(ValueError, match="unknown"):
        validate_manifest(manifest)


def test_manager_adapter_descriptor_does_not_import_implementation():
    descriptor = provider.descriptor()
    assert descriptor["api_version"] == "1.0"
    assert descriptor["kind"] == "model_weight_quantization_runtime"
    assert descriptor["implementation_imported"] is False
    assert descriptor["extension_id"] == BUNDLE_ID


def test_manager_configuration_is_closed():
    assert validate_manager_config({"enabled": False, "model": "/model"})["enabled"] is False
    with pytest.raises(ValueError, match="unknown"):
        validate_manager_config({"enabled": False, "model": "/model", "extra": True})


def test_typed_component_carrier_has_stable_experimental_surface():
    from vllm_ascend_quant_ext.adapters.vllm_hust.runtime import (
        AscendQuantRuntimeComponent,
    )

    assert AscendQuantRuntimeComponent.vllm_ascend_quantization_scheme_api_version == 1
    assert AscendQuantRuntimeComponent.component_id == COMPONENT_ID
    assert AscendQuantRuntimeComponent.contract == "vllm-ascend.quantization.scheme.v1"
    assert callable(AscendQuantRuntimeComponent.from_vllm_config)
    assert callable(AscendQuantRuntimeComponent.activate)
