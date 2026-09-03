import pytest

from vllm_ascend_quant_ext.manager import (
    load_manifest,
    provider,
    validate_manager_config,
    validate_manifest,
)


def test_manager_manifest_has_explicit_boundary():
    manifest = load_manifest()
    assert manifest["kind"] == "model_weight_quantization_runtime"
    assert manifest["host"]["provider"] == "vllm-ascend"
    assert manifest["lifecycle_owner"] == "vllm-ascend"
    assert manifest["permissions"] == []
    assert manifest["requires_services"] == []
    assert manifest["compatibility"]["kv_cache_quantization"] is False


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


def test_manager_configuration_is_closed():
    assert validate_manager_config({"enabled": False, "model": "/model"})["enabled"] is False
    with pytest.raises(ValueError, match="unknown"):
        validate_manager_config({"enabled": False, "model": "/model", "extra": True})
