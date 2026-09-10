import importlib
import sys

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
    assert manifest["kind"] == "in_process_plugin"
    assert manifest["host"] == {
        "provider": "vllm",
        "name": "vllm-ascend",
        "version_range": ">=0",
    }
    assert manifest["lifecycle_owner"] == "vllm"
    assert manifest["requires_services"] == []
    assert manifest["runtime"] == {
        "type": "python",
        "process_scope": "vllm-ascend-worker",
        "isolation": "trusted_in_process",
    }
    assert manifest["implementation"][0]["status"] == "import_only"
    assert manifest["components"][0]["component_id"] == (
        "ascend-quant-artifact-validator"
    )
    assert manifest["components"][0]["contracts"] == [
        "vllm.ascend.quantized-artifact-loader.v1"
    ]
    assert manifest["components"][0]["execution_planes"] == ["worker", "device"]
    assert manifest["components"][0]["permissions"] == [
        "filesystem_read",
        "device_access",
    ]
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
    assert descriptor["kind"] == "in_process_plugin"
    assert descriptor["implementation_imported"] is False
    assert descriptor["extension_id"] == BUNDLE_ID


def test_manager_configuration_is_closed():
    assert validate_manager_config({"enabled": False, "model": "/model"})["enabled"] is False
    with pytest.raises(ValueError, match="unknown"):
        validate_manager_config({"enabled": False, "model": "/model", "extra": True})


def test_import_only_implementation_ref_resolves_without_runtime_imports():
    manifest = load_manifest()
    implementation = manifest["implementation"][0]
    module = importlib.import_module(implementation["module"])
    carrier = getattr(module, implementation["object"])

    assert carrier.status == "import_only"
    assert manifest["components"][0]["implementation_ref"] == (
        f"{implementation['module']}:{implementation['object']}"
    )
    assert "torch" not in sys.modules
    assert "vllm" not in sys.modules
    assert "vllm_ascend" not in sys.modules


def test_component_identity_is_fully_namespaced():
    assert COMPONENT_ID == (
        "org.vllm-hust.ascend-quant-runtime/"
        "ascend-quant-artifact-validator"
    )
