"""Manifest 0.2 identity, enumeration and consistency tests (author guide 14.1)."""

from __future__ import annotations

import importlib
import json
import re
from importlib.metadata import entry_points
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from vllm_ascend_quant_ext import __version__

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "vllm_ascend_quant_ext"
MANIFEST_DIR = PACKAGE_ROOT / "manifests"
MANIFEST_PATH = MANIFEST_DIR / "vllm-hust-extension-v0.2.json"
PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"
LEGACY_MANIFEST = PACKAGE_ROOT / "extension-manifest.json"

BUNDLE_GROUP = "vllm_hust.extension_bundles"
BUNDLE_LOCATOR = "vllm_ascend_quant_ext.manifests"
EXPECTED_EXTENSION_ID = "org.vllm-hust.ascend-quant-runtime"

ALLOWED_KINDS = {
    "in_process_plugin",
    "scheduler_policy",
    "kv_connector",
    "kv_service_adapter",
    "control_plane_extension",
    "runtime_bridge",
}
ALLOWED_PROVIDERS = {"vllm", "mooncake", "production-stack"}
ALLOWED_RUNTIME_TYPES = {"python", "external_service", "oci", "kubernetes", "composite"}
ALLOWED_ISOLATION = {"trusted_in_process", "process_isolated"}
ALLOWED_LIFECYCLE_OWNERS = {"vllm", "host", "external_operator", "kubernetes", "user"}
ALLOWED_PERMISSIONS = {
    "device_access",
    "filesystem_read",
    "filesystem_write",
    "ipc",
    "network_egress",
    "shared_memory",
    "subprocess",
}
ALLOWED_IMPLEMENTATION_TYPES = {
    "python_entry_point",
    "python_module",
    "host_builtin",
    "external_service",
    "oci_image",
    "helm_values",
    "kubernetes_manifest",
    "crd",
    "controller",
}

# Exact package versions observed in the validated frozen v1 environment.
# See docs/evidence/w8a8-frozen-v1-npu-e2e-20260917.md.
FROZEN_VLLM = "0.28.1.post1.dev143+gf18cf803c.empty"
FROZEN_VLLM_ASCEND = "0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_directory_holds_exactly_one_supported_manifest() -> None:
    documents = sorted(path.name for path in MANIFEST_DIR.glob("*.json"))
    assert documents == ["vllm-hust-extension-v0.2.json"]
    assert not LEGACY_MANIFEST.exists(), "the retired legacy manifest must not ship"


def test_identity_fields_match_the_distribution(manifest: dict) -> None:
    assert manifest["schema_version"] == "0.2-experimental"
    assert manifest["extension_id"] == EXPECTED_EXTENSION_ID
    assert manifest["extension_version"] == __version__


def test_registration_name_equals_extension_id_and_locator_resolves(manifest: dict) -> None:
    text = PYPROJECT_PATH.read_text(encoding="utf-8")
    assert f'[project.entry-points."{BUNDLE_GROUP}"]' in text
    assert f'"{EXPECTED_EXTENSION_ID}" = "{BUNDLE_LOCATOR}"' in text

    module = importlib.import_module(BUNDLE_LOCATOR)
    manifest_from_locator = Path(module.__file__).resolve().parent / MANIFEST_PATH.name
    assert manifest_from_locator == MANIFEST_PATH
    assert manifest["extension_id"] == EXPECTED_EXTENSION_ID


def test_installed_metadata_registers_exactly_one_matching_bundle(
    installed_extension_versions: list[str],
) -> None:
    versions = installed_extension_versions
    if not versions:
        pytest.skip("vllm-ascend-quant-ext is not installed in this interpreter")
    assert versions == [__version__], (
        f"ambiguous installed metadata for vllm-ascend-quant-ext: {versions}; "
        "remove stale build artifacts (src/*.egg-info) and reinstall"
    )
    matching = [item for item in entry_points(group=BUNDLE_GROUP) if item.name == EXPECTED_EXTENSION_ID]
    assert len(matching) == 1, "duplicate or missing bundle registration"
    assert matching[0].value == BUNDLE_LOCATOR


def test_declared_enumerations_are_allowed(manifest: dict) -> None:
    assert manifest["kind"] in ALLOWED_KINDS
    assert manifest["host"]["provider"] in ALLOWED_PROVIDERS
    assert manifest["runtime"]["type"] in ALLOWED_RUNTIME_TYPES
    assert manifest["runtime"]["isolation"] in ALLOWED_ISOLATION
    assert manifest["lifecycle_owner"] in ALLOWED_LIFECYCLE_OWNERS
def test_host_declaration_does_not_fabricate_a_host_api_version(manifest: dict) -> None:
    host = manifest["host"]
    assert host["name"] == "vllm-ascend"
    assert Version(FROZEN_VLLM_ASCEND) in SpecifierSet(host["version_range"])
    assert "api_range" not in host, "the frozen host exposes no independent versioned plugin API"


def test_protocol_ranges_admit_the_tested_baseline(manifest: dict) -> None:
    protocols = {item["name"]: item["version_range"] for item in manifest["protocols"]}
    assert Version(FROZEN_VLLM) in SpecifierSet(protocols["vllm.general_plugins"])
    assert Version(FROZEN_VLLM_ASCEND) in SpecifierSet(
        protocols["vllm-ascend.quantization-scheme-surface"]
    )


def test_protocol_ranges_reject_a_foreign_host_line(manifest: dict) -> None:
    protocols = {item["name"]: item["version_range"] for item in manifest["protocols"]}
    assert Version("0.24.0") not in SpecifierSet(protocols["vllm-ascend.quantization-scheme-surface"])
    assert Version("0.29.0") not in SpecifierSet(protocols["vllm.general_plugins"])


def test_implementation_reference_resolves(manifest: dict) -> None:
    assert len(manifest["implementation"]) == 1
    module_name, _, attribute = manifest["components"][0]["implementation_ref"].partition(":")
    module = importlib.import_module(module_name)
    assert callable(getattr(module, attribute))


def test_components_are_unique_and_permissions_are_minimal(manifest: dict) -> None:
    component_ids = [item["component_id"] for item in manifest["components"]]
    assert len(component_ids) == len(set(component_ids))
    for component in manifest["components"]:
        assert component["permissions"], "device and weight access must be declared"
        assert set(component["permissions"]) <= ALLOWED_PERMISSIONS
        assert component["execution_planes"]
        assert component["isolation"] in ALLOWED_ISOLATION


def test_in_process_plugin_requires_no_external_service(manifest: dict) -> None:
    assert manifest["requires_services"] == []
    assert manifest["implementation"][0]["group"] == "vllm.general_plugins"


def test_activation_carries_no_deployment_specific_values(manifest: dict) -> None:
    activation = manifest["activation"]
    assert set(activation) == {"entry_points", "environment", "additional_config"}
    assert activation["environment"] == {"VLLM_ASCEND_QUANT_EXT_ENABLE": "1"}
    assert set(activation["additional_config"]) == {"_manager_runtime_qualification"}
    rendered = json.dumps(activation)
    assert "VLLM_ASCEND_QUANT_EXT_ARTIFACT" not in rendered
    assert not re.search(r"/(root|home|data|mnt)/", rendered)
    assert set(activation["entry_points"][0]) == {"group", "name"}
    assert activation["entry_points"][0]["name"] == manifest["implementation"][0]["name"]


def test_qualification_block_names_the_frozen_revisions(manifest: dict) -> None:
    from vllm_ascend_quant_ext.host_baseline import VLLM_ASCEND_HUST_COMMIT, VLLM_HUST_COMMIT

    qualification = manifest["activation"]["additional_config"]["_manager_runtime_qualification"]
    assert qualification["core_commit"] == VLLM_HUST_COMMIT
    assert qualification["ascend_platform_commit"] == VLLM_ASCEND_HUST_COMMIT
    assert qualification["accelerator"] == "ascend"
    assert qualification["status"] == "compatible_correctness_only"
    assert qualification["evidence"].startswith("docs/evidence/")


def test_manifest_contains_no_credentials(manifest: dict) -> None:
    text = json.dumps(manifest)
    for pattern in (r"sk-[A-Za-z0-9]{16,}", r"(?i)password", r"(?i)api[-_]?key", r"(?i)token"):
        assert not re.search(pattern, text), f"manifest leaks {pattern}"

    for item in manifest["implementation"]:
        assert item["type"] in ALLOWED_IMPLEMENTATION_TYPES
