"""Packaging tests: manifest shipping, retired artifacts, repo-level declaration."""

from __future__ import annotations

import importlib.util
import json
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from vllm_ascend_quant_ext import __version__

EXTENSION_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = EXTENSION_ROOT / "src" / "vllm_ascend_quant_ext"
MANIFEST_RELATIVE = "vllm_ascend_quant_ext/manifests/vllm-hust-extension-v0.2.json"
MANIFEST_PATH = EXTENSION_ROOT / "src" / MANIFEST_RELATIVE
PYPROJECT_PATH = EXTENSION_ROOT / "pyproject.toml"
MANIFEST_IN_PATH = EXTENSION_ROOT / "MANIFEST.in"
VERIFY_RELEASE_PATH = EXTENSION_ROOT / "tools" / "verify_release.py"
OPTIMIZATION_PATH = EXTENSION_ROOT.parent / ".vllm-hust" / "optimization.json"
DIST_DIR = EXTENSION_ROOT / "dist"

BUNDLE_LOCATOR = "vllm_ascend_quant_ext.manifests"
EXPECTED_EXTENSION_ID = "org.vllm-hust.ascend-quant-runtime"
RUNTIME_ENTRY_POINT = "vllm_ascend_quant_ext.plugin:register"


def load_release_verifier():
    spec = importlib.util.spec_from_file_location("verify_release_under_test", VERIFY_RELEASE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def verifier():
    return load_release_verifier()


def test_pyproject_declares_package_data_bundle_locator_and_dev_extra() -> None:
    text = PYPROJECT_PATH.read_text(encoding="utf-8")
    assert "[tool.setuptools.package-data]" in text
    assert 'vllm_ascend_quant_ext = ["manifests/*.json"]' in text
    assert f'"{EXPECTED_EXTENSION_ID}" = "{BUNDLE_LOCATOR}"' in text
    assert "[project.optional-dependencies]" in text
    assert '[project.entry-points."vllm.general_plugins"]' in text


def test_source_tree_carries_the_manifest_locator_package() -> None:
    assert MANIFEST_PATH.is_file()
    assert (PACKAGE_ROOT / "manifests" / "__init__.py").is_file()
    directives = MANIFEST_IN_PATH.read_text(encoding="utf-8")
    assert "recursive-include src/vllm_ascend_quant_ext/manifests *.json" in directives


def test_retired_artifacts_are_absent_from_the_source_tree() -> None:
    assert not (PACKAGE_ROOT / "extension-manifest.json").exists()
    assert not (PACKAGE_ROOT / "schemes" / "w8a8_pdmix.py").exists()


def test_release_verifier_constants_match_the_shipped_manifest(verifier) -> None:
    assert verifier.MANIFEST == MANIFEST_RELATIVE
    assert verifier.BUNDLE_ID == EXPECTED_EXTENSION_ID
    assert verifier.BUNDLE_GROUP == "vllm_hust.extension_bundles"
    assert verifier.ENTRY_NAME == "vllm_ascend_quant"
    assert verifier.LEGACY_MANIFEST == "vllm_ascend_quant_ext/extension-manifest.json"
    assert (EXTENSION_ROOT / "src" / verifier.MANIFEST).is_file()
    assert verifier.MANIFEST.endswith("/vllm-hust-extension-v0.2.json")


def test_repository_declaration_matches_the_distribution() -> None:
    document = json.loads(OPTIMIZATION_PATH.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["source_subdir"] == "runtime-extension"
    assert document["entrypoint"] == {"group": "vllm.general_plugins", "name": "vllm_ascend_quant"}
    assert document["activation"]["environment"] == {"VLLM_ASCEND_QUANT_EXT_ENABLE": "1"}
    assert "ascend" in document["activation"]["vllm_plugins"]
    assert document["compatibility"]["model_qualifications"][0]["status"] == "compatible"


def test_distribution_bundle_locator_resolves_the_manifest(
    installed_extension_versions: list[str],
) -> None:
    """Editable and wheel installs both resolve the locator through metadata."""
    versions = installed_extension_versions
    if not versions:
        pytest.skip("vllm-ascend-quant-ext is not installed in this interpreter")
    assert versions == [__version__], (
        f"ambiguous installed metadata for vllm-ascend-quant-ext: {versions}; "
        "remove stale build artifacts (src/*.egg-info) and reinstall"
    )

    matching = [
        item
        for item in entry_points(group="vllm_hust.extension_bundles")
        if item.name == EXPECTED_EXTENSION_ID
    ]
    assert len(matching) == 1
    module = importlib.import_module(matching[0].value)
    manifest_path = Path(module.__file__).resolve().parent / "vllm-hust-extension-v0.2.json"
    assert manifest_path.is_file()
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert document["extension_id"] == matching[0].name
    assert document["extension_version"] == versions[0]


def built_distribution(pattern: str) -> Path:
    matches = sorted(DIST_DIR.glob(pattern))
    if not matches:
        pytest.skip(f"no {pattern} present; run `python -m build` before this check")
    return matches[-1]


def test_built_wheel_passes_the_release_verifier(verifier) -> None:
    verifier._check_wheel(built_distribution(f"vllm_ascend_quant_ext-{__version__}-*.whl"))


def test_built_sdist_passes_the_release_verifier(verifier) -> None:
    verifier._check_sdist(built_distribution(f"vllm_ascend_quant_ext-{__version__}.tar.gz"))
