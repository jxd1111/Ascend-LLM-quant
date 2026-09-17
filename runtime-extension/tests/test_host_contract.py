"""Host contract tests: default-off activation, fail-closed admission, rollback."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from vllm_ascend_quant_ext import host_baseline, plugin
from vllm_ascend_quant_ext.plugin import (
    ARTIFACT_ENV,
    ENABLE_ENV,
    is_enabled,
    register,
    status,
)

EXTENSION_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    EXTENSION_ROOT / "src" / "vllm_ascend_quant_ext" / "manifests" / "vllm-hust-extension-v0.2.json"
)
FROZEN_VLLM_ASCEND = "0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27"

DEFAULT_OFF_PROBE = """
import sys
import vllm_ascend_quant_ext.plugin as plugin

assert plugin.register() is None, "registration must be inert while disabled"
assert plugin.register() is None, "the entry point callback must be re-entrant"
for name in ("torch", "vllm", "vllm_ascend"):
    assert name not in sys.modules, f"{name} must stay unimported while disabled"
print("default-off")
"""


def test_registration_is_default_off_and_device_free() -> None:
    """Runs in a fresh interpreter: the same contract vLLM will load."""
    probe = dict(os.environ, **{ENABLE_ENV: ""})
    probe.pop(ARTIFACT_ENV, None)
    result = subprocess.run(
        [sys.executable, "-c", DEFAULT_OFF_PROBE],
        capture_output=True,
        text=True,
        env=probe,
    )
    assert result.returncode == 0, result.stderr
    assert "default-off" in result.stdout


def test_enable_switch_is_read_per_call(monkeypatch) -> None:
    monkeypatch.delenv(ENABLE_ENV, raising=False)
    assert is_enabled() is False
    monkeypatch.setenv(ENABLE_ENV, "1")
    assert is_enabled() is True
    monkeypatch.delenv(ENABLE_ENV, raising=False)
    assert is_enabled() is False, "no cached enablement may survive a new process"
    for truthy in ("1", "true", "yes", "on"):
        monkeypatch.setenv(ENABLE_ENV, truthy)
        assert is_enabled() is True
    monkeypatch.setenv(ENABLE_ENV, "0")
    assert is_enabled() is False


def test_enabled_registration_requires_an_artifact(monkeypatch) -> None:
    monkeypatch.setenv(ENABLE_ENV, "1")
    monkeypatch.delenv(ARTIFACT_ENV, raising=False)
    with pytest.raises(RuntimeError, match=ARTIFACT_ENV):
        register()


def test_missing_artifact_fails_closed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(ENABLE_ENV, "1")
    monkeypatch.setenv(ARTIFACT_ENV, str(tmp_path / "does-not-exist"))
    with pytest.raises(RuntimeError, match="admission failed"):
        register()


def test_status_reports_the_frozen_host_and_entry_point() -> None:
    report = status()
    assert report["enabled"] is False
    assert report["enable_environment_variable"] == ENABLE_ENV
    assert report["artifact_environment_variable"] == ARTIFACT_ENV
    assert report["entry_point"] == "vllm.general_plugins/vllm_ascend_quant"
    assert report["host"] == host_baseline.host_description()


def test_manifest_and_runtime_agree_on_the_frozen_revisions() -> None:
    """The manifest may not claim a wider host than the runtime gate enforces."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    qualification = manifest["activation"]["additional_config"]["_manager_runtime_qualification"]
    assert qualification["core_commit"].startswith(host_baseline.FROZEN_HOST_REVISIONS["vllm"][1:])
    assert qualification["ascend_platform_commit"].startswith(
        host_baseline.FROZEN_HOST_REVISIONS["vllm_ascend"][1:]
    )
    assert Version(FROZEN_VLLM_ASCEND) in SpecifierSet(manifest["host"]["version_range"])


def test_plugin_exposes_no_hot_unload_api() -> None:
    for name in ("unregister", "unregister_schemes", "disable", "rollback", "stop"):
        assert not hasattr(plugin, name), "the vLLM host owns process lifecycle; no hot unload"


def test_only_the_two_declared_entry_point_groups_are_claimed() -> None:
    text = (EXTENSION_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    groups = sorted(re.findall(r'\[project\.entry-points\."([^"]+)"\]', text))
    assert groups == ["vllm.general_plugins", "vllm_hust.extension_bundles"]
