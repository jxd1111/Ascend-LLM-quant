from pathlib import Path

import pytest

from vllm_ascend_quant_ext.plugin import register, status


def test_plugin_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_QUANT_EXT_ENABLE", raising=False)
    monkeypatch.delenv("VLLM_ASCEND_QUANT_EXT_ARTIFACT", raising=False)
    assert register() is None


def test_enabled_plugin_requires_artifact(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_QUANT_EXT_ENABLE", "1")
    monkeypatch.delenv("VLLM_ASCEND_QUANT_EXT_ARTIFACT", raising=False)
    with pytest.raises(RuntimeError, match="ARTIFACT"):
        register()


def test_invalid_artifact_fails_before_runtime_import(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VLLM_ASCEND_QUANT_EXT_ENABLE", "1")
    monkeypatch.setenv("VLLM_ASCEND_QUANT_EXT_ARTIFACT", str(tmp_path))
    with pytest.raises(RuntimeError, match="admission failed"):
        register()


def test_status_exposes_bundle_and_runtime_entry_points():
    report = status()
    assert report["entry_point"] == "vllm.general_plugins/vllm_ascend_quant"
    assert report["bundle_entry_point"] == (
        "vllm_hust.extension_bundles/org.vllm-hust.ascend-quant"
    )
