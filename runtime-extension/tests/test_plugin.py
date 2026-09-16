from pathlib import Path

import pytest

from vllm_ascend_quant_ext.host_baseline import FROZEN_HOST_REVISIONS
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


def test_status_exposes_native_vllm_entry_point_and_frozen_host():
    report = status()
    assert report["entry_point"] == "vllm.general_plugins/vllm_ascend_quant"
    assert "v1/f18cf803c5" in report["host"]
    assert "74f0c0a27" in report["host"]
    assert FROZEN_HOST_REVISIONS == {
        "vllm": "gf18cf803c5",
        "vllm_ascend": "g74f0c0a27",
    }
