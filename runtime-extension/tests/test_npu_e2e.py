"""Safety invariants of the frozen-host NPU gate driver.

The gate itself needs an Ascend host, so these tests pin the properties that
must hold *before* it is ever pointed at hardware: the wheel is the only import
source, the previous install is restored, the device is isolated, and no host
path is hard-coded.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

EXTENSION_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = EXTENSION_ROOT / "tools" / "npu_e2e.py"
VERIFY_RELEASE_PATH = EXTENSION_ROOT / "tools" / "verify_release.py"
SOURCE_TREE = EXTENSION_ROOT / "src"

PROBES = ("SHIM", "DIST_PROBE", "RESOLUTION_PROBE", "LOADER_PROBE", "LEGACY_PROBE", "UNINSTALL_PROBE")


def load_tool():
    spec = importlib.util.spec_from_file_location("npu_e2e_under_test", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return load_tool()


@pytest.fixture
def plan(tool, tmp_path):
    wheel = tmp_path / "vllm_ascend_quant_ext-0.0.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    argv = [
        "--wheel",
        str(wheel),
        "--artifact",
        str(artifact),
        "--legacy-artifact",
        str(legacy),
        "--python",
        sys.executable,
        "--device",
        "6",
        "--port",
        "18003",
        "--env-script",
        "/opt/host/set_env.sh",
        "--host-source",
        "/opt/host/vllm-hust",
        "--dry-run",
    ]
    return "\n".join(tool.describe_plan(tool.build_parser().parse_args(argv), sys.executable, tmp_path / "s.log"))


def test_plan_never_imports_the_extension_from_the_source_tree(plan: str) -> None:
    assert "--force-reinstall" in plan
    assert "site-packages" in plan
    assert str(SOURCE_TREE) not in plan
    assert "/opt/host/vllm-hust" in plan


def test_plan_does_not_narrow_vllm_plugins(plan: str) -> None:
    # The frozen host needs its own platform and general plugins; narrowing
    # VLLM_PLUGINS by hand is forbidden in this repository.
    assert "VLLM_PLUGINS" not in plan


def test_plan_isolates_the_device_and_pins_the_preflight(plan: str) -> None:
    assert "ASCEND_RT_VISIBLE_DEVICES=6" in plan
    assert "VLLM_ASCEND_TORCH_PREFLIGHT=0" in plan
    assert "VLLM_ASCEND_QUANT_EXT_ENABLE=1" in plan
    assert "VLLM_ASCEND_QUANT_EXT_ARTIFACT=" in plan


def test_plan_uninstalls_and_restores_the_previous_install(plan: str) -> None:
    assert "pip uninstall -y vllm-ascend-quant-ext" in plan
    assert "--no-build-isolation" in plan
    assert "-e" in plan
    assert "compare every artifact file hash" in plan


def test_plan_serves_the_verified_flags(plan: str) -> None:
    assert "serve" in plan
    assert "--served-model-name w8a8-plugin-v1" in plan
    assert "--enforce-eager" in plan
    assert "--port 18003" in plan


def test_source_tree_on_pythonpath_is_refused(tool, tmp_path) -> None:
    wheel = tmp_path / "vllm_ascend_quant_ext-0.0.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    with pytest.raises(SystemExit) as excinfo:
        tool.main(
            [
                "--wheel",
                str(wheel),
                "--artifact",
                str(artifact),
                "--host-source",
                str(SOURCE_TREE),
                "--dry-run",
            ]
        )
    assert "must not be on PYTHONPATH" in str(excinfo.value)


def test_missing_arguments_fail_closed(tool) -> None:
    with pytest.raises(SystemExit):
        tool.main([])


def test_missing_wheel_fails_closed(tool, tmp_path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    with pytest.raises(SystemExit) as excinfo:
        tool.main(["--wheel", str(tmp_path / "absent.whl"), "--artifact", str(artifact), "--dry-run"])
    assert "wheel not found" in str(excinfo.value)


def test_probe_programs_compile(tool) -> None:
    for name in PROBES:
        compile(getattr(tool, name), name, "exec")


def test_probe_without_a_record_is_rejected(tool) -> None:
    with pytest.raises(ValueError):
        tool._probe("import json\nprint('nothing')\n")
    assert tool._probe(tool.LEGACY_PROBE) == tool.LEGACY_PROBE


def test_constants_agree_with_the_release_verifier(tool) -> None:
    spec = importlib.util.spec_from_file_location("verify_release_under_test", VERIFY_RELEASE_PATH)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    assert tool.DIST_NAME == verifier.DIST_NAME
    assert tool.ENTRY_NAME == verifier.ENTRY_NAME
    assert tool.BUNDLE_ID == verifier.BUNDLE_ID
    assert tool.BUNDLE_GROUP == verifier.BUNDLE_GROUP
    assert tool.MANIFEST == verifier.MANIFEST
    assert (SOURCE_TREE / tool.MANIFEST).is_file()
