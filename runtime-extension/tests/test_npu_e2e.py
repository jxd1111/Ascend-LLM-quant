"""Safety invariants of the frozen-host NPU gate driver.

The gate itself needs an Ascend host, so these tests pin the properties that
must hold *before* it is ever pointed at hardware: the wheel is the only import
source, the previous install is restored, the device is isolated, and no host
path is hard-coded.
"""

from __future__ import annotations

import importlib.util
import json
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


ONLINE_ENV = "VLLM_ASCEND_QUANT_EXT_SMOOTHING"
ARTIFACT_ENV = "VLLM_ASCEND_QUANT_EXT_ARTIFACT"
ATTACHMENT_MANIFEST = "ascend_quant_smoothing.json"


def online_argv(
    tool,
    tmp_path,
    *,
    description: bool = True,
    paired: bool = True,
    port: str = "18003",
    verified: bool = True,
    manifest: bool = True,
) -> tuple[list[str], Path, Path]:
    """Build a dry-run gate that also serves a generated ONLINE view."""

    wheel = tmp_path / "vllm_ascend_quant_ext-0.0.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    artifact = tmp_path / "artifact"
    artifact.mkdir(exist_ok=True)
    view = tmp_path / "view"
    view.mkdir(exist_ok=True)
    if description:
        (view / "quant_model_description.json").write_text("{}\n", encoding="utf-8")
    attachment = tmp_path / "attachment"
    attachment.mkdir(exist_ok=True)
    if manifest:
        (attachment / ATTACHMENT_MANIFEST).write_text(
            json.dumps({"verified": verified}) + "\n", encoding="utf-8"
        )
    argv = [
        "--wheel",
        str(wheel),
        "--artifact",
        str(artifact),
        "--online-view",
        str(view),
        "--port",
        port,
    ]
    if paired:
        argv += ["--smoothing", str(attachment)]
    argv += ["--dry-run"]
    return argv, view, attachment


def test_offline_plan_never_selects_the_attachment(plan: str) -> None:
    # The offline artifact carries its own factors; a leftover smoothing pointer
    # would make the two groups indistinguishable.
    assert ONLINE_ENV not in plan
    assert f"{ARTIFACT_ENV}=" in plan


def test_online_view_and_smoothing_are_paired(tool, tmp_path) -> None:
    argv, _, _ = online_argv(tool, tmp_path, paired=False)
    with pytest.raises(SystemExit) as excinfo:
        tool.main(argv)
    assert "pass both or neither" in str(excinfo.value)


def test_online_view_without_a_description_is_refused(tool, tmp_path) -> None:
    argv, _, _ = online_argv(tool, tmp_path, description=False)
    with pytest.raises(SystemExit) as excinfo:
        tool.main(argv)
    assert "quant_model_description.json" in str(excinfo.value)


def test_online_port_defaults_next_to_the_offline_port(tool, tmp_path, capsys) -> None:
    argv, view, attachment = online_argv(tool, tmp_path, port="18003")
    assert tool.main(argv) == 0
    printed = capsys.readouterr().out
    assert "--port 18004" in printed
    online_line = next(line for line in printed.splitlines() if f"serve {view}" in line)
    assert f"{ONLINE_ENV}={attachment}" in online_line
    # The ONLINE group resolves its factors from the attachment, never from the
    # offline artifact environment variable.
    assert f"{ARTIFACT_ENV}=" not in online_line


def test_online_environment_swaps_the_artifact_for_the_attachment(tool, tmp_path) -> None:
    argv, _, attachment = online_argv(tool, tmp_path)
    args = tool.build_parser().parse_args([item for item in argv if item != "--dry-run"])
    args.online_port = args.port + 1
    environment = tool._online_env(args, None)
    assert environment[ONLINE_ENV] == str(attachment)
    assert ARTIFACT_ENV not in environment
    assert environment["VLLM_ASCEND_QUANT_EXT_ENABLE"] == "1"


def test_online_plan_keeps_the_offline_phases(tool, tmp_path) -> None:
    argv, view, _ = online_argv(tool, tmp_path)
    args = tool.build_parser().parse_args([item for item in argv if item != "--dry-run"])
    args.online_port = args.port + 1
    lines = "\n".join(tool.describe_plan(args, sys.executable, tmp_path / "s.log"))
    assert f"serve {args.artifact}" in lines
    assert f"serve {view}" in lines
    assert "5b. serve the ONLINE view" in lines
    assert "stop the server and prove every entry point vanished" in lines


def test_online_attachment_without_a_manifest_is_refused(tool, tmp_path) -> None:
    argv, _, _ = online_argv(tool, tmp_path, manifest=False)
    with pytest.raises(SystemExit) as excinfo:
        tool.main(argv)
    assert ATTACHMENT_MANIFEST in str(excinfo.value)


def test_online_attachment_the_producer_did_not_verify_is_refused(tool, tmp_path) -> None:
    # The runtime refuses an attachment whose producer did not verify it, so the
    # gate must not spend device time discovering the same refusal.
    argv, _, _ = online_argv(tool, tmp_path, verified=False)
    with pytest.raises(SystemExit) as excinfo:
        tool.main(argv)
    assert "not producer-verified" in str(excinfo.value)


def test_online_attachment_verified_by_its_producer_passes(tool, tmp_path) -> None:
    argv, _, attachment = online_argv(tool, tmp_path)
    assert tool.main(argv) == 0
    assert tool._attachment_verified(attachment)


def _entry_point(name: str, value: str, distribution_name: str | None):
    from types import SimpleNamespace

    dist = SimpleNamespace(name=distribution_name) if distribution_name else None
    return SimpleNamespace(name=name, value=value, dist=dist)


def _leftovers(tool, monkeypatch, capsys, groups: dict[str, list]) -> list[str]:
    import builtins
    import importlib.metadata as metadata

    monkeypatch.setattr(metadata, "entry_points", lambda group=None: list(groups[group]))
    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "vllm_ascend_quant_ext":
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    exec(compile(tool.UNINSTALL_PROBE, "<uninstall-probe>", "exec"), {})
    return json.loads(capsys.readouterr().out.strip().removeprefix("PROBE "))["leftovers"]


def test_probes_judge_only_the_distribution_under_test(tool) -> None:
    # The frozen host shares both entry-point groups with unrelated packages
    # (``vllm_ascend_simllm`` ships its own bundle), so an environment-wide
    # comparison would blame this gate for somebody else's entry points.
    namespace: dict[str, object] = {}
    exec(compile(tool.DIST_SCOPE, "<dist-scope>", "exec"), namespace)
    ours = namespace["ours"]
    assert ours(_entry_point("bundle", "locator", "vllm_ascend_quant_ext"))
    assert ours(_entry_point("bundle", "locator", "vllm-ascend-quant-ext"))
    assert not ours(_entry_point("bundle", "locator", "vllm-ascend-simllm"))
    assert not ours(_entry_point("bundle", "locator", None))


def test_uninstall_probe_ignores_foreign_bundles(tool, monkeypatch, capsys) -> None:
    ours_general = _entry_point(
        "vllm_ascend_quant", "vllm_ascend_quant_ext.plugin:register", "vllm_ascend_quant_ext"
    )
    ours_bundle = _entry_point(
        "org.vllm-hust.ascend-quant-runtime", "vllm_ascend_quant_ext.manifests", "vllm_ascend_quant_ext"
    )
    foreign_general = _entry_point("simllm_plugin", "vllm_ascend_simllm:register", "vllm-ascend-simllm")
    foreign_bundle = _entry_point("org.vllm-hust.simllm", "vllm_ascend_simllm", "vllm-ascend-simllm")

    installed = {
        "vllm.general_plugins": [ours_general, foreign_general],
        "vllm_hust.extension_bundles": [ours_bundle, foreign_bundle],
    }
    assert _leftovers(tool, monkeypatch, capsys, installed) == [
        "org.vllm-hust.ascend-quant-runtime",
        "vllm_ascend_quant",
    ]

    # After a real uninstall only the host's own bundles may remain: they are not
    # this extension's leftovers and must not fail the phase.
    uninstalled = {
        "vllm.general_plugins": [foreign_general],
        "vllm_hust.extension_bundles": [foreign_bundle],
    }
    assert _leftovers(tool, monkeypatch, capsys, uninstalled) == []
