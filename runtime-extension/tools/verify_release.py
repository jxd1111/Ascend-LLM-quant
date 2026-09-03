#!/usr/bin/env python3
"""Verify wheel contents, isolated discovery, uninstall, and model immutability."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path

DIST_NAME = "vllm-ascend-quant-ext"
ENTRY_GROUP = "vllm.general_plugins"
ENTRY_NAME = "vllm_ascend_quant"


def _run(*args: str) -> str:
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout


def _model_snapshot(model: Path | None) -> dict[str, object] | None:
    if model is None:
        return None
    selected = [
        model / "ascend_quant_artifact.json",
        model / "quant_model_description.json",
        model / "config.json",
    ]
    files: dict[str, object] = {}
    for path in selected:
        if path.is_file():
            files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(model.glob("*.safetensors")):
        stat = path.stat()
        files[path.name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return files


def _check_wheel(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    suffixes = {
        "vllm_ascend_quant_ext/extension-manifest.json",
        "vllm_ascend_quant_ext/schemes/w8a8_pdmix.py",
        ".dist-info/entry_points.txt",
        ".data/data/share/vllm-hust/extensions/vllm-ascend-quant/extension-manifest.json",
    }
    missing = [suffix for suffix in suffixes if not any(name.endswith(suffix) for name in names)]
    if missing:
        raise RuntimeError(f"wheel is incomplete: {missing}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    model = args.model.resolve() if args.model else None
    _check_wheel(wheel)
    before = _model_snapshot(model)

    with tempfile.TemporaryDirectory(prefix="quant-ext-release-") as temporary:
        env_dir = Path(temporary) / "venv"
        venv.EnvBuilder(with_pip=True).create(env_dir)
        python = env_dir / "bin/python"
        pip = env_dir / "bin/pip"
        _run(str(pip), "install", "--no-deps", str(wheel))
        probe = r'''
import json
import sysconfig
from importlib.metadata import entry_points
from importlib.resources import files
from pathlib import Path

package_manifest = files("vllm_ascend_quant_ext").joinpath("extension-manifest.json")
manifest = json.loads(package_manifest.read_text(encoding="utf-8"))
installed_manifest = Path(sysconfig.get_path("data")) / "share/vllm-hust/extensions/vllm-ascend-quant/extension-manifest.json"
points = [ep for ep in entry_points(group="vllm.general_plugins") if ep.name == "vllm_ascend_quant"]
assert manifest["extension_id"] == "vllm-ascend-quant"
assert installed_manifest.is_file()
assert len(points) == 1
print(json.dumps({"manifest": str(installed_manifest), "entry_point": points[0].value}))
'''
        print(_run(str(python), "-c", probe).strip())
        _run(str(pip), "uninstall", "-y", DIST_NAME)
        uninstall_probe = r'''
from importlib.metadata import entry_points
assert not [ep for ep in entry_points(group="vllm.general_plugins") if ep.name == "vllm_ascend_quant"]
'''
        _run(str(python), "-c", uninstall_probe)

    after = _model_snapshot(model)
    if before != after:
        raise RuntimeError("model metadata changed during install/uninstall verification")
    print(json.dumps({"valid": True, "wheel": str(wheel), "model_unchanged": before == after}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
