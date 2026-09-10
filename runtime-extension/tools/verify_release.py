#!/usr/bin/env python3
"""Verify wheel contents, isolated discovery, uninstall, and model immutability."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path

DIST_NAME = "vllm-ascend-quant-ext"
ENTRY_GROUP = "vllm.general_plugins"
ENTRY_NAME = "vllm_ascend_quant"
BUNDLE_GROUP = "vllm_hust.extension_bundles"
BUNDLE_ID = "org.vllm-hust.ascend-quant-runtime"
MANIFEST = "vllm_ascend_quant_ext/manifests/vllm-hust-extension-v0.2.json"
LEGACY_MANIFEST = "vllm_ascend_quant_ext/extension-manifest.json"
RETIRED_SCHEME = "vllm_ascend_quant_ext/schemes/w8a8_pdmix.py"


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
        "vllm_ascend_quant_ext/_version.py",
        "vllm_ascend_quant_ext/manifests/__init__.py",
        MANIFEST,
        "vllm_ascend_quant_ext/adapters/vllm_hust/runtime.py",
        "vllm_ascend_quant_ext/schemes/w8a8.py",
        ".dist-info/entry_points.txt",
    }
    missing = [suffix for suffix in suffixes if not any(name.endswith(suffix) for name in names)]
    if missing:
        raise RuntimeError(f"wheel is incomplete: {missing}")
    if any(name.endswith(LEGACY_MANIFEST) for name in names):
        raise RuntimeError("wheel contains the retired legacy extension manifest")
    if any(name.endswith(RETIRED_SCHEME) for name in names):
        raise RuntimeError("wheel contains the retired PDMix-named scheme module")

    with zipfile.ZipFile(wheel) as archive:
        entry_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/entry_points.txt")
        )
        entries = archive.read(entry_name).decode("utf-8")
    if f"[{BUNDLE_GROUP}]" not in entries or f"{BUNDLE_ID} = vllm_ascend_quant_ext.manifests" not in entries:
        raise RuntimeError("wheel does not declare the Extension Bundle entry point")
    if f"[{ENTRY_GROUP}]" not in entries or f"{ENTRY_NAME} = vllm_ascend_quant_ext.plugin:register" not in entries:
        raise RuntimeError("wheel does not declare the vLLM runtime activation entry point")


def _check_sdist(sdist: Path) -> None:
    with tarfile.open(sdist, "r:gz") as archive:
        names = archive.getnames()
    suffixes = {
        "pyproject.toml",
        "src/vllm_ascend_quant_ext/_version.py",
        f"src/{MANIFEST}",
        "src/vllm_ascend_quant_ext/adapters/vllm_hust/runtime.py",
    }
    missing = [suffix for suffix in suffixes if not any(name.endswith(suffix) for name in names)]
    if missing:
        raise RuntimeError(f"sdist is incomplete: {missing}")
    if any(name.endswith(LEGACY_MANIFEST) for name in names):
        raise RuntimeError("sdist contains the retired legacy extension manifest")
    if any(name.endswith(RETIRED_SCHEME) for name in names):
        raise RuntimeError("sdist contains the retired PDMix-named scheme module")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--sdist", required=True, type=Path)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    sdist = args.sdist.resolve()
    model = args.model.resolve() if args.model else None
    _check_wheel(wheel)
    _check_sdist(sdist)
    before = _model_snapshot(model)

    with tempfile.TemporaryDirectory(prefix="quant-ext-release-") as temporary:
        env_dir = Path(temporary) / "venv"
        venv.EnvBuilder(with_pip=True).create(env_dir)
        python = env_dir / "bin/python"
        pip = env_dir / "bin/pip"
        _run(str(pip), "install", "--no-deps", str(wheel))
        probe = r'''
import json
import sys
from importlib.metadata import entry_points, version
from importlib.resources import files

package_manifest = files("vllm_ascend_quant_ext.manifests").joinpath("vllm-hust-extension-v0.2.json")
manifest = json.loads(package_manifest.read_text(encoding="utf-8"))
points = [ep for ep in entry_points(group="vllm.general_plugins") if ep.name == "vllm_ascend_quant"]
bundles = [ep for ep in entry_points(group="vllm_hust.extension_bundles") if ep.name == "org.vllm-hust.ascend-quant-runtime"]
assert manifest["extension_id"] == "org.vllm-hust.ascend-quant-runtime"
assert manifest["extension_version"] == version("vllm-ascend-quant-ext")
assert manifest["implementation"][0]["status"] == "import_only"
assert len(points) == 1
assert len(bundles) == 1
assert bundles[0].value == "vllm_ascend_quant_ext.manifests"
assert manifest["activation"] == {"entry_points": [], "environment": {}, "additional_config": {}}
assert "torch" not in sys.modules
assert "vllm" not in sys.modules
assert "vllm_ascend" not in sys.modules
print(json.dumps({"manifest": str(package_manifest), "runtime_entry_point": points[0].value, "bundle_entry_point": bundles[0].value}))
'''
        print(_run(str(python), "-c", probe).strip())
        _run(str(pip), "uninstall", "-y", DIST_NAME)
        uninstall_probe = r'''
from importlib.metadata import entry_points
assert not [ep for ep in entry_points(group="vllm.general_plugins") if ep.name == "vllm_ascend_quant"]
assert not [ep for ep in entry_points(group="vllm_hust.extension_bundles") if ep.name == "org.vllm-hust.ascend-quant-runtime"]
'''
        _run(str(python), "-c", uninstall_probe)

    after = _model_snapshot(model)
    if before != after:
        raise RuntimeError("model metadata changed during install/uninstall verification")
    print(json.dumps({"valid": True, "wheel": str(wheel), "sdist": str(sdist), "model_unchanged": before == after}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
