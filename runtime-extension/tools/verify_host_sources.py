#!/usr/bin/env python3
"""Verify checked-out host sources against the frozen v1 compatibility lock."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from vllm_ascend_quant_ext.host_baseline import (  # noqa: E402
    VLLM_ASCEND_HUST_COMMIT,
    VLLM_ASCEND_VERIFIED_CORE,
    VLLM_HUST_COMMIT,
    VLLM_HUST_FIRST_PARENT,
    VLLM_HUST_REF,
    VLLM_UPSTREAM_COMMIT,
)


class HostSourceError(RuntimeError):
    pass


def _git(source: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(source), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise HostSourceError(f"cannot inspect Git source {source}: {exc}") from exc


def _require_text(path: Path, fragments: tuple[str, ...]) -> None:
    if not path.is_file():
        raise HostSourceError(f"missing host source file: {path}")
    value = path.read_text(encoding="utf-8")
    missing = [fragment for fragment in fragments if fragment not in value]
    if missing:
        raise HostSourceError(f"host API mismatch in {path}: missing {missing}")


def verify(vllm_source: Path, ascend_source: Path) -> dict[str, object]:
    vllm_source = vllm_source.resolve()
    ascend_source = ascend_source.resolve()
    vllm_head = _git(vllm_source, "rev-parse", "HEAD")
    ascend_head = _git(ascend_source, "rev-parse", "HEAD")
    if vllm_head != VLLM_HUST_COMMIT:
        raise HostSourceError(
            f"wrong vLLM-HUST revision: {vllm_head}; required {VLLM_HUST_COMMIT}"
        )
    if ascend_head != VLLM_ASCEND_HUST_COMMIT:
        raise HostSourceError(
            "wrong vLLM-Ascend-HUST revision: "
            f"{ascend_head}; required {VLLM_ASCEND_HUST_COMMIT}"
        )

    parents = _git(vllm_source, "show", "-s", "--format=%P", "HEAD").split()
    if parents != [VLLM_HUST_FIRST_PARENT, VLLM_UPSTREAM_COMMIT]:
        raise HostSourceError(f"unexpected {VLLM_HUST_REF} parent chain: {parents}")
    verified_path = ascend_source / ".github/vllm-main-verified.commit"
    if not verified_path.is_file():
        raise HostSourceError(f"missing Ascend verified-core lock: {verified_path}")
    verified_core = verified_path.read_text(encoding="utf-8").strip()
    if verified_core != VLLM_ASCEND_VERIFIED_CORE:
        raise HostSourceError(
            f"Ascend verified core mismatch: {verified_core}; "
            f"required {VLLM_ASCEND_VERIFIED_CORE}"
        )

    _require_text(
        vllm_source / "vllm/plugins/__init__.py",
        (
            'DEFAULT_PLUGINS_GROUP = "vllm.general_plugins"',
            "def load_general_plugins()",
        ),
    )
    _require_text(
        ascend_source / "vllm_ascend/quantization/methods/__init__.py",
        (
            "AscendW8A8PDMixLinearMethod",
            '"register_scheme"',
            '"get_scheme_class"',
        ),
    )
    _require_text(
        ascend_source / "vllm_ascend/quantization/methods/w8a8/w8a8_pdmix.py",
        (
            '@register_scheme("W8A8_MIX", "linear")',
            "class AscendW8A8PDMixLinearMethod",
        ),
    )
    return {
        "valid": True,
        "vllm_hust": {"ref": VLLM_HUST_REF, "commit": vllm_head},
        "vllm_ascend_hust": {
            "commit": ascend_head,
            "verified_core": verified_core,
        },
        "runtime_entry_point": "vllm.general_plugins",
        "w8a8_surface": "AscendW8A8PDMixLinearMethod",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-source", required=True, type=Path)
    parser.add_argument("--vllm-ascend-source", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            verify(args.vllm_source, args.vllm_ascend_source),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
