from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import ContractError, validate_artifact
from .manager import load_manifest, manifest_path, plan, render
from .plugin import status


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vllm-ascend-quant-ext")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "plan", "render"):
        child = sub.add_parser(name)
        child.add_argument("--model", required=True)
    sub.add_parser("manifest")
    sub.add_parser("status")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "check":
            _print(validate_artifact(Path(args.model)))
        elif args.command == "plan":
            _print(plan(Path(args.model)))
        elif args.command == "render":
            _print(render(Path(args.model)))
        elif args.command == "manifest":
            _print({"path": str(manifest_path()), "manifest": load_manifest()})
        else:
            _print(status())
    except (ContractError, OSError, ValueError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
