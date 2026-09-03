"""Command-line interface for reproducible Ascend quantization workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifact import (
    inspect_artifact,
    prepare_runtime_metadata,
    restore_modelslim_metadata,
    write_manifest,
    write_runtime_contract,
)
from .doctor import collect_checks
from .evidence import load_evidence
from .modelslim import QuantizeRequest, run_quantization, validate_request
from .registry import get_recipe, list_recipes


def _request_from_args(args: argparse.Namespace) -> QuantizeRequest:
    return QuantizeRequest(
        recipe=get_recipe(args.recipe),
        model_path=Path(args.model).resolve(),
        output_path=Path(args.output).resolve(),
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        executable=args.executable,
    )


def command_list(_: argparse.Namespace) -> int:
    for recipe in list_recipes():
        print(f"{recipe.name}\t{recipe.description}")
    return 0


def command_plan(args: argparse.Namespace) -> int:
    request = _request_from_args(args)
    validate_request(request)
    print(request.rendered_command())
    return 0


def command_quantize(args: argparse.Namespace) -> int:
    request = _request_from_args(args)
    print(f"Recipe: {request.recipe.name}", flush=True)
    print(f"Command: {request.rendered_command()}", flush=True)
    return_code = run_quantization(request)
    if return_code != 0:
        print(f"msModelSlim exited with code {return_code}", file=sys.stderr)
        return return_code
    migration = prepare_runtime_metadata(request.output_path, request.recipe)
    print(
        f"Runtime metadata: {migration['runtime_quant_type']} "
        f"({migration['replacements']} replacements)",
        flush=True,
    )
    inspection = inspect_artifact(request.output_path)
    if not inspection["valid"]:
        print(json.dumps(inspection, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    destination = write_manifest(
        request.output_path,
        request.recipe,
        source_model=request.model_path,
    )
    print(f"Manifest: {destination}")
    contract = write_runtime_contract(request.output_path, request.recipe)
    print(f"Runtime contract: {contract}")
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    model_path = Path(args.model).resolve()
    inspection = inspect_artifact(model_path)
    if args.write_manifest:
        recipe = get_recipe(args.recipe)
        destination = write_manifest(model_path, recipe)
        inspection["manifest"] = str(destination)
    print(json.dumps(inspection, ensure_ascii=False, indent=2))
    return 0 if inspection["valid"] else 2


def command_doctor(args: argparse.Namespace) -> int:
    model_path = Path(args.path).resolve() if args.path else None
    checks = collect_checks(model_path)
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0 if checks["ready_for_modelslim"] else 2


def command_prepare_runtime(args: argparse.Namespace) -> int:
    model_path = Path(args.model).resolve()
    recipe = get_recipe(args.recipe)
    result = prepare_runtime_metadata(model_path, recipe)
    source_model = Path(args.source_model).resolve() if args.source_model else None
    destination = write_manifest(model_path, recipe, source_model=source_model)
    contract = write_runtime_contract(model_path, recipe)
    result["manifest"] = str(destination)
    result["runtime_contract"] = str(contract)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_restore_modelslim(args: argparse.Namespace) -> int:
    model_path = Path(args.model).resolve()
    recipe = get_recipe(args.recipe)
    result = restore_modelslim_metadata(model_path, recipe)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_export_contract(args: argparse.Namespace) -> int:
    model_path = Path(args.model).resolve()
    recipe = get_recipe(args.recipe)
    destination = write_runtime_contract(
        model_path,
        recipe,
        evidence_level=args.evidence_level,
        verified_profiles=args.verified_profile,
        evidence_results=args.evidence_result,
    )
    print(destination)
    return 0


def command_validate_evidence(args: argparse.Namespace) -> int:
    evidence = load_evidence(Path(args.file).resolve())
    print(json.dumps({"valid": True, "run_id": evidence["run_id"], "profile": evidence["profile"]}, ensure_ascii=False, indent=2))
    return 0


def _add_quant_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--executable", default="msmodelslim")
    parser.add_argument("--trust-remote-code", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jxd-quant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-recipes", help="List discovered recipes")
    list_parser.set_defaults(handler=command_list)

    plan_parser = subparsers.add_parser("plan", help="Validate and print the msModelSlim command")
    _add_quant_args(plan_parser)
    plan_parser.set_defaults(handler=command_plan)

    quantize_parser = subparsers.add_parser("quantize", help="Run an offline quantization recipe")
    _add_quant_args(quantize_parser)
    quantize_parser.set_defaults(handler=command_quantize)

    inspect_parser = subparsers.add_parser("inspect-model", help="Validate a quantized model artifact")
    inspect_parser.add_argument("--model", required=True)
    inspect_parser.add_argument("--write-manifest", action="store_true")
    inspect_parser.add_argument("--recipe", default="qwen25-w8a8-pdmix")
    inspect_parser.set_defaults(handler=command_inspect)

    prepare_parser = subparsers.add_parser(
        "prepare-runtime",
        help="Convert ModelSlim metadata to the plugin-owned runtime type",
    )
    prepare_parser.add_argument("--model", required=True)
    prepare_parser.add_argument("--recipe", default="qwen25-w8a8-pdmix")
    prepare_parser.add_argument("--source-model")
    prepare_parser.set_defaults(handler=command_prepare_runtime)

    restore_parser = subparsers.add_parser(
        "restore-modelslim",
        help="Restore the backed-up ModelSlim quantization description",
    )
    restore_parser.add_argument("--model", required=True)
    restore_parser.add_argument("--recipe", default="qwen25-w8a8-pdmix")
    restore_parser.set_defaults(handler=command_restore_modelslim)

    doctor_parser = subparsers.add_parser("doctor", help="Inspect the active Python environment")
    doctor_parser.add_argument("--path", help="Path whose filesystem capacity should be checked")
    doctor_parser.set_defaults(handler=command_doctor)

    contract_parser = subparsers.add_parser(
        "export-contract",
        help="Write the versioned runtime artifact contract during offline preparation",
    )
    contract_parser.add_argument("--model", required=True)
    contract_parser.add_argument("--recipe", default="qwen25-w8a8-pdmix")
    contract_parser.add_argument(
        "--evidence-level",
        choices=("schema_only", "correctness", "npu_e2e", "matched_benchmark"),
        default="schema_only",
    )
    contract_parser.add_argument(
        "--verified-profile",
        action="append",
        choices=("BF16", "W8A8", "W4A4", "W4A8"),
        default=[],
    )
    contract_parser.add_argument("--evidence-result", action="append", default=[])
    contract_parser.set_defaults(handler=command_export_contract)

    evidence_parser = subparsers.add_parser(
        "validate-evidence",
        help="Validate a closed BF16/W8A8/W4A4/W4A8 matched-result record",
    )
    evidence_parser.add_argument("--file", required=True)
    evidence_parser.set_defaults(handler=command_validate_evidence)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.handler(args))
    except (KeyError, ValueError, OSError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
