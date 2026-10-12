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
from .attachment import (
    ATTACHMENT_FILENAME,
    DEFAULT_QUANT_TYPE,
    SmoothingRequest,
    build_attachment,
    build_online_view,
    validate_attachment,
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


def command_build_smoothing(args: argparse.Namespace) -> int:
    request = SmoothingRequest(
        model_path=Path(args.model).resolve(),
        artifact_path=Path(args.artifact).resolve(),
        output_path=Path(args.output).resolve(),
        quant_type=args.quant_type,
        denom=args.denom,
        mag_frac=args.mag_frac,
        em_rounds=args.em_rounds,
        min_match_rate=args.min_match_rate,
        max_iterator_diff=args.max_iterator_diff,
        prefixes=tuple(args.prefix or ()),
    )
    every = max(int(args.progress_every), 0)

    def emit(done: int, total: int, prefix: str) -> None:
        if every and (done % every == 0 or done == total):
            print(f"[{done}/{total}] {prefix}", file=sys.stderr, flush=True)

    manifest = build_attachment(request, progress=emit)
    summary = manifest["summary"]
    print(f"Attachment: {request.output_path / ATTACHMENT_FILENAME}", flush=True)
    print(
        f"Replayed {summary['linears_replayed']}/{summary['linears_declared']} linears, "
        f"min INT8 match {summary['min_int8_match_rate'] * 100:.3f}%, "
        f"max diff {summary['max_iterator_diff']} LSB, "
        f"norms {summary['norms_recovered']}, packed aliases {summary['packed_aliases']}",
        flush=True,
    )
    if not manifest["verified"]:
        print(json.dumps(manifest["failures"], ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    return 0


def command_inspect_smoothing(args: argparse.Namespace) -> int:
    report = validate_attachment(
        Path(args.attachment).resolve(),
        allow_unverified=args.allow_unverified,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 2


def command_build_online_view(args: argparse.Namespace) -> int:
    output_path = Path(args.output).resolve()
    view = build_online_view(
        model_path=Path(args.model).resolve(),
        attachment_path=Path(args.attachment).resolve(),
        output_path=output_path,
        allow_unverified=args.allow_unverified,
    )
    description = view["description"]
    print(f"Serving view: {output_path / description['name']}", flush=True)
    print(
        f"Declared {description['linears']} online linears, {description['floats']} float weights, "
        f"attachment {view['attachment']['attachment_id']}",
        flush=True,
    )
    print(
        f"Activate with {view['activation']['enable_environment_variable']}=1 and "
        f"{view['activation']['attachment_environment_variable']}="
        f"{view['attachment']['path']}",
        flush=True,
    )
    return 0


def _add_quant_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--executable", default="msmodelslim")
    parser.add_argument("--trust-remote-code", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ascend-quant-toolkit")
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
    inspect_parser.add_argument("--recipe", default="qwen25-w8a8")
    inspect_parser.set_defaults(handler=command_inspect)

    prepare_parser = subparsers.add_parser(
        "prepare-runtime",
        help="Convert ModelSlim metadata to the plugin-owned runtime type",
    )
    prepare_parser.add_argument("--model", required=True)
    prepare_parser.add_argument("--recipe", default="qwen25-w8a8")
    prepare_parser.add_argument("--source-model")
    prepare_parser.set_defaults(handler=command_prepare_runtime)

    restore_parser = subparsers.add_parser(
        "restore-modelslim",
        help="Restore the backed-up ModelSlim quantization description",
    )
    restore_parser.add_argument("--model", required=True)
    restore_parser.add_argument("--recipe", default="qwen25-w8a8")
    restore_parser.set_defaults(handler=command_restore_modelslim)

    doctor_parser = subparsers.add_parser("doctor", help="Inspect the active Python environment")
    doctor_parser.add_argument("--path", help="Path whose filesystem capacity should be checked")
    doctor_parser.set_defaults(handler=command_doctor)

    contract_parser = subparsers.add_parser(
        "export-contract",
        help="Write the versioned runtime artifact contract during offline preparation",
    )
    contract_parser.add_argument("--model", required=True)
    contract_parser.add_argument("--recipe", default="qwen25-w8a8")
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

    smoothing_parser = subparsers.add_parser(
        "build-smoothing",
        help="Fit the iter_smooth replay attachment from a BF16 model and its artifact",
    )
    smoothing_parser.add_argument("--model", required=True, help="BF16 source model directory")
    smoothing_parser.add_argument("--artifact", required=True, help="quantized artifact directory")
    smoothing_parser.add_argument("--output", required=True, help="attachment output directory")
    smoothing_parser.add_argument("--quant-type", default=DEFAULT_QUANT_TYPE)
    smoothing_parser.add_argument("--denom", type=float, default=127.0)
    smoothing_parser.add_argument("--mag-frac", type=float, default=0.1)
    smoothing_parser.add_argument("--em-rounds", type=int, default=3)
    smoothing_parser.add_argument("--min-match-rate", type=float, default=0.90)
    smoothing_parser.add_argument("--max-iterator-diff", type=int, default=2)
    smoothing_parser.add_argument("--prefix", action="append", help="Restrict the fit to one prefix")
    smoothing_parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Report one progress line to stderr every N fitted linears; 0 disables it",
    )
    smoothing_parser.set_defaults(handler=command_build_smoothing)

    smoothing_inspect = subparsers.add_parser(
        "inspect-smoothing",
        help="Validate a fitted smoothing attachment without re-fitting it",
    )
    smoothing_inspect.add_argument("--attachment", required=True)
    smoothing_inspect.add_argument("--allow-unverified", action="store_true")
    smoothing_inspect.set_defaults(handler=command_inspect_smoothing)

    online_view = subparsers.add_parser(
        "build-online-view",
        help="Write the BF16 serving view that activates the online W8A8 scheme",
    )
    online_view.add_argument("--model", required=True, help="BF16 source model directory")
    online_view.add_argument("--attachment", required=True, help="fitted smoothing attachment directory")
    online_view.add_argument("--output", required=True, help="serving view output directory")
    online_view.add_argument("--allow-unverified", action="store_true")
    online_view.set_defaults(handler=command_build_online_view)
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
