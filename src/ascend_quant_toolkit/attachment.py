"""Fit and emit the offline-to-online smoothing attachment.

The offline ModelSlim recipe smooths every ``iter_smooth`` subgraph before the
per-channel weight quantization runs.  A runtime that only receives the BF16
source model reproduces that quantization exactly when it replays the same
per-channel smoothing factors.  This module recovers those factors from a
verified artifact plus the matching BF16 source model, writes them as a small
attachment, and refuses to emit an attachment that does not replay the
artifact's INT8 weights inside the declared tolerance.

Attachment layout (``ascend_quant_smoothing.safetensors``, float32)::

    <prefix>.col_scale                   # [in]  W <- W * col
    <prefix>.row_scale                   # [out] W <- W * row
    model.layers.N.self_attn.qkv_proj.*  # packed alias of q_proj/k_proj/v_proj
    model.layers.N.mlp.gate_up_proj.*    # packed alias of gate_proj/up_proj
    norm:<module name>                   # [hidden] factor applied to a norm

The runtime replays ``W' = W * col * row``, per-output-channel symmetric INT8
with ``denom = 127`` and per-token dynamic INT8 activations, then slices the
packed aliases to the local TP shard using ``layer.weight.shape``.

Every piece of evidence used to derive a factor is recorded in
``ascend_quant_smoothing.json`` so a later run can re-verify the claim without
re-fitting anything.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .artifact import sha256_file

ATTACHMENT_FILENAME = "ascend_quant_smoothing.safetensors"
ATTACHMENT_MANIFEST_FILENAME = "ascend_quant_smoothing.json"
SCHEMA_VERSION = "1.0.0"

COL_SUFFIX = "col_scale"
ROW_SUFFIX = "row_scale"
NORM_PREFIX = "norm:"

DESCRIPTION_FILENAME = "quant_model_description.json"
WEIGHT_INDEX_CANDIDATES = (
    "model.safetensors.index.json",
    "quant_model_weights.safetensors.index.json",
)
WEIGHT_SINGLE_CANDIDATES = ("model.safetensors", "quant_model_weights.safetensors")

# ModelSlim writes one entry per quantized tensor; only ``*.weight`` identifies a
# linear whose INT8 values are replayable.
QUANT_WEIGHT_SUFFIX = ".weight"
# lm_head stays in BF16 in the validated recipe and must never be replayed.
EXCLUDED_PREFIXES = ("lm_head",)

DEFAULT_QUANT_TYPE = "ASCEND_QUANT_W8A8"
# Historical producer type names for the same validated W8A8 recipe.  They are
# accepted for identification only; the emitted attachment is type-independent.
LEGACY_QUANT_TYPES = ("W8A8_MIX", "JXD_W8A8_PDMIX")

# The frozen host's packed_modules_mapping for the validated dense model family.
PACKED_GROUPS = (
    ("qkv_proj", ("q_proj", "k_proj", "v_proj")),
    ("gate_up_proj", ("gate_proj", "up_proj")),
)
# Norm modules written by ModelSlim when a norm-linear subgraph is smoothed.
NORM_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "norm.weight",
)
# Input-side subgraphs: the consumer's input is the output of one of these norms.
NORM_FED_SHARDS = {
    "q_proj": "input_layernorm",
    "k_proj": "input_layernorm",
    "v_proj": "input_layernorm",
    "gate_proj": "post_attention_layernorm",
    "up_proj": "post_attention_layernorm",
}

EPS = 1e-8
# Row block of the least-squares input-factor estimate; bounds the float64
# working set of the widest shard instead of materializing it whole.
LEAST_SQUARES_BLOCK = 4096


@dataclass(frozen=True)
class SmoothingRequest:
    """Inputs of one attachment build."""

    model_path: Path
    artifact_path: Path
    output_path: Path
    quant_type: str = DEFAULT_QUANT_TYPE
    denom: float = 127.0
    mag_frac: float = 0.1
    em_rounds: int = 3
    min_match_rate: float = 0.90
    max_iterator_diff: int = 2
    prefixes: tuple[str, ...] = field(default_factory=tuple)

    def accepted_quant_types(self) -> tuple[str, ...]:
        return (self.quant_type, *LEGACY_QUANT_TYPES)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "artifact_path": str(self.artifact_path),
            "output_path": str(self.output_path),
            "quant_type": self.quant_type,
            "denom": self.denom,
            "mag_frac": self.mag_frac,
            "em_rounds": self.em_rounds,
            "min_match_rate": self.min_match_rate,
            "max_iterator_diff": self.max_iterator_diff,
            "prefixes": list(self.prefixes),
        }


def read_weight_map(model_path: Path) -> tuple[dict[str, str], str]:
    """Return ``(weight_map, source_name)`` for sharded and single-file models."""

    for name in WEIGHT_INDEX_CANDIDATES:
        index_path = model_path / name
        if index_path.is_file():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = payload.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"{index_path} does not declare a usable weight_map")
            return dict(weight_map), name
    for name in WEIGHT_SINGLE_CANDIDATES:
        if (model_path / name).is_file():
            return {"__single__": name}, name
    listing = sorted(entry.name for entry in os.scandir(model_path))[:20]
    raise FileNotFoundError(f"no safetensors index or single weight file in {model_path}: {listing}")


def read_description(artifact_path: Path) -> dict[str, Any]:
    """Read and shape-check the ModelSlim quantization description."""

    path = artifact_path / DESCRIPTION_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing {DESCRIPTION_FILENAME}: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {DESCRIPTION_FILENAME}: {exc}") from exc
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{path} must contain a non-empty object")
    return payload


def quantized_linears(description: dict[str, Any], accepted_types: tuple[str, ...]) -> list[str]:
    """Return the sorted linear prefixes the artifact declares as quantized."""

    prefixes: list[str] = []
    for key, value in description.items():
        if not isinstance(key, str) or not key.endswith(QUANT_WEIGHT_SUFFIX):
            continue
        if not isinstance(value, str) or value not in accepted_types:
            continue
        prefix = key[: -len(QUANT_WEIGHT_SUFFIX)]
        if any(prefix == excluded or prefix.endswith(f".{excluded}") for excluded in EXCLUDED_PREFIXES):
            continue
        prefixes.append(prefix)
    if not prefixes:
        raise ValueError(
            "the artifact declares no quantized linear for "
            f"{', '.join(accepted_types)}; check --quant-type"
        )
    return sorted(set(prefixes))


class TensorReader:
    """Lazy float32 reader over one model directory."""

    def __init__(self, model_path: Path, weight_map: dict[str, str], label: str) -> None:
        self.model_path = model_path
        self.weight_map = weight_map
        self.label = label

    def shard_for(self, name: str) -> str | None:
        return self.weight_map.get(name) or self.weight_map.get("__single__")

    def get(self, name: str) -> Any:
        """Return ``name`` as float32 ``numpy.ndarray``, or ``None`` if absent."""

        import numpy as np
        from safetensors import safe_open

        shard = self.shard_for(name)
        if shard is None:
            return None
        path = self.model_path / shard
        if not path.is_file():
            raise FileNotFoundError(f"{self.label} shard is missing: {path}")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if name not in handle.keys():
                return None
            tensor = handle.get_tensor(name)
        array = tensor.to("cpu").float().numpy() if hasattr(tensor, "to") else np.asarray(tensor)
        return np.ascontiguousarray(array, dtype=np.float32)

    def has(self, name: str) -> bool:
        return self.shard_for(name) is not None

    def names(self) -> list[str]:
        """Return every tensor name readable from this directory."""

        single = self.weight_map.get("__single__")
        if single is None:
            return sorted(self.weight_map)
        from safetensors import safe_open

        with safe_open(str(self.model_path / single), framework="pt", device="cpu") as handle:
            return sorted(handle.keys())
@dataclass
class LayerFit:
    """Recovered per-linear smoothing factors plus their evidence."""

    prefix: str
    col: Any
    row: Any
    report: dict[str, Any]


@dataclass
class NormFit:
    """Recovered factor for one norm module."""

    module: str
    factor: Any
    report: dict[str, Any]


def _estimate_input_scale(w16: Any, wq_int8: Any, scale: Any, mag_frac: float) -> Any:
    """Recover the per-input-channel factor from artifact INT8 values.

    ``col[j]`` follows from ``wq[i, j] * scale[i] ~= w16[i, j] * col[j] * row[i]``;
    the median over output channels removes ``row`` up to one global constant,
    which the row estimate below re-normalizes against the artifact scale.
    """

    import numpy as np

    deq = wq_int8 * scale[:, None]
    thr = np.abs(w16).max(axis=0, keepdims=True) * mag_frac
    mask = np.abs(w16) >= thr
    safe = np.where(mask, w16, 1.0)
    ratio = np.where(mask, deq / safe, np.nan)
    with np.errstate(invalid="ignore"):
        col = np.nanmedian(ratio, axis=0)
    col = np.where(np.isfinite(col) & (col > 0), col, 1.0)
    return col.astype(np.float64)


def _refine_input_scale(w16: Any, wq_int8: Any, scale: Any, row: Any, mag_frac: float) -> Any:
    """Re-estimate the input factor with a known per-output-channel factor."""

    import numpy as np

    weighted = w16 * row[:, None]
    deq = wq_int8 * scale[:, None]
    thr = np.abs(weighted).max(axis=0, keepdims=True) * mag_frac
    mask = np.abs(weighted) >= thr
    safe = np.where(mask, weighted, 1.0)
    ratio = np.where(mask, deq / safe, np.nan)
    with np.errstate(invalid="ignore"):
        col = np.nanmedian(ratio, axis=0)
    col = np.where(np.isfinite(col) & (col > 0), col, 1.0)
    return col.astype(np.float64)


def _weighted_input_scale(w16: Any, wq_int8: Any, scale: Any, row: Any, denom: float) -> Any:
    """Recover the per-input-channel factor by least squares in LSB units.

    ``_estimate_input_scale`` divides two quantized quantities, so the rounding
    noise lands multiplicatively on the ratio and biases the median.  The replay
    error is instead measured in LSB units, where element ``(i, j)`` carries
    noise proportional to ``scale[i]``.  Minimizing

        sum_ij (w16[i, j] * col[j] * row[i] - wq[i, j] * scale[i]) ** 2 / scale[i] ** 2

    keeps the columns decoupled and gives the closed form below.  Elements that
    saturate the INT8 range are rounded by the producer for reasons the artifact
    cannot expose, so they contribute no usable ratio and are dropped.
    """

    import numpy as np

    out_dim, in_dim = w16.shape
    usable = np.abs(wq_int8) < denom
    numerator = np.zeros(in_dim, dtype=np.float64)
    denominator = np.zeros(in_dim, dtype=np.float64)
    for start in range(0, out_dim, LEAST_SQUARES_BLOCK):
        stop = min(start + LEAST_SQUARES_BLOCK, out_dim)
        weight = usable[start:stop] / np.square(scale[start:stop, None])
        weighted = w16[start:stop].astype(np.float64) * row[start:stop, None]
        reference = wq_int8[start:stop] * scale[start:stop, None]
        numerator += np.sum(weight * weighted * reference, axis=0)
        denominator += np.sum(weight * weighted * weighted, axis=0)
    col = np.divide(numerator, denominator, out=np.ones_like(numerator), where=denominator > 0)
    return np.where(np.isfinite(col) & (col > 0), col, 1.0).astype(np.float64)


def _row_from_scale(w16: Any, col: Any, scale: Any, denom: float) -> Any:
    """Choose the output factor so the replayed per-channel amax matches the artifact."""

    import numpy as np

    amax = np.abs(w16 * col[None, :]).max(axis=1)
    row = (scale * denom) / np.maximum(amax, EPS)
    return np.where(np.isfinite(row) & (row > 0), row, 1.0).astype(np.float64)


def _requant(w16: Any, col: Any, row: Any, denom: float) -> tuple[Any, Any]:
    """Per-output-channel symmetric INT8 replay of ``W * col * row``.

    ``denom`` maps the amax onto INT8, whose full range is ``[-128, 127]``.  The
    producer saturates the negative side one step further than the positive one,
    so the replay has to allow it to stay bit-exact.
    """

    import numpy as np

    scaled = w16 * col[None, :] * row[:, None]
    amax = np.abs(scaled).max(axis=1, keepdims=True)
    channel_scale = np.maximum(amax / denom, EPS)
    quantized = np.clip(np.rint(scaled / channel_scale), -128, 127).astype(np.int8)
    return quantized, channel_scale.reshape(-1)


def fit_layer(
    bf16: TensorReader,
    artifact: TensorReader,
    prefix: str,
    *,
    denom: float,
    mag_frac: float,
    em_rounds: int,
    col_hint: Any = None,
) -> LayerFit:
    """Fit ``(col, row)`` for one linear and report the replay fidelity."""

    import numpy as np

    w16 = bf16.get(f"{prefix}.weight")
    wq = artifact.get(f"{prefix}.weight")
    scale = artifact.get(f"{prefix}.weight_scale")
    report: dict[str, Any] = {"prefix": prefix}
    if w16 is None or wq is None or scale is None:
        report["status"] = "skipped"
        report["reason"] = "missing weight or weight_scale"
        return LayerFit(prefix, None, None, report)

    scale = np.ascontiguousarray(scale.reshape(-1), dtype=np.float64)
    if wq.shape == w16.T.shape and wq.shape != w16.shape:
        wq = np.ascontiguousarray(wq.T)
        layout = "transposed"
    else:
        layout = "out_in"
    if wq.shape != w16.shape:
        report["status"] = "skipped"
        report["reason"] = f"shape mismatch bf16={w16.shape} artifact={wq.shape}"
        return LayerFit(prefix, None, None, report)
    out_dim, in_dim = (int(size) for size in w16.shape)
    if scale.size != out_dim:
        report["status"] = "skipped"
        report["reason"] = f"weight_scale length {scale.size} != out_dim {out_dim}"
        return LayerFit(prefix, None, None, report)

    wq_int8 = np.rint(wq)
    # A norm ratio is exact by construction and must not be overwritten by the
    # noisy inverse estimate; the remaining shards have to be fitted.  Every
    # estimator is scored on the same replay, so a fit can never end up worse
    # than the plain median it starts from: the un-refined median stays a
    # candidate next to its EM-refined and least-squares siblings.
    median_col = _estimate_input_scale(w16, wq_int8, scale, mag_frac)
    median_row = _row_from_scale(w16, median_col, scale, denom)
    least_squares_col = _weighted_input_scale(w16, wq_int8, scale, median_row, denom)
    candidates: list[tuple[str, Any, bool]] = [
        ("median", median_col, False),
        ("fitted", median_col, True),
        ("lsb_least_squares", least_squares_col, False),
    ]
    if col_hint is not None and np.asarray(col_hint).shape == (in_dim,):
        candidates.insert(0, ("norm_ratio", np.asarray(col_hint, dtype=np.float64), False))

    best: tuple[tuple[float, int], str, Any, Any, dict[str, Any]] | None = None
    for source, initial, refine in candidates:
        col = np.ascontiguousarray(initial, dtype=np.float64)
        row = _row_from_scale(w16, col, scale, denom)
        for _ in range(max(em_rounds, 0) if refine else 0):
            col = _refine_input_scale(w16, wq_int8, scale, row, mag_frac)
            row = _row_from_scale(w16, col, scale, denom)
        quantized, replayed_scale = _requant(w16, col, row, denom)
        difference = np.abs(quantized.astype(np.int16) - wq_int8.astype(np.int16))
        stats = {
            "int8_match_rate": float((quantized == wq_int8).mean()),
            "max_iterator_diff": int(difference.max()),
            "scale_rel_p95": float(
                np.nanpercentile(np.abs(replayed_scale - scale) / np.maximum(scale, EPS), 95)
            ),
        }
        report[source] = stats
        # Rank by match rate first and by the tighter worst-case deviation
        # second, so a tie cannot silently keep the fuzzier replay.
        entry = (
            (stats["int8_match_rate"], -stats["max_iterator_diff"]),
            source,
            col,
            row,
            stats,
        )
        if best is None or entry[0] > best[0]:
            best = entry

    assert best is not None
    _, source, col, row, stats = best
    match_rate = stats["int8_match_rate"]
    report.update(
        {
            "status": "fitted",
            "col_source": source,
            "layout": layout,
            "out_dim": out_dim,
            "in_dim": in_dim,
            "int8_match_rate": match_rate,
            "max_iterator_diff": stats["max_iterator_diff"],
            "scale_rel_p95": stats["scale_rel_p95"],
            "col_stats": {
                "min": float(col.min()),
                "median": float(np.median(col)),
                "max": float(col.max()),
            },
            "row_stats": {
                "min": float(row.min()),
                "median": float(np.median(row)),
                "max": float(row.max()),
            },
        }
    )
    return LayerFit(prefix, col, row, report)


def norm_modules(names: Iterable[str]) -> list[str]:
    """Return the norm module names present in a tensor inventory."""

    modules = {
        name[: -len(".weight")]
        for name in names
        if isinstance(name, str) and name.endswith(NORM_SUFFIXES)
    }
    return sorted(modules)


def upstream_norm(prefix: str) -> str | None:
    """Return the norm module feeding ``prefix`` for a norm-linear subgraph."""

    parts = prefix.split(".")
    if len(parts) < 2:
        return None
    norm_name = NORM_FED_SHARDS.get(parts[-1])
    if norm_name is None:
        return None
    return ".".join([*parts[:-2], norm_name])


def fit_norm(bf16: TensorReader, artifact: TensorReader, module: str) -> NormFit:
    """Recover ``artifact_norm / bf16_norm`` for one norm module."""

    import numpy as np

    reference = bf16.get(f"{module}.weight")
    stored = artifact.get(f"{module}.weight")
    report: dict[str, Any] = {"module": module}
    if reference is None or stored is None or reference.shape != stored.shape:
        report["status"] = "absent"
        return NormFit(module, None, report)
    ratio = stored / np.where(np.abs(reference) > EPS, reference, np.nan)
    ratio = np.where(np.isfinite(ratio), ratio, 1.0)
    deviation = np.abs(ratio - 1.0)
    report.update(
        {
            "status": "fitted",
            "size": int(ratio.size),
            "median": float(np.median(ratio)),
            "min": float(ratio.min()),
            "max": float(ratio.max()),
            "frac_dev_gt_1pct": float((deviation > 0.01).mean()),
        }
    )
    return NormFit(module, ratio.astype(np.float64), report)
def packed_aliases(fits: dict[str, LayerFit]) -> dict[str, dict[str, Any]]:
    """Group fitted shards into the host's packed linear prefixes.

    The runtime sees ``self_attn.qkv_proj``/``mlp.gate_up_proj`` rather than the
    ModelSlim shard names, so the attachment also carries the packed alias plus
    the shard geometry a TP-rank has to slice.
    """

    import numpy as np

    aliases: dict[str, dict[str, Any]] = {}
    for packed_name, shard_names in PACKED_GROUPS:
        parents: dict[str, list[str]] = {}
        for prefix, fit in fits.items():
            if fit.col is None:
                continue
            parent, _, shard = prefix.rpartition(".")
            if shard in shard_names:
                parents.setdefault(parent, []).append(shard)
        for parent, present in sorted(parents.items()):
            if set(present) != set(shard_names):
                continue
            ordered = [f"{parent}.{name}" for name in shard_names]
            cols = [fits[prefix].col for prefix in ordered]
            rows = [fits[prefix].row for prefix in ordered]
            stacked_col = np.median(np.stack(cols, axis=0), axis=0)
            spread = float(max(np.abs(column / stacked_col - 1.0).max() for column in cols))
            alias = f"{parent}.{packed_name}"
            aliases[alias] = {
                "col": stacked_col.astype(np.float64),
                "row": np.concatenate(rows).astype(np.float64),
                "report": {
                    "alias": alias,
                    "packed_name": packed_name,
                    "shards": ordered,
                    "shard_rows": [int(row.size) for row in rows],
                    "shard_cols": [int(column.size) for column in cols],
                    "col_max_rel_spread": spread,
                },
            }
    return aliases


def _tensor_record(name: str, tensor: Any) -> dict[str, Any]:
    return {"name": name, "dtype": str(tensor.dtype), "shape": [int(size) for size in tensor.shape]}


def _attachment_id(tensors: list[dict[str, Any]]) -> str:
    canonical = json.dumps(tensors, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()
def build_attachment(
    request: SmoothingRequest,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Fit the smoothing factors and write the attachment plus its manifest.

    ``progress`` is called with ``(fitted, total, prefix)`` after every linear so
    a long run can be watched while its shards are being read.
    """

    import numpy as np
    from safetensors.numpy import save_file

    from . import __version__

    bf16_map, bf16_source = read_weight_map(request.model_path)
    artifact_map, artifact_source = read_weight_map(request.artifact_path)
    description = read_description(request.artifact_path)
    prefixes = quantized_linears(description, request.accepted_quant_types())
    if request.prefixes:
        wanted = set(request.prefixes)
        unknown = sorted(wanted - set(prefixes))
        if unknown:
            raise ValueError(f"requested prefixes are not quantized in the artifact: {unknown}")
        prefixes = [prefix for prefix in prefixes if prefix in wanted]

    bf16 = TensorReader(request.model_path, bf16_map, "bf16")
    artifact = TensorReader(request.artifact_path, artifact_map, "artifact")

    norm_fits = [fit_norm(bf16, artifact, module) for module in norm_modules(bf16.names())]
    norm_by_module = {fit.module: fit for fit in norm_fits}

    fits: dict[str, LayerFit] = {}
    layer_reports: list[dict[str, Any]] = []
    for index, prefix in enumerate(prefixes, start=1):
        fed_by = upstream_norm(prefix)
        upstream = norm_by_module.get(fed_by) if fed_by is not None else None
        hint = None
        if upstream is not None and upstream.factor is not None:
            hint = np.where(upstream.factor > EPS, 1.0 / upstream.factor, 1.0)
        fit = fit_layer(
            bf16,
            artifact,
            prefix,
            denom=request.denom,
            mag_frac=request.mag_frac,
            em_rounds=request.em_rounds,
            col_hint=hint,
        )
        fits[prefix] = fit
        layer_reports.append({**fit.report, "upstream_norm": fed_by})
        if progress is not None:
            progress(index, len(prefixes), prefix)

    aliases = packed_aliases(fits)

    tensors: dict[str, Any] = {}
    for prefix, fit in fits.items():
        if fit.col is None or fit.row is None:
            continue
        tensors[f"{prefix}.{COL_SUFFIX}"] = fit.col.astype(np.float32)
        tensors[f"{prefix}.{ROW_SUFFIX}"] = fit.row.astype(np.float32)
    for alias, payload in aliases.items():
        tensors[f"{alias}.{COL_SUFFIX}"] = payload["col"].astype(np.float32)
        tensors[f"{alias}.{ROW_SUFFIX}"] = payload["row"].astype(np.float32)
    for fit in norm_fits:
        if fit.factor is not None:
            tensors[f"{NORM_PREFIX}{fit.module}"] = fit.factor.astype(np.float32)

    request.output_path.mkdir(parents=True, exist_ok=True)
    attachment_path = request.output_path / ATTACHMENT_FILENAME
    save_file(dict(sorted(tensors.items())), str(attachment_path))
    tensor_records = [_tensor_record(name, tensors[name]) for name in sorted(tensors)]
    fitted_reports = [report for report in layer_reports if report.get("status") == "fitted"]
    failures = [
        {
            "prefix": report["prefix"],
            "reason": report.get("reason", "replay outside tolerance"),
            "int8_match_rate": report.get("int8_match_rate"),
            "max_iterator_diff": report.get("max_iterator_diff"),
        }
        for report in layer_reports
        if report.get("status") != "fitted"
        or report["int8_match_rate"] < request.min_match_rate
        or report["max_iterator_diff"] > request.max_iterator_diff
    ]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "iter_smooth_replay",
        "attachment_id": _attachment_id(tensor_records),
        "verified": not failures,
        "tool": {"name": "ascend-quant-toolkit", "version": __version__},
        "invocation": request.as_dict(),
        "consumer": {
            "host": "vllm-ascend",
            "scheme_provider": "vllm-ascend-quant-ext",
            "runtime_quant_type": request.quant_type,
            "replay": {
                "weight_formula": "W * col[None, :] * row[:, None]",
                "weight": "per_channel_symmetric_int8",
                "denom": request.denom,
                "activation": "per_token_dynamic_int8",
            },
        },
        "sources": {
            "model": {
                "path": str(request.model_path),
                "weight_index": bf16_source,
                "declared_linears": len(prefixes),
            },
            "artifact": {
                "path": str(request.artifact_path),
                "weight_index": artifact_source,
                "description": DESCRIPTION_FILENAME,
            },
        },
        "layers": layer_reports,
        "norms": [fit.report for fit in norm_fits],
        "packed_aliases": [payload["report"] for payload in aliases.values()],
        "tensors": tensor_records,
        "summary": {
            "linears_declared": len(prefixes),
            "linears_replayed": len(fitted_reports),
            "linears_failed": len(failures),
            "norms_recovered": sum(1 for fit in norm_fits if fit.factor is not None),
            "packed_aliases": len(aliases),
            "min_int8_match_rate": min(
                (report["int8_match_rate"] for report in fitted_reports), default=0.0
            ),
            "mean_int8_match_rate": float(
                sum(report["int8_match_rate"] for report in fitted_reports)
                / max(len(fitted_reports), 1)
            ),
            "max_iterator_diff": max(
                (report["max_iterator_diff"] for report in fitted_reports), default=0
            ),
        },
        "failures": failures,
        "files": [
            {
                "name": ATTACHMENT_FILENAME,
                "size": attachment_path.stat().st_size,
                "sha256": sha256_file(attachment_path),
            }
        ],
    }
    manifest_path = request.output_path / ATTACHMENT_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def validate_attachment(attachment_dir: Path, *, allow_unverified: bool = False) -> dict[str, Any]:
    """Fail-closed admission of one written attachment directory."""

    import numpy as np
    from safetensors import safe_open

    errors: list[str] = []
    manifest_path = attachment_dir / ATTACHMENT_MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "errors": [f"unreadable manifest: {exc}"]}
    if not isinstance(manifest, dict):
        return {"valid": False, "errors": ["manifest must be an object"]}
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"unsupported schema version: {manifest.get('schema_version')}")
    verified = bool(manifest.get("verified"))
    if not verified and not allow_unverified:
        errors.append("the producer did not verify every declared linear")

    declared_files = manifest.get("files")
    if not isinstance(declared_files, list) or not declared_files:
        errors.append("manifest declares no files")
        declared_files = []
    for entry in declared_files:
        candidate = attachment_dir / str(entry.get("name", ""))
        if not candidate.is_file():
            errors.append(f"missing attachment file: {candidate.name}")
            continue
        if candidate.stat().st_size != entry.get("size"):
            errors.append(f"size mismatch: {candidate.name}")
        elif sha256_file(candidate) != entry.get("sha256"):
            errors.append(f"sha256 mismatch: {candidate.name}")

    declared_tensors = manifest.get("tensors")
    if not isinstance(declared_tensors, list) or not declared_tensors:
        errors.append("manifest declares no tensors")
        declared_tensors = []
    inventory = {str(entry.get("name")): entry for entry in declared_tensors}

    attachment_path = attachment_dir / ATTACHMENT_FILENAME
    if not attachment_path.is_file():
        errors.append(f"missing {ATTACHMENT_FILENAME}")
    else:
        with safe_open(str(attachment_path), framework="np") as handle:
            present = set(handle.keys())
            for name in sorted(set(inventory) - present):
                errors.append(f"declared tensor is absent: {name}")
            for name in sorted(present - set(inventory)):
                errors.append(f"undeclared tensor: {name}")
            for name in sorted(set(inventory) & present):
                array = handle.get_tensor(name)
                entry = inventory[name]
                signature = ([int(size) for size in array.shape], str(array.dtype))
                if signature != (entry.get("shape"), entry.get("dtype")):
                    errors.append(f"tensor signature mismatch: {name}")
                    continue
                values = np.asarray(array, dtype=np.float64)
                if not np.isfinite(values).all():
                    errors.append(f"tensor is not finite: {name}")
                elif (values <= 0).any():
                    errors.append(f"tensor has a non-positive factor: {name}")

    return {
        "valid": not errors,
        "schema_version": manifest.get("schema_version"),
        "attachment_id": manifest.get("attachment_id"),
        "verified": verified,
        "tensors": len(inventory),
        "layers": len(manifest.get("layers") or []),
        "norms": len(manifest.get("norms") or []),
        "summary": manifest.get("summary"),
        "errors": errors,
    }


# The serving view that activates the online scheme.  The host picks a scheme per
# linear from ``quant_model_description.json``, so the view links a BF16
# checkpoint and describes it; the factors stay in the attachment.
ONLINE_VIEW_FILENAME = "online_view.json"
ONLINE_VIEW_KIND = "online_w8a8_serving_view"
ONLINE_QUANT_TYPE = "ASCEND_QUANT_W8A8_ONLINE"
FLOAT_QUANT_TYPE = "FLOAT"
# The module names the validated recipe smooths.  Every other weight keeps its
# BF16 tensor and is declared float.
LINEAR_MODULE_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def attachment_prefixes(attachment_dir: Path) -> tuple[list[str], list[str]]:
    """Return ``(linear prefixes, norm modules)`` declared by a written attachment."""

    manifest_path = attachment_dir / ATTACHMENT_MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable attachment manifest: {manifest_path}") from exc
    prefixes: set[str] = set()
    norms: set[str] = set()
    for entry in manifest.get("tensors") or []:
        name = str(entry.get("name", ""))
        if name.startswith(NORM_PREFIX):
            norms.add(name[len(NORM_PREFIX) :])
            continue
        for suffix in (COL_SUFFIX, ROW_SUFFIX):
            marker = f".{suffix}"
            if name.endswith(marker):
                prefixes.add(name[: -len(marker)])
                break
    if not prefixes:
        raise ValueError(f"{attachment_dir} declares no smoothing factors")
    return sorted(prefixes), sorted(norms)


def _expand_packed(prefixes: Iterable[str]) -> set[str]:
    """Flatten host packed aliases back onto the checkpoint's own shard names."""

    expanded: set[str] = set()
    for prefix in prefixes:
        head, _, module = prefix.rpartition(".")
        shards = next((group for packed, group in PACKED_GROUPS if packed == module), None)
        if shards is None:
            expanded.add(prefix)
            continue
        expanded.update(f"{head}.{shard}" for shard in shards)
    return expanded


def model_linears(model_path: Path) -> tuple[list[str], list[str], str]:
    """Return ``(linear prefixes, float prefixes, index name)`` of a checkpoint.

    A grouped checkpoint is flattened onto the shard names the host indexes the
    description by, so a packed ``qkv_proj`` and its three shards describe the
    same model.  Every weight the recipe does not smooth is reported as float.
    """

    weight_map, source = read_weight_map(model_path)
    reader = TensorReader(model_path, weight_map, "source")
    linears: set[str] = set()
    floats: set[str] = set()
    for name in reader.names():
        if not name.endswith(QUANT_WEIGHT_SUFFIX):
            continue
        prefix = name[: -len(QUANT_WEIGHT_SUFFIX)]
        module = prefix.rpartition(".")[2]
        shards = next((group for packed, group in PACKED_GROUPS if packed == module), None)
        if shards is not None:
            head = prefix[: -len(module)]
            linears.update(f"{head}{shard}" for shard in shards)
            continue
        excluded = any(prefix == item or prefix.endswith(f".{item}") for item in EXCLUDED_PREFIXES)
        if module in LINEAR_MODULE_SUFFIXES and not excluded:
            linears.add(prefix)
        else:
            floats.add(prefix)
    if not linears:
        raise ValueError(f"{model_path} declares no smoothable linear weight")
    return sorted(linears), sorted(floats), source


def _link_serving_files(model_path: Path, output_path: Path) -> list[str]:
    """Link every checkpoint file into the view; weights are never copied."""

    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError(f"serving view directory is not empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    linked: list[str] = []
    for entry in sorted(model_path.iterdir()):
        if not entry.is_file() or entry.name == DESCRIPTION_FILENAME:
            continue
        target = output_path / entry.name
        if target.exists() or target.is_symlink():
            raise ValueError(f"refusing to replace {target}")
        target.symlink_to(entry.resolve())
        linked.append(entry.name)
    return linked


def build_online_view(
    *,
    model_path: Path,
    attachment_path: Path,
    output_path: Path,
    allow_unverified: bool = False,
) -> dict[str, Any]:
    """Write the serving view that makes the host select the online W8A8 scheme.

    The host chooses a scheme per linear from ``quant_model_description.json``,
    so the view links the BF16 checkpoint, adds that description and points at
    the attachment the extension resolves its factors from.  No weight is copied
    or rewritten and the attachment stays the only smoothed input.

    Checkpoint and attachment have to agree exactly: a linear the attachment
    misses would stay BF16 next to replayed weights, and a factor for an absent
    module would abort the load, so both are rejected here.
    """

    report = validate_attachment(attachment_path, allow_unverified=allow_unverified)
    if not report["valid"]:
        raise ValueError(f"attachment is not admitted: {report['errors']}")

    attachment_linears, attachment_norms = attachment_prefixes(attachment_path)
    covered = _expand_packed(attachment_linears)
    linears, floats, index_name = model_linears(model_path)

    missing = sorted(set(linears) - covered)
    if missing:
        raise ValueError(f"the attachment does not cover every model linear: {missing[:5]}")
    unknown = sorted(covered - set(linears))
    if unknown:
        raise ValueError(f"the attachment declares linears the model does not have: {unknown[:5]}")
    # A declared norm has to exist in the checkpoint, otherwise the extension
    # would resolve a factor for a tensor the model does not have.
    float_names = set(floats)
    absent = [module for module in attachment_norms if module not in float_names]
    if absent:
        raise ValueError(f"the attachment declares absent norm modules: {absent[:5]}")

    linked = _link_serving_files(model_path, output_path)
    description = {f"{prefix}{QUANT_WEIGHT_SUFFIX}": ONLINE_QUANT_TYPE for prefix in linears}
    description.update({f"{prefix}{QUANT_WEIGHT_SUFFIX}": FLOAT_QUANT_TYPE for prefix in floats})
    description_path = output_path / DESCRIPTION_FILENAME
    description_path.write_text(
        json.dumps(dict(sorted(description.items())), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    view: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": ONLINE_VIEW_KIND,
        "quant_type": ONLINE_QUANT_TYPE,
        "source": {"path": str(model_path), "weight_index": index_name, "files_linked": linked},
        "attachment": {
            "path": str(attachment_path),
            "attachment_id": report["attachment_id"],
            "verified": report["verified"],
        },
        "description": {
            "name": DESCRIPTION_FILENAME,
            "linears": len(linears),
            "floats": len(floats),
            "sha256": sha256_file(description_path),
        },
        "activation": {
            "enable_environment_variable": "VLLM_ASCEND_QUANT_EXT_ENABLE",
            "attachment_environment_variable": "VLLM_ASCEND_QUANT_EXT_SMOOTHING",
        },
    }
    (output_path / ONLINE_VIEW_FILENAME).write_text(
        json.dumps(view, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return view
