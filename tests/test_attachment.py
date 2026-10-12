"""Tests for the offline-to-online smoothing attachment producer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

from ascend_quant_toolkit.artifact import sha256_file
from ascend_quant_toolkit.attachment import (
    ATTACHMENT_FILENAME,
    DESCRIPTION_FILENAME,
    FLOAT_QUANT_TYPE,
    ONLINE_QUANT_TYPE,
    ONLINE_VIEW_FILENAME,
    ONLINE_VIEW_KIND,
    SmoothingRequest,
    TensorReader,
    _requant,
    _weighted_input_scale,
    build_attachment,
    build_online_view,
    validate_attachment,
)

HIDDEN = 32
SHARD_OUT = 16
LAYER = "model.layers.0"
INPUT_NORM = f"{LAYER}.input_layernorm"
POST_NORM = f"{LAYER}.post_attention_layernorm"
NORM_FACTORS = {
    INPUT_NORM: np.linspace(0.25, 0.8, HIDDEN).astype(np.float32),
    POST_NORM: np.linspace(0.40, 1.1, HIDDEN).astype(np.float32),
}
# shard -> (output size, upstream subgraph). ``None`` means the input side has to
# be fitted from the artifact INT8 values instead of the norm ratio.
SHARDS = {
    f"{LAYER}.self_attn.q_proj": INPUT_NORM,
    f"{LAYER}.self_attn.k_proj": INPUT_NORM,
    f"{LAYER}.self_attn.v_proj": INPUT_NORM,
    f"{LAYER}.self_attn.o_proj": None,
    f"{LAYER}.mlp.gate_proj": POST_NORM,
    f"{LAYER}.mlp.up_proj": POST_NORM,
    f"{LAYER}.mlp.down_proj": None,
}


def _quantize(weight: np.ndarray, col: np.ndarray, row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaled = weight * col[None, :] * row[:, None]
    scale = np.maximum(np.abs(scaled).max(axis=1, keepdims=True) / 127.0, 1e-8)
    # ModelSlim saturates INT8 asymmetrically, so the stand-in artifact has to
    # keep -128 alive for the replay to be compared against the same contract.
    quantized = np.clip(np.rint(scaled / scale), -128, 127).astype(np.int8)
    return quantized, scale.reshape(-1).astype(np.float32)


def _build_pair(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(20260928)
    source: dict[str, np.ndarray] = {}
    artifact: dict[str, np.ndarray] = {}
    description: dict[str, str] = {}
    for module, factor in NORM_FACTORS.items():
        source[f"{module}.weight"] = np.ones(HIDDEN, dtype=np.float32)
        artifact[f"{module}.weight"] = factor
    for prefix, norm_module in SHARDS.items():
        weight = rng.normal(size=(SHARD_OUT, HIDDEN)).astype(np.float32)
        col = (
            1.0 / NORM_FACTORS[norm_module]
            if norm_module is not None
            else np.linspace(0.9, 1.4, HIDDEN).astype(np.float32)
        )
        row = np.linspace(1.0, 2.0, SHARD_OUT).astype(np.float32)
        quantized, scale = _quantize(weight, col, row)
        source[f"{prefix}.weight"] = weight
        artifact[f"{prefix}.weight"] = quantized
        artifact[f"{prefix}.weight_scale"] = scale
        for suffix in ("weight", "weight_scale"):
            description[f"{prefix}.{suffix}"] = "ASCEND_QUANT_W8A8"

    model_path = tmp_path / "bf16"
    artifact_path = tmp_path / "artifact"
    model_path.mkdir()
    artifact_path.mkdir()
    save_file(source, str(model_path / "model.safetensors"))
    save_file(artifact, str(artifact_path / "model.safetensors"))
    (artifact_path / "quant_model_description.json").write_text(
        json.dumps(description), encoding="utf-8"
    )
    return model_path, artifact_path


def _write_bf16_shard(path: Path, name: str, values: np.ndarray) -> None:
    """Write one real BF16 safetensors shard without a torch dependency.

    ``safetensors.numpy`` cannot tag a BF16 tensor (numpy has no such dtype), so
    the header and the widened payload are written by hand: values are truncated
    to their high 16 bits, exactly as a BF16 producer stores them.
    """

    words = (values.astype(np.float32).view(np.uint32) >> 16).astype("<u2")
    header = json.dumps(
        {
            name: {
                "dtype": "BF16",
                "shape": list(values.shape),
                "data_offsets": [0, int(words.nbytes)],
            }
        }
    ).encode("utf-8")
    padded = header + b" " * (-len(header) % 8)
    path.write_bytes(len(padded).to_bytes(8, "little") + padded + words.tobytes())


def _request(tmp_path: Path) -> SmoothingRequest:
    model_path, artifact_path = _build_pair(tmp_path)
    return SmoothingRequest(
        model_path=model_path,
        artifact_path=artifact_path,
        output_path=tmp_path / "attachment",
        # The synthesized shards are far smaller than a real layer, so the
        # inverse estimate carries more noise than the 14B replay does.
        min_match_rate=0.80,
    )


def test_build_attachment_replays_every_linear(tmp_path: Path):
    manifest = build_attachment(_request(tmp_path))

    assert manifest["verified"] is True
    assert manifest["failures"] == []
    assert manifest["summary"]["linears_declared"] == len(SHARDS)
    assert manifest["summary"]["linears_replayed"] == len(SHARDS)
    assert manifest["summary"]["max_iterator_diff"] <= 2
    by_prefix = {report["prefix"]: report for report in manifest["layers"]}
    # Norm-fed shards replay exactly: their input factor is the artifact norm ratio.
    for prefix in (
        f"{LAYER}.self_attn.q_proj",
        f"{LAYER}.self_attn.k_proj",
        f"{LAYER}.self_attn.v_proj",
        f"{LAYER}.mlp.gate_proj",
        f"{LAYER}.mlp.up_proj",
    ):
        assert by_prefix[prefix]["col_source"] == "norm_ratio"
        assert by_prefix[prefix]["int8_match_rate"] == 1.0
        assert by_prefix[prefix]["max_iterator_diff"] == 0
    # Pair-smoothing shards have no norm to derive from and stay fitted.
    for prefix in (f"{LAYER}.self_attn.o_proj", f"{LAYER}.mlp.down_proj"):
        assert by_prefix[prefix]["col_source"] == "fitted"
        assert by_prefix[prefix]["int8_match_rate"] >= 0.80


def test_attachment_carries_norm_factors_and_packed_aliases(tmp_path: Path):
    request = _request(tmp_path)
    manifest = build_attachment(request)

    attachment_path = request.output_path / ATTACHMENT_FILENAME
    with safe_open(str(attachment_path), framework="np") as handle:
        names = set(handle.keys())
        norm_factor = handle.get_tensor(f"norm:{INPUT_NORM}")
        qkv_row = handle.get_tensor(f"{LAYER}.self_attn.qkv_proj.row_scale")
        gate_up_col = handle.get_tensor(f"{LAYER}.mlp.gate_up_proj.col_scale")
        q_col = handle.get_tensor(f"{LAYER}.self_attn.q_proj.col_scale")
        o_col = handle.get_tensor(f"{LAYER}.self_attn.o_proj.col_scale")
        o_row = handle.get_tensor(f"{LAYER}.self_attn.o_proj.row_scale")

    assert names
    assert np.allclose(norm_factor, NORM_FACTORS[INPUT_NORM], rtol=1e-6)
    assert np.allclose(q_col, 1.0 / NORM_FACTORS[INPUT_NORM], rtol=1e-5)
    # A fitted input factor is only defined up to one global constant: the row
    # estimate absorbs it, so the replayable quantity is the outer product.  The
    # residual is the inverse estimate's own quantization noise.
    col_true = np.linspace(0.9, 1.4, HIDDEN).astype(np.float32)
    row_true = np.linspace(1.0, 2.0, SHARD_OUT).astype(np.float32)
    assert np.allclose(np.outer(o_row, o_col), np.outer(row_true, col_true), rtol=0.05)
    assert qkv_row.shape == (3 * SHARD_OUT,)
    assert gate_up_col.shape == (HIDDEN,)
    aliases = {report["alias"]: report for report in manifest["packed_aliases"]}
    assert aliases[f"{LAYER}.self_attn.qkv_proj"]["shard_rows"] == [SHARD_OUT] * 3
    assert aliases[f"{LAYER}.mlp.gate_up_proj"]["shards"] == [
        f"{LAYER}.mlp.gate_proj",
        f"{LAYER}.mlp.up_proj",
    ]
    assert f"{LAYER}.self_attn.qkv_proj.{ATTACHMENT_FILENAME}" not in names
    assert validate_attachment(request.output_path)["valid"] is True


def test_attachment_validation_rejects_tampering(tmp_path: Path):
    request = _request(tmp_path)
    build_attachment(request)

    attachment_path = request.output_path / ATTACHMENT_FILENAME
    payload = bytearray(attachment_path.read_bytes())
    payload[-1] ^= 0xFF
    attachment_path.write_bytes(bytes(payload))

    report = validate_attachment(request.output_path)

    assert report["valid"] is False
    assert any("sha256 mismatch" in error or "size mismatch" in error for error in report["errors"])


def test_progress_reports_every_fitted_linear(tmp_path: Path):
    request = _request(tmp_path)
    seen: list[tuple[int, int, str]] = []

    manifest = build_attachment(
        request,
        progress=lambda done, total, prefix: seen.append((done, total, prefix)),
    )

    declared = manifest["summary"]["linears_declared"]
    assert [done for done, _, _ in seen] == list(range(1, declared + 1))
    assert {total for _, total, _ in seen} == {declared}
    assert seen[-1][2] == manifest["layers"][-1]["prefix"]


def test_every_candidate_is_scored_and_the_best_one_wins(tmp_path: Path):
    """A chosen factor can never replay the artifact worse than the median does.

    The un-refined median stays a candidate next to its EM-refined and
    least-squares siblings, so a fit that improves nothing still falls back to
    a replay that is at least as exact.
    """

    request = _request(tmp_path)
    manifest = build_attachment(request)

    for report in manifest["layers"]:
        assert report["status"] == "fitted"
        assert report["col_source"] in {
            "norm_ratio",
            "median",
            "fitted",
            "lsb_least_squares",
        }
        baseline = report["median"]
        if report["col_source"] == "norm_ratio":
            # A norm ratio is exact by construction and has to win outright.
            assert report["int8_match_rate"] == 1.0
            continue
        assert report["int8_match_rate"] >= baseline["int8_match_rate"]
        if report["int8_match_rate"] == baseline["int8_match_rate"]:
            assert report["max_iterator_diff"] <= baseline["max_iterator_diff"]


def test_weighted_input_scale_recovers_the_planted_factor_up_to_one_constant():
    """The least-squares solve inverts the planted smoothing factor.

    The replayable quantity is the outer product, so the recovered column is
    only defined up to the constant the row estimate absorbs; every column has
    to agree on that same constant.
    """

    rng = np.random.default_rng(20261010)
    weight = rng.uniform(0.2, 1.0, size=(64, HIDDEN)).astype(np.float32)
    planted_col = rng.uniform(0.5, 3.0, size=HIDDEN)
    planted_row = rng.uniform(0.8, 1.2, size=64)
    quantized, scale = _quantize(weight, planted_col, planted_row)

    recovered = _weighted_input_scale(
        weight,
        quantized.astype(np.float64),
        scale.astype(np.float64),
        planted_row.astype(np.float64),
        127.0,
    )

    ratio = recovered / planted_col
    assert np.allclose(ratio, ratio.mean(), rtol=0.05)


def test_requant_keeps_the_artifacts_asymmetric_int8_floor():
    """The replay has to reach -128, because the producer saturates there.

    A symmetric ``[-127, 127]`` clamp collapses that floor one step short, which
    is exactly the LSB the artifact carries on its most negative element.
    """

    weight = np.full((2, 4), -1.0, dtype=np.float32)
    col = np.ones(4, dtype=np.float32)
    row = np.ones(2, dtype=np.float32)

    quantized, scale = _requant(weight, col, row, 128.0)

    assert int(quantized.min()) == -128
    assert np.allclose(scale, 1.0 / 128.0)


def test_online_view_declares_every_smoothed_linear_and_links_the_checkpoint(tmp_path: Path):
    """The host indexes the description by ``<prefix>.weight`` for every linear.

    One missing entry would leave that layer in BF16 next to replayed
    neighbours, so the view has to describe the whole model while keeping the
    weights in the checkpoint instead of copying them.
    """

    request = _request(tmp_path)
    build_attachment(request)
    output_path = tmp_path / "view"

    view = build_online_view(
        model_path=request.model_path,
        attachment_path=request.output_path,
        output_path=output_path,
    )

    description = json.loads((output_path / DESCRIPTION_FILENAME).read_text(encoding="utf-8"))
    expected = {f"{prefix}.weight" for prefix in SHARDS}
    expected.update(f"{module}.weight" for module in NORM_FACTORS)
    assert set(description) == expected
    assert all(description[f"{prefix}.weight"] == ONLINE_QUANT_TYPE for prefix in SHARDS)
    assert all(description[f"{module}.weight"] == FLOAT_QUANT_TYPE for module in NORM_FACTORS)
    # The attachment carries the host's packed alias; the description must not,
    # because the checkpoint itself only has the three shards.
    assert f"{LAYER}.self_attn.qkv_proj.weight" not in description
    assert view["description"]["linears"] == len(SHARDS)
    assert view["description"]["floats"] == len(NORM_FACTORS)
    assert view["description"]["sha256"] == sha256_file(output_path / DESCRIPTION_FILENAME)
    assert json.loads((output_path / ONLINE_VIEW_FILENAME).read_text(encoding="utf-8"))["kind"] == (
        ONLINE_VIEW_KIND
    )
    assert view["attachment"]["attachment_id"] == validate_attachment(request.output_path)["attachment_id"]
    # The weights stay in the BF16 checkpoint: the view only links them.
    linked = output_path / "model.safetensors"
    assert linked.is_symlink()
    assert linked.resolve() == (request.model_path / "model.safetensors").resolve()


def test_online_view_fails_closed_on_an_uncovered_linear(tmp_path: Path):
    """A linear the attachment misses must abort the build, not silently stay BF16."""

    request = _request(tmp_path)
    build_attachment(request)
    weights = load_file(str(request.model_path / "model.safetensors"))
    weights["model.layers.1.self_attn.o_proj.weight"] = weights[f"{LAYER}.self_attn.o_proj.weight"]
    save_file(weights, str(request.model_path / "model.safetensors"))

    with pytest.raises(ValueError, match="does not cover every model linear"):
        build_online_view(
            model_path=request.model_path,
            attachment_path=request.output_path,
            output_path=tmp_path / "view",
        )


def test_online_view_refuses_to_reuse_an_occupied_directory(tmp_path: Path):
    request = _request(tmp_path)
    build_attachment(request)
    output_path = tmp_path / "view"
    output_path.mkdir()
    (output_path / "stale.txt").write_text("occupied", encoding="utf-8")

    with pytest.raises(ValueError, match="is not empty"):
        build_online_view(
            model_path=request.model_path,
            attachment_path=request.output_path,
            output_path=output_path,
        )


def test_reader_decodes_bf16_weights_without_pytorch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The offline host that fits the factors has no PyTorch installed.

    A checkpoint weight is BF16 and numpy has no bfloat16 dtype, so reading it
    through a framework backend would import torch; the reader widens the raw
    16-bit pattern instead.  Earlier fixtures were float32, which hid the torch
    dependency until the attachment fixture ran on a runner without a torch
    runtime.
    """

    # These values are exactly representable in BF16, so a wrong widening shows
    # up as a mismatch instead of as rounding noise.
    values = np.array([[0.5, -1.25, 3.0], [0.0, 2.5, -0.75]], dtype=np.float32)
    layer = f"{LAYER}.self_attn.q_proj.weight"
    model_path = tmp_path / "bf16"
    model_path.mkdir()
    _write_bf16_shard(model_path / "model.safetensors", layer, values)

    # ``None`` in ``sys.modules`` turns any ``import torch`` into an ImportError.
    monkeypatch.setitem(sys.modules, "torch", None)
    reader = TensorReader(model_path, {"__single__": "model.safetensors"}, "source")

    assert reader.names() == [layer]
    read_back = reader.get(layer)
    assert read_back.dtype == np.float32
    assert np.array_equal(read_back, values)
    assert reader.get(f"{LAYER}.mlp.down_proj.weight") is None

