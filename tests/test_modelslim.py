from pathlib import Path

import pytest

from jxd_ascend_quant.modelslim import QuantizeRequest, validate_request
from jxd_ascend_quant.registry import get_recipe


def make_request(model: Path, output: Path) -> QuantizeRequest:
    return QuantizeRequest(
        recipe=get_recipe("qwen25-w8a8"),
        model_path=model,
        output_path=output,
        device="npu:7",
        executable="python",
    )


def test_validate_request_accepts_new_output(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    validate_request(make_request(model, tmp_path / "output"))


def test_validate_request_rejects_nonempty_output(tmp_path: Path):
    model = tmp_path / "model"
    output = tmp_path / "output"
    model.mkdir()
    output.mkdir()
    (output / "existing").write_text("keep me")

    with pytest.raises(ValueError, match="not empty"):
        validate_request(make_request(model, output))


def test_validate_request_rejects_output_inside_model(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()

    with pytest.raises(ValueError, match="child"):
        validate_request(make_request(model, model / "quantized"))
