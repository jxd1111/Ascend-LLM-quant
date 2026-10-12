"""Registration and replay tests for the online W8A8 scheme."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

from vllm_ascend_quant_ext.attachment import (
    COL_SUFFIX,
    ROW_SUFFIX,
    AttachmentError,
    SmoothingAttachment,
)

MODULE_NAME = "vllm_ascend_quant_ext.schemes.online_w8a8"

# The extension imports torch lazily inside each method - the host provides it -
# and the CI/publish jobs install no torch, vLLM or vLLM-Ascend.  Every assertion
# below replays real torch tensors, so the module skips as a whole on a torch-free
# runner; the import-free admission half stays covered everywhere by
# `test_online_attachment.py` and the tensor half by the frozen-host NPU gate.
torch = pytest.importorskip(
    "torch", reason="the ONLINE replay assertions build and check real torch tensors"
)


class _NativeDynamic:
    """Stand-in for the host's native dynamic W8A8 scheme."""

    def __init__(self) -> None:
        self.post_process_calls = 0

    def get_weight(self, input_size, output_size, params_dtype):
        return {"weight": torch.empty(output_size, input_size, dtype=params_dtype)}

    def process_weights_after_loading(self, layer):
        self.post_process_calls += 1


class _InMemoryAttachment(SmoothingAttachment):
    """Admitted-looking attachment backed by tensors instead of a file."""

    def __init__(self, tensors, packed_geometry=None) -> None:
        object.__setattr__(self, "directory", Path("."))
        object.__setattr__(
            self,
            "manifest",
            {
                "attachment_id": "in-memory",
                "consumer": {"runtime_quant_type": "ASCEND_QUANT_W8A8_ONLINE"},
            },
        )
        object.__setattr__(self, "tensors", {name: {} for name in tensors})
        object.__setattr__(self, "packed_geometry", packed_geometry or {})
        object.__setattr__(self, "_tensors", tensors)

    def open_tensor(self, name):
        return self._tensors[name]


class _FakeLayer:
    def __init__(
        self, prefix: str, weight: torch.Tensor, tp_rank: int = 0, device: str = "cpu"
    ) -> None:
        self.prefix = prefix
        self.tp_rank = tp_rank
        self.weight = torch.nn.Parameter(weight)
        output_size = int(weight.shape[0])
        self.weight_scale = torch.nn.Parameter(
            torch.zeros(output_size, 1, dtype=torch.float32, device=device)
        )
        self.weight_offset = torch.nn.Parameter(
            torch.zeros(output_size, 1, dtype=torch.float32, device=device)
        )


def _load_scheme_module(monkeypatch):
    registry: dict[tuple[str, str], type] = {}

    def get_scheme_class(quant_type, layer_type):
        return registry.get((quant_type, layer_type))

    def register_scheme(quant_type, layer_type):
        def decorate(cls):
            registry[(quant_type, layer_type)] = cls
            return cls

        return decorate

    vllm_ascend = types.ModuleType("vllm_ascend")
    quantization = types.ModuleType("vllm_ascend.quantization")
    methods = types.ModuleType("vllm_ascend.quantization.methods")
    vllm_ascend.__path__ = []
    quantization.__path__ = []
    methods.__path__ = []
    methods.get_scheme_class = get_scheme_class
    methods.register_scheme = register_scheme
    methods.AscendW8A8DynamicLinearMethod = _NativeDynamic
    monkeypatch.setitem(sys.modules, "vllm_ascend", vllm_ascend)
    monkeypatch.setitem(sys.modules, "vllm_ascend.quantization", quantization)
    monkeypatch.setitem(sys.modules, "vllm_ascend.quantization.methods", methods)
    monkeypatch.delitem(sys.modules, MODULE_NAME, raising=False)
    module = importlib.import_module(MODULE_NAME)
    monkeypatch.setattr(module, "_ACTIVE", None)
    return module, registry


def test_registration_requires_an_admitted_attachment(monkeypatch):
    module, _ = _load_scheme_module(monkeypatch)
    with pytest.raises(AttachmentError, match="requires an admitted smoothing attachment"):
        module.register_schemes()


def test_registration_is_idempotent(monkeypatch):
    module, registry = _load_scheme_module(monkeypatch)
    attachment = _InMemoryAttachment({})
    assert module.register_schemes(attachment) == ["ASCEND_QUANT_W8A8_ONLINE/linear"]
    assert module.register_schemes(attachment) == []
    assert len(registry) == 1


def test_registration_collision_fails_closed(monkeypatch):
    module, registry = _load_scheme_module(monkeypatch)
    registry[(module.RUNTIME_QUANT_TYPE, "linear")] = type("ForeignLinear", (), {})
    with pytest.raises(RuntimeError, match="registration collision"):
        module.register_schemes(_InMemoryAttachment({}))
def test_replay_keeps_the_artifacts_asymmetric_int8_floor(monkeypatch):
    """ModelSlim stores -128, so a symmetric clamp would shift that element."""

    module, _ = _load_scheme_module(monkeypatch)
    assert (module._QUANT_MIN, module._QUANT_MAX) == (-128, 127)


def _expected_quantization(weight, col, row):
    scaled = weight.to(torch.float32) * col[None, :] * row[:, None]
    scale = (scaled.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-8)
    quantized = torch.round(scaled / scale).clamp(-128, 127).to(torch.int8)
    return quantized, scale


def test_replay_quantizes_the_loaded_float_weight(monkeypatch):
    module, _ = _load_scheme_module(monkeypatch)
    prefix = "model.layers.0.self_attn.q_proj"
    weight = torch.tensor([[1.0, -2.0, 3.0, -4.0], [2.0, 1.0, -1.0, 0.5]])
    col = torch.tensor([1.0, 2.0, 0.5, 4.0])
    row = torch.tensor([1.0, 0.5])
    module.register_schemes(
        _InMemoryAttachment({f"{prefix}.{COL_SUFFIX}": col, f"{prefix}.{ROW_SUFFIX}": row})
    )
    layer = _FakeLayer(prefix, weight)
    method = module.AscendQuantW8A8OnlineLinearMethod()

    method.process_weights_after_loading(layer)

    expected_weight, expected_scale = _expected_quantization(weight, col, row)
    assert layer.weight.dtype == torch.int8
    assert torch.equal(layer.weight.data, expected_weight)
    assert torch.allclose(layer.weight_scale.data, expected_scale)
    assert method.post_process_calls == 1
    assert torch.count_nonzero(layer.weight_offset.data) == 0


def test_packed_layer_slices_each_shard_for_its_rank(monkeypatch):
    module, _ = _load_scheme_module(monkeypatch)
    alias = "model.layers.0.self_attn.qkv_proj"
    col = torch.arange(1.0, 5.0)
    # q_proj(4) + k_proj(2) + v_proj(2) in the host's packed order.
    row = torch.arange(1.0, 9.0)
    attachment = _InMemoryAttachment(
        {f"{alias}.{COL_SUFFIX}": col, f"{alias}.{ROW_SUFFIX}": row},
        packed_geometry={alias: [4, 2, 2]},
    )
    module.register_schemes(attachment)
    weight = torch.ones(4, 4)
    layer = _FakeLayer(alias, weight, tp_rank=1)

    module.AscendQuantW8A8OnlineLinearMethod().process_weights_after_loading(layer)

    local_row = torch.tensor([3.0, 4.0, 6.0, 8.0])
    expected_weight, expected_scale = _expected_quantization(weight, col, local_row)
    assert torch.equal(layer.weight.data, expected_weight)
    assert torch.allclose(layer.weight_scale.data, expected_scale)


def test_unadmitted_prefix_fails_closed(monkeypatch):
    module, _ = _load_scheme_module(monkeypatch)
    module.register_schemes(_InMemoryAttachment({}))
    layer = _FakeLayer("model.layers.0.mlp.down_proj", torch.ones(2, 2))

    with pytest.raises(AttachmentError, match="carries no smoothing factors"):
        module.AscendQuantW8A8OnlineLinearMethod().process_weights_after_loading(layer)


def test_replay_follows_the_weight_device(monkeypatch):
    """The factors are read on the CPU, so they must move to the weight's device.

    ``meta`` is the only second device a unit test can build without an
    accelerator, and it propagates through every replay operator, so a CPU-side
    factor surfaces as a device mismatch instead of a silently wrong result.
    """

    module, _ = _load_scheme_module(monkeypatch)
    prefix = "model.layers.0.self_attn.q_proj"
    module.register_schemes(
        _InMemoryAttachment(
            {
                f"{prefix}.{COL_SUFFIX}": torch.arange(1.0, 5.0),
                f"{prefix}.{ROW_SUFFIX}": torch.arange(1.0, 3.0),
            }
        )
    )
    layer = _FakeLayer(prefix, torch.ones(2, 4, device="meta"), device="meta")
    method = module.AscendQuantW8A8OnlineLinearMethod()

    method.process_weights_after_loading(layer)

    assert layer.weight.device.type == "meta"
    assert layer.weight.dtype == torch.int8
    assert layer.weight_scale.device.type == "meta"
    assert layer.weight_offset.device.type == "meta"
    assert method.post_process_calls == 1

