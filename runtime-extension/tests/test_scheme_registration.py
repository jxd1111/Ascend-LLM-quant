import importlib
import sys
import types

import pytest


def _load_scheme_module(monkeypatch):
    registry = {}

    def get_scheme_class(runtime_type, layer_type):
        return registry.get((runtime_type, layer_type))

    def register_scheme(runtime_type, layer_type):
        def decorate(cls):
            registry[(runtime_type, layer_type)] = cls
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
    methods.AscendW8A8PDMixLinearMethod = type("NativeLinear", (), {})
    monkeypatch.setitem(sys.modules, "vllm_ascend", vllm_ascend)
    monkeypatch.setitem(sys.modules, "vllm_ascend.quantization", quantization)
    monkeypatch.setitem(sys.modules, "vllm_ascend.quantization.methods", methods)
    module_name = "vllm_ascend_quant_ext.schemes.w8a8"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    return importlib.import_module(module_name), registry


def test_registration_is_idempotent(monkeypatch):
    module, registry = _load_scheme_module(monkeypatch)
    assert module.register_schemes() == ["ASCEND_QUANT_W8A8/linear"]
    assert module.register_schemes() == []
    assert len(registry) == 1


def test_registration_collision_fails_closed(monkeypatch):
    module, registry = _load_scheme_module(monkeypatch)
    registry[(module.RUNTIME_QUANT_TYPE, "linear")] = type("ForeignLinear", (), {})
    with pytest.raises(RuntimeError, match="registration collision"):
        module.register_schemes()
