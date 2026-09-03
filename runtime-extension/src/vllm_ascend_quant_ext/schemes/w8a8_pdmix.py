"""W8A8 PDMix implementation using vLLM-Ascend's public scheme contract."""

from __future__ import annotations

from typing import Any

import torch
from vllm.config import get_current_vllm_config
from vllm_ascend.quantization.methods import get_scheme_class, register_scheme
from vllm_ascend.quantization.methods.base import AscendLinearScheme
from vllm_ascend.quantization.methods.w8a8_dynamic import (
    AscendW8A8DynamicFusedMoEMethod,
    AscendW8A8DynamicLinearMethod,
)
from vllm_ascend.quantization.methods.w8a8_static import AscendW8A8LinearMethod

RUNTIME_QUANT_TYPE = "JXD_W8A8_PDMIX"


class JXDW8A8PDMixLinearMethod(AscendLinearScheme):
    def __init__(self):
        self._static_method = AscendW8A8LinearMethod()
        self._dynamic_method = AscendW8A8DynamicLinearMethod()
        kv_transfer_config = get_current_vllm_config().kv_transfer_config
        self._is_kv_consumer = bool(
            kv_transfer_config is not None and kv_transfer_config.is_kv_consumer
        )

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        return self._static_method.get_weight(input_size, output_size, params_dtype)

    def get_pertensor_param(self, params_dtype: torch.dtype) -> dict[str, Any]:
        return self._static_method.get_pertensor_param(params_dtype)

    def get_perchannel_param(self, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        return self._static_method.get_perchannel_param(output_size, params_dtype)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None, tp_rank: int | None = 0) -> torch.Tensor:
        if layer.is_kv_consumer:
            return self._static_method.apply(layer, x, bias, tp_rank)
        return self._dynamic_method.apply(layer, x, bias, tp_rank)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._static_method.process_weights_after_loading(layer)
        layer.weight_scale_fp32 = layer.weight_scale.data.to(torch.float32)
        layer.is_kv_consumer = self._is_kv_consumer


class JXDW8A8PDMixFusedMoeMethod(AscendW8A8DynamicFusedMoEMethod):
    def get_dynamic_quant_param(self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype) -> dict[str, Any]:
        params = super().get_dynamic_quant_param(
            num_experts, intermediate_size_per_partition, hidden_sizes, params_dtype
        )
        params["w2_deq_scale"] = torch.empty(num_experts, hidden_sizes, dtype=torch.float32)
        params["w13_deq_scale"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, dtype=torch.float32
        )
        params["w2_input_offset"] = torch.empty(num_experts, 1, dtype=torch.int8)
        params["w13_input_offset"] = torch.empty(num_experts, 1, dtype=torch.int8)
        return params


def register_schemes() -> list[str]:
    registered: list[str] = []
    if get_scheme_class(RUNTIME_QUANT_TYPE, "linear") is None:
        register_scheme(RUNTIME_QUANT_TYPE, "linear")(JXDW8A8PDMixLinearMethod)
        registered.append(f"{RUNTIME_QUANT_TYPE}/linear")
    if get_scheme_class(RUNTIME_QUANT_TYPE, "moe") is None:
        register_scheme(RUNTIME_QUANT_TYPE, "moe")(JXDW8A8PDMixFusedMoeMethod)
        registered.append(f"{RUNTIME_QUANT_TYPE}/moe")
    return registered
