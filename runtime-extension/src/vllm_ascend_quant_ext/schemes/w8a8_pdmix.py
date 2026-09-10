"""Namespaced adapter for vLLM-Ascend's native W8A8 PDMix schemes.

The PDMix algorithm and operators remain owned by vLLM-Ascend. This module
only registers plugin-owned aliases, so it does not duplicate or fork the host
implementation.
"""

from __future__ import annotations

from vllm_ascend.quantization.methods import get_scheme_class, register_scheme
from vllm_ascend.quantization.methods.w8a8_pdmix import (
    AscendW8A8PDMixFusedMoeMethod,
    AscendW8A8PDMixLinearMethod,
)

RUNTIME_QUANT_TYPE = "JXD_W8A8_PDMIX"


class JXDW8A8PDMixLinearMethod(AscendW8A8PDMixLinearMethod):
    """Plugin-owned alias of the host's native PDMix linear scheme."""


class JXDW8A8PDMixFusedMoeMethod(AscendW8A8PDMixFusedMoeMethod):
    """Plugin-owned alias of the host's native PDMix MoE scheme."""


def register_schemes() -> list[str]:
    registered: list[str] = []
    linear = get_scheme_class(RUNTIME_QUANT_TYPE, "linear")
    if linear is None:
        register_scheme(RUNTIME_QUANT_TYPE, "linear")(JXDW8A8PDMixLinearMethod)
        registered.append(f"{RUNTIME_QUANT_TYPE}/linear")
    elif linear is not JXDW8A8PDMixLinearMethod:
        raise RuntimeError(
            f"scheme registration collision: {RUNTIME_QUANT_TYPE}/linear"
        )
    moe = get_scheme_class(RUNTIME_QUANT_TYPE, "moe")
    if moe is None:
        register_scheme(RUNTIME_QUANT_TYPE, "moe")(JXDW8A8PDMixFusedMoeMethod)
        registered.append(f"{RUNTIME_QUANT_TYPE}/moe")
    elif moe is not JXDW8A8PDMixFusedMoeMethod:
        raise RuntimeError(f"scheme registration collision: {RUNTIME_QUANT_TYPE}/moe")
    return registered
