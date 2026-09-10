"""Namespaced adapter for the validated vLLM-Ascend W8A8 implementation.

The selected algorithm and operators remain owned by vLLM-Ascend. This module
only registers extension-owned aliases, so it does not duplicate or fork the
host implementation.
"""

from __future__ import annotations

from vllm_ascend.quantization.methods import get_scheme_class, register_scheme
from vllm_ascend.quantization.methods.w8a8_pdmix import (
    AscendW8A8PDMixFusedMoeMethod,
    AscendW8A8PDMixLinearMethod,
)

RUNTIME_QUANT_TYPE = "ASCEND_QUANT_W8A8"


class AscendQuantW8A8LinearMethod(AscendW8A8PDMixLinearMethod):
    """Extension-owned alias of the selected native W8A8 linear scheme."""


class AscendQuantW8A8FusedMoeMethod(AscendW8A8PDMixFusedMoeMethod):
    """Extension-owned alias of the selected native W8A8 MoE scheme."""


def register_schemes() -> list[str]:
    registered: list[str] = []
    linear = get_scheme_class(RUNTIME_QUANT_TYPE, "linear")
    if linear is None:
        register_scheme(RUNTIME_QUANT_TYPE, "linear")(AscendQuantW8A8LinearMethod)
        registered.append(f"{RUNTIME_QUANT_TYPE}/linear")
    elif linear is not AscendQuantW8A8LinearMethod:
        raise RuntimeError(
            f"scheme registration collision: {RUNTIME_QUANT_TYPE}/linear"
        )
    moe = get_scheme_class(RUNTIME_QUANT_TYPE, "moe")
    if moe is None:
        register_scheme(RUNTIME_QUANT_TYPE, "moe")(AscendQuantW8A8FusedMoeMethod)
        registered.append(f"{RUNTIME_QUANT_TYPE}/moe")
    elif moe is not AscendQuantW8A8FusedMoeMethod:
        raise RuntimeError(f"scheme registration collision: {RUNTIME_QUANT_TYPE}/moe")
    return registered
