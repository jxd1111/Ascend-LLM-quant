"""Namespaced adapter for the validated vLLM-Ascend W8A8 implementation.

The selected algorithm and operators remain owned by vLLM-Ascend. This module
only registers extension-owned aliases, so it does not duplicate or fork the
host implementation.
"""

from __future__ import annotations

from vllm_ascend.quantization.methods import (
    AscendW8A8PDMixLinearMethod,
    get_scheme_class,
    register_scheme,
)

RUNTIME_QUANT_TYPE = "ASCEND_QUANT_W8A8"


class AscendQuantW8A8LinearMethod(AscendW8A8PDMixLinearMethod):
    """Extension-owned alias of the selected native W8A8 linear scheme."""


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
    return registered
