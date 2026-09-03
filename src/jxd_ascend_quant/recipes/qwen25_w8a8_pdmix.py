"""Qwen2.5 W8A8 PDMix recipe."""

from jxd_ascend_quant.config import Recipe


def get_recipe() -> Recipe:
    return Recipe(
        name="qwen25-w8a8-pdmix",
        description=(
            "Qwen2.5-14B W8A8 PDMix with SmoothQuant alpha=0.30, "
            "C4 calibration, per-channel weights, and lm_head excluded"
        ),
        model_type="Qwen2.5-14B-Instruct",
        config_resource="recipes/yaml/qwen25-w8a8-pdmix.yaml",
        producer_quant_type="W8A8_MIX",
        runtime_quant_type="JXD_W8A8_PDMIX",
        calibration_dataset="c4",
        weight_dtype="int8",
        activation_dtype="int8",
        quant_scheme="W8A8",
    )
