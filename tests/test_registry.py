from jxd_ascend_quant.registry import get_recipe, list_recipes


def test_builtin_recipe_is_discoverable():
    assert "qwen25-w8a8-pdmix" in {recipe.name for recipe in list_recipes()}
    recipe = get_recipe("qwen25-w8a8-pdmix")
    assert recipe.producer_quant_type == "W8A8_MIX"
    assert recipe.runtime_quant_type == "JXD_W8A8_PDMIX"
    assert recipe.quant_scheme == "W8A8"
