from ascend_quant_toolkit.registry import get_recipe, list_recipes


def test_builtin_recipe_is_discoverable():
    assert "qwen25-w8a8" in {recipe.name for recipe in list_recipes()}
    recipe = get_recipe("qwen25-w8a8")
    assert recipe.producer_quant_type == "W8A8_MIX"
    assert recipe.runtime_quant_type == "ASCEND_QUANT_W8A8"
    assert recipe.quant_scheme == "W8A8"
