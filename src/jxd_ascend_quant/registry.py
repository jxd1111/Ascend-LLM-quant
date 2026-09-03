"""Recipe discovery through Python entry points with source-tree fallback."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Callable

from .config import Recipe
from .recipes.qwen25_w8a8_pdmix import get_recipe as get_builtin_pdmix

ENTRY_POINT_GROUP = "jxd_ascend_quant.recipes"


def _builtins() -> dict[str, Callable[[], Recipe]]:
    return {"qwen25-w8a8-pdmix": get_builtin_pdmix}


def _entry_point_factories() -> dict[str, Callable[[], Recipe]]:
    factories: dict[str, Callable[[], Recipe]] = {}
    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        factories[entry_point.name] = entry_point.load()
    return factories


def recipe_factories() -> dict[str, Callable[[], Recipe]]:
    factories = _builtins()
    factories.update(_entry_point_factories())
    return factories


def list_recipes() -> list[Recipe]:
    return [factory() for _, factory in sorted(recipe_factories().items())]


def get_recipe(name: str) -> Recipe:
    try:
        recipe = recipe_factories()[name]()
    except KeyError as exc:
        available = ", ".join(sorted(recipe_factories()))
        raise KeyError(f"Unknown recipe {name!r}. Available recipes: {available}") from exc
    if recipe.name != name:
        raise ValueError(
            f"Recipe entry point {name!r} returned recipe named {recipe.name!r}"
        )
    return recipe

