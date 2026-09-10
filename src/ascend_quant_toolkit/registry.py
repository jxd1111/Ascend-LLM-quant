"""Recipe discovery through Python entry points with source-tree fallback."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Callable

from .config import Recipe
from .recipes.qwen25_w8a8 import get_recipe as get_builtin_w8a8

ENTRY_POINT_GROUP = "ascend_quant_toolkit.recipes"


def _builtins() -> dict[str, Callable[[], Recipe]]:
    return {"qwen25-w8a8": get_builtin_w8a8}


def _entry_point_factories() -> dict[str, Callable[[], Recipe]]:
    factories: dict[str, Callable[[], Recipe]] = {}
    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        distribution = getattr(entry_point, "dist", None)
        distribution_name = getattr(distribution, "name", "")
        if distribution_name.lower().replace("_", "-") == "ascend-quant-toolkit":
            # Built-ins above are authoritative for the current source tree.
            # Ignoring this distribution's installed entry point also makes an
            # editable-install upgrade safe after a recipe rename.
            continue
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
