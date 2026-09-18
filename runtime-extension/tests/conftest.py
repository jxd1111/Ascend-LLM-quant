"""Shared fixtures for the runtime extension test suite."""

from __future__ import annotations

from importlib.metadata import distributions

import pytest
from packaging.utils import canonicalize_name

DIST_NAME = "vllm-ascend-quant-ext"


@pytest.fixture(scope="session")
def installed_extension_versions() -> list[str]:
    """Every installed distribution metadata version for this package.

    Editable checkouts and in-tree build leftovers can leave more than one
    metadata record for the same distribution. Returning all distinct versions
    lets a test fail closed instead of silently reading a stale record.
    """

    target = canonicalize_name(DIST_NAME)
    versions = {
        str(distribution.version)
        for distribution in distributions()
        if distribution.metadata.get("Name")
        and canonicalize_name(distribution.metadata["Name"]) == target
    }
    return sorted(versions)
