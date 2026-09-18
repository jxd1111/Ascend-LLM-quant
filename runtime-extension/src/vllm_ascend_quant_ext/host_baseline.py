"""Single source of truth for the frozen vLLM-HUST v1 host baseline."""

VLLM_HUST_REF = "v1"
VLLM_HUST_COMMIT = "f18cf803c5f63625e2c71253ddaf8b0bad0bad1a"
VLLM_HUST_FIRST_PARENT = "a67f6a5dda6e6b81eb42d0ef82c7fca864ca969f"
VLLM_UPSTREAM_COMMIT = "bfb443a6b6f670e68e211112a141d089f1cf956f"

VLLM_ASCEND_HUST_COMMIT = "74f0c0a272376412b51e1c1864803d5f3a0f1b5f"
VLLM_ASCEND_VERIFIED_CORE = VLLM_HUST_FIRST_PARENT

FROZEN_HOST_REVISIONS = {
    # These tokens match each repository's own configured Git abbreviation
    # and therefore the local version segment emitted by its build backend.
    "vllm": "gf18cf803c5",
    "vllm_ascend": "g74f0c0a27",
}

# Full commits are authoritative.  The generated local-version token is not:
# setuptools-scm may abbreviate the same commit to a different unambiguous
# length depending on the repository/object database used for the build.
FROZEN_HOST_COMMITS = {
    "vllm": VLLM_HUST_COMMIT,
    "vllm_ascend": VLLM_ASCEND_HUST_COMMIT,
}


def host_description() -> str:
    return (
        f"vllm-hust@{VLLM_HUST_REF}/{VLLM_HUST_COMMIT[:10]} + "
        f"vllm-ascend-hust@{VLLM_ASCEND_HUST_COMMIT[:10]}"
    )
