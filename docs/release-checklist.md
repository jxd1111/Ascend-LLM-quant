# Runtime Extension release checklist

- [ ] Version in `_version.py`, changelog, tag and release title matches.
- [ ] Tag has form `runtime-v<version>` and targets a commit contained in `main`.
- [ ] Only `vllm.general_plugins/vllm_ascend_quant` is present.
- [ ] No `vllm_hust.extension_bundles` entry point or Manager manifest exists.
- [ ] Frozen vLLM-HUST and vLLM-Ascend-HUST revisions are documented and tested.
- [ ] `verify_host_sources.py` passes for the exact checked-out host commits.
- [ ] Installation and plugin discovery are default-off and device-free.
- [ ] Invalid artifact, wrong host revision and duplicate scheme fail closed.
- [ ] Unit tests and Ruff pass on Python 3.10, 3.11 and 3.12.
- [ ] Wheel and sdist pass `twine check`.
- [ ] Isolated install/discovery/uninstall verification passes.
- [ ] Wheel contains no Toolkit calibration/conversion code or model data.
- [ ] Real NPU validation records exact wheel hash, host revisions and model.
- [ ] Release notes do not claim unmatched performance evidence.
- [ ] PyPI release uses GitHub OIDC Trusted Publishing; no token is stored.

PyPI artifacts are immutable. A failed release is corrected with a new version
or yanked according to project policy; files are never overwritten.
