# Runtime Extension release checklist

Manifest identity

- [ ] `_version.py`, the Manifest 0.2 `extension_version`, the changelog and the
      tag agree; the tag has form `runtime-v<version>` and targets a commit
      contained in `main`.
- [ ] The `vllm_hust.extension_bundles` registration name is byte-identical to
      the manifest `extension_id`, and its value is the static manifest locator
      `vllm_ascend_quant_ext.manifests`.
- [ ] `kind`, `host`, `runtime`, `lifecycle_owner`, `protocols`, `components`
      and `permissions` describe the code that actually runs; no fabricated
      host API or protocol version is declared.
- [ ] The manifest `host.version_range` admits the tested baseline and the
      runtime exact-revision gate still fails closed outside it.

Runtime integration

- [ ] `vllm.general_plugins/vllm_ascend_quant` is the only runtime activation
      entry point; no new `vllm.*` namespace is claimed.
- [ ] Installation and discovery are default-off and device-free: `torch`,
      `vllm` and `vllm_ascend` stay unimported.
- [ ] Frozen vLLM-HUST and vLLM-Ascend-HUST revisions are documented and
      tested; `verify_host_sources.py` passes for the exact host commits.
- [ ] Invalid artifact, wrong host revision and duplicate scheme fail closed.

Packaging

- [ ] Wheel and sdist both contain
      `vllm_ascend_quant_ext/manifests/vllm-hust-extension-v0.2.json`; the
      retired legacy manifest and the retired PDMix-named scheme module are
      absent.
- [ ] `validate-evidence`-style manifest tests cover fields, enumerations,
      entry-point identity, implementation references and permission sets.
- [ ] Wheel and sdist pass `twine check`.
- [ ] Wheel contains no Toolkit calibration/conversion code or model data.
- [ ] Isolated install/discovery/uninstall verification passes
      (`runtime-extension/tools/verify_release.py`).
- [ ] Unit tests and Ruff pass on Python 3.10, 3.11 and 3.12.

Lifecycle and evidence

- [ ] Enable intent, disable plus new-process rollback, forget and uninstall
      leave no residual bundle registration or stale enabled intent.
- [ ] Real NPU validation records the exact wheel hash, host revisions, model,
      accelerator and raw log paths.
- [ ] The frozen-host gate (`runtime-extension/tools/npu_e2e.py`) ran against
      the exact wheel with no failed phase: the wheel resolved from
      `site-packages`, the loader activation record was captured, the
      disabled contrast left the scheme unregistered, two identical completions
      matched, uninstall left no entry point, and the artifact hashes were
      unchanged.
- [ ] Release notes and the manifest qualification block do not claim matched
      performance evidence that was not measured.

Publishing

- [ ] Distributions are rehearsed on TestPyPI first, then installed from
      TestPyPI at the exact version and re-verified before PyPI.
- [ ] PyPI release uses GitHub OIDC Trusted Publishing; no token is stored.

PyPI artifacts are immutable. A failed release is corrected with a new version
or yanked according to project policy; files are never overwritten.

