# Changelog

## vllm-ascend-quant-ext 0.5.0 - 2026-10-12

- Bump the Runtime Extension to `0.5.0`: the ONLINE serving path is a new
  capability with its own frozen-host evidence, so it ships as a new immutable
  distribution instead of reusing the already published `0.4.1a4` files.
- Generate the ONLINE serving view the replay path needs:
  `ascend-quant-toolkit build-online-view --model <bf16> --attachment <dir>
  --output <view>` links the BF16 checkpoint (no weight is copied or rewritten),
  writes the per-linear `quant_model_description.json` the host looks up as
  `<prefix>.weight` (`ASCEND_QUANT_W8A8_ONLINE` for every linear the attachment
  covers, `FLOAT` for every other weight) and fails closed when the checkpoint
  and the attachment disagree in either direction.
- Teach `tools/npu_e2e.py` the ONLINE group: `--online-view`/`--smoothing` serve
  the generated view with `VLLM_ASCEND_QUANT_EXT_SMOOTHING` on `--online-port`
  (`--port + 1`) and require the online answer to equal the offline artifact's.
  The gate refuses an attachment without `ascend_quant_smoothing.json` or with
  `verified: false` before it installs anything, so a producer that never
  verified its fit cannot be mistaken for a device result.
- Judge entry points per distribution in the gate: `RESOLUTION_PROBE` and
  `UNINSTALL_PROBE` filter `vllm.general_plugins` and `vllm_hust.extension_bundles`
  to `vllm-ascend-quant-ext`, record the environment-wide bundle list separately
  as `bundles_environment`, and no longer report a frozen host's unrelated bundle
  (for example `vllm_ascend_simllm`) as this extension's leftover or as a
  wheel-resolution mismatch.
- Score every input-factor estimator on the same replay and keep the most
  exact one: the un-refined median now stays a candidate next to its
  EM-refined and LSB-weighted least-squares siblings, and equal match rates are
  broken by the tighter worst-case LSB deviation. A fit can therefore no longer
  end up worse than the plain median it starts from, and a tie can no longer
  keep the fuzzier replay.
- Replay the artifact's INT8 range faithfully on both sides of the contract: the
  online scheme and `build-smoothing` now saturate at `[-128, 127]` like the
  ModelSlim producer, while the previous symmetric `[-127, 127]` clamp rejected
  the most negative element of a shard and cost one LSB on that channel.
- Replay the smoothing factors on the weight's device: the attachment is read
  from a CPU-side safetensors file while the host has already placed every
  parameter on the accelerator for `process_weights_after_loading`, so
  `col_scale`/`row_scale` now follow `weight.device` before the replay
  multiplies them, exactly as the norm side already did.  Without this the
  ONLINE serve died with `Expected all tensors to be on the same device` on the
  first replayed linear, after the offline half of the gate had passed.
- Require the ONLINE modules in the release check: `verify_release.py` now fails
  when `vllm_ascend_quant_ext/attachment.py` or
  `vllm_ascend_quant_ext/schemes/online_w8a8.py` is missing from the wheel or the
  sdist, so a build that predates the ONLINE path can never be published as the
  release that claims it.

## ascend-quant-toolkit 0.4.2 - 2026-10-12

- Bump the offline Toolkit to `0.4.2` and publish it as its own immutable
  distribution: `publish-toolkit.yml` (tag `toolkit-v<version>`) and
  `rehearse-toolkit-testpypi.yml` mirror the runtime pair. The ONLINE serving
  path needs the view generator, so the Toolkit can no longer stay outside the
  PyPI release; the `0.4.1a1` note that kept it outside described that release
  only.
- Derive the distribution version from `ascend_quant_toolkit.__version__` in
  `pyproject.toml` instead of repeating it, so the publish-time tag check and
  the installed metadata cannot drift, and declare the Homepage, Documentation
  and Issues URLs.
- Generate the ONLINE serving view: `ascend-quant-toolkit build-online-view
  --model <bf16> --attachment <dir> --output <view>` links the BF16 checkpoint
  without copying a weight, writes the per-linear `quant_model_description.json`
  that the host resolves as `<prefix>.weight` (`ASCEND_QUANT_W8A8_ONLINE`), and
  fails closed when the checkpoint and the attachment disagree in either
  direction.
- Add `ascend_quant_toolkit.attachment`: fit and emit the per-channel smoothing
  factors (`build-smoothing`) with the producer-side replay verification the
  runtime requires, record every piece of derivation evidence in
  `ascend_quant_smoothing.json`, and saturate the replayed INT8 range at
  `[-128, 127]` like the ModelSlim producer instead of the previous symmetric
  clamp.
- Keep the wheel dependency-free: the Toolkit runtime modules import only the
  standard library, and the publish workflow proves it by installing the built
  wheel with `--no-deps` and running the console script.

## vllm-ascend-quant-ext 0.4.1a4 - 2026-09-18

- Bump the Runtime Extension to `0.4.1a4` so the Manifest 0.2-aware packaging
  can be published as a new immutable distribution.
- Register the Runtime Extension as a vLLM-HUST Extension Manager bundle: ship
  `manifests/vllm-hust-extension-v0.2.json`, declare
  `vllm_hust.extension_bundles/org.vllm-hust.ascend-quant-runtime` as a static
  manifest locator, and add the repository-level `.vllm-hust/optimization.json`.
- Declare `kind: in_process_plugin`, `lifecycle_owner: vllm`, the frozen
  host package range, the consumed `vllm.general_plugins` and
  `vllm-ascend.quantization-scheme-surface` protocols, and the minimal
  `device_access`/`filesystem_read` permissions. No host API version and no
  deployment path is fabricated in the manifest.
- Keep `vllm.general_plugins/vllm_ascend_quant` as the only runtime activation
  entry point; the bundle registration is a manifest directory locator, not an
  import-and-execute callback.
- Admit abbreviated `setuptools-scm` local-version tokens. The frozen v1 host
  installs as `0.28.1.post1.dev143+gf18cf803c.empty` while the allowlist token
  is `gf18cf803c5`, so the previous substring check rejected the exact frozen
  host; revision admission now parses the `g<sha>` token and compares full
  commits.
- Add `tests/test_manifest.py`, `tests/test_packaging.py` and
  `tests/test_host_contract.py`: manifest fields, enumerations and identity,
  entry-point/`extension_id` agreement, wheel and sdist manifest shipping,
  discovery without importing the implementation, default-off registration and
  fail-closed admission.
- `verify_release.py` now requires the Manifest 0.2 in the wheel and the sdist,
  requires the bundle registration to resolve to the manifest locator, and still
  rejects the retired legacy manifest and the retired PDMix-named scheme module.
- Rewrite `docs/release-checklist.md` around Manifest identity, runtime
  integration, packaging, lifecycle and TestPyPI-first publishing.
- Make the bundle acceptable to the Extension Manager, verified with its own
  validator and CLI: `components[].contracts` stays inside the `vllm.`
  namespace, `protocols[].version_range` is declared as absent because the
  frozen host exposes no independently versioned protocol surface, and the
  runtime qualification profile no longer shadows the Manager-reserved `status`
  key (moved to `qualification_scope`).
- Add `tools/npu_e2e.py`, a reproducible frozen-host gate that installs the
  wheel, activates it through vLLM's loader, serves the artifact, then
  uninstalls, restores the previous install and re-hashes the artifact; add
  `tests/test_npu_e2e.py` for its safety invariants.

## vllm-ascend-quant-ext 0.4.1a3 - 2026-09-16

- Retarget the runtime contract to the frozen vLLM-HUST `v1` ref at
  `f18cf803c5` and the adjacent vLLM-Ascend-HUST snapshot `74f0c0a272`.
- Align the declared stack with torch `2.13.0`, torch-npu `2.13.0rc1` and
  CANN `9.1.x`.
- Update the native W8A8 adapter to the frozen Ascend public import surface.
- Restrict the runtime alias to dense linear layers because that snapshot does
  not expose a PDMix fused-MoE implementation.
- Keep earlier validation records immutable; v1 NPU end-to-end validation is
  required before this alpha can be published as verified.
- Add an executable host-source lock and CI job for exact revisions, merge
  ancestry, Ascend verified-core provenance and the public plugin/W8A8 APIs.

## vllm-ascend-quant-ext 0.4.1a2 - 2026-09-16

- Make vLLM's native `vllm.general_plugins` entry point the sole runtime
  integration surface.
- Remove the experimental Extension Manager Bundle, manifest and adapter.
- Pin admission to vLLM-HUST revision `6cff125127ba` and
  vLLM-Ascend-HUST revision `203a33e677ac`.
- Preserve default-off installation, fail-closed artifact admission, scheme
  collision checks, rollback and isolated uninstall verification.

## vllm-ascend-quant-ext 0.4.1a1 - 2026-09-15

- Prepare the first public PyPI alpha of the Runtime Extension.
- Keep the Extension Manager integration fail-closed and `import_only` until
  the typed host protocols are accepted.
- Publish only the independent `runtime-extension` distribution; the offline
  Toolkit remains outside the vLLM process and this PyPI release.

- Bump Toolkit and Runtime Extension to `0.4.1` and artifact contract to
  `1.1.0`.
- Bind config, quantization description, safetensors shards/indexes, and
  claimed evidence records by exact size and SHA-256; derive `artifact_id`
  from the canonical model-file inventory.
- Reject out-of-bounds, overlapping, non-contiguous, or dtype/shape-inconsistent
  safetensors payloads and inconsistent shard indexes.
- Freeze scheme-specific W8A8/W4A4/W4A8 semantics, the required operator set,
  and the runtime scheme provider.
- Add a closed evaluation-evidence schema and stricter evidence/profile,
  hardware, request-count, metric, HBM, raw-log, and artifact-identity checks.
- Restrict the experimental Bundle host version and reduce the import-only
  validator permission set to filesystem read.
- Reject conflicting duplicate installed-package metadata during software
  compatibility admission instead of selecting an order-dependent version.
- Record the Runtime Extension 0.4.1 / contract 1.1 Ascend NPU revalidation,
  including disabled fail-closed and native rollback checks.

- Remove personal-name branding from the Toolkit's import package, CLI entry
  point, recipe entry-point group, artifact manifest, and package authorship;
  use the neutral `ascend_quant_toolkit` / `ascend-quant-toolkit` identities.
- Present the validated capability uniformly as W8A8: rename the built-in
  recipe to `qwen25-w8a8` and the extension-owned runtime type to
  `ASCEND_QUANT_W8A8`; the selected ModelSlim/backend algorithm remains an
  internal implementation detail.
- Bump Toolkit and Runtime Extension to `0.4.0`. Artifacts prepared with the
  retired runtime identifier must restore their ModelSlim metadata and export
  fresh runtime metadata and a fresh artifact contract.
- Align the Extension Bundle with Manifest `0.2-experimental` identity
  `org.vllm-hust.ascend-quant-runtime` and the `in_process_plugin` host model.
- Replace the advertised runtime carrier with an import-only artifact validator;
  enabled Manager plan/render now fail closed until vLLM-Ascend publishes the
  quantized-artifact loader and operator-selection protocols.
- Add duplicate scheme-registration rejection, source provenance, Python 3.12
  CI, linting, package metadata checks, and updated Manager hand-off documents.

All notable changes are documented here. The project uses semantic versioning
for each Python distribution.

## 0.3.0 - 2026-09-03

- separated the offline Toolkit and runtime extension distributions;
- added versioned, fail-closed Ascend quantized artifact admission;
- added namespaced W8A8 `ASCEND_QUANT_W8A8` aliases that delegate to the native
  vLLM-Ascend `W8A8_MIX` implementations;
- added proposed Extension Manager manifest and pure-data adapter;
- documented current vLLM/vLLM-Ascend architecture and plugin I/O;
- added wheel discovery/uninstall/model-immutability verification;
- recorded traceable W8A8 NPU E2E, PPL and ShareGPT evidence.

The Extension Manager schema remains experimental and formal Manager admission
is not claimed in this release.
