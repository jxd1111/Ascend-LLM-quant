# Changelog

## Unreleased

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
