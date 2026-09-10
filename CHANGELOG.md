# Changelog

## Unreleased

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
