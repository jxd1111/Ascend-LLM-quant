# Changelog

All notable changes are documented here. The project uses semantic versioning
for each Python distribution.

## 0.3.0 - 2026-09-03

- separated the offline Toolkit and runtime extension distributions;
- added versioned, fail-closed Ascend quantized artifact admission;
- added namespaced W8A8 `JXD_W8A8_PDMIX` aliases that delegate to the native
  vLLM-Ascend `W8A8_MIX` implementations;
- added proposed Extension Manager manifest and pure-data adapter;
- documented current vLLM/vLLM-Ascend architecture and plugin I/O;
- added wheel discovery/uninstall/model-immutability verification;
- recorded traceable W8A8 NPU E2E, PPL and ShareGPT evidence.

The Extension Manager schema remains experimental and formal Manager admission
is not claimed in this release.
