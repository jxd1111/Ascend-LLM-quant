# ADR 0001: Runtime quantization plugin boundary

- Status: accepted
- Date: 2026-09-03

## Context

ModelSlim calibration/conversion and vLLM-Ascend runtime quantization currently
have different lifecycle and dependency requirements. Packaging the whole
offline repository as a vLLM plugin would put datasets, conversion dependencies
and artifact mutation into the serving environment.

## Decision

Maintain two distributions:

- `ascend-quant-toolkit` owns offline artifact production and evidence;
- `vllm-ascend-quant-ext` owns read-only artifact admission, namespaced runtime
  schemes and vLLM-native registration.

vLLM-Ascend retains platform, operator, parallelism, compiler and typed host SPI
ownership. Adaptive Quantized KV remains an independent extension.

Production activation uses the native `vllm.general_plugins` entry point and
is explicit through `VLLM_PLUGINS` plus extension-owned admission variables.
Unsupported or ambiguous artifacts fail closed.

## Consequences

- quantized artifacts survive plugin disable/uninstall;
- the runtime wheel stays small and avoids calibration dependencies;
- extracting additional algorithms depends on a stable vLLM-Ascend SPI;
- the frozen host revisions and artifact contracts must be versioned together;
- a host/plugin NPU E2E test is required before claiming integration.
