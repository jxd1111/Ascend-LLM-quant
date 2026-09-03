# ADR 0001: Runtime quantization plugin boundary

- Status: proposed
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
  schemes and Manager-facing lifecycle data.

vLLM-Ascend retains platform, operator, parallelism, compiler and typed host SPI
ownership. Adaptive Quantized KV remains an independent extension.

Production activation is Manager-managed and explicit. Direct environment
activation is diagnostic only. Unsupported or ambiguous artifacts fail closed.

## Consequences

- quantized artifacts survive plugin disable/uninstall;
- the runtime wheel stays small and avoids calibration dependencies;
- extracting additional algorithms depends on a stable vLLM-Ascend SPI;
- host-version ranges and artifact contracts must be versioned together;
- a joint Manager/host/plugin E2E test is required before claiming formal
  integration.
