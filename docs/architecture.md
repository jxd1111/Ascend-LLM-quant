# Architecture and interface boundary

## Toolkit

The root `ascend-quant-toolkit` distribution owns offline work only:

1. calibration and recipe selection;
2. ModelSlim invocation;
3. conversion and artifact construction;
4. runtime-contract and provenance export;
5. PPL, accuracy, throughput and HBM evidence collection.

It has no vLLM entry point. Importing or installing it cannot alter a serving
process.

## Runtime extension

`runtime-extension/` builds the independent `vllm-ascend-quant-ext` wheel. Its
responsibilities are deliberately narrow:

1. identify `ascend_quant_artifact.json`;
2. validate the closed contract and safetensors headers;
3. enforce host/software/model/shape compatibility before implementation import;
4. map the admitted runtime quant type to a typed vLLM-Ascend scheme;
5. select only declared operators;
6. fail closed when no compatible path exists.

The implementation uses `vllm_ascend.quantization.methods.register_scheme` and
does not monkey patch scheduler, loader, model, or operator modules.

## Extension Manager

The wheel contains a static 0.2-experimental manifest. Manager operations are
pure discover/check/plan/render operations. They produce environment and plugin
selection data but do not modify model files, install calibration assets, or
import implementation modules during admission.

## Adaptive Quantized KV

This repository owns model-weight quantization only. KV-cache storage formats,
KV allocation, compression, eviction, request scheduling, and the
`vllm-ascend-adaptive-quantized-kv-hust` package remain independent. There is no
shared manifest, activation flag, lifecycle owner, or compatibility claim.
