# Architecture and interface boundary

## Offline Toolkit

The root `ascend-quant-toolkit` distribution owns calibration, ModelSlim
conversion, artifact construction, contract export and evaluation evidence. It
has no vLLM entry point and never runs in the serving process.

## Runtime extension

`runtime-extension/` builds the independent `vllm-ascend-quant-ext` wheel. It:

1. validates `ascend_quant_artifact.json` and all bound files;
2. admits only the frozen vLLM-HUST/vLLM-Ascend-HUST revisions;
3. registers `ASCEND_QUANT_W8A8` through vLLM's native
   `vllm.general_plugins` callback;
4. delegates implementation to the installed vLLM-Ascend W8A8 linear and MoE
   schemes;
5. fails closed on unknown format, shape, software or registration collision.

The extension does not monkey patch the scheduler, loader, model or operator
modules. vLLM owns the plugin process lifecycle. Installation is default-off;
activation applies only to a newly started vLLM process.

## Frozen host

The supported source pair is vLLM-HUST `6cff125127ba` and
vLLM-Ascend-HUST `203a33e677ac`. Compatibility is intentionally not generalized
to every release in the same numeric version family.

## Adaptive Quantized KV

KV-cache formats, allocation, compression, eviction and request scheduling are
outside this repository. No manifest, activation flag or compatibility claim
is shared with Adaptive Quantized KV.
