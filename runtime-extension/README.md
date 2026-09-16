# vllm-ascend-quant-ext

Runtime-only W8A8 extension for the project-frozen vLLM-HUST Ascend stack.
The package contains no calibration data, ModelSlim conversion, PPL evaluation
or benchmark orchestration.

## Supported host baseline

This alpha intentionally supports one tested source pair:

| Component | Frozen revision |
|---|---|
| vLLM-HUST | `6cff125127bac512488dc90a9812dcafddb65298` |
| vLLM-Ascend-HUST | `203a33e677ac6728108473e749069244bea00373` |
| torch / torch-npu | `2.9.0` |
| CANN | `>=8.5,<8.6` |

Admission checks both the declared package version and the Git revision token
embedded in the installed vLLM distributions. Unknown revisions fail closed.

## Integration boundary

The wheel uses vLLM's native Python plugin entry point:

```toml
[project.entry-points."vllm.general_plugins"]
vllm_ascend_quant = "vllm_ascend_quant_ext.plugin:register"
```

It does not register an Extension Manager Bundle and does not monkey patch
vLLM. vLLM owns process startup and shutdown; vLLM-Ascend continues to own
weight loading, parameter layout, W8A8 execution and NPU operators. The plugin
owns artifact admission and the namespaced `ASCEND_QUANT_W8A8` scheme alias.

Installation and discovery are side-effect free. The registration callback is
idempotent and remains disabled unless explicitly enabled.

## Install and inspect

```bash
python -m pip install vllm-ascend-quant-ext==0.4.1a2

vllm-ascend-quant-ext check --model /path/to/w8a8-model
vllm-ascend-quant-ext status
vllm-ascend-quant-ext render --model /path/to/w8a8-model
```

## Start vLLM

```bash
export VLLM_ASCEND_QUANT_EXT_ENABLE=1
export VLLM_ASCEND_QUANT_EXT_ARTIFACT=/path/to/w8a8-model

vllm-hust serve /path/to/w8a8-model \
  --host 0.0.0.0 \
  --port 18000
```

Do not set `VLLM_PLUGINS` to only `vllm_ascend_quant`: that variable filters
every plugin group and would suppress required vLLM-Ascend platform/general
plugins. The frozen host discovers installed general plugins automatically;
the extension-owned enable switch keeps this callback default-off.

The contract is validated before importing the vLLM-Ascend scheme module.
Missing files, hashes, tensors, software versions, frozen revisions or scheme
providers abort startup.

## Disable and uninstall

The plugin is process scoped; it is not hot-unloaded. Stop the old vLLM
process, then start a new process without the selection variables:

```bash
unset VLLM_ASCEND_QUANT_EXT_ENABLE
unset VLLM_ASCEND_QUANT_EXT_ARTIFACT
python -m pip uninstall -y vllm-ascend-quant-ext
```

Installation, validation and uninstall never modify the model directory. A
model whose active metadata names `ASCEND_QUANT_W8A8` still requires the
plugin; restoring native `W8A8_MIX` metadata is a separate offline Toolkit
operation.

This extension quantizes model weights and linear/MoE activations only. It does
not own KV-cache format, allocation, compression or request scheduling.
