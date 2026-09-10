# vllm-ascend-quant-ext

Runtime-only extension for loading and executing versioned Ascend quantized
model artifacts. It contains no calibration datasets, ModelSlim invocation,
model conversion, PPL evaluation, or benchmark orchestration.

Installation alone changes no vLLM behavior. Admission is explicit and
fail-closed:

```bash
vllm-ascend-quant-ext check --model /path/to/model
vllm-ascend-quant-ext plan --model /path/to/model
vllm-ascend-quant-ext render --model /path/to/model
```

`plan` and `render` describe an explicit direct-diagnostic path. They do not
produce Extension Manager activation evidence. The plugin validates the model
contract before importing vLLM-Ascend implementation modules and never modifies
the model directory.

This extension is for model-weight quantization. It does not own KV-cache
format, allocation, compression, scheduling, or the lifecycle of
`vllm-ascend-adaptive-quantized-kv-hust`.

The proposed Manager adapter is exposed as a pure-data symbol:

```python
from vllm_ascend_quant_ext.manager import provider

provider.descriptor()
provider.check({"enabled": True, "model": "/path/to/model"})
provider.plan({"enabled": True, "model": "/path/to/model"})
provider.render({"enabled": True, "model": "/path/to/model"})
```

This adapter is import-only: inspection and artifact checking are supported,
but enabled `plan` and `render` fail closed until vLLM-Ascend publishes the
declared typed Host protocols. See `../docs/extension-manager-handoff.md`.

## Extension Bundle identity

```text
PyPI distribution: vllm-ascend-quant-ext
Python package:     vllm_ascend_quant_ext
Bundle ID:          org.vllm-hust.ascend-quant-runtime
Component ID:       ascend-quant-artifact-validator
Full component ID:  org.vllm-hust.ascend-quant-runtime/ascend-quant-artifact-validator
Contract:           vllm.ascend.quantized-artifact-loader.v1
Execution planes:   worker, device
Status:             import_only
```

The wheel exposes two distinct entry points:

- `vllm_hust.extension_bundles` discovers static Bundle metadata;
- `vllm.general_plugins` provides an explicit direct-diagnostic registration path.

The first is the Manager discovery target. The Bundle does not automatically
select the second entry point or inject environment/additional-config values.
That prevents partial integration from starting an unsafe legacy path. Formal
activation remains blocked until vLLM-Ascend accepts and materializes both
declared Host protocols.

The namespaced `JXD_W8A8_PDMIX` registration delegates to vLLM-Ascend's native
`W8A8_MIX` linear and MoE implementations. The extension owns admission and
alias registration, while vLLM-Ascend continues to own weight loading,
parameter layout, execution-role selection, and NPU operators.

The Bundle manifest is located at:

```text
vllm_ascend_quant_ext/manifests/vllm-hust-extension-v0.2.json
```

The identifiers use the current `0.2-experimental` Manager baseline. This is
still an alpha/NO-GO integration: `import_only` must not be changed to an active
status before the Extension Manager and vLLM-Ascend owners approve the Host
loader/operator contracts.
