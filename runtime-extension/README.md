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

`render` produces environment values for a supervisor or the vLLM-HUST
Extension Manager. The plugin validates the model contract before importing
vLLM-Ascend implementation modules. It never modifies the model directory.

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

This adapter is an extension-side hand-off contract, not an assertion that the
experimental Manager schema has been accepted. See
`../docs/extension-manager-handoff.md` before integration.

## Extension Bundle identity

```text
PyPI distribution: vllm-ascend-quant-ext
Python package:     vllm_ascend_quant_ext
Bundle ID:          org.vllm-hust.ascend-quant
Component ID:       w8a8-runtime
Full component ID:  org.vllm-hust.ascend-quant/w8a8-runtime
Contract:           vllm-ascend.quantization.scheme.v1
Execution plane:    model_worker
```

The wheel exposes two distinct entry points:

- `vllm_hust.extension_bundles` discovers static Bundle metadata;
- `vllm.general_plugins` provides an explicit direct-diagnostic registration path.

The first is the production discovery target. The Bundle does not automatically
select the second entry point or inject environment/additional-config values.
That avoids a partial Manager integration starting an unsafe legacy path. The
typed carrier becomes active only after vLLM-Ascend accepts and materializes the
quantization component contract.

The namespaced `JXD_W8A8_PDMIX` registration delegates to vLLM-Ascend's native
`W8A8_MIX` linear and MoE implementations. The extension owns admission and
alias registration, while vLLM-Ascend continues to own weight loading,
parameter layout, execution-role selection, and NPU operators.

The Bundle manifest is located at:

```text
vllm_ascend_quant_ext/manifests/vllm-hust-extension-v0.2.json
```

The identifiers and the `model_weight_quantization_runtime` kind are
provisional under the `0.2-experimental` schema and must be synchronized with
the Extension Manager and vLLM-Ascend owners before a public alpha release.
