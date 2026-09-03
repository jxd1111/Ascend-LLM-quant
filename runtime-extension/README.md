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
