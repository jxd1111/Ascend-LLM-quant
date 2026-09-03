# Extension Manager hand-off contract

Status: **extension-side integration ready; Manager API confirmation pending**.

This document is the hand-off boundary between the quantization team and the
vLLM-HUST Extension Manager team. It does not claim that the experimental
manifest schema has already been accepted by the Manager implementation.

## Identity and ownership

| Field | Value |
|---|---|
| Distribution | `vllm-ascend-quant-ext` |
| Extension ID | `vllm-ascend-quant` |
| Proposed kind | `model_weight_quantization_runtime` |
| Host | `vllm-ascend` |
| Lifecycle owner | `vllm-ascend` |
| Runtime scope | trusted Python in-process, model worker |
| vLLM entry point | `vllm.general_plugins/vllm_ascend_quant` |
| Proposed adapter symbol | `vllm_ascend_quant_ext.manager:provider` |
| Adapter API | `1.0` |

The Toolkit is not a plugin. The runtime extension contains no calibration,
conversion, dataset, PPL, benchmark, or KV-cache lifecycle.

## Discovery

The wheel contains the manifest both as a package resource and as an installed
data file:

```text
vllm_ascend_quant_ext/extension-manifest.json
share/vllm-hust/extensions/vllm-ascend-quant/extension-manifest.json
```

The Manager must discover and validate static metadata without importing
`vllm_ascend_quant_ext.plugin`, torch, torch-npu, vLLM, or vLLM-Ascend.

## Configuration

The extension-owned adapter accepts a closed object:

```json
{
  "enabled": true,
  "model": "/absolute/path/to/quantized/model"
}
```

Unknown or missing fields fail closed. The final Manager may use a different
external configuration schema, but its vLLM-Ascend host provider must map that
schema to this object.

## Pure-data calls

Until the Manager publishes a stable typed provider protocol, its host-provider
adapter can call:

```python
from vllm_ascend_quant_ext.manager import provider

provider.descriptor()
provider.check(configuration)
provider.plan(configuration)
provider.render(configuration)
```

These calls are read-only. An enabled `check` validates the complete artifact,
software compatibility, safetensors metadata, operators, model identity and
shape contract before any implementation import.

For an enabled extension, `render` returns:

```json
{
  "environment_set": {
    "VLLM_ASCEND_QUANT_EXT_ENABLE": "1",
    "VLLM_ASCEND_QUANT_EXT_ARTIFACT": "/absolute/model/path"
  },
  "environment_unset": [],
  "vllm_plugins_add": ["vllm_ascend_quant"],
  "vllm_plugins_remove": []
}
```

For a disabled extension it returns the two extension variables in
`environment_unset` and only `vllm_ascend_quant` in `vllm_plugins_remove`.
The Manager must merge plugin selections and preserve the Ascend platform
plugin; it must never replace `VLLM_PLUGINS` with only the quant plugin.

## Required Manager behavior

The Manager team owns:

1. acceptance of the final manifest schema and typed kind;
2. duplicate-ID, permissions, host/API and version-range admission;
3. deterministic discover/check/plan/render lifecycle and state storage;
4. safe merging of environment and plugin selections from multiple extensions;
5. process launch, disable, uninstall and rollback orchestration;
6. ensuring ordinary `import vllm` never invokes the Manager.

The Manager must not modify the model artifact or invoke ModelSlim.

## Items requiring an explicit framework-team decision

- replace or accept `schema_version=0.2-experimental`;
- accept the proposed kind and host-provider names;
- confirm the installed manifest discovery path;
- define the official typed provider/materializer interface;
- decide whether the adapter symbol becomes a registered Manager entry point;
- publish the final CLI syntax for configure/check/plan/render/run.

No extension release may silently guess these decisions. A schema change must
be followed by a new extension version and compatibility tests.

## Joint acceptance sequence

1. Install the runtime wheel into the same environment as vLLM-Ascend.
2. Discover and validate the static manifest without implementation import.
3. Configure the verified W8A8 artifact and run Manager `check`.
4. Inspect `plan` and `render`; verify that no model file changed.
5. Start vLLM-Ascend and complete a deterministic inference request.
6. Disable the extension and verify that its environment/plugin selection is
   absent on the next start.
7. Uninstall the wheel and verify that its entry point and Manager descriptor
   disappear while model hashes remain unchanged.
8. Start the original vLLM-Ascend/ModelSlim path to demonstrate rollback.
