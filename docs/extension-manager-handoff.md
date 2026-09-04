# Extension Manager hand-off contract

Status: **extension-side integration ready; Manager API confirmation pending**.

This document is the hand-off boundary between the quantization team and the
vLLM-HUST Extension Manager team. It does not claim that the experimental
manifest schema has already been accepted by the Manager implementation.

## Identity and ownership

| Field | Value |
|---|---|
| Distribution | `vllm-ascend-quant-ext` |
| Bundle/Extension ID | `org.vllm-hust.ascend-quant` |
| Component ID | `w8a8-runtime` |
| Full component ID | `org.vllm-hust.ascend-quant/w8a8-runtime` |
| Proposed kind | `model_weight_quantization_runtime` |
| Host | `vllm-ascend` |
| Lifecycle owner | `host` (the vLLM-Ascend provider) |
| Runtime scope | trusted Python in-process, model worker |
| Component contract | `vllm-ascend.quantization.scheme.v1` |
| Execution plane | `model_worker` |
| Bundle discovery | `vllm_hust.extension_bundles/org.vllm-hust.ascend-quant` |
| vLLM entry point | `vllm.general_plugins/vllm_ascend_quant` |
| Proposed adapter symbol | `vllm_ascend_quant_ext.manager:provider` |
| Adapter API | `1.0` |

The Toolkit is not a plugin. The runtime extension contains no calibration,
conversion, dataset, PPL, benchmark, or KV-cache lifecycle.

## Discovery

The wheel registers the manifest package through the Extension Bundle entry
point:

```text
vllm_hust.extension_bundles:
  org.vllm-hust.ascend-quant = vllm_ascend_quant_ext.manifests

vllm_ascend_quant_ext/manifests/__init__.py
vllm_ascend_quant_ext/manifests/vllm-hust-extension-v0.2.json
```

The Manager must discover and validate this static metadata without importing
`vllm_ascend_quant_ext.plugin`, torch, torch-npu, vLLM, or vLLM-Ascend.

The manifest declares the provisional typed component
`org.vllm-hust.ascend-quant/w8a8-runtime`. Its Python carrier is
`vllm_ascend_quant_ext.adapters.vllm_hust.runtime:AscendQuantRuntimeComponent`
with `implementation.status=active`. Host materialization must still fail
closed until the proposed kind and component contract are supported.

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

The `JXD_W8A8_PDMIX` carrier is a namespaced alias of vLLM-Ascend's native
`W8A8_MIX` linear and MoE schemes. The extension does not own or fork weight
loading, tensor layout conversion, execution-role selection, or NPU operators.

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
- accept or revise the provisional Bundle/component/contract identifiers;
- define how the model path and immutable artifact identity reach the model
  worker component;
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
8. Run Toolkit `restore-modelslim`, then start the original
   vLLM-Ascend/ModelSlim `W8A8_MIX` path to demonstrate rollback. Uninstalling
   the wheel alone must not be treated as metadata rollback.
