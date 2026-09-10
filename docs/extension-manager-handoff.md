# Extension Manager hand-off contract

Status: **Manager-discoverable `import_only` baseline; runtime activation NO-GO**.

This document is the hand-off boundary between the quantization team and the
vLLM-HUST Extension Manager team. It does not claim that the experimental
manifest schema has already been accepted by the Manager implementation.

## Identity and ownership

| Field | Value |
|---|---|
| Distribution | `vllm-ascend-quant-ext` |
| Bundle/Extension ID | `org.vllm-hust.ascend-quant-runtime` |
| Component ID | `ascend-quant-artifact-validator` |
| Full component ID | `org.vllm-hust.ascend-quant-runtime/ascend-quant-artifact-validator` |
| Kind | `in_process_plugin` |
| Host | provider `vllm`, name `vllm-ascend` |
| Lifecycle owner | `vllm` |
| Runtime scope | trusted Python in-process, `vllm-ascend-worker` |
| Component contract | `vllm.ascend.quantized-artifact-loader.v1` |
| Execution planes | `worker`, `device` |
| Implementation status | `import_only` |
| Bundle discovery | `vllm_hust.extension_bundles/org.vllm-hust.ascend-quant-runtime` |
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
  org.vllm-hust.ascend-quant-runtime = vllm_ascend_quant_ext.manifests

vllm_ascend_quant_ext/manifests/__init__.py
vllm_ascend_quant_ext/manifests/vllm-hust-extension-v0.2.json
```

The Manager must discover and validate this static metadata without importing
`vllm_ascend_quant_ext.plugin`, torch, torch-npu, vLLM, or vLLM-Ascend.

The manifest declares an import-only artifact validator. Its Python carrier is
`vllm_ascend_quant_ext.contract:ArtifactContractValidator`. Loading it performs
no torch, vLLM, vLLM-Ascend, device, or model side effects. The older runtime
carrier remains unadvertised and diagnostic-only.

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
shape contract before any implementation import, then reports
`enable_allowed=false`. Enabled `plan` and `render` raise an `import_only`
error. This refusal is intentional.

The `ASCEND_QUANT_W8A8` carrier is a namespaced alias of the selected
vLLM-Ascend W8A8 linear and MoE schemes. The extension does not own or fork weight
loading, tensor layout conversion, execution-role selection, or NPU operators.

For a disabled extension, `render` returns the two diagnostic variables in
`environment_unset` and only `vllm_ascend_quant` in `vllm_plugins_remove`.
No enabled Manager render exists in the import-only baseline. The extension's
CLI render output is a separate direct-diagnostic aid and must never be
materialized by the Manager.

## Required Manager behavior

The Manager team owns:

1. validation of Manifest `0.2-experimental` and the `in_process_plugin` kind;
2. duplicate-ID, permissions, host/API and version-range admission;
3. deterministic discover/check/plan/render lifecycle and state storage;
4. safe merging of environment and plugin selections from multiple extensions;
5. process launch, disable, uninstall and rollback orchestration;
6. ensuring ordinary `import vllm` never invokes the Manager.

The Manager must not modify the model artifact or invoke ModelSlim.

## Items requiring an explicit framework-team decision

- graduate or replace `schema_version=0.2-experimental`;
- confirm the installed manifest discovery path and identifiers;
- define the official typed provider/materializer interface;
- publish `vllm.ascend.quantized-artifact-loader.v1`;
- publish `vllm.ascend.quantized-operator-selection.v1`;
- define how the model path and immutable artifact identity reach the model
  worker component;
- publish the final CLI syntax for configure/check/plan/render/run.

No extension release may silently guess these decisions. A schema change must
be followed by a new extension version and compatibility tests.

## Joint acceptance sequence

1. Install the runtime wheel into the same environment as vLLM-Ascend.
2. Discover and validate the static manifest without implementation import.
3. Configure the verified W8A8 artifact and run Manager `check`; confirm it is
   valid but not enableable.
4. Confirm enabled Manager `plan` and `render` fail closed as `import_only`.
5. After both Host protocols are approved in a future version, start
   vLLM-Ascend and complete a deterministic inference request.
6. Disable the extension and verify that its environment/plugin selection is
   absent on the next start.
7. Uninstall the wheel and verify that its entry point and Manager descriptor
   disappear while model hashes remain unchanged.
8. Run Toolkit `restore-modelslim`, then start the original
   vLLM-Ascend/ModelSlim `W8A8_MIX` path to demonstrate rollback. Uninstalling
   the wheel alone must not be treated as metadata rollback.
