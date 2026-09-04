# Ascend model quantization runtime plugin design

Status: design baseline for review before the next implementation phase.

## Goals

1. Package model weight/activation runtime support independently from the
   offline Toolkit and vLLM-Ascend release cadence.
2. Admit only versioned, compatible artifacts and fail before model loading.
3. Register schemes through typed public host interfaces without monkey patch.
4. Keep installation inert; activation must be explicit and reversible.
5. Make W8A8 the first verified profile while reserving contract space for
   W4A4/W4A8 without claiming they are verified.

## Non-goals

- calibration, GPTQ, SmoothQuant, conversion or dataset handling in vLLM;
- KV-cache quantization, compression, allocation or request scheduling;
- ownership of vLLM model runner, NPU platform or CANN operators;
- silently replacing built-in `ascend` schemes;
- automatic fallback that changes numerical behavior without reporting it.

## Components

```text
Extension Manager
  -> vllm_hust.extension_bundles static manifest discovery
  -> QuantRuntimeProvider.check/plan/render
       -> ArtifactReader + ContractValidator
       -> CapabilityMatcher
       -> RuntimePlan
  -> start model worker with explicit activation
       -> plugin register()
       -> repeat artifact admission
       -> SchemeProvider.register(host_registry)
       -> vLLM-Ascend creates/loads/applies the selected scheme
```

The Manager check and worker admission intentionally duplicate validation. The
first produces an operator-visible plan; the second protects against changed
files or environment between planning and process start.

## Public plugin input

### Manager configuration

```json
{
  "enabled": true,
  "model": "/absolute/path/to/artifact"
}
```

This remains closed: unknown or missing keys fail. The final Manager schema may
wrap this object, but the typed host provider must map to it deterministically.

### Artifact input

Required files:

```text
config.json
quant_model_description.json
ascend_quant_artifact.json
one or more declared *.safetensors shards
```

The contract supplies model identity, logical shapes, scheme, weight layout,
scale/zero-point semantics, runtime quant type, required operators, software
ranges and evidence level.

### Runtime capability input

The host provider must supply or make discoverable:

```text
host/provider API version
CANN version
torch-npu version and required operator availability
vLLM and vLLM-Ascend versions
model dtype and architecture
TP/EP configuration
execution_role = standalone | kv_producer | kv_consumer
```

The artifact independently declares `quantization.scheme: W8A8` and
`activation.granularity: pd_mix`. PDMix is the offline quantization
algorithm/profile; `execution_role` is host runtime input and does not rename
or redefine that profile. For a PDMix artifact it selects the static or dynamic
runtime path. It does not authorize KV-cache quantization and is independent of
the Adaptive Quantized KV extension.

### Layer input

```text
layer_type: linear | moe
prefix: stable logical layer name
input_size K, output_size N
partition sizes and TP rank
parameter dtype
runtime activation x [..., K]
optional bias [N]
```

## Public plugin output

### Admission report

```json
{
  "admitted": true,
  "artifact_id": "...",
  "scheme": "W8A8",
  "runtime_quant_type": "JXD_W8A8_PDMIX",
  "required_operators": ["torch_npu.npu_dynamic_quant", "torch_npu.npu_quant_matmul"],
  "implementation_imported": false
}
```

Failures are structured and terminal: unsupported schema/model/shape/layout,
missing tensor, dtype/scale mismatch, incompatible software, unavailable
operator or duplicate scheme registration.

### Runtime plan

The render result contains environment additions/removals, plugin additions and
the admitted immutable artifact identity. It never contains a command that
modifies the model.

### Scheme registration

The current extension registers only the namespaced PDMix aliases:

```text
JXD_W8A8_PDMIX/linear
JXD_W8A8_PDMIX/moe
```

They subclass the native vLLM-Ascend `W8A8_MIX` implementations instead of
copying or forking their algorithm code. The extension does not register or
overwrite host-owned keys such as `W8A8`, `W8A8_DYNAMIC` or `W8A8_MIX`.

### Extension Bundle identity

The `0.2-experimental` packaging prototype uses the following provisional
identifiers pending framework-team confirmation:

```text
Bundle ID:          org.vllm-hust.ascend-quant
Component ID:       w8a8-runtime
Full component ID:  org.vllm-hust.ascend-quant/w8a8-runtime
Contract:           vllm-ascend.quantization.scheme.v1
Execution plane:    model_worker
```

The Bundle is discovered through `vllm_hust.extension_bundles`. The existing
`vllm.general_plugins/vllm_ascend_quant` entry point is retained only for direct
diagnostics; it is not selected by Bundle activation and is not the Bundle
discovery mechanism.

### Tensor output

For linear input `x [...,K]`, the scheme returns `y [...,N]` in the declared
model output dtype. Parameter-specification calls return named tensors plus
layout/partition metadata; post-load processing produces the exact runtime
layout required by the admitted operator.

## W8A8 algorithm policy

| Mode | Activation policy | Weight policy | Operator path |
|---|---|---|---|
| static | offline per-tensor activation scale/offset | offline INT8 per-channel | vLLM quantize + NPU quant matmul |
| dynamic | runtime per-token scale | offline INT8 per-channel | NPU dynamic quant + NPU quant matmul |
| PDMix (`activation.granularity=pd_mix`) | static for `kv_consumer`, dynamic otherwise | static-parameter superset | selected using the independent `execution_role` input |

A standalone service therefore uses the dynamic runtime path, but its artifact
and quantization profile remain PDMix.

The PDMix algorithm, operator rounding/saturation and physical FRACTAL_NZ
layout remain host-owned contracts. The plugin validates the artifact and
provides a namespaced alias; it does not fork or emulate the host algorithm.

## Startup modes

### Disabled/default

Installing the wheel registers metadata but changes no serving behavior.
No artifact is inspected and no W8A8 implementation module is imported.

### Manager-managed production mode

```text
install -> discover -> validate -> configure -> check -> plan -> render
-> launch vLLM-Ascend worker -> worker re-admission -> scheme registration
```

This is the only production target. The Manager must merge plugin selections
and preserve the `ascend` platform plugin.

### Direct diagnostic mode

Environment variables may activate the plugin for development/E2E tests. This
mode is not evidence of Extension Manager integration and must be documented as
diagnostic only.

## Safety and rollback

- all artifact reads are read-only;
- check happens before implementation imports;
- unknown contract fields fail closed;
- no silent BF16 fallback for a declared quantized layer;
- disable removes only this extension's variables and plugin selection;
- uninstall removes registration without touching artifacts;
- the original ModelSlim/vLLM-Ascend path is restored only by the Toolkit's
  explicit, fail-closed `restore-modelslim` operation; uninstall is not a
  metadata migration.

## Host API required before full extraction

The framework team must freeze:

1. manifest schema and discovery path;
2. typed `model_weight_quantization_runtime` materializer;
3. stable scheme registry and duplicate-registration behavior;
4. stable parameter-spec and post-load interfaces;
5. runtime capability descriptor, including execution role;
6. error/result schemas for check/plan/render.

Until those interfaces are accepted, the plugin remains integration-ready but
must not copy more vLLM-Ascend internals into its own package.

## Versioning

- artifact contract: semantic version, fail on unsupported major;
- Manager adapter: independent API version;
- runtime scheme IDs: namespaced and immutable once published;
- wheel: semantic version with explicit compatible host ranges;
- any packing/layout change requires a new contract/scheme version.
