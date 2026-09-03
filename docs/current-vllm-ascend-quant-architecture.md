# Current vLLM-HUST / vLLM-Ascend quantization architecture

Status: source audit for plugin design. This document describes the local
source snapshots inspected on 2026-09-03:

- vLLM-HUST: branch `feature-qt`, commit `6cff125127ba`;
- vLLM-Ascend-HUST: branch `feature-qt`, commit `203a33e677ac` plus an
  uncommitted working tree. The dirty files are not copied into this repository.

The scope below is model weights and linear/MoE activations. KV-cache
quantization is a separate capability and is not part of the proposed plugin.

## Runtime ownership today

| Layer | Current owner | Responsibility |
|---|---|---|
| CLI/model configuration | vLLM | Carries `model`, dtype and quantization selection |
| Platform detection | vLLM-Ascend | Detects `quant_model_description.json` and selects `ascend` |
| Artifact description parser | vLLM-Ascend | Loads ModelSlim per-tensor quantization type mappings |
| Layer-to-scheme mapping | vLLM-Ascend | Maps model prefixes/fused modules and selects a registered scheme |
| Parameter lifecycle | vLLM plus vLLM-Ascend adapter | Allocates parameters, loads shards and invokes post-load transforms |
| W8A8 algorithms | vLLM-Ascend | Static, dynamic and P/D-role-mixed implementations |
| Device operators | torch-npu/vLLM-Ascend | Activation quantization, quantized matmul and graph fusion |
| Offline calibration/GPTQ | ModelSlim/Toolkit | Produces weights and scales; never runs in vLLM |

## End-to-end control flow

```text
vllm serve MODEL
  -> NPU platform plugin is selected
  -> detect quant_model_description.json
  -> model_config.quantization = "ascend"
  -> vLLM get_quant_config()
  -> AscendModelSlimConfig.maybe_update_config(MODEL)
  -> parse per-layer quant type and model/fused-prefix mappings
  -> each LinearBase/FusedMoE asks config.get_quant_method(layer, prefix)
  -> create_scheme_for_layer(quant_type, layer_type)
  -> registry lookup: (quant_type, layer_type) -> scheme class
  -> method adapter calls scheme.get_*() and registers parameters
  -> checkpoint loader copies safetensors into allocated parameters
  -> process_weights_after_loading(): transpose/flatten/NZ conversion
  -> layer.forward(x)
  -> method adapter -> scheme.apply(layer, x, bias, tp_rank)
  -> torch-npu quantize/matmul operator
  -> output tensor
```

Primary source points:

- vLLM `vllm/config/vllm.py`: creates and updates `QuantizationConfig`;
- vLLM `vllm/model_executor/model_loader/weight_utils.py`: resolves the config;
- vLLM `vllm/model_executor/layers/linear.py`: create/load/forward lifecycle;
- vLLM-Ascend `vllm_ascend/quantization/utils.py`: format detection;
- vLLM-Ascend `vllm_ascend/quantization/modelslim_config.py`: description,
  prefix and layer mapping;
- vLLM-Ascend `vllm_ascend/quantization/method_adapters.py`: vLLM adapters;
- vLLM-Ascend `vllm_ascend/quantization/methods/registry.py`: typed scheme registry;
- vLLM-Ascend `vllm_ascend/quantization/methods/w8a8_*.py`: W8A8 logic.

## Common linear interface

For a logical linear operation with input width `K` and output width `N`:

```text
input x:          [..., K], BF16/FP16 unless already quantized
disk weight:      [N, K], INT8 for W8A8
runtime weight:   [K, N], INT8, optionally FRACTAL_NZ
optional bias:    [N]
output y:         [..., N], normally same floating dtype as x
```

The vLLM-Ascend scheme interface separates four stages:

| Method | Input | Output/effect |
|---|---|---|
| `get_weight(K,N,dtype)` | dimensions and model dtype | empty weight tensors and packing metadata |
| `get_pertensor_param(dtype)` | model dtype | activation scale/offset tensors |
| `get_perchannel_param(N,dtype)` | output dimension | weight scale/offset, dequant scale and quant bias |
| `get_pergroup_param(K,N,...)` | dimensions and layer type | group scales/offsets and auxiliary tensors |
| `process_weights_after_loading(layer)` | populated layer parameters | runtime layout/shape/dtype conversion |
| `apply(layer,x,bias,tp_rank)` | runtime activation and loaded parameters | floating output tensor |

`AscendLinearMethod` is the vLLM adapter. It registers returned tensors as
parameters, attaches TP/loading metadata, chooses the TP rank and delegates the
forward call to the scheme.

## W8A8 static

Current registration: `("W8A8", "linear")`.

Artifact-side parameters observed in the verified Qwen2.5-14B model:

| Name | Artifact dtype/shape | Runtime purpose |
|---|---|---|
| `weight` | INT8 `[N,K]` | pre-quantized weight |
| `input_scale` | FP32 `[1]` | static activation scale |
| `input_offset` | FP32 `[1]` | producer offset; runtime parameter is converted for the quant op |
| `weight_scale` | FP32 `[N,1]` | per-output-channel weight scale |
| `weight_offset` | FP32 `[N,1]` | per-output-channel producer offset |
| `deq_scale` | FP32 `[N]` | matmul dequantization factor |
| `quant_bias` | INT32 `[N]` | correction in integer accumulation domain |

Post-load processing repeats activation scale/offset to width `K`, creates the
reciprocal scale, transposes weight from `[N,K]` to `[K,N]`, optionally converts
it to FRACTAL_NZ, and flattens channel parameters.

Forward logic:

```text
if x is not INT8:
    qx = torch.ops.vllm.quantize(x, scale, reciprocal_scale, offset)
y = torch_npu.npu_quant_matmul(qx, runtime_weight, deq_scale,
                               bias=quant_bias, output_dtype=model_dtype)
```

Conceptually, activation quantization is affine INT8 quantization followed by
integer GEMM and dequantization. Exact rounding, saturation and offset behavior
belongs to the torch-npu/CANN operator contract and must not be redefined by the
plugin.

## W8A8 dynamic

Current registrations: `("W8A8_DYNAMIC", "linear")` and
`("W8A8_DYNAMIC", "moe")`.

Linear artifact/runtime parameters are INT8 weight `[N,K]`, weight scale
`[N,1]` and weight offset `[N,1]`. There is no fixed input scale.

```text
(qx, per_token_scale) = torch_npu.npu_dynamic_quant(x)
y = torch_npu.npu_quant_matmul(qx, runtime_weight, weight_scale,
                               pertoken_scale=per_token_scale,
                               bias=bias, output_dtype=x.dtype)
```

If the NPU operator returns a singleton middle dimension, the implementation
squeezes it before matmul and restores it afterwards. Weights are transposed,
optionally converted to FRACTAL_NZ, and channel scales are flattened after
loading.

The MoE path allocates expert INT8 weights and per-expert channel scales, then
passes them to the selected fused-expert communication/operator path. It also
contains EPLB and fused-MC2-specific preparation; these are host capabilities,
not portable quantization algorithm semantics.

## W8A8 PDMix

Current registrations: `("W8A8_MIX", "linear")` and
`("W8A8_MIX", "moe")`.

The PDMix name originates from the offline ModelSlim activation-quantization
configuration `act.scope: pd_mix`. It identifies the quantization
algorithm/profile and the resulting parameter set; it is not derived from the
runtime deployment role.

The linear implementation composes the static and dynamic implementations and
allocates the superset of static parameters. The selection is:

```text
KV-transfer consumer process -> static W8A8 apply
all other processes          -> dynamic W8A8 apply
```

This runtime branch is a **deployment-role decision**, not the definition or
origin of PDMix and not a token-by-token prefill/decode branch. A normal
non-disaggregated server follows the dynamic path for both phases while the
artifact remains a PDMix artifact. The proposed plugin therefore preserves
PDMix as the artifact profile and uses a separate explicit `execution_role`
capability to select its runtime path.

## Other weight/activation schemes in the same host

The same registry currently contains W8A8 MXFP8, W8A16, W4A8 dynamic, W4A16,
W4A4 FlatQuant and W4A4 LAOS schemes. They share the adapter lifecycle but have
different packing, group-scale and operator contracts. They remain inventory
items for later phases; only W8A8 is hardware-verified by this project today.

## Boundaries that must stay in vLLM-Ascend

- NPU platform registration and device capability discovery;
- stable vLLM `QuantizationConfig` and layer method adapters;
- public scheme registry/SPI;
- torch-npu operator availability and CANN compatibility reporting;
- TP/EP/EPLB and communication ownership;
- graph compiler and fusion-pass ownership.

## Logic suitable for the external plugin

- versioned quantized artifact contract and admission;
- namespaced scheme definitions and algorithm policy;
- artifact quant-type to scheme mapping;
- parameter specification and scheme-local post-load transforms;
- capability negotiation and deterministic runtime plan;
- Manager-facing metadata and enable/disable lifecycle.

Moving host platform, scheduler, model runner or offline calibration code into
the plugin would violate this boundary.
