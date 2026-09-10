# Source provenance

This repository separates externally owned quantization implementations from
HUST-owned integration code.

| Area | Source/owner | Treatment in this repository |
|---|---|---|
| Offline calibration and model conversion | msModelSlim project and its upstream license | Invoked as an external dependency; implementation code is not vendored into the runtime wheel. |
| W8A8 PDMix linear/MoE implementations and NPU operators | installed vLLM-Ascend distribution | Imported and subclassed by a namespaced diagnostic alias; algorithm/kernel code is not copied. |
| Artifact contract, fail-closed validator, Bundle metadata and packaging | Ascend-LLM-quant contributors | Authored and maintained in this repository under its declared license. |
| Extension Manager discovery/lifecycle | vLLM-HUST Extension Manager | Consumed as an external Host contract; Manager source is not vendored. |
| Adaptive Quantized KV | separate project/team | Not included in this package, manifest, lifecycle, or compatibility claims. |

The `runtime-extension` distribution contains no calibration dataset, model
weights, generated quantized artifact, or copied proprietary kernel. Any future
code import must record its upstream repository, exact revision, original
license, files copied, and local modifications before merge.

The current Manager Bundle is `import_only`. The direct diagnostic adapter
delegates to the installed Host and is not advertised as a production Manager
component.
