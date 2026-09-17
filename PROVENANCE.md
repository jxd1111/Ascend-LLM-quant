# Source provenance

This repository separates externally owned quantization implementations from
HUST-owned integration code.

| Area | Source/owner | Treatment in this repository |
|---|---|---|
| Offline calibration and model conversion | msModelSlim project and its upstream license | Invoked as an external dependency; implementation code is not vendored into the runtime wheel. |
| W8A8 linear implementation and NPU operators | installed vLLM-Ascend distribution | Imported and subclassed by a namespaced runtime alias; algorithm/kernel code is not copied. |
| Artifact contract, fail-closed validator, plugin registration and packaging | Ascend-LLM-quant contributors | Authored and maintained in this repository under its declared license. |
| Runtime discovery/lifecycle | vLLM-HUST `vllm.general_plugins` plus the org `vllm_hust.extension_bundles` manifest locator | Consumed through Python distribution metadata; host and Manager source are not vendored. |
| Adaptive Quantized KV | separate project/team | Not included in this package, manifest, lifecycle, or compatibility claims. |

The `runtime-extension` distribution contains no calibration dataset, model
weights, generated quantized artifact, or copied proprietary kernel. Any future
code import must record its upstream repository, exact revision, original
license, files copied, and local modifications before merge.

The runtime entry point is default-off and delegates to the frozen installed
Host only after artifact and software admission. The distribution is a static
Manifest 0.2 `in_process_plugin` bundle: the Extension Manager owns discovery,
configuration and enable intent, while the vLLM host owns the process lifecycle
and the actual scheme registration.
