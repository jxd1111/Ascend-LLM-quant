# Ascend Quant Toolkit

> Classification: **Tool** — offline development lifecycle only.

> Project status: W8A8 runtime-extension prototype and NPU E2E are available;
> the Extension Manager manifest/provider API is awaiting framework-team review.

## 中文设计入口

首次了解本项目，建议按以下顺序阅读：

1. [中文设计概览](docs/design-overview.zh-CN.md)
2. [文档导航](docs/README.md)
3. [当前 vLLM-Ascend 量化架构](docs/current-vllm-ascend-quant-architecture.md)
4. [Runtime Extension 详细设计](docs/runtime-plugin-design.md)
5. [Extension Manager 对接契约](docs/extension-manager-handoff.md)
6. [验收矩阵](docs/acceptance-matrix.md)

当前状态：

- W8A8 PDMix 的 Toolkit → Runtime Extension → Ascend NPU 推理已验证；
- Runtime Extension 原型、产物契约和 fail-closed 校验已实现；
- Extension Manager 正式接入及 vLLM-Ascend 公共 Scheme API 待框架团队评审；
- W4A4/W4A8 尚未声明为真实硬件验证能力。

跨团队设计评审和待确认接口记录在
[Issue #1](https://github.com/jxd1111/Ascend-LLM-quant/issues/1)。

Design is the gate for further runtime extraction. Start with:

- [`docs/current-vllm-ascend-quant-architecture.md`](docs/current-vllm-ascend-quant-architecture.md)
- [`docs/runtime-plugin-design.md`](docs/runtime-plugin-design.md)
- [`docs/runtime-plugin-roadmap.md`](docs/runtime-plugin-roadmap.md)
- [`docs/adr/0001-runtime-plugin-boundary.md`](docs/adr/0001-runtime-plugin-boundary.md)

This repository owns calibration, ModelSlim conversion, quantized artifact
generation, provenance, PPL/accuracy evaluation inputs, and matched benchmark
evidence. It is not itself a vLLM plugin and its root distribution does not
register a `vllm.general_plugins` entry point.

The runtime component is a separate Python distribution under
[`runtime-extension/`](runtime-extension/): **`vllm-ascend-quant-ext`**. That
package is classified as a **Plugin** and contains no calibration, conversion,
dataset, PPL, or benchmark lifecycle.

## Ownership boundary

| Component | Classification | Lifecycle owner | May modify/build model artifacts |
|---|---|---|---|
| Ascend Quant Toolkit | Tool | offline developer/operator | yes, only during explicit offline conversion/export |
| vLLM Ascend Quant Extension | Plugin | vLLM-Ascend | no; runtime access is read-only |
| vLLM-HUST Extension Manager | Manager | operator/manager | no; discover/check/plan/render only |
| Adaptive Quantized KV | separate capability | its own runtime owner | outside this repository |

## Install the offline Toolkit

```bash
python -m pip install -e /root/jxd-ascend-quant \
  --no-deps --no-build-isolation
```

The compatibility alias `jxd-quant` remains available, while the canonical
command is `ascend-quant-toolkit`.

```bash
ascend-quant-toolkit doctor --path /data/jxd
ascend-quant-toolkit list-recipes
ascend-quant-toolkit plan \
  --recipe qwen25-w8a8-pdmix \
  --model /root/models/Qwen2.5-14B-Instruct \
  --output /data/jxd/models/Qwen2.5-14B-Instruct-w8a8-pdmix \
  --device npu:7
```

Replace `plan` with `quantize` to run the offline job. Successful quantization
writes both provenance and the frozen runtime contract:

- `jxd_quant_manifest.json`
- `ascend_quant_artifact.json`
- `quant_model_description.json`
- `quant_model_description.modelslim.json` (legacy rollback copy)

For a previously produced plugin-format model, export the contract explicitly:

```bash
ascend-quant-toolkit export-contract \
  --model /path/to/model \
  --recipe qwen25-w8a8-pdmix \
  --evidence-level schema_only
```

Do not claim NPU or benchmark verification by changing the evidence flag alone;
the referenced result files must exist and follow the matched protocol in
[`docs/acceptance-matrix.md`](docs/acceptance-matrix.md).

Matched BF16/W8A8/W4A4/W4A8 records use
[`evidence/example-result-v1.json`](evidence/example-result-v1.json) and are
checked with:

```bash
ascend-quant-toolkit validate-evidence --file result.json
```

## Install the runtime extension

```bash
python -m pip install -e /root/jxd-ascend-quant/runtime-extension \
  --no-build-isolation

vllm-ascend-quant-ext check --model /path/to/model
vllm-ascend-quant-ext plan --model /path/to/model
vllm-ascend-quant-ext render --model /path/to/model
```

Installation alone has no runtime effect. `check`, `plan`, and `render` are
read-only. The rendered environment explicitly selects the artifact and enables
the extension for the next vLLM start. Contract admission completes before
torch or vLLM-Ascend implementation modules are imported.

The runtime wheel registers the experimental Extension Bundle
`org.vllm-hust.ascend-quant` through `vllm_hust.extension_bundles`. Its static
manifest is packaged under
`vllm_ascend_quant_ext/manifests/vllm-hust-extension-v0.2.json`; discovery does
not import the runtime implementation.

Target Manager flow, once the host provider admits the
`model_weight_quantization_runtime` kind:

```bash
pip install vllm-hust-ext
pip install vllm-ascend-quant-ext
vllm-hust-ext extension check org.vllm-hust.ascend-quant
```

The provisional Bundle/component identities used during framework review are:

```text
Bundle:    org.vllm-hust.ascend-quant
Component: org.vllm-hust.ascend-quant/w8a8-runtime
Contract:  vllm-ascend.quantization.scheme.v1
Plane:     model_worker
```

Use the Bundle ID, rather than the Python distribution name, in Manager
commands once the proposed kind/provider is admitted:

```bash
vllm-hust-ext extension inspect org.vllm-hust.ascend-quant
vllm-hust-ext extension check org.vllm-hust.ascend-quant
```

Until that Manager kind/materializer is available in the deployed Manager,
use the extension's own `check/plan/render` commands; do not describe this as
Manager admission evidence.

The framework-team hand-off is documented in
[`docs/extension-manager-handoff.md`](docs/extension-manager-handoff.md), and
release ownership/gates are tracked in
[`docs/release-checklist.md`](docs/release-checklist.md). Existing traceable
W8A8 results are recorded in
[`docs/w8a8-evidence.md`](docs/w8a8-evidence.md).

## Artifact contract and failure policy

The closed JSON contract is documented in [`contracts/`](contracts/). It covers
W8A8, W4A4, and W4A8 declarations for:

- weight packing/layout and signed nibble semantics;
- scale granularity and shape;
- zero-point presence, dtype, and semantics;
- supported model identity and rank/alignment constraints;
- CANN, torch-npu, vLLM-HUST and vLLM-Ascend-HUST compatibility ranges;
- ModelSlim loader, scheme provider, and required operators.

Unknown fields, unknown schemes/operators/layouts, incomplete tensors, dtype or
shape mismatches, hash mismatches, unsupported software versions, and absent
scheme providers fail closed.

W4A8 hierarchical per-group artifacts are deliberately not admitted by
contract v1; the current validator rejects them instead of interpreting their
secondary scale tensors as ordinary per-group scales.

## Rollback and uninstall

The Manager and runtime extension never rewrite model files. Disabling the
extension removes its environment selection on the next start. Uninstalling
the runtime wheel removes its registration, but an artifact whose active
metadata contains `JXD_W8A8_PDMIX` still requires the extension. Returning that
artifact to the native vLLM-Ascend `W8A8_MIX` path is therefore an explicit
offline Toolkit operation:

```bash
ascend-quant-toolkit restore-modelslim \
  --model /path/to/model \
  --recipe qwen25-w8a8-pdmix
```

The command atomically restores `quant_model_description.json` byte-for-byte
from the preserved `quant_model_description.modelslim.json`. It fails closed if the
active metadata no longer exactly matches the migration generated by the
Toolkit. Uninstall itself never modifies the model directory.

This repository never combines model-weight quantization with Adaptive
Quantized KV manifests, lifecycle, compatibility claims, or configuration.
