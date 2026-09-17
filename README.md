# Ascend Quant Toolkit and W8A8 Runtime Extension

This repository contains two separately packaged components:

- **Ascend Quant Toolkit**: offline calibration, ModelSlim conversion,
  contract export and evaluation evidence.
- **vllm-ascend-quant-ext**: read-only W8A8 runtime plugin loaded through
  vLLM's native `vllm.general_plugins` interface and registered with the
  vLLM-HUST Extension Manager as a static Manifest 0.2 bundle.

The Toolkit is not installed in the serving process. The Runtime Extension
contains no dataset, calibration or model-conversion lifecycle. Model-weight
quantization is also independent of Adaptive Quantized KV.

## Frozen runtime baseline

The current alpha targets the frozen `v1` release baseline and its closest
documented Ascend platform snapshot:

| Component | Supported revision/version |
|---|---|
| vLLM-HUST | ref `v1`, commit `f18cf803c5f63625e2c71253ddaf8b0bad0bad1a` |
| vLLM-Ascend-HUST | commit `74f0c0a272376412b51e1c1864803d5f3a0f1b5f` |
| torch | `2.13.0` |
| torch-npu | `2.13.0rc1` |
| CANN | `>=9.1,<9.2` |

The plugin intentionally fails closed for a different vLLM/vLLM-Ascend Git
revision, even if its numeric package version appears compatible.

## Offline quantization

```bash
git clone https://github.com/jxd1111/Ascend-LLM-quant.git
cd Ascend-LLM-quant
python -m pip install -e . --no-deps --no-build-isolation

ascend-quant-toolkit doctor --path /path/to/output-volume
ascend-quant-toolkit plan \
  --recipe qwen25-w8a8 \
  --model /path/to/Qwen2.5-14B-Instruct \
  --output /path/to/Qwen2.5-14B-Instruct-w8a8 \
  --device npu:7
```

Replace `plan` with `quantize` to execute conversion. A successful export
writes the provenance, ModelSlim description and closed runtime contract. The
contract binds config, indexes, weight shards and evidence files by size and
SHA-256. It is validation metadata; it does not rewrite weight tensors.

## Runtime plugin

Install the independent distribution:

```bash
python -m pip install vllm-ascend-quant-ext==0.4.1a4
vllm-ascend-quant-ext check --model /path/to/w8a8-model
```

Enable it for the next vLLM process:

```bash
export VLLM_ASCEND_QUANT_EXT_ENABLE=1
export VLLM_ASCEND_QUANT_EXT_ARTIFACT=/path/to/w8a8-model

vllm-hust serve /path/to/w8a8-model \
  --host 0.0.0.0 \
  --port 18000
```

Leave `VLLM_PLUGINS` unset unless the deployment already maintains a complete
allowlist for every required Ascend platform and general plugin. The
extension-owned enable flag is the activation gate.

The installed wheel exposes exactly two entry points:

```text
vllm.general_plugins/vllm_ascend_quant
vllm_hust.extension_bundles/org.vllm-hust.ascend-quant-runtime
```

The first is the runtime activation hook vLLM loads in every process. The
second is a static Manifest 0.2 locator: the vLLM-HUST Extension Manager
resolves it from distribution metadata and validates
`manifests/vllm-hust-extension-v0.2.json` before importing any implementation
module. Its registration name is identical to the manifest `extension_id`, and
no new `vllm.*` entry-point namespace is claimed.

The extension does not modify vLLM or vLLM-Ascend source. The plugin validates
the artifact and frozen host first, then registers the namespaced
`ASCEND_QUANT_W8A8` linear scheme by delegating to the host's selected W8A8
implementation.

### Extension Manager integration

When the Extension Manager tooling is available, discovery, configuration and
enable intent belong to it while vLLM keeps the process lifecycle:

```bash
python -m pip install 'vllm-hust-ext @ git+https://github.com/vLLM-HUST/extension-manager.git'
vllm-hust-ext extension list
vllm-hust-ext extension validate org.vllm-hust.ascend-quant-runtime
vllm-hust-ext extension enable org.vllm-hust.ascend-quant-runtime
vllm-hust-ext run --dry-run -- vllm serve /path/to/w8a8-model
```

Installation alone has no runtime effect. The callback is idempotent and
default-off. Unknown contract fields, incompatible versions/revisions, missing
files, tensor mismatches and registration collisions abort activation.

## Disable, recover and uninstall

Stop the current vLLM process and start a new one after clearing selection:

```bash
unset VLLM_ASCEND_QUANT_EXT_ENABLE
unset VLLM_ASCEND_QUANT_EXT_ARTIFACT
python -m pip uninstall -y vllm-ascend-quant-ext
```

The extension never modifies model files. If the active model description
uses `ASCEND_QUANT_W8A8`, restoring the native `W8A8_MIX` description is an
explicit offline operation:

```bash
ascend-quant-toolkit restore-modelslim \
  --model /path/to/model \
  --recipe qwen25-w8a8
```

## Documentation

- [Architecture](docs/architecture.md)
- [Current vLLM-Ascend quantization flow](docs/current-vllm-ascend-quant-architecture.md)
- [Runtime plugin design](docs/runtime-plugin-design.md)
- [Artifact contract](contracts/README.md)
- [Acceptance matrix](docs/acceptance-matrix.md)
- [W8A8 evidence](docs/w8a8-evidence.md)
- [Runtime packaging and release](docs/runtime-extension-packaging-and-release.zh-CN.md)
- [Frozen v1 NPU E2E evidence](docs/evidence/w8a8-frozen-v1-npu-e2e-20260917.md)
- [Release checklist](docs/release-checklist.md)

Historical validation records describe the exact behavior of their published
version and are not retroactively rewritten when integration policy changes.
