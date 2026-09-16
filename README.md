# Ascend Quant Toolkit and W8A8 Runtime Extension

This repository contains two separately packaged components:

- **Ascend Quant Toolkit**: offline calibration, ModelSlim conversion,
  contract export and evaluation evidence.
- **vllm-ascend-quant-ext**: read-only W8A8 runtime plugin loaded through
  vLLM's native `vllm.general_plugins` interface.

The Toolkit is not installed in the serving process. The Runtime Extension
contains no dataset, calibration or model-conversion lifecycle. Model-weight
quantization is also independent of Adaptive Quantized KV.

## Frozen runtime baseline

The current alpha supports exactly the source pair used by the project:

| Component | Supported revision/version |
|---|---|
| vLLM-HUST | `6cff125127bac512488dc90a9812dcafddb65298` |
| vLLM-Ascend-HUST | `203a33e677ac6728108473e749069244bea00373` |
| torch | `2.9.0` |
| torch-npu | `2.9.0` |
| CANN | `>=8.5,<8.6` |

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
python -m pip install vllm-ascend-quant-ext==0.4.1a2
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

The installed wheel exposes only:

```text
vllm.general_plugins/vllm_ascend_quant
```

It does not expose an Extension Manager Bundle and does not modify vLLM or
vLLM-Ascend source. The plugin validates the artifact and frozen host first,
then registers the namespaced `ASCEND_QUANT_W8A8` linear/MoE schemes by
delegating to the host's selected W8A8 implementation.

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

Historical validation records describe the exact behavior of their published
version and are not retroactively rewritten when integration policy changes.
