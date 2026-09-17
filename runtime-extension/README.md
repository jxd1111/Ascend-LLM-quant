# vllm-ascend-quant-ext

Runtime-only W8A8 extension for the project-frozen vLLM-HUST Ascend stack.
The package contains no calibration data, ModelSlim conversion, PPL evaluation
or benchmark orchestration.

## Supported host baseline

This alpha targets the frozen `v1` release baseline:

| Component | Frozen revision |
|---|---|
| vLLM-HUST | ref `v1`, commit `f18cf803c5f63625e2c71253ddaf8b0bad0bad1a` |
| vLLM-Ascend-HUST | commit `74f0c0a272376412b51e1c1864803d5f3a0f1b5f` |
| torch / torch-npu | `2.13.0` / `2.13.0rc1` |
| CANN | `>=9.1,<9.2` |

Admission checks both the declared package version and the Git revision token
embedded in the installed vLLM distributions. Unknown revisions fail closed.

## Integration boundary

The wheel declares exactly two entry points:

```toml
[project.entry-points."vllm.general_plugins"]
vllm_ascend_quant = "vllm_ascend_quant_ext.plugin:register"

[project.entry-points."vllm_hust.extension_bundles"]
"org.vllm-hust.ascend-quant-runtime" = "vllm_ascend_quant_ext.manifests"
```

The first is the runtime activation hook that vLLM loads in every process. The
second is a static Manifest 0.2 locator consumed by the vLLM-HUST Extension
Manager: it resolves to `manifests/vllm-hust-extension-v0.2.json`, which the
Manager validates before importing any implementation module. The registration
name is identical to the manifest `extension_id`, and no new `vllm.*`
entry-point namespace is claimed.

The extension does not monkey patch vLLM or vLLM-Ascend. vLLM owns process
startup and shutdown; vLLM-Ascend continues to own weight loading, parameter
layout, W8A8 execution and NPU operators. The plugin owns artifact admission and
the namespaced `ASCEND_QUANT_W8A8` scheme alias.

Installation and discovery are side-effect free: after installation `torch`,
`vllm` and `vllm_ascend` stay unimported. The registration callback is
idempotent and remains disabled unless explicitly enabled.

## Manifest

`manifests/vllm-hust-extension-v0.2.json` declares:

| Field | Value |
|---|---|
| `schema_version` | `0.2-experimental` |
| `extension_id` | `org.vllm-hust.ascend-quant-runtime` |
| `kind` | `in_process_plugin` |
| `host` | `vllm` / `vllm-ascend` / `>=0.25.1rc1,<0.25.2` |
| `runtime` | `python`, `vllm_engine_and_ascend_worker`, `trusted_in_process` |
| `lifecycle_owner` | `vllm` |
| `protocols` | `vllm.general_plugins`, `vllm-ascend.quantization-scheme-surface` |
| `components[].permissions` | `device_access`, `filesystem_read` |

No host API version is declared because the frozen host exposes no independently
versioned plugin API, and the deployment artifact path is user configuration
that is deliberately kept out of the static manifest.

Inspect it with the Extension Manager when that tooling is available:

```bash
python -m pip install 'vllm-hust-ext @ git+https://github.com/vLLM-HUST/extension-manager.git'
vllm-hust-ext extension list
vllm-hust-ext extension inspect org.vllm-hust.ascend-quant-runtime
vllm-hust-ext extension validate org.vllm-hust.ascend-quant-runtime
```

## Install and inspect

```bash
python -m pip install vllm-ascend-quant-ext==0.4.1a4

vllm-ascend-quant-ext check --model /path/to/w8a8-model
vllm-ascend-quant-ext status
vllm-ascend-quant-ext render --model /path/to/w8a8-model
```

## Start vLLM

```bash
export VLLM_ASCEND_QUANT_EXT_ENABLE=1
export VLLM_ASCEND_QUANT_EXT_ARTIFACT=/path/to/w8a8-model

vllm-hust serve /path/to/w8a8-model \
  --host 0.0.0.0 \
  --port 18000
```

Do not set `VLLM_PLUGINS` to only `vllm_ascend_quant`: that variable filters
every plugin group and would suppress required vLLM-Ascend platform/general
plugins. The frozen host discovers installed general plugins automatically;
the extension-owned enable switch keeps this callback default-off.

The contract is validated before importing the vLLM-Ascend scheme module.
Missing files, hashes, tensors, software versions, frozen revisions or scheme
providers abort startup.

## Frozen-host NPU gate

`tools/npu_e2e.py` is the release gate that proves the *wheel* serves the
reference artifact on the frozen host. It installs the wheel non-editable with
the extension source tree removed from `PYTHONPATH`, activates it through
vLLM's own general-plugin loader, serves the artifact, compares two identical
completions, then uninstalls, restores the previous install and re-hashes the
artifact. Host paths are parameters; nothing about a specific host is
hard-coded.

```bash
python runtime-extension/tools/npu_e2e.py \
  --wheel runtime-extension/dist/vllm_ascend_quant_ext-0.4.1a4-py3-none-any.whl \
  --artifact /path/to/w8a8-model \
  --legacy-artifact /path/to/pre-v1-model \
  --python /path/to/host/python \
  --device 6 --port 18003 \
  --env-script /path/to/cann/set_env.sh \
  --env-script '/path/to/atb/set_env.sh --cxx_abi=0' \
  --host-source /path/to/vllm-hust \
  --host-source /path/to/vllm-ascend-hust \
  --summary npu_e2e_summary.json
```

`--dry-run` prints every command without touching the host. The gate fails
closed: the summary reports `"ok": false` plus the failed phase list, and the
previous install is restored even after an error. `--keep-installed` leaves the
wheel in place instead of restoring, and `--skip-hash` skips the two full
artifact hash passes.

The gate never narrows `VLLM_PLUGINS`, never imports the extension from the
source tree, and refuses a `--host-source` that resolves inside this
repository's own `src` directory.

## Disable and uninstall

The plugin is process scoped; it is not hot-unloaded. Stop the old vLLM
process, then start a new process without the selection variables:

```bash
unset VLLM_ASCEND_QUANT_EXT_ENABLE
unset VLLM_ASCEND_QUANT_EXT_ARTIFACT
python -m pip uninstall -y vllm-ascend-quant-ext
```

When the Extension Manager owns the deployment, keep the same order it
documents: disable, restart and verify the built-in path, then forget, then
uninstall.

```bash
vllm-hust-ext extension disable org.vllm-hust.ascend-quant-runtime
# stop the old process, then start a new one and verify the built-in path
vllm-hust-ext extension forget org.vllm-hust.ascend-quant-runtime
python -m pip uninstall -y vllm-ascend-quant-ext
```

Installation, validation and uninstall never modify the model directory. A
model whose active metadata names `ASCEND_QUANT_W8A8` still requires the
plugin; restoring native `W8A8_MIX` metadata is a separate offline Toolkit
operation.

This extension loads model weights and registers the validated dense-linear
W8A8 activation/weight path only. It does
not own KV-cache format, allocation, compression or request scheduling.
