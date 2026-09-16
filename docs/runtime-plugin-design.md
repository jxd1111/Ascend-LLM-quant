# W8A8 Runtime Plugin Design

## Goal

`vllm-ascend-quant-ext` is a bounded, in-process plugin for W8A8 model
artifacts. It extends one behavior: after explicit admission, it registers the
extension-owned `ASCEND_QUANT_W8A8` scheme name with the frozen
vLLM-Ascend-HUST scheme registry.

It is not a vLLM fork, an Extension Manager Bundle, a conversion toolkit or a
KV-cache plugin.

## Host contract

The plugin is discovered through:

```text
group:  vllm.general_plugins
name:   vllm_ascend_quant
target: vllm_ascend_quant_ext.plugin:register
```

Supported host:

```text
vLLM-HUST        6cff125127bac512488dc90a9812dcafddb65298
vLLM-Ascend-HUST 203a33e677ac6728108473e749069244bea00373
torch            2.9.0
torch-npu        2.9.0
CANN             >=8.5,<8.6
```

No compatibility is claimed for other revisions. The embedded Git revision
tokens are checked in addition to numeric package requirements.

## Inputs

Activation inputs are three environment variables:

```text
VLLM_ASCEND_QUANT_EXT_ENABLE=1
VLLM_ASCEND_QUANT_EXT_ARTIFACT=/absolute/model/path
```

The model directory must contain contract 1.1, ModelSlim description, config,
indexes and weight shards matching the declared size, hash, dtype and shape.
Unknown fields and undeclared files fail closed.

The frozen host automatically discovers installed general plugins. Do not set
`VLLM_PLUGINS` to only this plugin: it filters every plugin group and would
suppress required Ascend platform/general entry points. Operators that already
maintain an allowlist must include the complete frozen-host plugin set.

## Outputs and effects

Successful registration adds at most two registry entries:

```text
(ASCEND_QUANT_W8A8, linear)
(ASCEND_QUANT_W8A8, moe)
```

Both are namespaced aliases of the frozen host implementation. Existing
foreign registrations are rejected; repeated registration by the same plugin
is idempotent. No model file is written.

## Lifecycle

1. `pip install` records the entry point but changes no serving behavior.
2. vLLM discovers the entry point in each process.
3. Without the explicit enable flag, `register()` returns without imports or
   device access.
4. When enabled, the plugin validates the artifact and software first.
5. Only then does it import vLLM-Ascend and register the W8A8 aliases.
6. Disable applies to a newly started process; hot unload is not promised.
7. Uninstall removes the entry point but never rewrites the model.

## Failure policy

Missing activation input, incompatible host revision, CANN/torch-npu mismatch,
invalid artifact, absent host scheme or registration collision aborts startup.
There is no silent fallback from `ASCEND_QUANT_W8A8` to BF16 or another
quantization method.

## Security and data boundary

The plugin reads only the configured model directory and local distribution /
CANN metadata. It does not require network access, subprocesses, writable model
storage or prompt/token logging.
