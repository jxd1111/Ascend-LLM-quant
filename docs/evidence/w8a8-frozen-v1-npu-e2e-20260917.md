# W8A8 Runtime Extension frozen v1 NPU E2E evidence (2026-09-17)

Wheel-hash-level correctness validation of the released artifact against the
frozen vLLM-HUST `v1` host on the documented Ascend platform snapshot. This is
not a matched BF16/W8A8 performance comparison and it does not widen the
supported host range.

## Object under test

- Runtime Extension source: revision `c6f42e5` on
  `feature/manifest-0.2-extension-bundles`.
- Artifact under test: `vllm-ascend-quant-ext==0.4.1a4`,
  wheel `44d73089f02062534f9464edff3e926ed8af0cbc2d115f42bb5a1c8b90642d04`,
  sdist `9f3a12d190dc6a29ba8a619d21a2a212c012a28db05d270942a342b680d39413`.
- Install under test: the wheel above installed non-editable into
  `/data/jxd/envs/vllm-hust-v1-cann91`. The source tree was removed from
  `PYTHONPATH`, and `vllm_ascend_quant_ext.__file__` resolved to
  `.../site-packages/vllm_ascend_quant_ext/__init__.py` in every process.
- Manifest: `org.vllm-hust.ascend-quant-runtime`, `kind: in_process_plugin`,
  `lifecycle_owner: vllm`,
  `runtime.process_scope: vllm_engine_and_ascend_worker`.
- Model artifact: Qwen2.5-14B-Instruct W8A8, contract `1.1.0`,
  `artifact_id qwen2:W8A8:289309eb1322e19c`, 2595 tensors.
- Host: vLLM-HUST ref `v1`, commit `f18cf803c5f63625e2c71253ddaf8b0bad0bad1a`,
  installed distribution `0.28.1.post1.dev143+gf18cf803c.empty`.
- Platform: vLLM-Ascend-HUST commit `74f0c0a272376412b51e1c1864803d5f3a0f1b5f`,
  installed distribution `0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`.
- torch `2.13.0+cpu`, torch-npu `2.13.0rc1`, CANN `9.1.0`
  (`/data/jxd/Ascend-9.1.0/cann-9.1.0`, inner `V100R001C11SPC001B243`).
- Driver `25.3.rc1`; NPU Ascend 910B3, device 6, TP=1, `--enforce-eager`,
  port 18003. Device 7 held the earlier editable-install serving run.

## Results

| Check | Result | Note |
|---|---|---|
| Wheel and sdist build, `twine check` | PASS | hashes above |
| Wheel replaces the editable install | PASS | module resolved from `site-packages`; no `__editable__*vllm_ascend_quant_ext*` left |
| Installed metadata exposes both entry-point groups | PASS | `vllm.general_plugins/vllm_ascend_quant` and `vllm_hust.extension_bundles/org.vllm-hust.ascend-quant-runtime` |
| Manifest discoverable from installed metadata | PASS | `.../site-packages/vllm_ascend_quant_ext/manifests/vllm-hust-extension-v0.2.json`, also listed in `RECORD` |
| Manifest, packaging and host-contract tests | PASS | 70 passed, 0 skipped |
| Toolkit unit tests and Ruff | PASS | 17 passed; `ruff check` clean |
| vLLM general-plugin loader activates the wheel | PASS | in-process record `vllm_ascend_quant_ext.plugin INFO Ascend quant runtime admitted ...; registered=['ASCEND_QUANT_W8A8/linear']` |
| Scheme registry answers only when enabled | PASS | enabled -> `AscendQuantW8A8LinearMethod`; with `VLLM_ASCEND_QUANT_EXT_ENABLE` unset -> `None` |
| MoE scheme stays fail-closed | PASS | `get_scheme_class("ASCEND_QUANT_W8A8", "moe") -> None` |
| Legacy artifact fail-closed | PASS | at plugin-load time: `Ascend quant artifact admission failed: incompatible CANN: installed 9.1.0, required >=8.5,<8.6` |
| CANN gate fail-closed | PASS | without `ASCEND_TOOLKIT_HOME` on 9.1: `incompatible CANN: installed 8.5.0, required >=9.1,<9.2` |
| Default-off registration | PASS | `status.enabled=false`; `torch`, `vllm` and `vllm_ascend` stay unimported |
| Frozen host revision admission | PASS | abbreviated token `+gf18cf803c.empty` admitted by full-commit comparison |
| Artifact admission under the frozen baseline | PASS | 2595 tensors, all four software checks (reported by `vllm-ascend-quant-ext check`; the wheel run performs the same gate before registering) |
| NPU weight load from the wheel | PASS | 4 safetensors shards; device 6 `61579/65536` MiB loaded |
| Deterministic chat completion from the wheel | PASS | two identical requests returned exactly `OK`; `system_fingerprint vllm-0.28.1.post1.dev143+gf18cf803c-fe28e53e` |
| Server stop and NPU release | PASS | 0 processes left; device 6 back to `3423/65536` MiB |
| Uninstall proves artifact provenance | PASS | `import vllm_ascend_quant_ext` -> `ModuleNotFoundError`; both entry-point groups disappear |
| Host restored after the run | PASS | editable install, `17 passed` and `70 passed`, `verify_release.py` valid again |
| Model artifact unchanged | PASS | per-file sha256 list identical before and after the run |
| Frozen-host gate driver re-run | PASS | `runtime-extension/tools/npu_e2e.py`, 13 phases, `"ok": true`, `"failed_phases": []`, exit 0, device 6, port 18003 |
| Manager lifecycle gates | NOT RUN | `vllm-hust-ext` is unavailable on this host |

## Limitations

- The quantized weights predate this validation; the artifact contract was
  re-exported by the Toolkit running under the frozen baseline. The offline
  `msmodelslim` conversion step was not re-run under CANN 9.1.
- This host configures only the `vllm` logger tree, so the extension's own INFO
  record is dropped at the default root level. The activation line above comes
  from a loader-level check that raises the level for the extension logger
  through a `sitecustomize.py` shim on `PYTHONPATH`. The shim changes neither
  the wheel nor the host, but the serving process itself did not re-print the
  record. Re-check it through the Manager once `vllm-hust-ext` is available.
- The Extension Manager CLI (`vllm-hust-ext`) was unavailable on this host, so
  `extension validate`, `enable`, `disable`, `forget` and `run --dry-run`
  remain outstanding.
- `ASCEND_TOOLKIT_HOME` must select the 9.1 toolkit. The default shell exports
  `/usr/local/Ascend/cann-8.5.0`, which fails admission by design.
- `VLLM_ASCEND_TORCH_PREFLIGHT=0` is inherited from the earlier validation
  method: the frozen host's preflight timeout is shorter than a cold import on
  this machine. Weight loading, KV-cache initialization and inference all
  completed on the NPU afterwards, so this is not a substitute for NPU
  execution.
- Correctness-only: single device, TP=1, `--enforce-eager`, one artifact. No
  matched BF16/W8A8 throughput, TTFT, TPOT or PPL comparison exists.

## Raw evidence

The phases below are reproducible with one command:
`runtime-extension/tools/npu_e2e.py` (see the runtime-extension README). The
per-phase transcripts were produced before the driver existed and are kept as
the primary record.

```text
/data/jxd/ascend-quant-validation/20260917-frozen-v1/wheel-e2e/npu_e2e_summary.json  driver summary, "ok": true
/data/jxd/ascend-quant-validation/20260917-frozen-v1/wheel-e2e/npu_e2e_plan.txt      driver plan of the same run
/data/jxd/ascend-quant-validation/20260917-frozen-v1/wheel-e2e/wheel_npu_e2e.txt               install, discovery, launch, inference, uninstall, restore
/data/jxd/ascend-quant-validation/20260917-frozen-v1/wheel-e2e/loader_check.txt                loader-level activation and enabled/disabled contrast
/data/jxd/ascend-quant-validation/20260917-frozen-v1/wheel-e2e/legacy_and_reference_control.txt legacy-artifact rejection and reference-artifact admission
/data/jxd/ascend-quant-validation/20260917-frozen-v1/wheel-e2e/chat_wheel.json                 request payload
/data/jxd/ascend-quant-validation/20260917-frozen-v1/wheel-e2e/serving-wheel-041a4.log         copy of the serving log of the wheel run
/data/jxd/validation/w8a8-plugin-v1-cann91-wheel-041a4.log                                    serving log of the wheel run
/data/jxd/validation/npu-e2e-vllm_ascend_quant_ext-0.4.1a4-py3-none-any-port18003.log         serving log of the driver run
/data/jxd/validation/w8a8-plugin-v1-cann91-triton322b.log                                     earlier editable-install run
/data/jxd/ascend-quant-validation/20260917-frozen-v1/w8a8/                                    hardlink copy with the re-exported contract
/data/jxd/validation/Qwen2.5-14B-Instruct-w8a8-plugin-v1/                                     artifact used for the serving run
```
