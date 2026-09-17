# W8A8 Runtime Extension frozen v1 NPU E2E evidence (2026-09-17)

Correctness-only NPU validation against the frozen vLLM-HUST `v1` host on the
documented Ascend platform snapshot. This is not a matched BF16/W8A8
performance comparison and it does not widen the supported host range.

## Object under test

- Runtime Extension source: revision `b6b9981` plus the working-tree Manifest 0.2
  packaging changes, installed editable (no wheel-hash-level NPU rerun yet).
- Planned release: `vllm-ascend-quant-ext==0.4.1a4`.
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
- Driver `25.3.rc1`; NPU Ascend 910B3, device 7, TP=1, `--enforce-eager`.

## Results

| Check | Result | Note |
|---|---|---|
| Wheel and sdist build, `twine check` | PASS | wheel `44d73089f02062534f9464edff3e926ed8af0cbc2d115f42bb5a1c8b90642d04`, sdist `9f3a12d190dc6a29ba8a619d21a2a212c012a28db05d270942a342b680d39413` |
| Isolated install, entry-point discovery, uninstall | PASS | both entry-point groups discovered from the wheel; `extension_id` equals the registration name; uninstall leaves no residue |
| Manifest 0.2 discoverable from distribution metadata | PASS | the bundle locator resolved the manifest without importing the implementation |
| Manifest, packaging and host-contract tests | PASS | 70 passed, 0 skipped |
| Toolkit unit tests and Ruff | PASS | 17 passed; `ruff check` clean |
| Frozen host revision admission | PASS | abbreviated token `+gf18cf803c.empty` admitted by full-commit comparison |
| Artifact admission under the frozen baseline | PASS | 2595 tensors, all four software checks |
| Legacy artifact fail-closed | PASS | rejected with `incompatible torch_npu: installed 2.13.0rc1, required >=2.9,<2.10` |
| CANN gate fail-closed | PASS | without `ASCEND_TOOLKIT_HOME` on 9.1: `incompatible CANN: installed 8.5.0, required >=9.1,<9.2` |
| Default-off registration | PASS | `status.enabled=false`; `torch`, `vllm` and `vllm_ascend` stay unimported |
| NPU weight load and deterministic inference | PASS | chat completion returned exactly `OK`; `system_fingerprint vllm-0.28.1.post1.dev143+gf18cf803c` |
| NPU HBM | observed | idle 3414 MiB, loaded 61584 MiB (single sample, not a peak series) |

## Limitations

- The quantized weights predate this validation; the artifact contract was
  re-exported by the Toolkit running under the frozen baseline. The offline
  `msmodelslim` conversion step was not re-run under CANN 9.1.
- NPU inference used the editable install rather than the `0.4.1a4` wheel. A
  wheel-hash-level NPU rerun is required before a public release.
- The Extension Manager CLI (`vllm-hust-ext`) was unavailable on this host, so
  `extension validate`, `enable`, `disable`, `forget` and `run --dry-run`
  remain outstanding.
- `ASCEND_TOOLKIT_HOME` must select the 9.1 toolkit. The default shell exports
  `/usr/local/Ascend/cann-8.5.0`, which fails admission by design.
- No matched BF16/W8A8 throughput, TTFT, TPOT or PPL comparison exists.

## Raw evidence

```text
/data/jxd/ascend-quant-validation/20260917-frozen-v1/w8a8/   hardlink copy with the re-exported contract
/data/jxd/validation/Qwen2.5-14B-Instruct-w8a8-plugin-v1/    artifact used for the serving run
/data/jxd/validation/w8a8-plugin-v1-cann91-triton322b.log    serving log
```
