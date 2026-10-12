# W8A8 verification evidence

This page records only results that can be traced to existing raw logs. It
deliberately does not label unmatched or incomplete measurements as a matched
BF16/W8A8 benchmark.

## Runtime Extension 0.4.1 / contract 1.1 NPU revalidation

The contract 1.1 artifact passed complete admission, NPU 7 model loading,
deterministic chat completion, disabled fail-closed behavior, native
`W8A8_MIX` rollback, and install/uninstall immutability checks. The bound
artifact evidence level is `npu_e2e`.

See [`validation-w8a8-v0.4.1-20260914.md`](validation-w8a8-v0.4.1-20260914.md)
for the exact artifact identity, software versions, HBM snapshot, raw-log
locations, and remaining matched-benchmark limitations.

## Runtime Extension 0.5.0 ONLINE serving-view replay

The Toolkit-generated ONLINE view (the BF16 checkpoint plus the per-linear
`quant_model_description.json`) was served with
`VLLM_ASCEND_QUANT_EXT_SMOOTHING` on the same frozen `v1` host and answered
exactly like the converted W8A8 artifact. The gate installed the built wheel,
still refused the pre-v1 artifact control, compared the online answer with the
offline artifact's, uninstalled the wheel again and left both artifact hashes
unchanged.

See
[`evidence/w8a8-online-view-npu-e2e-20261012.md`](evidence/w8a8-online-view-npu-e2e-20261012.md)
for the wheel hash, the view and attachment identities, the phase table and the
raw log paths.

This establishes ONLINE correctness replay, not `matched_benchmark`: no BF16 or
W8A8 throughput, TTFT, TPOT or PPL comparison was measured under this protocol.

## Fresh Toolkit to Runtime Extension E2E

- Public artifact name: `Qwen2.5-14B-Instruct-w8a8`
- Artifact ID: `qwen2:W8A8:c9f0c8f014c84ae2`
- Contract SHA-256:
  `d73a901bdd1bf93369d725aac75bf88b0f14a380944929ea8f90172ddd6915e3`
- Profile: W8A8 / `ASCEND_QUANT_W8A8`
- Extension: `vllm-ascend-quant-ext==0.3.0`
- torch-npu: `2.9.0`
- vLLM-HUST: `0.17.2.post2.dev27+g54ee69103`
- vLLM-Ascend-HUST: `0.1.dev2790+g95c956c6a`
- Device: NPU 7, tensor parallel 1
- Result: model load, health request, chat completion and shutdown passed
- Quantization log: retained in the local W8A8 experiment archive
- Runtime log: retained in the private experiment archive; it must be attached
  to a GitHub release or committed as a redacted evidence artifact before this
  result can be independently reproduced.

This establishes `npu_e2e`, not `matched_benchmark`.

## Existing W8A8 PPL run

The retained log identifies the evaluated artifact as the W8A8 prototype used
before public identifier normalization:

| Dataset | PPL |
|---|---:|
| WikiText-2 | 5.930791293525971 |
| C4 | 9.523090100630093 |

Raw log: retained in the local W8A8 experiment archive.

A BF16 run using the identical evaluator, tokenizer, sequence length and data
revision is still required before publishing a PPL delta.

## Existing W8A8 ShareGPT-V3 runs

Dataset file: `ShareGPT_V3_unfiltered_cleaned_split.json`

Dataset SHA-256:
`35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`

All reported values are medians of three recorded runs with 200 successful and
zero failed requests. The tokenizer is the unquantized
`Qwen/Qwen2.5-14B-Instruct` tokenizer corresponding to the evaluated model.

| RPS / concurrency | Req/s | Output tok/s | Total tok/s | Mean TTFT ms | P99 TTFT ms | Mean TPOT ms | P99 TPOT ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 / 8 | 0.79 | 155.34 | 323.62 | 145.70 | 166.69 | 48.78 | 53.17 |
| 8 / 80 | 3.37 | 668.13 | 1384.92 | 159.70 | 268.67 | 57.62 | 65.57 |

Raw logs are retained in the local W8A8 ShareGPT experiment archive.

These are valid W8A8 single-profile performance records. They are not yet a
matched comparison because corresponding BF16 raw logs and HBM measurements
have not been retained under the same protocol.

The measurements predate the public rename to `W8A8`; new release evidence
must regenerate the artifact contract and record the normalized identifiers.

## Remaining evidence actions

1. Run BF16 with the exact same PPL and ShareGPT commands.
2. Record NPU model and CANN version in each raw run header.
3. Record idle, loaded and peak HBM for both BF16 and W8A8.
4. Convert both profiles to schema-valid evidence JSON files.
5. Only then promote the artifact evidence level to `matched_benchmark`.
