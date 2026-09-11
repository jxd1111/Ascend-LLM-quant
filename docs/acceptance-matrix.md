# Acceptance and evidence matrix

Results must be collected with matched model prompts, request arrival process,
tokenizer, input/output caps, device allocation, CANN/torch-npu/vLLM versions,
warm-up, run count, and aggregation method. A schema-valid artifact is not
automatically hardware-verified.

| Profile | Correctness | Accuracy/PPL | Throughput/TTFT | HBM | Status |
|---|---|---|---|---|---|
| BF16 | required | required baseline | required baseline | required baseline | pending repository evidence |
| W8A8 | passed on Qwen2.5-14B | single-profile PPL retained; matched BF16 delta pending | single-profile ShareGPT retained; matched BF16 comparison pending | pending matched HBM record | fresh Toolkit → Runtime Extension NPU E2E passed |
| W4A4 | required | required | required | required | pending |
| W4A8 | required | required | required | required | pending |

Current W8A8 NPU E2E evidence:

- public artifact name: `Qwen2.5-14B-Instruct-w8a8`;
- offline log: retained in the local W8A8 experiment archive;
- runtime log: retained in the private experiment archive and not yet published;
- contract level: `npu_e2e`, verified profile: `W8A8`;
- W4A4 and W4A8 remain unverified and must not be presented as supported hardware results.

Traceable W8A8 PPL and ShareGPT values, raw log locations, and the precise
remaining comparison gaps are recorded in [`w8a8-evidence.md`](w8a8-evidence.md).

Required result fields:

- immutable model/artifact identifier and hashes;
- contract schema and extension version;
- CANN, torch-npu, vLLM-HUST and vLLM-Ascend-HUST versions;
- NPU model/count and tensor-parallel configuration;
- dataset name, revision/hash, prompt count and sampling configuration;
- success/failure count, correctness metric and tolerance;
- PPL or task accuracy with BF16 delta;
- request/output/total token throughput, TTFT, TPOT and ITL percentiles;
- idle, loaded and peak HBM;
- raw log paths and aggregation script revision.

Contract 1.1 additionally requires immutable size/SHA-256 records for all
model files and every evidence result referenced by the artifact contract.

Promotion gates:

1. `schema_only`: JSON and static profile exist; no runtime claim.
2. `correctness`: deterministic output/tolerance gate passes.
3. `npu_e2e`: real Ascend load, inference, shutdown and rollback pass.
4. `matched_benchmark`: BF16 and quantized profiles pass the same protocol and publish raw evidence.

Negative gates must cover unknown fields, unsupported format, missing tensors,
wrong dtype, wrong packing/layout, scale/offset shape mismatch, unsupported
model/shape, incompatible software, unavailable loader/operator/scheme,
disabled startup, and uninstall rollback.
