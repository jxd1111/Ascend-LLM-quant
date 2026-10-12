# ONLINE W8A8 serving view NPU E2E evidence (2026-10-12)

Serving-view replay of the fitted smoothing factors on the frozen `v1` host:
the BF16 checkpoint plus a generated `quant_model_description.json` is served
with `VLLM_ASCEND_QUANT_EXT_SMOOTHING` and has to answer exactly like the
converted W8A8 artifact. This is a correctness gate, not a benchmark.

**Verdict: PASS**

## Object under test

- Runtime Extension wheel `vllm-ascend-quant-ext==0.5.0`, sha256 `29ee5e510685909ea164a28a83378e4d1716b31bb1211d3ab97b4495823419cb`, installed from the built wheel (not the source tree).
- ONLINE view `/data/jxd/ascend-online-w8a8/view-v042`: `quant_model_description.json` sha256 `83927dad6fae3853ef3150c681cda52534f8810876d82b42b2ca73a28f43b7b6`, 336 replayed linears, 99 FLOAT weights.
- Attachment `/data/jxd/ascend-online-w8a8/attachment-v042`: `attachment_id` `c71ca0a7c41286525a734b71c2fcd8be4eec3136e77477d5002acb173b43e3f2`,
  `verified` `True` after the producer's own replay check at `max_iterator_diff <= 2` and `min_match_rate >= 0.9`,
  worst replayed match rate `0.914941`, worst LSB deviation `2`.
- Offline artifact compared against: `/data/jxd/validation/Qwen2.5-14B-Instruct-w8a8-plugin-v1`
 (tree sha256 `f802a0c59ea16fe6e3be31a47b3a2c577c72014e932f26b62554e940366a1828`).
- Device 0, port 18003, ONLINE port 18004 (`--port + 1` unless `--online-port` overrides it).

## Results

| Phase | Result | Observed |
|---|---|---|
| `artifact-hash-before` | PASS | files=17; tree_sha256="f802a0c59ea16fe6e3be31a47b3a2c577c72014e932f26b62554e940366a1828" |
| `previous-install` | PASS | editable="/root/jxd-ascend-quant/runtime-extension"; installed=true; module_file="/root/jxd-ascend-quant/runtime-extension/src/vllm_ascend_quant_ext/__init__.py" |
| `wheel-resolution` | PASS | bundles=[["org.vllm-hust.ascend-quant-runtime", "vllm_ascend_quant_ext.manifests"]]; bundles_environment=[["org.vllm-hust.ascend-quant-runtime", "vllm_ascend_quant_ext.manifests"], ["org.vllm-hust.simllm", "vllm_ascend_simllm"]]; from_site_packages=true |
| `loader-enabled` | PASS | admission_line="2026-10-12 05:25:45,687 vllm_ascend_quant_ext.plugin INFO Ascend quant runtime admitted /data/jxd/validation/Qwen2.5-14B-Instruct-w8a8-p...; admitted_record=true; linear="AscendQuantW8A8LinearMethod" |
| `loader-disabled` | PASS | linear="None"; linear_module=null; moe=null |
| `loader-legacy-artifact` | PASS | legacy_artifact="/data/jxd/models/Qwen2.5-14B-Instruct-w8a8-runtime-v041"; reason="Ascend quant artifact admission failed: incompatible CANN: installed 9.1.0, required >=8.5,<8.6"; rejected=true |
| `server-health` | PASS | status=200; exit_code=null; log="/data/jxd/validation/npu-e2e-online-0.5.0-port18003.log" |
| `deterministic-inference` | PASS | content="OK"; identical=true; prompt="Reply with exactly the two letters OK and nothing else." |
| `server-stop` | PASS | - |
| `online-server-health` | PASS | status=200; exit_code=null; log="/data/jxd/validation/npu-e2e-online-0.5.0-port18003-online.log" |
| `online-deterministic-inference` | PASS | content="OK"; identical=true; scheme="ASCEND_QUANT_W8A8_ONLINE" |
| `online-matches-offline` | PASS | offline="OK"; online="OK" |
| `online-server-stop` | PASS | - |
| `uninstall` | PASS | leftovers=[] |
| `restore-editable` | PASS | editable="/root/jxd-ascend-quant/runtime-extension"; expected="/root/jxd-ascend-quant/runtime-extension"; installed=true |
| `artifact-hash-after` | PASS | changed=0; files=17; tree_sha256="f802a0c59ea16fe6e3be31a47b3a2c577c72014e932f26b62554e940366a1828" |

## Reproduction

```bash
python runtime-extension/tools/npu_e2e.py \
  --wheel runtime-extension/dist/vllm_ascend_quant_ext-0.5.0-py3-none-any.whl \
  --artifact /data/jxd/validation/Qwen2.5-14B-Instruct-w8a8-plugin-v1 \
  --online-view /data/jxd/ascend-online-w8a8/view-v042 \
  --smoothing /data/jxd/ascend-online-w8a8/attachment-v042 \
  --device <device> --port <port> \
  --env-script <cann-9.1>/set_env.sh --env-script '<atb>/set_env.sh --cxx_abi=0' \
  --log /data/jxd/validation/npu-e2e-online-0.5.0-port18003.log --summary /data/jxd/validation/npu-e2e-online-0.5.0-summary.json
```

The view is the artefact of `ascend-quant-toolkit build-online-view` against the same
BF16 source, and the attachment is the artefact of
`ascend-quant-toolkit build-smoothing` against the offline artifact; the gate never
creates either one itself.

## Limitations

- Correctness only: single device, TP=1, `--enforce-eager`, one artifact and one
  attachment; no matched BF16/W8A8 throughput, TTFT, TPOT or PPL comparison.
- The ONLINE factors are fitted from the BF16 checkpoint and the converted artifact;
  the offline ModelSlim conversion step is not re-run here.
- The gate proves the served answer matches the artifact's answer on one prompt; it
  does not compare whole-vocabulary logits against the artifact.
