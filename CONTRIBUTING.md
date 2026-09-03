# Contributing

## Design-first gate

Changes to runtime behavior, artifact layout, scheme IDs, plugin inputs/outputs,
startup lifecycle or compatibility ranges require a design update before code.

1. Update or add an ADR under `docs/adr/`.
2. Update `docs/runtime-plugin-design.md` and the artifact contract when needed.
3. Obtain review from the quantization owner and the affected host/Manager owner.
4. Implement the smallest change behind the approved interface.
5. Add unit, negative and NPU E2E evidence appropriate to the change.

Do not add monkey patches or put offline calibration/conversion into the runtime
extension. Do not combine model-weight and KV-cache lifecycle.

## Local checks

```bash
python -m pytest -q
python -m pytest -q runtime-extension/tests

python -m pip wheel . --no-deps --no-build-isolation -w dist
python -m pip wheel ./runtime-extension \
  --no-deps --no-build-isolation -w runtime-extension/dist

python runtime-extension/tools/verify_release.py \
  --wheel runtime-extension/dist/vllm_ascend_quant_ext-0.3.0-py3-none-any.whl
```

For NPU changes, record the exact CANN, torch-npu, vLLM-HUST and
vLLM-Ascend-HUST versions and retain raw logs.

## Pull requests

The submitter must understand and review every changed line. PRs must state:

- the approved design/ADR;
- whether public input/output or artifact format changes;
- commands and environments used for tests;
- NPU correctness/performance evidence, when applicable;
- rollback behavior;
- whether AI assistance was used.

Use signed-off commits:

```bash
git commit -s -m "docs: define W8A8 runtime plugin architecture"
```
