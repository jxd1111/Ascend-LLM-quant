# Quant extension release checklist

## Quantization-team gate

- [x] Toolkit and Runtime Extension are separate distributions.
- [x] Runtime installation is default-off.
- [x] No monkey patch is used.
- [x] Static manifest package is included in the wheel.
- [x] `vllm_hust.extension_bundles` entry point matches the Bundle ID.
- [x] Bundle declares an import-only artifact validator with empty activation.
- [x] Import-only validator requests filesystem-read permission only.
- [x] Experimental manifest host range is restricted to the validated vLLM-Ascend line.
- [x] Closed W8A8 artifact contract and negative tests exist.
- [x] Contract 1.1 binds config, description, index, weight and evidence files by size/SHA-256.
- [x] Safetensors payload bounds, dtype/shape byte spans and overlap/contiguity fail closed.
- [x] Admission runs before implementation imports.
- [x] Manager-facing descriptor/check/plan/render adapter exists.
- [x] Enabled Manager plan/render fail closed while status is import-only.
- [x] Disabled render is deterministic and removes only extension-owned state.
- [x] Namespaced W8A8 aliases delegate to host-owned implementations.
- [x] Native `W8A8_MIX` metadata restoration is explicit and documented.
- [x] W8A8 Toolkit to Runtime Extension 0.4.0 NPU E2E has passed.
- [x] Runtime Extension 0.4.1 / contract 1.1 NPU E2E revalidation has passed.
- [x] Direct plugin-disabled `ASCEND_QUANT_W8A8` startup fails closed.
- [x] Direct plugin-disabled native `W8A8_MIX` startup and inference pass.
- [x] Wheel/sdist content and isolated install/discovery/uninstall verification is scripted.
- [ ] Repository URL, maintainers and release owner confirmed.
- [ ] Git history, signed tag and published wheel created.
- [ ] Framework team accepts the non-experimental Manager schema/kind/API.
- [ ] PyPI project ownership and scoped publishing credentials are confirmed.
- [ ] A protected tag/release publishing workflow is enabled.
- [ ] The published version passes a no-cache PyPI installation smoke test.

## Evidence gate

- [x] W8A8 PPL raw log retained.
- [x] W8A8 ShareGPT RPS 1/concurrency 8 raw log retained.
- [x] W8A8 ShareGPT RPS 8/concurrency 80 raw log retained.
- [ ] Matched BF16 runs retained under the same protocol.
- [ ] Idle, loaded and peak HBM retained for BF16 and W8A8.
- [ ] Evidence records pass validation and reference immutable dataset/model IDs.

## Joint Manager gate

- [x] Isolated wheel discovery reads the manifest without importing runtime code.
- [ ] Manager discovers installed manifest without importing implementation.
- [ ] Manager rejects incompatible manifest/host/version/permissions.
- [ ] Manager check rejects an invalid artifact before implementation import.
- [ ] Future active Manager plan/render preserve other plugins, including `ascend`.
- [ ] Manager-enabled W8A8 startup and inference pass.
- [ ] Disable and uninstall rollback pass without changing model hashes.

Unchecked Manager items are owned by or require the Extension Manager and
vLLM-Ascend teams. They do not block an import-only extension-side hand-off,
but they block active runtime enablement and any claim of complete formal
Manager integration.
