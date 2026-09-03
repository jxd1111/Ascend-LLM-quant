# Quant extension release checklist

## Quantization-team gate

- [x] Toolkit and Runtime Extension are separate distributions.
- [x] Runtime installation is default-off.
- [x] No monkey patch is used.
- [x] Static manifest is included in the wheel and installed data directory.
- [x] Closed W8A8 artifact contract and negative tests exist.
- [x] Admission runs before implementation imports.
- [x] Manager-facing descriptor/check/plan/render adapter exists.
- [x] Enable and disable renders are deterministic and read-only.
- [x] W8A8 Toolkit to NPU runtime E2E has passed.
- [x] Wheel clean-install/discovery/uninstall verification is scripted.
- [ ] Repository URL, maintainers and release owner confirmed.
- [ ] Git history, signed tag and published wheel created.
- [ ] Framework team accepts the non-experimental Manager schema/kind/API.

## Evidence gate

- [x] W8A8 PPL raw log retained.
- [x] W8A8 ShareGPT RPS 1/concurrency 8 raw log retained.
- [x] W8A8 ShareGPT RPS 8/concurrency 80 raw log retained.
- [ ] Matched BF16 runs retained under the same protocol.
- [ ] Idle, loaded and peak HBM retained for BF16 and W8A8.
- [ ] Evidence records pass validation and reference immutable dataset/model IDs.

## Joint Manager gate

- [ ] Manager discovers installed manifest without importing implementation.
- [ ] Manager rejects incompatible manifest/host/version/permissions.
- [ ] Manager check rejects an invalid artifact before implementation import.
- [ ] Manager plan/render preserve other plugins, including `ascend`.
- [ ] Manager-enabled W8A8 startup and inference pass.
- [ ] Disable and uninstall rollback pass without changing model hashes.

Unchecked Manager items are owned by or require the Extension Manager team.
They do not block extension-side hand-off, but they do block a claim of complete
formal Manager integration.
