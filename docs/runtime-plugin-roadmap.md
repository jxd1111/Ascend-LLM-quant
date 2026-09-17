# Runtime plugin roadmap

## Completed

- independent runtime wheel and vLLM native entry point;
- Manifest 0.2 bundle registration (`vllm_hust.extension_bundles`) with a static
  manifest locator, package data and a repository-level `.vllm-hust` declaration;
- default-off, idempotent W8A8 registration;
- exact frozen-host revision admission, including abbreviated setuptools-scm
  tokens;
- contract 1.1 file/tensor/hash validation;
- isolated wheel/sdist install, discovery and uninstall checks;
- real Ascend NPU startup, inference, disabled-path and native recovery tests
  for the historical `0.4.1a2` host pair;
- correctness-only NPU validation on the frozen `v1` host pair;
- PyPI Trusted Publishing.

## Next evidence work

1. Run the remaining Manager lifecycle gates (`validate`, `configure`, `enable`,
   `run --dry-run`, `disable`, `forget`) once `vllm-hust-ext` leaves its frozen
   alpha state, against the published wheel hash.
2. Publish matched BF16/W8A8 throughput, TTFT, accuracy/PPL and HBM evidence.
3. Add a clean frozen-host integration job when an authorized NPU runner is
   available; do not use an unmanaged self-hosted GitHub runner.
4. Admit a new host revision only through an explicit compatibility PR and
   new evidence record.

W4A4 and W4A8 declarations remain format-level work until independently
validated on target hardware.
