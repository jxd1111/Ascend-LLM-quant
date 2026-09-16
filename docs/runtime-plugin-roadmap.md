# Runtime plugin roadmap

## Completed

- independent runtime wheel and vLLM native entry point;
- contract 1.1 file/tensor/hash validation;
- default-off, idempotent W8A8 registration;
- exact frozen-host revision admission;
- isolated wheel/sdist install, discovery and uninstall checks;
- real Ascend NPU startup, inference, disabled-path and native recovery tests;
- PyPI Trusted Publishing.

## Next evidence work

1. Repeat NPU E2E for `0.4.1a2` built from the released wheel.
2. Publish matched BF16/W8A8 throughput, TTFT, accuracy/PPL and HBM evidence.
3. Add a clean frozen-host integration job when an authorized NPU runner is
   available; do not use an unmanaged self-hosted GitHub runner.
4. Admit a new host revision only through an explicit compatibility PR and
   new evidence record.

W4A4 and W4A8 declarations remain format-level work until independently
validated on target hardware.
