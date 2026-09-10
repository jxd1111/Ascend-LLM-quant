# Runtime plugin roadmap after architecture review

## Priority 0: design and host agreement

- review `current-vllm-ascend-quant-architecture.md` with the vLLM-Ascend owner;
- review `runtime-plugin-design.md` with the Extension Manager owner;
- freeze manifest, capability and typed scheme-provider interfaces;
- decide whether scheme registration remains in vLLM-Ascend or becomes a
  supported external extension point;
- record decisions as ADRs before moving additional algorithms.

Exit criterion: both teams approve the input/output and lifecycle boundary.

## Priority 1: W8A8 extraction

- replace imports of concrete host W8A8 implementations with an accepted
  public SPI or plugin-owned namespaced implementations;
- add the generic `ASCEND_QUANT_W8A8` runtime profile while preserving the
  validated offline recipe semantics;
- keep the W8A8 artifact profile separate from runtime execution role, and
  test standalone/producer/consumer path selection;
- validate operator availability before model allocation;
- cover Linear and, separately, MoE parameter shapes and TP/EP behavior.

Exit criterion: W8A8 runs from an installed wheel without private host imports,
monkey patching or changes to model files.

## Priority 2: performance and maintainability

- profile activation quantization, transpose/NZ conversion and scale handling;
- evaluate fusing norm plus activation quantization using host graph passes;
- avoid CPU/NPU synchronization and repeated scale/layout conversion;
- cache deterministic post-load transforms where host contracts permit it;
- publish matched BF16/W8A8 throughput, TTFT, PPL/accuracy and HBM evidence;
- add version/shape/operator negative tests and performance regression gates.

Each optimization requires a profile showing the bottleneck and an unchanged
artifact/API contract, or a versioned contract update.

## Priority 3: additional formats

- evaluate W4A8 only after the target Ascend hardware/operator path is proven;
- evaluate W4A4 separately because packing and activation semantics differ;
- add formats one at a time with independent evidence and promotion gates;
- never infer support from schema validation alone.

## Work explicitly assigned elsewhere

- Extension Manager implementation/materializer: framework team;
- NPU platform and typed host SPI: vLLM-Ascend team;
- KV-cache quantization/scheduling: Adaptive Quantized KV team;
- offline quantization recipes and calibration: Toolkit team.

## Open technical questions

1. Can vLLM-Ascend expose parameter specifications as stable typed records
   instead of dictionaries of empty tensors?
2. Can physical weight layout conversion be selected through a public operator
   capability rather than importing `maybe_trans_nz`?
3. Should fused-module prefix mapping be host metadata or artifact metadata?
4. How should Manager plans bind an artifact hash so worker admission detects a
   plan/start time-of-check-to-time-of-use change?
5. Which host API reports the execution role used by the W8A8 runtime-path
   selector without coupling this plugin to
   KV-cache implementation details?
6. What is the supported fallback policy when an optimized kernel is absent?
   The default proposed policy is fail closed, not silent BF16 execution.
