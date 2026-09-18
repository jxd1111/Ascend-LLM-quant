# W8A8 Runtime Extension frozen v1 NPU E2E evidence (2026-09-17)

Wheel-hash-level correctness validation of the released artifact against the
frozen vLLM-HUST `v1` host on the documented Ascend platform snapshot. This is
not a matched BF16/W8A8 performance comparison and it does not widen the
supported host range.

## Object under test

- Runtime Extension source: revision `c6f42e5` on
  `feature/manifest-0.2-extension-bundles`.
- Artifact under test: `vllm-ascend-quant-ext==0.4.1a4`,
  wheel `14a72b41031a6b4ca07aed5209a369d40c6d0b180f3d87b3d8485bdd4746f22e`,
  sdist `097394523e609a668cdf52daaf00ff40cdb278e84a71bdee0a6d390d7998d04d`.
  The wheel was rebuilt after the Manager-gate fixes below; the earlier
  per-phase transcripts were captured with the pre-fix build
  `44d73089f02062534f9464edff3e926ed8af0cbc2d115f42bb5a1c8b90642d04`. The same
  phases are re-run with `tools/npu_e2e.py` against this final build and
  archived as `npu_e2e_summary_final.json`.
- Published artifacts: release `runtime-v0.4.1a4` (pre-release) on tag commit
  `cdd8eea`, built by `publish-runtime.yml` run #3.
  - PyPI wheel
    `24ab724c6e526206c504d6e0baf0b5a98ddc0a160fdc01d7e3c1764857169f5c`
  - PyPI sdist
    `c1e7dd455980b498a674a679897a86a47a0f1a1d01fa0cfdb9fae72d7569125c`
  - TestPyPI rehearsal wheel
    `053810b370522f3bca3e45b14f08efd935d1930cc3cccbf68d7ea6507b6e12af`
  - The two wheels are not byte-identical (ZIP framing differs) but every
    uncompressed file matches exactly, so the gated and the published artifact
    carry identical content.
- Install under test: the wheel above installed non-editable into
  `/data/jxd/envs/vllm-hust-v1-cann91`. The source tree was removed from
  `PYTHONPATH`, and `vllm_ascend_quant_ext.__file__` resolved to
  `.../site-packages/vllm_ascend_quant_ext/__init__.py` in every process.
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
- Driver `25.3.rc1`; NPU Ascend 910B3, device 6, TP=1, `--enforce-eager`,
  port 18003. Device 7 held the earlier editable-install serving run.

## Results

| Check | Result | Note |
|---|---|---|
| Wheel and sdist build, `twine check` | PASS | hashes above |
| Wheel replaces the editable install | PASS | module resolved from `site-packages`; no `__editable__*vllm_ascend_quant_ext*` left |
| Installed metadata exposes both entry-point groups | PASS | `vllm.general_plugins/vllm_ascend_quant` and `vllm_hust.extension_bundles/org.vllm-hust.ascend-quant-runtime` |
| Manifest discoverable from installed metadata | PASS | `.../site-packages/vllm_ascend_quant_ext/manifests/vllm-hust-extension-v0.2.json`, also listed in `RECORD` |
| Manifest, packaging and host-contract tests | PASS | 70 passed, 0 skipped |
| Toolkit unit tests and Ruff | PASS | 17 passed; `ruff check` clean |
| vLLM general-plugin loader activates the wheel | PASS | in-process record `vllm_ascend_quant_ext.plugin INFO Ascend quant runtime admitted ...; registered=['ASCEND_QUANT_W8A8/linear']` |
| Scheme registry answers only when enabled | PASS | enabled -> `AscendQuantW8A8LinearMethod`; with `VLLM_ASCEND_QUANT_EXT_ENABLE` unset -> `None` |
| MoE scheme stays fail-closed | PASS | `get_scheme_class("ASCEND_QUANT_W8A8", "moe") -> None` |
| Legacy artifact fail-closed | PASS | at plugin-load time: `Ascend quant artifact admission failed: incompatible CANN: installed 9.1.0, required >=8.5,<8.6` |
| CANN gate fail-closed | PASS | without `ASCEND_TOOLKIT_HOME` on 9.1: `incompatible CANN: installed 8.5.0, required >=9.1,<9.2` |
| Default-off registration | PASS | `status.enabled=false`; `torch`, `vllm` and `vllm_ascend` stay unimported |
| Frozen host revision admission | PASS | abbreviated token `+gf18cf803c.empty` admitted by full-commit comparison |
| Artifact admission under the frozen baseline | PASS | 2595 tensors, all four software checks (reported by `vllm-ascend-quant-ext check`; the wheel run performs the same gate before registering) |
| NPU weight load from the wheel | PASS | 4 safetensors shards; device 6 `61579/65536` MiB loaded |
| Deterministic chat completion from the wheel | PASS | two identical requests returned exactly `OK`; `system_fingerprint vllm-0.28.1.post1.dev143+gf18cf803c-fe28e53e` |
| Server stop and NPU release | PASS | 0 processes left; device 6 back to `3423/65536` MiB |
| Uninstall proves artifact provenance | PASS | `import vllm_ascend_quant_ext` -> `ModuleNotFoundError`; both entry-point groups disappear |
| Host restored after the run | PASS | editable install, `17 passed` and `70 passed`, `verify_release.py` valid again |
| Model artifact unchanged | PASS | per-file sha256 list identical before and after the run |
| Frozen-host gate driver re-run | PASS | `runtime-extension/tools/npu_e2e.py`, 13 phases, `"ok": true`, `"failed_phases": []`, exit 0, device 6, port 18003 |
| Extension Manager lifecycle gates | PASS with caveats | `vllm-hust-ext` `0.2.0.dev0` at `cf1ea71`, isolated client environment; see below |
| PyPI publication | PASS | release `runtime-v0.4.1a4` -> tag commit `cdd8eea`, `publish-runtime.yml` run #3 green, wheel `24ab724c…`, sdist `c1e7dd45…` |
| Published wheel vs gated wheel | PASS | every uncompressed file identical to `053810b3…`; only ZIP framing differs |
| NPU gate on the published wheel | PASS | `tools/npu_e2e.py` against `24ab724c…`: 13/13 phases, `"ok": true`, two identical `OK` completions |

## Extension Manager lifecycle gates

The Manager is not published yet (its support matrix says "尚未发布；发布冻结"), but
it installs from source, so these gates were run against a recorded revision
instead of being left open.

- Client: `vllm-hust-ext 0.2.0.dev0` at commit
  `cf1ea71e3e2cb81ab06267ef05eddb3e580ea20b`.
- Environment: scratch venv with `--system-site-packages` (Python 3.11) that
  sees this distribution; state isolated through `VLLM_HUST_EXT_CONFIG`.
  The Manager's tested client matrix lists Python 3.10/3.12, so 3.11 is a
  deviation that must be re-checked after the Manager releases.
- No global state was touched: the Manager config was written under `/tmp`, and
  the frozen host was not reconfigured.

| Command | Result | Note |
|---|---|---|
| `extension list --json` | PASS | discovered the bundle from the editable install; `torch` stayed unimported |
| `extension inspect` / `validate` | PASS | `activation_blocker: null` |
| `extension status` / `check` | PASS | `states: installed, discovered, compatible, configured, enabled`; evidence: host range satisfies, both protocols not independently versioned, required runtime qualification profile passed |
| `extension plan` / `render` | PASS | non-mutating `configure_launch`; rendered environment is exactly `VLLM_ASCEND_QUANT_EXT_ENABLE=1` |
| `extension env` | PASS | publishes the enabled-bundle marker |
| `extension configure` + `enable` | PASS | `configure` alone leaves `enabled: false`; the operator profile must carry `status: passed` |
| `run --dry-run -- vllm serve <artifact>` | PASS | exit 0; environment `VLLM_ASCEND_QUANT_EXT_ENABLE=1`, `VLLM_HUST_EXT_ENABLED_BUNDLES` marker, `native_extension_manifests: {}` (no fabricated host API range) |
| `extension disable` / `forget` | PASS | state returns to `{}`; `forget` refuses while the extension is enabled |

This gate found three real manifest defects, all fixed and now covered by tests
(see ADR 0002):

1. a host-side protocol name inside `components[].contracts` (the Manager
   validates that list against the `vllm.` namespace only);
2. concrete `protocols[].version_range` values for surfaces the frozen host does
   not version independently, which made the Manager report the extension as
   unverifiable and refuse `run`;
3. a `status` key inside the runtime qualification profile, which the Manager
   reserves for the operator-supplied configuration and therefore can never
   match.

## Release retrospective

Two workflow defects were found by the release process itself rather than by
review, and both are fixed:

- `rehearse-runtime-testpypi.yml` failed on its first dispatch at the test step
  because it installed only the build dependencies, so the subprocess probe in
  `tests/test_host_contract.py` could not import the package (fixed by #10);
- `publish-runtime.yml` failed the same way on the first PyPI release attempt.
  Nothing was published and no version was consumed, so the fix (#11) was
  followed by moving the tag and re-creating the release instead of bumping the
  version. Tag `runtime-v0.4.1a4` therefore points at `cdd8eea`, the commit that
  carries both fixes.

Both workflows now mirror `ci.yml`: install `jsonschema` and both local
distributions editable before running the tests.

## Limitations

- The quantized weights predate this validation; the artifact contract was
  re-exported by the Toolkit running under the frozen baseline. The offline
  `msmodelslim` conversion step was not re-run under CANN 9.1.
- This host configures only the `vllm` logger tree, so the extension's own INFO
  record is dropped at the default root level. The activation line above comes
  from a loader-level check that raises the level for the extension logger
  through a `sitecustomize.py` shim on `PYTHONPATH`. The shim changes neither
  the wheel nor the host, but the serving process itself did not re-print the
  record. Re-check it through the Manager once `vllm-hust-ext` is available.
- The Extension Manager CLI (`vllm-hust-ext`) was unavailable on this host, so
  `extension validate`, `enable`, `disable`, `forget` and `run --dry-run`
  remain outstanding.
- `ASCEND_TOOLKIT_HOME` must select the 9.1 toolkit. The default shell exports
  `/usr/local/Ascend/cann-8.5.0`, which fails admission by design.
- `VLLM_ASCEND_TORCH_PREFLIGHT=0` is inherited from the earlier validation
  method: the frozen host's preflight timeout is shorter than a cold import on
  this machine. Weight loading, KV-cache initialization and inference all
  completed on the NPU afterwards, so this is not a substitute for NPU
  execution.
- Correctness-only: single device, TP=1, `--enforce-eager`, one artifact. No
  matched BF16/W8A8 throughput, TTFT, TPOT or PPL comparison exists.

## Raw evidence

The phases below are reproducible with one command:
`runtime-extension/tools/npu_e2e.py` (see the runtime-extension README). The
per-phase transcripts were produced before the driver existed and are kept as
the primary record.

```text
/data/jxd/ascend-quant-validation/20260917-frozen-v1/wheel-e2e/PYPI_WHEEL_GATE_VERDICT.txt        gate verdict for the published wheel
/data/jxd/ascend-quant-validation/20260917-frozen-v1/wheel-e2e/pypi_npu_e2e_summary.json          gate summary for the published wheel
/data/jxd/ascend-quant-validation/20260917-frozen-v1/wheel-e2e/pypi-wheel.sha256                 hash of the downloaded published wheel
/data/jxd/ascend-quant-validation/20260917-frozen-v1/wheel-e2e/TESTPYPI_WHEEL_GATE_VERDICT.txt    gate verdict for the rehearsal wheel
/data/jxd/ascend-quant-validation/20260917-frozen-v1/wheel-e2e/npu_e2e_summary_final.json         driver summary for the local final build
/data/jxd/ascend-quant-validation/20260917-frozen-v1/wheel-e2e/manager-lifecycle-*.txt    Manager CLI lifecycle transcripts
/data/jxd/ascend-quant-validation/20260917-frozen-v1/wheel-e2e/npu_e2e_summary.json  driver summary, "ok": true
/data/jxd/ascend-quant-validation/20260917-frozen-v1/wheel-e2e/npu_e2e_plan.txt      driver plan of the same run
/data/jxd/ascend-quant-validation/20260917-frozen-v1/wheel-e2e/wheel_npu_e2e.txt               install, discovery, launch, inference, uninstall, restore
/data/jxd/ascend-quant-validation/20260917-frozen-v1/wheel-e2e/loader_check.txt                loader-level activation and enabled/disabled contrast
/data/jxd/ascend-quant-validation/20260917-frozen-v1/wheel-e2e/legacy_and_reference_control.txt legacy-artifact rejection and reference-artifact admission
/data/jxd/ascend-quant-validation/20260917-frozen-v1/wheel-e2e/chat_wheel.json                 request payload
/data/jxd/ascend-quant-validation/20260917-frozen-v1/wheel-e2e/serving-wheel-041a4.log         copy of the serving log of the wheel run
/data/jxd/validation/w8a8-plugin-v1-cann91-wheel-041a4.log                                    serving log of the wheel run
/data/jxd/validation/npu-e2e-vllm_ascend_quant_ext-0.4.1a4-py3-none-any-port18003.log         serving log of the driver run
/data/jxd/validation/w8a8-plugin-v1-cann91-triton322b.log                                     earlier editable-install run
/data/jxd/ascend-quant-validation/20260917-frozen-v1/w8a8/                                    hardlink copy with the re-exported contract
/data/jxd/validation/Qwen2.5-14B-Instruct-w8a8-plugin-v1/                                     artifact used for the serving run
```
