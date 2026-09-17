# ADR 0002: Manifest 0.2 extension bundle registration

- Status: accepted
- Date: 2026-09-17
- Amends: ADR 0001 (same package boundary; adds a discovery-only registration and
  corrects the `VLLM_PLUGINS` wording)

## Context

The runtime extension was discoverable only through the vLLM runtime hook
`vllm.general_plugins/vllm_ascend_quant`. The main repository therefore could not
locate or describe the plugin from installed distribution metadata, and the
release tooling asserted that no Extension Manager metadata existed at all.

The Extension Manager defines Manifest 0.2 with `kind: in_process_plugin` and the
`vllm_hust.extension_bundles` entry-point group for exactly this in-process case.
The Extension Manager itself is still a frozen alpha, and the frozen host exposes
no independently versioned plugin API, so this decision must not make the runtime
depend on Manager tooling.

## Decision

Publish the extension as a Manifest 0.2 bundle in addition to the runtime hook:

- `vllm_hust.extension_bundles/org.vllm-hust.ascend-quant-runtime` resolves to the
  static manifest locator `vllm_ascend_quant_ext.manifests`;
- the manifest declares `kind: in_process_plugin`, host `vllm-ascend`,
  `lifecycle_owner: vllm`, `runtime.process_scope:
  vllm_engine_and_ascend_worker`, the consumed protocols `vllm.general_plugins`
  and `vllm-ascend.quantization-scheme-surface`, and the minimal permissions
  `device_access` and `filesystem_read`;
- `vllm.general_plugins/vllm_ascend_quant` stays the only runtime activation entry
  point and no new `vllm.*` namespace is claimed;
- activation stays explicit and extension-owned: the Manager declares the
  environment it needs, the extension keeps `VLLM_ASCEND_QUANT_EXT_ENABLE`, and
  `VLLM_PLUGINS` is never narrowed by hand because that variable filters every
  plugin group;
- versioning stays two-track: manifest `host.version_range` is the package range
  while `host_baseline.py` keeps the exact frozen revisions fail-closed;
- the deployment artifact path stays user configuration and is deliberately not
  part of the static manifest.

## Consequences

- the main repository can locate, validate and describe the plugin from
  distribution metadata without importing the implementation;
- the wheel and sdist carry `manifests/vllm-hust-extension-v0.2.json`, and the
  release verifier fails when it is missing or when the registration name and the
  manifest `extension_id` diverge;
- Manager CLI lifecycle gates (`extension validate`, `enable`, `disable`,
  `forget`, `run --dry-run`) remain outstanding while the Manager is frozen;
  disable, uninstall and native recovery are verified at runtime instead;
- renaming the registration namespace is a breaking interface change and needs a
  new ADR plus a synchronized manifest, packaging and test update.

## Manager-validated manifest rules

The Manager does not read this manifest the way the 0.2 description alone
suggests: it re-parses `components` with its Bundle v1 component schema and it
reserves the `status` key inside the runtime qualification profile. Three rules
follow, each covered by a test:

- every `components[].contracts` entry must stay inside the `vllm.` namespace.
  The Ascend-side surface therefore lives under `protocols` only; mixing it into
  `contracts` makes the Manager reject the whole manifest.
- `protocols[].version_range` is `null`. The frozen host exposes no independently
  versioned protocol surface, so a concrete range is unverifiable and makes the
  Manager refuse to launch this `trusted_in_process` extension. `null` makes the
  Manager defer to `host.version_range` plus the acceptance evidence, which is
  where the real compatibility statement already lives.
- `activation.additional_config._manager_runtime_qualification` must not carry a
  `status` key. The Manager requires the operator to configure
  `status: "passed"` and then compares every manifest key against that
  configuration, so an embedded `status` can never match; the scope statement is
  declared as `qualification_scope` instead.

Verified with the Manager at `cf1ea71` in an isolated client environment; see
`docs/evidence/w8a8-frozen-v1-npu-e2e-20260917.md`. Keeping
`vllm.general_plugins/vllm_ascend_quant` registered is still required: the
bundle registration only makes the extension discoverable and describable.
