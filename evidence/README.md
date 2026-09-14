# Evaluation evidence

`example-result-v1.json` is a deliberately invalid template: values such as
`replace-me` prevent an unfinished record from being accepted accidentally.
Copy it to a result file, replace every placeholder, and keep raw logs at safe
relative paths beside the record (or use immutable HTTPS release-asset URLs).

Validate a completed record with:

```bash
ascend-quant-toolkit validate-evidence --file /path/to/result.json
```

Validation requires:

- 64-character lowercase SHA-256 values for the artifact file inventory and dataset;
- non-empty software, hardware and dataset identities;
- one unique device ID per declared NPU;
- `successful + failed == prompts`;
- finite, non-negative performance/latency/HBM values;
- `idle_mib <= loaded_mib <= peak_mib`;
- at least one safe relative raw-log path or HTTPS URL.

Referencing a result from `ascend_quant_artifact.json` additionally binds that
result file by size and SHA-256. A missing or modified evidence record fails
runtime admission.
