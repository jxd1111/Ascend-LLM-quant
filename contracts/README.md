# Ascend quantized artifact contract

`ascend-quant-artifact-v1.schema.json` is the closed, versioned interchange
contract between the offline Ascend Quant Toolkit and the separately installed
`vllm-ascend-quant-ext` runtime extension.

The JSON schema freezes the declarative surface. Runtime validation additionally
checks safetensors headers and cross-field invariants that JSON Schema cannot
express: packed logical dimensions, parameter dtype/shape, scale granularity,
zero-point tensors, model identity, software compatibility, and loader/scheme
availability.

Unknown fields and unknown enum values are intentionally rejected. Adding a new
layout or operator therefore requires a new compatible contract revision rather
than silently changing the meaning of an existing artifact.

The initial contract contains profiles for W8A8, W4A4 and W4A8. Only profiles
listed in an artifact's `evidence.verified_profiles` may be described as
hardware-verified; schema support alone is not performance evidence.

Contract v1 admits unpacked per-channel W8A8, packed signed-nibble W4A4
(per-channel or single-level per-group scale), and packed signed-nibble
per-channel W4A8 with the v1 `scale_bias` tensor. Legacy W4A8 `version=1.0`
and hierarchical per-group `weight_scale_second/weight_offset_second` are not
silently generalized: they fail closed until a separately frozen profile is
added and verified.
