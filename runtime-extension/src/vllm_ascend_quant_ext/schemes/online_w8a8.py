"""Online W8A8: rebuild the artifact's INT8 weights from a BF16 checkpoint.

Only the weight side is replayed; activation quantization stays host-owned.
``apply`` and the transpose/NZ/flatten half of ``process_weights_after_loading``
are inherited unchanged from the host's native dynamic W8A8 scheme, so the
online path executes the same operators as a ModelSlim artifact would.

Replay contract, per linear::

    W'    = W * col[None, :] * row[:, None]
    scale = amax_j(|W'|) / 127        # per output channel, symmetric
    q     = clip(round(W' / scale), -128, 127)

which is exactly what ``ascend-quant-toolkit build-smoothing`` verified against
the artifact.  The norm side is owned by ``plugin.py``: every admitted
``norm:<module>`` factor is multiplied into that norm weight after loading, so
the activation entering a replayed linear is ``x / s`` and the pair
``(x / s) @ (W * s)`` stays numerically identical to the artifact.

Packed host layers (``qkv_proj``, ``gate_up_proj``) are sliced with shape-only
arithmetic: a factor vector that covers more channels than the local partition
is divided into per-shard TP blocks in the host's checkpoint order, which keeps
the module independent of any vLLM linear class.
"""

from __future__ import annotations

from typing import Any

from vllm_ascend.quantization.methods import (
    AscendW8A8DynamicLinearMethod,
    get_scheme_class,
    register_scheme,
)

from ..attachment import COL_SUFFIX, ROW_SUFFIX, AttachmentError, SmoothingAttachment

RUNTIME_QUANT_TYPE = "ASCEND_QUANT_W8A8_ONLINE"
DENOM = 127.0
EPS = 1e-8
# ModelSlim saturates INT8 asymmetrically: the most negative element lands on
# -128 while the positive side stops at 127.  A symmetric clamp would reject the
# artifact's most negative element and cost one LSB on that channel.
_QUANT_MIN = -128
_QUANT_MAX = 127

_ACTIVE: SmoothingAttachment | None = None


def set_active_attachment(attachment: SmoothingAttachment | None) -> None:
    """Publish the admitted attachment to every scheme instance of this process."""

    global _ACTIVE
    _ACTIVE = attachment


def active_attachment() -> SmoothingAttachment | None:
    return _ACTIVE


def _slice_factors(vector: Any, local_size: int, tp_rank: int, shard_sizes: list[int] | None) -> Any:
    """Slice a full-length factor vector down to one TP partition."""

    import torch

    total = int(vector.numel())
    if total == local_size:
        return vector
    if local_size <= 0 or total % local_size != 0:
        raise AttachmentError(f"cannot slice {total} factors into {local_size} local channels")
    tp_size = total // local_size
    if not 0 <= tp_rank < tp_size:
        raise AttachmentError(f"tp_rank {tp_rank} is outside the admitted tp_size {tp_size}")
    if shard_sizes and sum(shard_sizes) == total and len(shard_sizes) > 1:
        pieces = []
        offset = 0
        for size in shard_sizes:
            if size % tp_size != 0:
                raise AttachmentError(f"packed shard of {size} channels is not divisible by {tp_size}")
            block = size // tp_size
            pieces.append(vector[offset + tp_rank * block : offset + (tp_rank + 1) * block])
            offset += size
        return torch.cat(pieces)
    block = total // tp_size
    return vector[tp_rank * block : (tp_rank + 1) * block]


class AscendQuantW8A8OnlineLinearMethod(AscendW8A8DynamicLinearMethod):
    """Online W8A8 linear scheme with host-owned execution."""

    def get_weight(self, input_size: int, output_size: int, params_dtype: Any) -> dict[str, Any]:
        """Declare a float placeholder so the BF16 checkpoint loads unchanged."""

        import torch

        return {"weight": torch.empty(output_size, input_size, dtype=params_dtype)}

    def get_perchannel_param(self, output_size: int, params_dtype: Any) -> dict[str, Any]:
        import torch

        return {
            "weight_scale": torch.empty(output_size, 1, dtype=params_dtype),
            "weight_offset": torch.empty(output_size, 1, dtype=params_dtype),
        }
    def process_weights_after_loading(self, layer: Any) -> None:
        """Replace the loaded float weight with the replayed INT8 quantization."""

        import torch

        attachment = active_attachment()
        if attachment is None:
            raise AttachmentError("the online W8A8 scheme requires an admitted attachment")
        prefix = str(getattr(layer, "prefix", ""))
        if not prefix:
            raise AttachmentError("the online W8A8 scheme needs a layer prefix to select factors")
        if not attachment.has_prefix(prefix):
            raise AttachmentError(f"the attachment carries no smoothing factors for {prefix}")

        weight = layer.weight.data.to(torch.float32)
        local_out, local_in = (int(size) for size in weight.shape)
        tp_rank = int(getattr(layer, "tp_rank", 0) or 0)
        geometry = attachment.packed_geometry.get(prefix)
        # The attachment is read from a CPU-side file while the checkpoint weight
        # already lives on the accelerator, so every factor has to follow the
        # weight's device before the replay multiplies the two - the host keeps
        # the parameters on the target device for this call.  ``plugin`` applies
        # the same rule to the norm side.
        device = weight.device
        col = attachment.open_tensor(f"{prefix}.{COL_SUFFIX}").to(torch.float32)
        row = attachment.open_tensor(f"{prefix}.{ROW_SUFFIX}").to(torch.float32)
        col = _slice_factors(col, local_in, tp_rank, None).to(device=device)
        row = _slice_factors(row, local_out, tp_rank, geometry).to(device=device)

        scaled = weight * col[None, :] * row[:, None]
        channel_scale = (scaled.abs().amax(dim=1, keepdim=True) / DENOM).clamp_min(EPS)
        quantized = torch.round(scaled / channel_scale).clamp_(_QUANT_MIN, _QUANT_MAX)
        quantized = quantized.to(torch.int8)

        scale_dtype = layer.weight_scale.dtype
        layer.weight = torch.nn.Parameter(quantized, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            channel_scale.to(scale_dtype), requires_grad=False
        )
        layer.weight_offset = torch.nn.Parameter(
            torch.zeros_like(channel_scale, dtype=scale_dtype), requires_grad=False
        )
        # Host-owned tail: transpose(0, 1) -> maybe_trans_nz -> flatten -> fp32.
        super().process_weights_after_loading(layer)


def register_schemes(attachment: SmoothingAttachment | None = None) -> list[str]:
    """Register the online scheme under the extension-owned runtime type."""

    resolved = attachment or active_attachment()
    if resolved is None:
        raise AttachmentError("register_schemes requires an admitted smoothing attachment")
    set_active_attachment(resolved)

    registered: list[str] = []
    linear = get_scheme_class(RUNTIME_QUANT_TYPE, "linear")
    if linear is None:
        register_scheme(RUNTIME_QUANT_TYPE, "linear")(AscendQuantW8A8OnlineLinearMethod)
        registered.append(f"{RUNTIME_QUANT_TYPE}/linear")
    elif linear is not AscendQuantW8A8OnlineLinearMethod:
        raise RuntimeError(f"scheme registration collision: {RUNTIME_QUANT_TYPE}/linear")
    return registered
