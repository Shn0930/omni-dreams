# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Avoid spatially redundant AdaLN-LoRA GEMMs in CP=1 causal training.

``CosmosCausalDiT`` builds one timestep embedding per latent frame and then
repeats it over every spatial token before entering the transformer blocks.
The three per-block AdaLN-LoRA MLPs therefore receive thousands of identical
rows per frame.  A pointwise MLP commutes with that repeat:

    mlp(repeat_interleave(x, S)) == repeat_interleave(mlp(x), S)

This sample-side patch evaluates each MLP on the per-frame rows and expands
its result afterwards.  It is mathematically equivalent and preserves
parameter names and checkpoint layout.  Reduced-precision GEMMs can differ
slightly because their gradient reduction order changes.  The optimization is
deliberately limited to CP=1 because non-contiguous CP layouts do not, in
general, keep a complete frame on one rank.
"""

from __future__ import annotations

from typing import Any

from omnidreams._src.omnidreams.networks import causal_cosmos
from torch import nn

_ORIGINAL_BLOCK_INIT = causal_cosmos.CausalCosmosBlock.__init__
_ORIGINAL_BLOCK_FORWARD = causal_cosmos.CausalCosmosBlock.forward
_MODULATION_NAMES = (
    "adaln_modulation_self_attn",
    "adaln_modulation_cross_attn",
    "adaln_modulation_mlp",
)
_INSTALLED = False


class SpatiallyRepeatedSequential(nn.Sequential):
    """Run a pointwise sequence on one row per frame, then spatially repeat."""

    spatial_repeat_factor: int = 1

    def set_spatial_repeat_factor(self, repeat_factor: int) -> None:
        if repeat_factor < 1:
            raise ValueError(f"repeat_factor must be positive, got {repeat_factor}")
        self.spatial_repeat_factor = repeat_factor

    def forward(self, input):  # noqa: ANN001, ANN201 - match torch.nn.Sequential
        repeat_factor = self.spatial_repeat_factor
        if repeat_factor == 1:
            return super().forward(input)
        if input.ndim != 3:
            raise ValueError(
                "spatially repeated AdaLN input must have shape [B, L, D], "
                f"got {tuple(input.shape)}"
            )
        if input.shape[1] % repeat_factor != 0:
            raise ValueError(
                f"sequence length {input.shape[1]} is not divisible by spatial "
                f"repeat factor {repeat_factor}"
            )

        per_frame_input = input[:, ::repeat_factor, :]
        per_frame_output = super().forward(per_frame_input)
        return per_frame_output.repeat_interleave(repeat_factor, dim=1)


def _replace_modulation_sequences(block: nn.Module) -> None:
    for name in _MODULATION_NAMES:
        module = getattr(block, name)
        if isinstance(module, SpatiallyRepeatedSequential):
            continue
        if not isinstance(module, nn.Sequential):
            raise TypeError(f"{name} must be nn.Sequential, got {type(module)!r}")
        unsupported_children = [
            type(child).__name__
            for child in module.children()
            if not isinstance(child, nn.SiLU | nn.Linear)
        ]
        if unsupported_children:
            raise TypeError(
                f"{name} contains operations not validated for repeat elision: "
                f"{unsupported_children}"
            )
        # Keep children at the same hierarchy level so state-dict keys remain
        # e.g. ``adaln_modulation_self_attn.1.weight``.
        setattr(block, name, SpatiallyRepeatedSequential(*module.children()))


def _optimized_block_init(self, *args: Any, **kwargs: Any) -> None:
    _ORIGINAL_BLOCK_INIT(self, *args, **kwargs)
    _replace_modulation_sequences(self)


def _extract_video_size(args: tuple[Any, ...], kwargs: dict[str, Any]):
    if "video_size" in kwargs:
        return kwargs["video_size"]
    # CausalCosmosBlock.forward has 14 positional arguments after ``self``;
    # ``video_size`` is the last one.
    return args[13] if len(args) > 13 else None


def _extract_inference_range(args: tuple[Any, ...], kwargs: dict[str, Any]):
    kv_cache = kwargs.get("kv_cache", args[7] if len(args) > 7 else None)
    current_start = kwargs.get("current_start", args[9] if len(args) > 9 else 0)
    current_end = kwargs.get("current_end", args[10] if len(args) > 10 else 0)
    return kv_cache, int(current_start), int(current_end)


def _validate_frame_alignment(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    repeat_factor: int,
) -> None:
    kv_cache, current_start, current_end = _extract_inference_range(args, kwargs)
    if kv_cache is None:
        return
    if current_start % repeat_factor != 0 or current_end % repeat_factor != 0:
        raise ValueError(
            "repeated-AdaLN optimization requires KV-cache chunks to start and "
            "end on latent-frame boundaries: "
            f"current_start={current_start}, current_end={current_end}, "
            f"tokens_per_frame={repeat_factor}"
        )


def _optimized_block_forward(self, *args: Any, **kwargs: Any):
    video_size = _extract_video_size(args, kwargs)
    cp_size = getattr(self, "cp_size", None)
    if video_size is not None and cp_size in (None, 1):
        repeat_factor = int(video_size.H) * int(video_size.W)
        _validate_frame_alignment(args, kwargs, repeat_factor)
    else:
        repeat_factor = 1

    for name in _MODULATION_NAMES:
        module = getattr(self, name)
        if not isinstance(module, SpatiallyRepeatedSequential):
            raise RuntimeError(
                f"{name} was not prepared for repeated-AdaLN optimization; "
                "install the optimization before constructing the model"
            )
        module.set_spatial_repeat_factor(repeat_factor)
    return _ORIGINAL_BLOCK_FORWARD(self, *args, **kwargs)


def install_repeated_adaln_optimization() -> None:
    """Install the CP=1 transform before constructing any causal blocks."""
    global _INSTALLED
    if _INSTALLED:
        return
    causal_cosmos.CausalCosmosBlock.__init__ = _optimized_block_init
    causal_cosmos.CausalCosmosBlock.forward = _optimized_block_forward
    _INSTALLED = True
