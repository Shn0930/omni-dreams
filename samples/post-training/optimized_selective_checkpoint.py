# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Sample-side selective-checkpoint policy for the Predict2 2B MLP.

The release ``predict2_2b_720_aggressive`` policy saves attention outputs and
recomputes every matrix multiplication.  In the current 2B block, the gated
MLP residual needs the second linear layer's output during backward, so the
checkpoint replay runs both MLP GEMMs.  Saving only the 8192 -> 2048 FC2 output
removes that replay GEMM while retaining far less activation memory than
saving the 2048 -> 8192 FC1 output.

This is deliberately a sample-side installer: the release tree remains
unchanged and the optimization is enabled only through an explicit launcher
environment variable.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

_MLP_HIDDEN_DIM = 8192
_MODEL_DIM = 2048


def _is_predict2_2b_mlp_fc2(
    function: object,
    arguments: Sequence[object],
) -> bool:
    """Return whether an ATen call is the bias-free 8192 -> 2048 MLP FC2.

    The leading dimension is intentionally unconstrained so this policy keeps
    working with different batch sizes, sequence lengths, and CP degrees.
    """

    if function != torch.ops.aten.mm.default or len(arguments) < 2:
        return False
    lhs, rhs = arguments[:2]
    if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor):
        return False
    return (
        lhs.ndim == 2
        and rhs.ndim == 2
        and lhs.shape[1] == _MLP_HIDDEN_DIM
        and rhs.shape[0] == _MLP_HIDDEN_DIM
        and rhs.shape[1] == _MODEL_DIM
        and lhs.dtype == rhs.dtype
        and lhs.device == rhs.device
    )


def predict2_2b_720_context_fn_save_mlp_fc2():
    """Save attention and main-MLP FC2 outputs; recompute everything else."""

    def policy_fn(ctx, function, *args, **kwargs):
        del ctx, kwargs
        # Preserve the release aggressive policy's attention MUST_SAVE rule.
        if "flash_attn" in str(function):
            return CheckpointPolicy.MUST_SAVE
        if _is_predict2_2b_mlp_fc2(function, args):
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy_fn)


def install_mlp_fc2_selective_checkpoint() -> None:
    """Replace only the 2B aggressive policy factory before model creation."""

    from omnidreams._src.predict2.networks import minimal_v4_dit

    minimal_v4_dit.predict2_2B_720_context_fn_aggressive = (
        predict2_2b_720_context_fn_save_mlp_fc2
    )
