# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Sample-side optimized autograd for FA3 block-causal prefix attention.

The release implementation intentionally composes ordinary tensor slices,
FlashAttention calls, and ``torch.cat``.  Its generic autograd graph expands
every prefix gradient to the full sequence before accumulation.  This module
keeps the same FA3 kernels and mask semantics, but accumulates the ragged
prefix gradients directly.
"""

from __future__ import annotations

from importlib.metadata import version

import torch
from flash_attn_3_nv.flash_attn_interface import _flash_attn_backward, flash_attn_func
from omnidreams._src.omnidreams.modules.block_causal_flash_attention import (
    _validate_flash3_inputs,
)
from torch.autograd.function import once_differentiable

_SUPPORTED_FA3_VERSION_PREFIX = "1.0.3"


def _validate_private_fa3_abi() -> None:
    """Fail closed when the private FA3 backward ABI has not been validated."""
    installed_version = version("flash-attn-3-nv")
    if not installed_version.startswith(_SUPPORTED_FA3_VERSION_PREFIX):
        raise RuntimeError(
            "The optimized prefix-gradient path uses the private "
            "flash-attn-3-nv _flash_attn_backward ABI and has only been "
            f"validated with {_SUPPORTED_FA3_VERSION_PREFIX}.x; got {installed_version!r}."
        )


class _BlockCausalPrefixAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        tokens_per_block: int,
    ) -> torch.Tensor:
        scale = query.shape[-1] ** -0.5
        sequence_length = query.shape[1]
        outputs: list[torch.Tensor] = []
        softmax_lses: list[torch.Tensor] = []

        for block_start in range(0, sequence_length, tokens_per_block):
            block_end = min(block_start + tokens_per_block, sequence_length)
            output, softmax_lse = flash_attn_func(
                query[:, block_start:block_end],
                key[:, :block_end],
                value[:, :block_end],
                softmax_scale=scale,
                causal=False,
                return_attn_probs=True,
            )
            outputs.append(output)
            softmax_lses.append(softmax_lse)

        ctx.tokens_per_block = tokens_per_block
        ctx.scale = scale
        ctx.num_blocks = len(outputs)
        ctx.save_for_backward(query, key, value, *outputs, *softmax_lses)
        return torch.cat(outputs, dim=1)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        saved = ctx.saved_tensors
        query, key, value = saved[:3]
        outputs = saved[3 : 3 + ctx.num_blocks]
        softmax_lses = saved[3 + ctx.num_blocks :]

        grad_query = torch.empty_like(query, memory_format=torch.contiguous_format)
        # The last block covers the full K/V sequence, so its FA3 gradients can
        # initialize these buffers directly.  Earlier, shorter prefixes are
        # accumulated in place without full-sequence zero/copy tensors.
        grad_key = torch.empty_like(key, memory_format=torch.contiguous_format)
        grad_value = torch.empty_like(value, memory_format=torch.contiguous_format)

        sequence_length = query.shape[1]
        block_starts = list(range(0, sequence_length, ctx.tokens_per_block))
        for block_index in range(ctx.num_blocks - 1, -1, -1):
            block_start = block_starts[block_index]
            block_end = min(block_start + ctx.tokens_per_block, sequence_length)

            query_slice = query[:, block_start:block_end]
            key_prefix = key[:, :block_end]
            value_prefix = value[:, :block_end]
            grad_output_slice = grad_output[:, block_start:block_end]

            grad_query_view = grad_query[:, block_start:block_end]
            if grad_query_view.is_contiguous():
                grad_query_slice = grad_query_view
            else:
                grad_query_slice = torch.empty(
                    query_slice.shape,
                    dtype=query.dtype,
                    device=query.device,
                )

            initializes_full_kv = block_index == ctx.num_blocks - 1 and block_end == key.shape[1]
            if initializes_full_kv:
                grad_key_prefix = grad_key
                grad_value_prefix = grad_value
            else:
                grad_key_prefix = torch.empty(
                    key_prefix.shape,
                    dtype=key.dtype,
                    device=key.device,
                )
                grad_value_prefix = torch.empty(
                    value_prefix.shape,
                    dtype=value.dtype,
                    device=value.device,
                )

            _flash_attn_backward(
                grad_output_slice,
                query_slice,
                key_prefix,
                value_prefix,
                outputs[block_index],
                softmax_lses[block_index],
                dq=grad_query_slice,
                dk=grad_key_prefix,
                dv=grad_value_prefix,
                softmax_scale=ctx.scale,
                is_causal=False,
                window_size_left=-1,
                window_size_right=-1,
                softcap=0.0,
                deterministic=False,
                sm_margin=0,
            )

            if grad_query_slice is not grad_query_view:
                grad_query_view.copy_(grad_query_slice)
            if not initializes_full_kv:
                grad_key[:, :block_end].add_(grad_key_prefix)
                grad_value[:, :block_end].add_(grad_value_prefix)

        return grad_query, grad_key, grad_value, None


def optimized_block_causal_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    tokens_per_block: int,
) -> torch.Tensor:
    """Run exact block-causal FA3 with direct ragged-prefix gradient accumulation."""
    _validate_private_fa3_abi()
    _validate_flash3_inputs(query, key, value, tokens_per_block=tokens_per_block)
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError(
            "query, key, and value shapes must match, got "
            f"{query.shape}, {key.shape}, {value.shape}"
        )
    return _BlockCausalPrefixAttention.apply(query, key, value, tokens_per_block)


def install_optimized_block_causal_flash_attention() -> None:
    """Patch only the CP=1 training call site, leaving CP backends untouched."""
    from omnidreams._src.omnidreams.networks import causal_cosmos

    _validate_private_fa3_abi()
    causal_cosmos.block_causal_flash_attention = optimized_block_causal_flash_attention
