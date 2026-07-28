# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Autograd-safe sequence/head all-to-all transforms for Ulysses CP."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


def _validate_4d_tensor(x: torch.Tensor) -> None:
    if x.ndim != 4:
        raise ValueError(f"Ulysses CP expects [batch, sequence, heads, head_dim], got shape {tuple(x.shape)}")


def _sequence_to_head_impl(x: torch.Tensor, process_group: ProcessGroup) -> torch.Tensor:
    """[B, S/P, H, D] -> [B, S, H/P, D]."""
    _validate_4d_tensor(x)
    world_size = dist.get_world_size(process_group)
    batch_size, local_sequence_length, num_heads, head_dim = x.shape
    if num_heads % world_size != 0:
        raise ValueError(f"Ulysses CP requires num_heads ({num_heads}) to be divisible by cp_size ({world_size})")
    local_heads = num_heads // world_size

    # The leading dimension is the destination rank (head shard).
    send = (
        x.reshape(batch_size, local_sequence_length, world_size, local_heads, head_dim)
        .permute(2, 0, 1, 3, 4)
        .contiguous()
    )
    recv = torch.empty_like(send)
    with torch.autograd.profiler.record_function("ulysses_sequence_to_head_a2a"):
        dist.all_to_all_single(recv, send, group=process_group)

    # The leading receive dimension is the source rank (sequence shard).
    return recv.permute(1, 0, 2, 3, 4).reshape(
        batch_size,
        local_sequence_length * world_size,
        local_heads,
        head_dim,
    )


def _head_to_sequence_impl(x: torch.Tensor, process_group: ProcessGroup) -> torch.Tensor:
    """[B, S, H/P, D] -> [B, S/P, H, D]."""
    _validate_4d_tensor(x)
    world_size = dist.get_world_size(process_group)
    batch_size, sequence_length, local_heads, head_dim = x.shape
    if sequence_length % world_size != 0:
        raise ValueError(
            f"Ulysses CP requires sequence length ({sequence_length}) to be divisible by cp_size ({world_size})"
        )
    local_sequence_length = sequence_length // world_size

    # The leading dimension is the destination rank (sequence shard).
    send = (
        x.reshape(batch_size, world_size, local_sequence_length, local_heads, head_dim)
        .permute(1, 0, 2, 3, 4)
        .contiguous()
    )
    recv = torch.empty_like(send)
    with torch.autograd.profiler.record_function("ulysses_head_to_sequence_a2a"):
        dist.all_to_all_single(recv, send, group=process_group)

    # The leading receive dimension is the source rank (head shard).
    return recv.permute(1, 2, 0, 3, 4).reshape(
        batch_size,
        local_sequence_length,
        local_heads * world_size,
        head_dim,
    )


class _SequenceToHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, process_group: ProcessGroup) -> torch.Tensor:
        ctx.process_group = process_group
        return _sequence_to_head_impl(x, process_group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return _head_to_sequence_impl(grad_output.contiguous(), ctx.process_group), None


class _HeadToSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, process_group: ProcessGroup) -> torch.Tensor:
        ctx.process_group = process_group
        return _head_to_sequence_impl(x, process_group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return _sequence_to_head_impl(grad_output.contiguous(), ctx.process_group), None


def sequence_to_head(x: torch.Tensor, process_group: ProcessGroup) -> torch.Tensor:
    """Gather sequence shards while scattering attention heads."""
    if dist.get_world_size(process_group) == 1:
        return x
    return _SequenceToHead.apply(x, process_group)


def sequence_to_head_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    process_group: ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather sequence/scatter heads for Q/K/V with one packed all-to-all."""
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError(
            "Packed Ulysses Q/K/V all-to-all requires matching shapes, "
            f"got {query.shape}, {key.shape}, and {value.shape}"
        )
    packed = torch.cat((query, key, value), dim=0)
    full_packed = sequence_to_head(packed, process_group)
    return full_packed.chunk(3, dim=0)


def head_to_sequence(x: torch.Tensor, process_group: ProcessGroup) -> torch.Tensor:
    """Restore local sequence shards while gathering attention heads."""
    if dist.get_world_size(process_group) == 1:
        return x
    return _HeadToSequence.apply(x, process_group)
