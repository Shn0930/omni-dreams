# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""FlashAttention-3 implementations of block-causal training attention."""

from typing import Literal

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor, Partial, Shard
from torch.distributed.tensor.device_mesh import DeviceMesh

from omnidreams._src.imaginaire.attention.flash3 import FLASH3_SUPPORTED, flash3_attention
from omnidreams._src.imaginaire.utils.context_parallel import restore_zigzag_sequence_order
from omnidreams._src.omnidreams.modules.ulysses_attention import (
    head_to_sequence,
    sequence_to_head_qkv,
)


def _validate_flash3_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    tokens_per_block: int,
) -> None:
    if not FLASH3_SUPPORTED:
        raise RuntimeError(
            "FlashAttention-3 training was requested, but flash-attn-3-nv is unavailable. "
            "Install the CUDA/PyTorch-matched FA3 environment."
        )
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must use [batch, sequence, heads, head_dim] layout")
    if key.shape != value.shape:
        raise ValueError(f"key and value shapes must match, got {key.shape} and {value.shape}")
    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError(
            f"query, key, and value batch/head dimensions must match, got {query.shape}, {key.shape}, and {value.shape}"
        )
    if query.device != key.device or query.device != value.device:
        raise ValueError(f"query, key, and value devices must match, got {query.device}, {key.device}, {value.device}")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError(f"query, key, and value dtypes must match, got {query.dtype}, {key.dtype}, {value.dtype}")
    if tokens_per_block <= 0:
        raise ValueError(f"tokens_per_block must be positive, got {tokens_per_block}")
    if query.device.type != "cuda":
        raise ValueError(f"FlashAttention-3 requires CUDA tensors, got {query.device}")
    if query.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError(f"FlashAttention-3 training requires float16 or bfloat16, got {query.dtype}")
    if torch.cuda.get_device_capability(query.device) != (9, 0):
        raise RuntimeError(
            "The flash-attn-3-nv backend currently requires a Hopper GPU "
            f"(compute capability 9.0), got {torch.cuda.get_device_capability(query.device)}"
        )


def block_causal_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    tokens_per_block: int,
) -> torch.Tensor:
    """Apply full attention within a block and causal attention across blocks.

    The inputs use ``[batch, sequence, heads, head_dim]`` layout.  Each query
    block attends to the key/value prefix ending at that block.  Prefixes are
    tensor views, so this expresses the same mask as OmniDreams' non-interleaved
    FlexAttention path without materializing duplicated K/V tensors.

    This helper is process-group agnostic: it supports a full sequence with
    either all heads (CP=1) or a local head shard produced by Ulysses CP.
    Interleaved layouts are rejected by the caller.
    """
    _validate_flash3_inputs(query, key, value, tokens_per_block=tokens_per_block)
    if query.shape != key.shape:
        raise ValueError(f"query, key, and value shapes must match, got {query.shape}, {key.shape}, {value.shape}")

    sequence_length = query.shape[1]
    outputs = []
    for block_start in range(0, sequence_length, tokens_per_block):
        block_end = min(block_start + tokens_per_block, sequence_length)
        outputs.append(
            flash3_attention(
                query=query[:, block_start:block_end],
                key=key[:, :block_end],
                value=value[:, :block_end],
                is_causal=False,
            )
        )

    return torch.cat(outputs, dim=1)


def _all_gather_sequence_with_grad(x: torch.Tensor, process_group: ProcessGroup) -> torch.Tensor:
    """Differentiably gather a rank-local sequence shard."""
    device_mesh = DeviceMesh.from_group(process_group, device_type=x.device.type)
    local = DTensor.from_local(x, device_mesh, [Shard(1)])
    return local.full_tensor(grad_placements=[Partial()])


def query_sharded_block_causal_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    tokens_per_block: int,
    process_group: ProcessGroup,
    cp_layout: Literal["contiguous", "zigzag"],
) -> torch.Tensor:
    """Run exact FA3 for sequence-sharded queries and replicated full K/V.

    ``contiguous`` is the unbalanced CP baseline.  ``zigzag`` assigns each rank
    a symmetric pair of the ``2 * CP`` query micro-chunks.  K/V are gathered
    differentiably and restored to chronological order before each local query
    segment is evaluated against its legal block prefix.
    """
    if process_group is None:
        raise ValueError("Query-sharded block-causal attention requires a context-parallel process group")
    if cp_layout not in {"contiguous", "zigzag"}:
        raise ValueError(f"Unknown CP layout {cp_layout!r}; expected 'contiguous' or 'zigzag'")
    _validate_flash3_inputs(query, key, value, tokens_per_block=tokens_per_block)
    if query.shape != key.shape:
        raise ValueError(f"Rank-local query, key, and value shapes must match, got {query.shape} and {key.shape}")

    cp_size = dist.get_world_size(process_group)
    cp_rank = dist.get_rank(process_group)
    local_sequence_length = query.shape[1]
    global_sequence_length = local_sequence_length * cp_size

    with torch.autograd.profiler.record_function("query_sharded_kv_allgather"):
        full_key = _all_gather_sequence_with_grad(key, process_group)
        full_value = _all_gather_sequence_with_grad(value, process_group)
        if cp_layout == "zigzag":
            full_key = restore_zigzag_sequence_order(full_key, seq_dim=1, cp_size=cp_size)
            full_value = restore_zigzag_sequence_order(full_value, seq_dim=1, cp_size=cp_size)

    if cp_layout == "contiguous":
        local_chunks = ((query, cp_rank * local_sequence_length),)
    else:
        if local_sequence_length % 2 != 0:
            raise ValueError(
                "Zigzag CP requires two equal rank-local query chunks, "
                f"got local sequence length {local_sequence_length}"
            )
        micro_chunk_size = local_sequence_length // 2
        chunk_ids = (cp_rank, 2 * cp_size - cp_rank - 1)
        local_chunks = (
            (
                query[:, local_chunk_idx * micro_chunk_size : (local_chunk_idx + 1) * micro_chunk_size],
                global_chunk_idx * micro_chunk_size,
            )
            for local_chunk_idx, global_chunk_idx in enumerate(chunk_ids)
        )

    outputs = []
    with torch.autograd.profiler.record_function(f"{cp_layout}_query_sharded_fa3"):
        for local_query_chunk, global_chunk_start in local_chunks:
            local_cursor = 0
            global_cursor = global_chunk_start
            global_chunk_end = global_chunk_start + local_query_chunk.shape[1]
            while global_cursor < global_chunk_end:
                block_end = min(
                    ((global_cursor // tokens_per_block) + 1) * tokens_per_block,
                    global_sequence_length,
                )
                segment_end = min(block_end, global_chunk_end)
                segment_length = segment_end - global_cursor
                outputs.append(
                    flash3_attention(
                        query=local_query_chunk[:, local_cursor : local_cursor + segment_length],
                        key=full_key[:, :block_end],
                        value=full_value[:, :block_end],
                        is_causal=False,
                    )
                )
                local_cursor += segment_length
                global_cursor = segment_end

    return torch.cat(outputs, dim=1)


def ulysses_block_causal_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    tokens_per_block: int,
    process_group: ProcessGroup,
) -> torch.Tensor:
    """Run exact block-causal FA3 with Ulysses sequence/head redistribution.

    Inputs and output use the rank-local sequence layout
    ``[B, S / CP, H, D]``.  The two all-to-all phases temporarily transform it
    into ``[B, S, H / CP, D]`` so every rank evaluates the complete causal
    sequence for an equal subset of heads.
    """
    if process_group is None:
        raise ValueError("Ulysses block-causal attention requires a context-parallel process group")

    full_query, full_key, full_value = sequence_to_head_qkv(query, key, value, process_group)
    full_output = block_causal_flash_attention(
        full_query,
        full_key,
        full_value,
        tokens_per_block=tokens_per_block,
    )
    return head_to_sequence(full_output, process_group)
