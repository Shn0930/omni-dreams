# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""FlashAttention-3 implementations of block-causal training attention."""

import torch

from omnidreams._src.imaginaire.attention.flash3 import FLASH3_SUPPORTED, flash3_attention
from omnidreams._src.omnidreams.modules.ulysses_attention import UlyssesCPManager


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

    This helper is process-group agnostic: it accepts either every attention
    head for CP=1 or a local head shard produced by Ulysses CP.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must use [batch, sequence, heads, head_dim] layout")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError(f"query, key, and value shapes must match, got {query.shape}, {key.shape}, {value.shape}")
    if query.device != key.device or query.device != value.device:
        raise ValueError(f"query, key, and value devices must match, got {query.device}, {key.device}, {value.device}")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError(f"query, key, and value dtypes must match, got {query.dtype}, {key.dtype}, {value.dtype}")
    if tokens_per_block <= 0:
        raise ValueError(f"tokens_per_block must be positive, got {tokens_per_block}")
    if query.shape[1] == 0:
        raise ValueError("FlashAttention-3 block-causal attention requires a non-empty sequence")
    if query.device.type != "cuda":
        raise ValueError(f"FlashAttention-3 requires CUDA tensors, got {query.device}")
    if query.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError(f"FlashAttention-3 training requires float16 or bfloat16, got {query.dtype}")
    if not FLASH3_SUPPORTED:
        raise RuntimeError(
            "FlashAttention-3 training was requested, but flash-attn-3-nv is unavailable. "
            "Install the CUDA/PyTorch-matched FA3 environment."
        )
    device_capability = torch.cuda.get_device_capability(query.device)
    if device_capability != (9, 0):
        raise RuntimeError(
            "The flash-attn-3-nv backend currently requires a Hopper GPU "
            f"(compute capability 9.0), got {device_capability}"
        )

    sequence_length = query.shape[1]
    outputs = []
    for block_start in range(0, sequence_length, tokens_per_block):
        block_end = min(block_start + tokens_per_block, sequence_length)
        output = flash3_attention(
            query=query[:, block_start:block_end],
            key=key[:, :block_end],
            value=value[:, :block_end],
            is_causal=False,
        )
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("FlashAttention-3 unexpectedly returned logsumexp output")
        outputs.append(output)

    return torch.cat(outputs, dim=1)


def ulysses_block_causal_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    tokens_per_block: int,
    cp_manager: UlyssesCPManager,
) -> torch.Tensor:
    """Run exact block-causal FA3 through a CP1/CP>1 Ulysses adapter.

    With CP>1, inputs and output use the rank-local sequence layout
    ``[B, S / CP, H, D]``. Packed Q/K/V all-to-all temporarily transforms it
    into ``[B, S, H / CP, D]`` so every rank evaluates the full causal
    sequence for an equal subset of heads. CP1 has no process group and takes
    the same API's communication-free identity path.
    """
    if not cp_manager.is_distributed:
        return block_causal_flash_attention(
            query,
            key,
            value,
            tokens_per_block=tokens_per_block,
        )

    full_query, full_key, full_value = cp_manager.sequence_to_head_qkv(query, key, value)
    full_output = block_causal_flash_attention(
        full_query,
        full_key,
        full_value,
        tokens_per_block=tokens_per_block,
    )
    return cp_manager.head_to_sequence(full_output)
