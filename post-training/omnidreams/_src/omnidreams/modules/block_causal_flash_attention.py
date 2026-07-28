# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""FlashAttention-3 implementation of CP=1 block-causal attention."""

import torch

from omnidreams._src.imaginaire.attention.flash3 import FLASH3_SUPPORTED, flash3_attention


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

    This helper intentionally implements only the CP=1, non-interleaved mask.
    Context-parallel and interleaved layouts require a separate distributed
    implementation and are rejected by the caller.
    """
    if not FLASH3_SUPPORTED:
        raise RuntimeError(
            "FlashAttention-3 training was requested, but flash-attn-3-nv is unavailable. "
            "Install the CUDA/PyTorch-matched FA3 environment."
        )
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
    if query.device.type != "cuda":
        raise ValueError(f"FlashAttention-3 requires CUDA tensors, got {query.device}")
    if query.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError(f"FlashAttention-3 training requires float16 or bfloat16, got {query.dtype}")
    if torch.cuda.get_device_capability(query.device) != (9, 0):
        raise RuntimeError(
            "The flash-attn-3-nv backend currently requires a Hopper GPU "
            f"(compute capability 9.0), got {torch.cuda.get_device_capability(query.device)}"
        )

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
