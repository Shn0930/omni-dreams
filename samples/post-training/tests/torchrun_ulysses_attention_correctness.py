# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed forward/gradient oracle for Ulysses FlashAttention-3.

Run from ``post-training`` in the CUDA 12.8 environment:

    uv run --extra cu128 torchrun --standalone --nproc-per-node=4 \
      ../samples/post-training/tests/torchrun_ulysses_attention_correctness.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from omnidreams._src.omnidreams.modules.block_causal_flash_attention import (
    block_causal_flash_attention,
    ulysses_block_causal_flash_attention,
)
from omnidreams._src.omnidreams.modules.ulysses_attention import (
    UlyssesCPManager,
    head_to_sequence,
    sequence_to_head,
)


def _local_sequence_shard(x: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    return x.chunk(world_size, dim=1)[rank].contiguous()


def _gather_sequence(x: torch.Tensor, world_size: int) -> torch.Tensor:
    gathered = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x)
    return torch.cat(gathered, dim=1)


def _test_a2a_identity(device: torch.device, rank: int, world_size: int) -> None:
    batch, local_sequence, heads, head_dim = 1, 6, 2 * world_size, 4
    global_sequence = local_sequence * world_size
    global_x = torch.arange(
        batch * global_sequence * heads * head_dim,
        dtype=torch.float32,
        device=device,
    ).reshape(batch, global_sequence, heads, head_dim)
    local_x = _local_sequence_shard(global_x, rank, world_size).clone().requires_grad_(True)

    head_shard = sequence_to_head(local_x, dist.group.WORLD)
    expected = global_x[:, :, rank * 2 : (rank + 1) * 2]
    torch.testing.assert_close(head_shard, expected)

    restored = head_to_sequence(head_shard, dist.group.WORLD)
    torch.testing.assert_close(restored, local_x)
    weights = (
        torch.arange(restored.numel(), dtype=torch.float32, device=device).reshape_as(restored)
        + rank
    )
    (restored * weights).sum().backward()
    torch.testing.assert_close(local_x.grad, weights)


def _test_flash3_oracle(device: torch.device, rank: int, world_size: int) -> None:
    batch, sequence, heads, head_dim = 1, 24 * world_size, 2 * world_size, 64
    tokens_per_block = 8
    generator = torch.Generator(device=device).manual_seed(20260918)
    global_inputs = [
        torch.randn(
            batch,
            sequence,
            heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        for _ in range(3)
    ]
    output_gradient = torch.randn(
        batch,
        sequence,
        heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    local_inputs = [
        _local_sequence_shard(x, rank, world_size).detach().clone().requires_grad_(True)
        for x in global_inputs
    ]

    local_output = ulysses_block_causal_flash_attention(
        *local_inputs,
        tokens_per_block=tokens_per_block,
        cp_manager=UlyssesCPManager(dist.group.WORLD),
    )
    reference_inputs = [x.detach().clone().requires_grad_(True) for x in global_inputs]
    reference_output = block_causal_flash_attention(
        *reference_inputs,
        tokens_per_block=tokens_per_block,
    )
    torch.testing.assert_close(
        _gather_sequence(local_output.detach(), world_size).float(),
        reference_output.float(),
        atol=3e-2,
        rtol=3e-2,
    )

    local_output.backward(_local_sequence_shard(output_gradient, rank, world_size))
    reference_output.backward(output_gradient)
    for actual, reference in zip(local_inputs, reference_inputs, strict=True):
        expected_gradient = _local_sequence_shard(reference.grad, rank, world_size)
        torch.testing.assert_close(
            actual.grad.float(),
            expected_gradient.float(),
            atol=6e-2,
            rtol=6e-2,
        )


def main() -> None:
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise RuntimeError("The Ulysses correctness oracle requires at least two ranks")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)

    try:
        _test_a2a_identity(device, dist.get_rank(), world_size)
        dist.barrier()
        _test_flash3_oracle(device, dist.get_rank(), world_size)
        if dist.get_rank() == 0:
            print(f"PASS: Ulysses CP={world_size} forward and gradients", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
