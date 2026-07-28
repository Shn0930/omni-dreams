# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Four-rank forward/gradient oracle for CP attention layouts.

Run from ``post-training`` in the FA3 environment:

    torchrun --standalone --nproc-per-node=4 \
      ../samples/post-training/tests/torchrun_cp_attention_correctness.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from omnidreams._src.imaginaire.utils.context_parallel import restore_zigzag_sequence_order
from omnidreams._src.omnidreams.modules.block_causal_flash_attention import (
    block_causal_flash_attention,
    query_sharded_block_causal_flash_attention,
    ulysses_block_causal_flash_attention,
)
from omnidreams._src.omnidreams.modules.ulysses_attention import head_to_sequence, sequence_to_head


def _local_sequence_shard(x: torch.Tensor, rank: int, world_size: int, layout: str) -> torch.Tensor:
    if layout == "contiguous":
        return x.chunk(world_size, dim=1)[rank].contiguous()
    num_chunks = 2 * world_size
    chunk_size = x.shape[1] // num_chunks
    chunks = x.reshape(x.shape[0], num_chunks, chunk_size, *x.shape[2:])
    index = torch.tensor([rank, num_chunks - rank - 1], device=x.device)
    return (
        chunks.index_select(1, index).reshape(x.shape[0], 2 * chunk_size, *x.shape[2:]).contiguous()
    )


def _gather_output(x: torch.Tensor, world_size: int, layout: str) -> torch.Tensor:
    gathered = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x)
    output = torch.cat(gathered, dim=1)
    if layout == "zigzag":
        output = restore_zigzag_sequence_order(output, seq_dim=1, cp_size=world_size)
    return output


def _test_a2a_identity(device: torch.device, rank: int, world_size: int) -> None:
    batch, local_sequence, heads, head_dim = 1, 6, 8, 4
    global_sequence = local_sequence * world_size
    global_x = torch.arange(
        batch * global_sequence * heads * head_dim,
        dtype=torch.float32,
        device=device,
    ).reshape(batch, global_sequence, heads, head_dim)
    local_x = global_x.chunk(world_size, dim=1)[rank].clone().requires_grad_(True)

    head_shard = sequence_to_head(local_x, dist.group.WORLD)
    expected_head_shard = global_x[
        :, :, rank * (heads // world_size) : (rank + 1) * (heads // world_size)
    ]
    torch.testing.assert_close(head_shard, expected_head_shard)

    restored = head_to_sequence(head_shard, dist.group.WORLD)
    torch.testing.assert_close(restored, local_x)
    weights = (
        torch.arange(restored.numel(), dtype=torch.float32, device=device).reshape_as(restored)
        + rank
    )
    (restored * weights).sum().backward()
    torch.testing.assert_close(local_x.grad, weights)


def _test_distributed_flash3(
    device: torch.device,
    rank: int,
    world_size: int,
    strategy: str,
) -> None:
    batch, sequence, heads, head_dim = 1, 96, 8, 64
    tokens_per_block = 8
    generator = torch.Generator(device=device).manual_seed(20260728)
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
    layout = "zigzag" if strategy == "zigzag" else "contiguous"
    local_inputs = [
        _local_sequence_shard(x, rank, world_size, layout).detach().clone().requires_grad_(True)
        for x in global_inputs
    ]

    if strategy == "ulysses":
        local_output = ulysses_block_causal_flash_attention(
            *local_inputs,
            tokens_per_block=tokens_per_block,
            process_group=dist.group.WORLD,
        )
    else:
        local_output = query_sharded_block_causal_flash_attention(
            *local_inputs,
            tokens_per_block=tokens_per_block,
            process_group=dist.group.WORLD,
            cp_layout=layout,
        )

    reference_inputs = [x.detach().clone().requires_grad_(True) for x in global_inputs]
    reference_output = block_causal_flash_attention(
        *reference_inputs,
        tokens_per_block=tokens_per_block,
    )
    gathered_output = _gather_output(local_output.detach(), world_size, layout)
    torch.testing.assert_close(
        gathered_output.float(), reference_output.float(), atol=3e-2, rtol=3e-2
    )

    local_output_gradient = _local_sequence_shard(output_gradient, rank, world_size, layout)
    local_output.backward(local_output_gradient)
    reference_output.backward(output_gradient)
    for actual, reference in zip(local_inputs, reference_inputs, strict=True):
        expected_local_gradient = _local_sequence_shard(reference.grad, rank, world_size, layout)
        torch.testing.assert_close(
            actual.grad.float(),
            expected_local_gradient.float(),
            atol=6e-2,
            rtol=6e-2,
        )


def main() -> None:
    if int(os.environ["WORLD_SIZE"]) != 4:
        raise RuntimeError("This correctness oracle requires exactly four ranks")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)

    try:
        _test_a2a_identity(device, dist.get_rank(), dist.get_world_size())
        for strategy in ("contiguous", "zigzag", "ulysses"):
            dist.barrier()
            _test_distributed_flash3(device, dist.get_rank(), dist.get_world_size(), strategy)
            if dist.get_rank() == 0:
                print(f"PASS: {strategy} CP=4 forward and gradients", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
