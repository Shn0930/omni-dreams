# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed forward/gradient oracle for Ulysses attention backends.

Run from ``post-training`` in the CUDA 12.8 environment:

    CP_SIZE=${CP_SIZE:-2}
    uv run --extra cu128 torchrun --standalone --nproc-per-node="$CP_SIZE" \
      ../samples/post-training/tests/torchrun_ulysses_attention_correctness.py

Set ``ATTENTION_BACKEND=flex`` to test FlexAttention. Add
``BENCHMARK_FLEX_CP=1`` to compare legacy and Ulysses CP after correctness.
"""

from __future__ import annotations

import math
import os
import statistics
from copy import deepcopy

import torch
import torch.distributed as dist
from omnidreams._src.omnidreams.modules.attention_backend import (
    FLASH_ATTENTION_BACKENDS,
    TRAINING_ATTENTION_BACKENDS,
)
from omnidreams._src.omnidreams.modules.block_causal_flash_attention import (
    block_causal_flash_attention,
    ulysses_block_causal_flash_attention,
)
from omnidreams._src.omnidreams.modules.flex_attention import (
    flex_attention_cp,
    ulysses_flex_attention,
)
from omnidreams._src.omnidreams.modules.framewise_adaln import (
    apply_adaln_modulation,
    make_token_frame_indices,
)
from omnidreams._src.omnidreams.modules.ulysses_attention import (
    UlyssesCPManager,
    head_to_sequence,
    sequence_to_head,
)
from omnidreams._src.omnidreams.networks.causal_cosmos import CosmosCausalDiT
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

compiled_flex_attention = torch.compile(flex_attention, dynamic=False)


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


def _test_framewise_adaln_oracle(
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    """Validate compact AdaLN when CP shards start and end inside frames."""
    batch, num_frames, embedding_dim = 2, world_size + 1, 16
    tokens_per_frame = world_size
    generator = torch.Generator(device=device).manual_seed(20260921)
    frame_embedding = torch.randn(
        batch,
        num_frames,
        embedding_dim,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    frame_lora = torch.randn(
        batch,
        num_frames,
        3 * embedding_dim,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    torch.manual_seed(20260921)
    module = torch.nn.Sequential(
        torch.nn.SiLU(),
        torch.nn.Linear(embedding_dim, embedding_dim, bias=False),
        torch.nn.Linear(embedding_dim, 3 * embedding_dim, bias=False),
    ).to(device)
    reference_module = deepcopy(module)
    reference_embedding = frame_embedding.detach().clone().requires_grad_(True)
    reference_lora = frame_lora.detach().clone().requires_grad_(True)

    global_indices = make_token_frame_indices(
        num_frames,
        tokens_per_frame,
        device=device,
    )
    local_indices = _local_sequence_shard(
        global_indices.unsqueeze(0),
        rank,
        world_size,
    ).squeeze(0)
    local_output = apply_adaln_modulation(
        module,
        frame_embedding,
        frame_lora,
        token_frame_indices=local_indices,
        sequence_length=local_indices.numel(),
    )
    reference_output = reference_module(reference_embedding.index_select(1, global_indices))
    reference_output = reference_output + reference_lora.index_select(1, global_indices)
    torch.testing.assert_close(
        _gather_sequence(local_output.detach(), world_size),
        reference_output,
    )

    output_gradient = torch.randn(
        reference_output.shape,
        device=device,
        generator=generator,
    )
    local_output.backward(_local_sequence_shard(output_gradient, rank, world_size))
    reference_output.backward(output_gradient)

    distributed_gradients = [frame_embedding.grad, frame_lora.grad]
    distributed_gradients.extend(parameter.grad for parameter in module.parameters())
    reference_gradients = [reference_embedding.grad, reference_lora.grad]
    reference_gradients.extend(parameter.grad for parameter in reference_module.parameters())
    for actual, expected in zip(distributed_gradients, reference_gradients, strict=True):
        dist.all_reduce(actual)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    if rank == 0:
        print(
            f"PASS: framewise AdaLN CP={world_size} forward and gradients",
            flush=True,
        )


def _test_flash_attention_oracle(
    device: torch.device,
    rank: int,
    world_size: int,
    attention_backend: str,
) -> None:
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
        attention_backend=attention_backend,
    )
    reference_inputs = [x.detach().clone().requires_grad_(True) for x in global_inputs]
    reference_output = block_causal_flash_attention(
        *reference_inputs,
        tokens_per_block=tokens_per_block,
        attention_backend=attention_backend,
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


def _dense_block_causal_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    tokens_per_block: int,
) -> torch.Tensor:
    scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float()) / math.sqrt(
        query.shape[-1]
    )
    positions = torch.arange(query.shape[1], device=query.device)
    query_block = torch.div(positions, tokens_per_block, rounding_mode="floor")
    key_block = torch.div(positions, tokens_per_block, rounding_mode="floor")
    keep = key_block[None, :] <= query_block[:, None]
    probabilities = scores.masked_fill(~keep[None, None], float("-inf")).softmax(dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probabilities, value.float())


def _dense_interleave_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    frame_sequence: int,
    num_frames_per_block: int,
    num_interleave: int,
) -> torch.Tensor:
    """Independent dense oracle for the model's interleaved causal mask."""
    scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float()) / math.sqrt(
        query.shape[-1]
    )
    positions = torch.arange(query.shape[1], device=query.device)
    frame_chunks = torch.div(positions, frame_sequence, rounding_mode="floor")
    num_frame_types = num_interleave + 1
    frame_types = frame_chunks % num_frame_types
    block_indices = torch.div(
        frame_chunks,
        num_frames_per_block * num_frame_types,
        rounding_mode="floor",
    )
    same_block_and_type = (block_indices[:, None] == block_indices[None, :]) & (
        frame_types[:, None] == frame_types[None, :]
    )
    previous_conditioning_block = (frame_types[None, :] == num_interleave) & (
        block_indices[None, :] < block_indices[:, None]
    )
    keep = same_block_and_type | previous_conditioning_block
    probabilities = scores.masked_fill(~keep[None, None], float("-inf")).softmax(dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probabilities, value.float())


def _test_flex_attention_oracle(
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    batch, sequence, heads, head_dim = 1, 24 * world_size, 2 * world_size, 64
    tokens_per_block = 8
    padded_sequence = ((sequence + 127) // 128) * 128
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

    def block_causal_mask(b, h, q_idx, kv_idx):
        del b, h
        valid = (q_idx < sequence) & (kv_idx < sequence)
        return valid & (kv_idx // tokens_per_block <= q_idx // tokens_per_block)

    block_mask = create_block_mask(
        block_causal_mask,
        B=None,
        H=None,
        Q_LEN=padded_sequence,
        KV_LEN=padded_sequence,
        device=device,
        _compile=True,
    )
    local_output = ulysses_flex_attention(
        *local_inputs,
        block_mask=block_mask,
        cp_manager=UlyssesCPManager(dist.group.WORLD),
        flex_attention_fn=compiled_flex_attention,
    )
    reference_inputs = [x.detach().clone().requires_grad_(True) for x in global_inputs]
    reference_output = _dense_block_causal_reference(
        *reference_inputs,
        tokens_per_block=tokens_per_block,
    )
    torch.testing.assert_close(
        _gather_sequence(local_output.detach(), world_size).float(),
        reference_output,
        atol=3e-2,
        rtol=3e-2,
    )

    local_output.backward(_local_sequence_shard(output_gradient, rank, world_size))
    reference_output.backward(output_gradient.float())
    for actual, reference in zip(local_inputs, reference_inputs, strict=True):
        expected_gradient = _local_sequence_shard(reference.grad, rank, world_size)
        torch.testing.assert_close(
            actual.grad.float(),
            expected_gradient.float(),
            atol=6e-2,
            rtol=6e-2,
        )


def _test_flex_interleave_oracle(
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    batch, sequence, heads, head_dim = 1, 24 * world_size, 2 * world_size, 64
    frame_sequence = 4
    num_frames_per_block = 2
    num_interleave = 1
    num_frames = sequence // (frame_sequence * (num_interleave + 1))
    generator = torch.Generator(device=device).manual_seed(20260921)
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

    block_mask = CosmosCausalDiT._prepare_blockwise_causal_attn_mask(
        device=device,
        num_frames=num_frames,
        frame_seqlen=frame_sequence,
        num_frame_per_block=num_frames_per_block,
        num_interleave=num_interleave,
        cp_size=1,
    )
    local_output = ulysses_flex_attention(
        *local_inputs,
        block_mask=block_mask,
        cp_manager=UlyssesCPManager(dist.group.WORLD),
        flex_attention_fn=compiled_flex_attention,
    )
    reference_inputs = [x.detach().clone().requires_grad_(True) for x in global_inputs]
    reference_output = _dense_interleave_reference(
        *reference_inputs,
        frame_sequence=frame_sequence,
        num_frames_per_block=num_frames_per_block,
        num_interleave=num_interleave,
    )
    torch.testing.assert_close(
        _gather_sequence(local_output.detach(), world_size).float(),
        reference_output,
        atol=3e-2,
        rtol=3e-2,
    )

    local_output.backward(_local_sequence_shard(output_gradient, rank, world_size))
    reference_output.backward(output_gradient.float())
    for actual, reference in zip(local_inputs, reference_inputs, strict=True):
        expected_gradient = _local_sequence_shard(reference.grad, rank, world_size)
        torch.testing.assert_close(
            actual.grad.float(),
            expected_gradient.float(),
            atol=6e-2,
            rtol=6e-2,
        )

    if rank == 0:
        print(
            f"PASS: flex Ulysses CP={world_size} num_interleave={num_interleave} "
            "forward and gradients",
            flush=True,
        )


def _benchmark_flex_cp(
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    sequence = int(os.getenv("BENCHMARK_SEQUENCE", "4096"))
    tokens_per_block = int(os.getenv("BENCHMARK_TOKENS_PER_BLOCK", "512"))
    repeats = int(os.getenv("BENCHMARK_REPEATS", "10"))
    heads, head_dim = 16, 128
    if sequence % world_size != 0:
        raise ValueError(f"BENCHMARK_SEQUENCE={sequence} must be divisible by CP={world_size}")
    if sequence % 128 != 0 or sequence // world_size % 128 != 0:
        raise ValueError("Flex CP benchmark sequence lengths must be multiples of 128")

    def block_causal_mask(b, h, q_idx, kv_idx):
        del b, h
        return kv_idx // tokens_per_block <= q_idx // tokens_per_block

    block_mask = create_block_mask(
        block_causal_mask,
        B=None,
        H=None,
        Q_LEN=sequence,
        KV_LEN=sequence,
        device=device,
        _compile=True,
    )
    generator = torch.Generator(device=device).manual_seed(20260920)
    local_sequence = sequence // world_size
    inputs = [
        torch.randn(
            1,
            local_sequence,
            heads,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
            requires_grad=True,
        )
        for _ in range(3)
    ]
    output_gradient = torch.randn(
        1,
        local_sequence,
        heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    cp_manager = UlyssesCPManager(dist.group.WORLD)

    def step(cp_backend: str) -> None:
        for tensor in inputs:
            tensor.grad = None
        if cp_backend == "legacy":
            output = flex_attention_cp(
                inputs[0].transpose(2, 1),
                inputs[1].transpose(2, 1),
                inputs[2].transpose(2, 1),
                process_group=dist.group.WORLD,
                block_mask=block_mask,
                flex_attention_fn=compiled_flex_attention,
            ).transpose(2, 1)
        else:
            output = ulysses_flex_attention(
                *inputs,
                block_mask=block_mask,
                cp_manager=cp_manager,
                flex_attention_fn=compiled_flex_attention,
            )
        output.backward(output_gradient)

    for cp_backend in ("legacy", "ulysses"):
        for _ in range(3):
            step(cp_backend)
    dist.barrier()

    samples = {"legacy": [], "ulysses": []}
    for iteration in range(repeats):
        order = ("legacy", "ulysses") if iteration % 2 == 0 else ("ulysses", "legacy")
        for cp_backend in order:
            dist.barrier()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            step(cp_backend)
            end.record()
            end.synchronize()
            elapsed = torch.tensor(start.elapsed_time(end), device=device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            samples[cp_backend].append(float(elapsed))

    if rank == 0:
        for cp_backend in ("legacy", "ulysses"):
            values = samples[cp_backend]
            print(
                f"BENCH: flex+{cp_backend} CP={world_size} S={sequence} "
                f"median_ms={statistics.median(values):.3f} "
                f"min_ms={min(values):.3f} max_ms={max(values):.3f}",
                flush=True,
            )
        print(
            "BENCH: flex Ulysses / legacy speedup="
            f"{statistics.median(samples['legacy']) / statistics.median(samples['ulysses']):.4f}x",
            flush=True,
        )


def main() -> None:
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise RuntimeError("The Ulysses correctness oracle requires at least two ranks")
    attention_backend = os.getenv("ATTENTION_BACKEND", "flash_attn_3")
    if attention_backend not in TRAINING_ATTENTION_BACKENDS:
        raise ValueError(
            f"ATTENTION_BACKEND must be one of {sorted(TRAINING_ATTENTION_BACKENDS)}, "
            f"got {attention_backend!r}"
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)

    try:
        _test_a2a_identity(device, dist.get_rank(), world_size)
        dist.barrier()
        _test_framewise_adaln_oracle(device, dist.get_rank(), world_size)
        dist.barrier()
        if attention_backend in FLASH_ATTENTION_BACKENDS:
            _test_flash_attention_oracle(
                device,
                dist.get_rank(),
                world_size,
                attention_backend,
            )
        else:
            _test_flex_attention_oracle(device, dist.get_rank(), world_size)
            _test_flex_interleave_oracle(device, dist.get_rank(), world_size)
            if os.getenv("BENCHMARK_FLEX_CP", "0") == "1":
                _benchmark_flex_cp(device, dist.get_rank(), world_size)
        if dist.get_rank() == 0:
            print(
                f"PASS: {attention_backend} Ulysses CP={world_size} forward and gradients",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
