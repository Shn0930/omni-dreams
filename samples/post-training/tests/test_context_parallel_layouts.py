# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU correctness tests for contiguous and zigzag CP index mappings."""

import pytest
import torch
from omnidreams._src.imaginaire.utils import context_parallel
from omnidreams._src.omnidreams.modules.flex_attention import rewrite_mask_mod_for_cp
from omnidreams._src.omnidreams.networks.causal_cosmos import CausalSelfAttention


class _FakeProcessGroup:
    def __init__(self, rank: int):
        self._rank = rank

    def rank(self) -> int:
        return self._rank


def test_ulysses_attention_strategy_requires_flash3() -> None:
    with pytest.raises(ValueError, match="requires training_attention_backend='flash_attn_3'"):
        CausalSelfAttention(
            query_dim=64,
            n_heads=1,
            head_dim=64,
            training_attention_backend="flex",
            training_context_parallel_strategy="ulysses",
        )


def test_zigzag_split_and_inverse_gather(monkeypatch: pytest.MonkeyPatch) -> None:
    cp_size = 4
    monkeypatch.setattr(
        context_parallel, "get_process_group_ranks", lambda group: list(range(cp_size))
    )
    full = torch.arange(2 * 32 * 3).reshape(2, 32, 3)

    shards = [
        context_parallel.split_inputs_cp(
            full,
            seq_dim=1,
            cp_group=_FakeProcessGroup(rank),
            layout="zigzag",
        )
        for rank in range(cp_size)
    ]

    expected_chunk_ids = ((0, 7), (1, 6), (2, 5), (3, 4))
    chunks = full.reshape(2, 2 * cp_size, 4, 3)
    for shard, chunk_ids in zip(shards, expected_chunk_ids, strict=True):
        expected = chunks[:, list(chunk_ids)].reshape(2, 8, 3)
        torch.testing.assert_close(shard, expected)

    physical_rank_order = torch.cat(shards, dim=1)
    restored = context_parallel.restore_zigzag_sequence_order(
        physical_rank_order,
        seq_dim=1,
        cp_size=cp_size,
    )
    torch.testing.assert_close(restored, full)


def test_zigzag_restore_is_differentiable() -> None:
    cp_size = 4
    logical = torch.arange(16, dtype=torch.float32)
    physical_to_logical = torch.tensor([0, 7, 1, 6, 2, 5, 3, 4])
    physical = (
        logical.reshape(8, 2).index_select(0, physical_to_logical).reshape(16).requires_grad_(True)
    )
    weights = torch.arange(1, 17, dtype=torch.float32)

    restored = context_parallel.restore_zigzag_sequence_order(physical, seq_dim=0, cp_size=cp_size)
    (restored * weights).sum().backward()

    expected_grad = weights.reshape(8, 2).index_select(0, physical_to_logical).reshape(16)
    torch.testing.assert_close(physical.grad, expected_grad)


def test_zigzag_restore_reuses_permutation_index() -> None:
    context_parallel._zigzag_restore_index.cache_clear()
    physical = torch.arange(16)

    context_parallel.restore_zigzag_sequence_order(physical, seq_dim=0, cp_size=4)
    context_parallel.restore_zigzag_sequence_order(physical, seq_dim=0, cp_size=4)

    cache_info = context_parallel._zigzag_restore_index.cache_info()
    assert cache_info.misses == 1
    assert cache_info.hits == 1


def test_zigzag_flex_mask_mapping_with_rank_padding() -> None:
    cp_size = 4
    valid_shard_size = 6
    padded_shard_size = 8
    tokens_per_block = 4

    def contiguous_physical_mask(b, h, q_idx, kv_idx):
        del b, h
        q_rank = q_idx // padded_shard_size
        q_off = q_idx % padded_shard_size
        kv_rank = kv_idx // padded_shard_size
        kv_off = kv_idx % padded_shard_size
        q_valid = q_off < valid_shard_size
        kv_valid = kv_off < valid_shard_size
        q_logical = q_rank * valid_shard_size + q_off
        kv_logical = kv_rank * valid_shard_size + kv_off
        return (
            q_valid & kv_valid & (kv_logical // tokens_per_block <= q_logical // tokens_per_block)
        )

    micro_chunk_size = valid_shard_size // 2
    for rank in range(cp_size):
        mapped_mask = rewrite_mask_mod_for_cp(
            contiguous_physical_mask,
            rank=rank,
            shard_size=padded_shard_size,
            world_size=cp_size,
            valid_shard_size=valid_shard_size,
            cp_layout="zigzag",
        )
        for local_q_idx in range(padded_shard_size):
            q_valid = local_q_idx < valid_shard_size
            if local_q_idx < micro_chunk_size:
                q_logical = rank * micro_chunk_size + local_q_idx
            else:
                q_logical = (
                    (2 * cp_size - rank - 1) * micro_chunk_size + local_q_idx - micro_chunk_size
                )

            for physical_kv_idx in range(cp_size * padded_shard_size):
                source_rank = physical_kv_idx // padded_shard_size
                source_off = physical_kv_idx % padded_shard_size
                kv_valid = source_off < valid_shard_size
                if source_off < micro_chunk_size:
                    kv_logical = source_rank * micro_chunk_size + source_off
                else:
                    kv_logical = (
                        (2 * cp_size - source_rank - 1) * micro_chunk_size
                        + source_off
                        - micro_chunk_size
                    )
                expected = (
                    q_valid
                    and kv_valid
                    and kv_logical // tokens_per_block <= q_logical // tokens_per_block
                )
                actual = mapped_mask(
                    None,
                    None,
                    torch.tensor(local_q_idx),
                    torch.tensor(physical_kv_idx),
                )
                assert bool(actual) is expected
