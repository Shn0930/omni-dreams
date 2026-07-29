# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Strict tests for the fused exact FA4 block-causal sample backend."""

from __future__ import annotations

import math
import os
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest
import torch

import fa4_exact_block_causal_attention as backend


def _fa4_install_is_available() -> bool:
    try:
        return version("flash-attn-4") == backend._SUPPORTED_FA4_VERSION
    except PackageNotFoundError:
        return False


def _hopper_fa4_is_available() -> bool:
    return (
        _fa4_install_is_available()
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (9, 0)
    )


@pytest.mark.gpu
@pytest.mark.skipif(
    not _hopper_fa4_is_available(),
    reason="Commit-pinned FA4 CuTeDSL and a Hopper SM90 GPU are required",
)
def test_fa4_runtime_modules_resolve_from_isolated_overlay() -> None:
    backend._fa4_symbols()
    overlay = Path(os.environ["OMNI_FA4_OVERLAY"]).resolve()
    for module_name in (
        "flash_attn.cute",
        "flash_attn.cute.block_sparsity",
        "flash_attn.cute.interface",
        "cutlass",
    ):
        module_path = Path(import_module(module_name).__file__).resolve()
        assert module_path.is_relative_to(overlay), (module_name, module_path, overlay)


def _metadata_to_dense(
    metadata: backend.SparseBlockMetadata,
    num_rows: int,
    num_columns: int,
) -> torch.Tensor:
    dense = torch.zeros(num_rows, num_columns, dtype=torch.bool)
    for row in range(num_rows):
        for count, indices in (
            (metadata.mask_block_cnt[0, 0, row], metadata.mask_block_idx[0, 0, row]),
            (metadata.full_block_cnt[0, 0, row], metadata.full_block_idx[0, 0, row]),
        ):
            selected = indices[: int(count)].cpu().long()
            dense[row, selected] = True
    return dense


@pytest.mark.parametrize(
    ("num_blocks", "sparse_tiles_per_block"),
    [
        (1, 1),
        (7, 1),
        (5, 3),
        (17, 2),
        # Production geometry: 12 blocks, 7040 tokens/block, 84480 total.
        (12, 55),
    ],
)
def test_metadata_is_the_exact_block_causal_mask_and_backward_transpose(
    num_blocks: int,
    sparse_tiles_per_block: int,
) -> None:
    tokens_per_block = sparse_tiles_per_block * backend._SPARSE_TILE_SIZE
    sequence_length = num_blocks * tokens_per_block
    metadata = backend.build_exact_block_causal_metadata(
        sequence_length,
        tokens_per_block,
        device="cpu",
    )
    num_sparse_tiles = num_blocks * sparse_tiles_per_block

    forward = _metadata_to_dense(
        metadata.forward,
        num_sparse_tiles,
        num_sparse_tiles,
    )
    backward = _metadata_to_dense(
        metadata.backward,
        num_sparse_tiles,
        num_sparse_tiles,
    )

    tile_ids = torch.arange(num_sparse_tiles)
    expected = torch.div(
        tile_ids[:, None], sparse_tiles_per_block, rounding_mode="floor"
    ) >= torch.div(tile_ids[None, :], sparse_tiles_per_block, rounding_mode="floor")
    assert torch.equal(forward, expected)
    assert torch.equal(backward, expected.T)
    assert torch.count_nonzero(metadata.forward.mask_block_cnt) == 0
    assert torch.count_nonzero(metadata.backward.mask_block_cnt) == 0
    assert metadata.forward.full_block_cnt.shape == (1, 1, num_sparse_tiles)
    assert metadata.backward.full_block_cnt.shape == (1, 1, num_sparse_tiles)


@pytest.mark.parametrize(
    ("sequence_length", "tokens_per_block", "error_type", "match"),
    [
        (0, 128, ValueError, "sequence_length must be positive"),
        (128, 0, ValueError, "tokens_per_block must be positive"),
        (256, 64, ValueError, "aligned to 128"),
        (384, 256, ValueError, "partial final block"),
        (True, 128, TypeError, "sequence_length must be an integer"),
        (128, False, TypeError, "tokens_per_block must be an integer"),
    ],
)
def test_metadata_builder_fails_closed_on_unsupported_geometry(
    sequence_length: int,
    tokens_per_block: int,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        backend.build_exact_block_causal_metadata(
            sequence_length,
            tokens_per_block,
            device="cpu",
        )


def test_attention_rejects_partial_final_block_before_runtime_dispatch() -> None:
    tensors = [torch.randn(1, 384, 2, 64) for _ in range(3)]
    with pytest.raises(ValueError, match="partial final block"):
        backend.fa4_exact_block_causal_attention(
            *tensors,
            tokens_per_block=256,
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
    query_blocks = torch.div(positions, tokens_per_block, rounding_mode="floor")
    key_blocks = torch.div(positions, tokens_per_block, rounding_mode="floor")
    keep = query_blocks[:, None] >= key_blocks[None, :]
    probabilities = scores.masked_fill(~keep[None, None], float("-inf")).softmax(dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probabilities, value.float())


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("shape", "tokens_per_block", "dtype"),
    [
        ((1, 384, 2, 64), 128, torch.bfloat16),
        # Five logical blocks and the production head dimension.
        ((1, 1280, 2, 128), 256, torch.bfloat16),
        # Metadata broadcasts over batch/head; validate the accepted FP16 path.
        ((2, 384, 2, 64), 128, torch.float16),
    ],
)
@pytest.mark.skipif(
    not _hopper_fa4_is_available(),
    reason="Commit-pinned FA4 CuTeDSL and a Hopper SM90 GPU are required",
)
def test_fused_fa4_matches_dense_forward_and_gradients(
    shape: tuple[int, int, int, int],
    tokens_per_block: int,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(41)
    actual_inputs = [
        torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)
    ]
    reference_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in actual_inputs]
    output_gradient = torch.randn(shape, device="cuda", dtype=dtype)

    actual = backend.fa4_exact_block_causal_attention(
        *actual_inputs,
        tokens_per_block=tokens_per_block,
    )
    reference = _dense_block_causal_reference(
        *reference_inputs,
        tokens_per_block=tokens_per_block,
    )
    actual.backward(output_gradient)
    reference.backward(output_gradient.float())

    torch.testing.assert_close(actual.float(), reference, atol=3e-2, rtol=3e-2)
    for actual_input, reference_input in zip(actual_inputs, reference_inputs, strict=True):
        torch.testing.assert_close(
            actual_input.grad.float(),
            reference_input.grad.float(),
            atol=7e-2,
            rtol=7e-2,
        )


@pytest.mark.gpu
@pytest.mark.skipif(
    not _hopper_fa4_is_available(),
    reason="Commit-pinned FA4 CuTeDSL and a Hopper SM90 GPU are required",
)
def test_aggressive_sac_must_saves_fa4_and_does_not_recompute_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omnidreams._src.predict2.networks.minimal_v4_dit import (
        predict2_2B_720_context_fn_aggressive,
    )
    from torch.utils.checkpoint import checkpoint

    forward_calls = 0
    original_forward = backend._run_fa4_forward

    def counted_forward(*args, **kwargs):
        nonlocal forward_calls
        forward_calls += 1
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(backend, "_run_fa4_forward", counted_forward)
    inputs = [
        torch.randn(
            1,
            384,
            2,
            64,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        for _ in range(3)
    ]

    output = checkpoint(
        lambda query, key, value: backend.fa4_exact_block_causal_attention(
            query,
            key,
            value,
            tokens_per_block=128,
        ),
        *inputs,
        use_reentrant=False,
        context_fn=predict2_2B_720_context_fn_aggressive,
        preserve_rng_state=False,
    )
    output.sum().backward()

    assert forward_calls == 1
    assert all(tensor.grad is not None for tensor in inputs)
    assert "flash_attn" in str(backend._flash_attn4_exact_block_causal_op)


@pytest.mark.gpu
@pytest.mark.skipif(
    not _hopper_fa4_is_available(),
    reason="Commit-pinned FA4 CuTeDSL and a Hopper SM90 GPU are required",
)
def test_installer_patches_only_cp1_network_binding() -> None:
    from importlib import import_module

    from omnidreams._src.omnidreams.networks import causal_cosmos

    module = import_module("omnidreams._src.omnidreams.modules.block_causal_flash_attention")
    original_module_full = module.block_causal_flash_attention
    original_module_query_sharded = module.query_sharded_block_causal_flash_attention
    original_module_ulysses = module.ulysses_block_causal_flash_attention
    original_network_full = causal_cosmos.block_causal_flash_attention
    original_network_query_sharded = causal_cosmos.query_sharded_block_causal_flash_attention
    original_network_ulysses = causal_cosmos.ulysses_block_causal_flash_attention
    try:
        backend.install_fa4_exact_block_causal_attention()

        assert module.block_causal_flash_attention is original_module_full
        assert module.query_sharded_block_causal_flash_attention is original_module_query_sharded
        assert module.ulysses_block_causal_flash_attention is original_module_ulysses
        assert (
            causal_cosmos.block_causal_flash_attention is backend.fa4_exact_block_causal_attention
        )
        assert (
            causal_cosmos.query_sharded_block_causal_flash_attention
            is original_network_query_sharded
        )
        assert causal_cosmos.ulysses_block_causal_flash_attention is original_network_ulysses
    finally:
        module.block_causal_flash_attention = original_module_full
        module.query_sharded_block_causal_flash_attention = original_module_query_sharded
        module.ulysses_block_causal_flash_attention = original_module_ulysses
        causal_cosmos.block_causal_flash_attention = original_network_full
        causal_cosmos.query_sharded_block_causal_flash_attention = original_network_query_sharded
        causal_cosmos.ulysses_block_causal_flash_attention = original_network_ulysses
