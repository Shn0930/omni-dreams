# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Correctness tests for the full-sequence FlashAttention-3 block-causal backend."""

import math

import pytest
import torch


def _hopper_is_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


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
    scores = scores.masked_fill(~keep[None, None], float("-inf"))
    probabilities = scores.softmax(dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probabilities, value.float())


@pytest.mark.parametrize(
    ("shape", "tokens_per_block"),
    [
        ((1, 12, 2, 64), 4),
        # Production head_dim, B>1, and a partial final block.
        ((2, 10, 4, 128), 4),
        # A single partial block larger than the sequence is full attention.
        ((1, 5, 2, 64), 8),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("implementation", ["release", "optimized"])
@pytest.mark.skipif(
    not _hopper_is_available(),
    reason="FlashAttention-3 requires a Hopper (SM90) GPU",
)
def test_flash3_matches_dense_block_causal_forward_and_gradients(
    shape: tuple[int, int, int, int],
    tokens_per_block: int,
    dtype: torch.dtype,
    implementation: str,
) -> None:
    pytest.importorskip("flash_attn_3_nv")
    from omnidreams._src.omnidreams.modules.block_causal_flash_attention import (
        block_causal_flash_attention,
    )

    from optimized_block_causal_flash_attention import (
        optimized_block_causal_flash_attention,
    )

    torch.manual_seed(7)
    actual_inputs = [
        torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)
    ]
    reference_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in actual_inputs]
    output_gradient = torch.randn(shape, device="cuda", dtype=dtype)

    attention_fn = (
        block_causal_flash_attention
        if implementation == "release"
        else optimized_block_causal_flash_attention
    )
    actual = attention_fn(
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
            atol=6e-2,
            rtol=6e-2,
        )


@pytest.mark.parametrize(
    ("shape", "tokens_per_block"),
    [
        # Exercise the production temporal block count with BF16 accumulation.
        ((1, 384, 4, 128), 16),
        # Exercise non-contiguous per-block batch views and a partial final block.
        ((2, 10, 4, 128), 4),
    ],
)
@pytest.mark.skipif(
    not _hopper_is_available(),
    reason="FlashAttention-3 requires a Hopper (SM90) GPU",
)
def test_optimized_prefix_grad_strictly_matches_release(
    shape: tuple[int, int, int, int],
    tokens_per_block: int,
) -> None:
    pytest.importorskip("flash_attn_3_nv")
    from omnidreams._src.omnidreams.modules.block_causal_flash_attention import (
        block_causal_flash_attention,
    )

    from optimized_block_causal_flash_attention import (
        optimized_block_causal_flash_attention,
    )

    torch.manual_seed(23)
    release_inputs = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    optimized_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in release_inputs]
    output_gradient = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    release = block_causal_flash_attention(
        *release_inputs,
        tokens_per_block=tokens_per_block,
    )
    optimized = optimized_block_causal_flash_attention(
        *optimized_inputs,
        tokens_per_block=tokens_per_block,
    )
    release.backward(output_gradient)
    optimized.backward(output_gradient)

    # Both paths execute the same FA3 forward kernels.
    torch.testing.assert_close(optimized, release, atol=0, rtol=0)
    # Allow only small kernel scheduling/low-precision accumulation variation,
    # substantially tighter than the comparison against the FP32 dense oracle.
    for optimized_input, release_input in zip(
        optimized_inputs,
        release_inputs,
        strict=True,
    ):
        torch.testing.assert_close(
            optimized_input.grad,
            release_input.grad,
            atol=5e-4,
            rtol=5e-4,
        )


@pytest.mark.skipif(
    not _hopper_is_available(),
    reason="FlashAttention-3 requires a Hopper (SM90) GPU",
)
def test_optimized_installer_patches_only_cp1_network_binding() -> None:
    pytest.importorskip("flash_attn_3_nv")
    from omnidreams._src.omnidreams.modules import block_causal_flash_attention as backend
    from omnidreams._src.omnidreams.networks import causal_cosmos

    from optimized_block_causal_flash_attention import (
        install_optimized_block_causal_flash_attention,
        optimized_block_causal_flash_attention,
    )

    original_backend_full = backend.block_causal_flash_attention
    original_network_full = causal_cosmos.block_causal_flash_attention
    original_backend_query_sharded = backend.query_sharded_block_causal_flash_attention
    original_network_query_sharded = causal_cosmos.query_sharded_block_causal_flash_attention
    try:
        install_optimized_block_causal_flash_attention()

        assert backend.block_causal_flash_attention is original_backend_full
        assert causal_cosmos.block_causal_flash_attention is optimized_block_causal_flash_attention
        assert backend.query_sharded_block_causal_flash_attention is original_backend_query_sharded
        assert (
            causal_cosmos.query_sharded_block_causal_flash_attention
            is original_network_query_sharded
        )
    finally:
        backend.block_causal_flash_attention = original_backend_full
        causal_cosmos.block_causal_flash_attention = original_network_full


@pytest.mark.skipif(
    not _hopper_is_available(),
    reason="FlashAttention-3 requires a Hopper (SM90) GPU",
)
def test_optimized_prefix_grad_is_compatible_with_selective_checkpoint() -> None:
    pytest.importorskip("flash_attn_3_nv")
    from omnidreams._src.predict2.networks.minimal_v4_dit import (
        predict2_2B_720_context_fn_aggressive,
    )
    from torch.utils.checkpoint import checkpoint

    from optimized_block_causal_flash_attention import (
        optimized_block_causal_flash_attention,
    )

    shape = (1, 12, 2, 64)
    torch.manual_seed(17)
    plain_inputs = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    checkpoint_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in plain_inputs]
    output_gradient = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    plain = optimized_block_causal_flash_attention(
        *plain_inputs,
        tokens_per_block=4,
    )
    checkpointed = checkpoint(
        lambda query, key, value: optimized_block_causal_flash_attention(
            query,
            key,
            value,
            tokens_per_block=4,
        ),
        *checkpoint_inputs,
        use_reentrant=False,
        context_fn=predict2_2B_720_context_fn_aggressive,
        preserve_rng_state=False,
    )
    plain.backward(output_gradient)
    checkpointed.backward(output_gradient)

    torch.testing.assert_close(plain, checkpointed, atol=0, rtol=0)
    for plain_input, checkpoint_input in zip(
        plain_inputs,
        checkpoint_inputs,
        strict=True,
    ):
        torch.testing.assert_close(
            plain_input.grad,
            checkpoint_input.grad,
            atol=0,
            rtol=0,
        )


def test_spatially_repeated_adaln_matches_pointwise_reference() -> None:
    from copy import deepcopy

    from optimized_repeated_adaln import SpatiallyRepeatedSequential

    torch.manual_seed(31)
    repeat_factor = 7
    per_frame = torch.randn(2, 3, 32, dtype=torch.float32, requires_grad=True)
    optimized_input = per_frame.detach().clone().requires_grad_(True)
    reference = torch.nn.Sequential(
        torch.nn.SiLU(),
        torch.nn.Linear(32, 8, bias=False),
        torch.nn.Linear(8, 96, bias=False),
    )
    optimized = SpatiallyRepeatedSequential(*deepcopy(list(reference.children())))
    optimized.set_spatial_repeat_factor(repeat_factor)
    output_gradient = torch.randn(2, 3 * repeat_factor, 96)

    reference_output = reference(per_frame.repeat_interleave(repeat_factor, dim=1))
    optimized_output = optimized(optimized_input.repeat_interleave(repeat_factor, dim=1))
    reference_output.backward(output_gradient)
    optimized_output.backward(output_gradient)

    torch.testing.assert_close(optimized_output, reference_output)
    torch.testing.assert_close(optimized_input.grad, per_frame.grad, atol=2e-5, rtol=2e-5)
    for optimized_parameter, reference_parameter in zip(
        optimized.parameters(),
        reference.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            optimized_parameter.grad,
            reference_parameter.grad,
            atol=2e-5,
            rtol=2e-5,
        )


def test_spatially_repeated_adaln_preserves_state_dict_layout() -> None:
    from optimized_repeated_adaln import SpatiallyRepeatedSequential

    reference = torch.nn.Sequential(
        torch.nn.SiLU(),
        torch.nn.Linear(16, 4, bias=False),
        torch.nn.Linear(4, 48, bias=False),
    )
    optimized = SpatiallyRepeatedSequential(*list(reference.children()))

    assert list(optimized.state_dict()) == ["1.weight", "2.weight"]


def test_repeated_adaln_extracts_positional_and_keyword_call_context() -> None:
    from optimized_repeated_adaln import (
        _extract_inference_range,
        _extract_video_size,
    )

    positional = tuple(range(14))
    assert _extract_video_size(positional, {}) == 13
    assert _extract_video_size(positional, {"video_size": "keyword"}) == "keyword"
    assert _extract_inference_range(positional, {}) == (7, 9, 10)
    assert _extract_inference_range(
        positional,
        {
            "kv_cache": "cache",
            "current_start": 21,
            "current_end": 42,
        },
    ) == ("cache", 21, 42)


def test_repeated_adaln_rejects_partial_frame_kv_cache_chunks() -> None:
    from optimized_repeated_adaln import _validate_frame_alignment

    _validate_frame_alignment(
        (),
        {"kv_cache": {}, "current_start": 32, "current_end": 64},
        16,
    )
    with pytest.raises(ValueError, match="latent-frame boundaries"):
        _validate_frame_alignment(
            (),
            {"kv_cache": {}, "current_start": 1, "current_end": 33},
            16,
        )
