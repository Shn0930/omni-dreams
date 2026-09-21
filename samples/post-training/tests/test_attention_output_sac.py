# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the backend-neutral attention-output SAC policy."""

import pytest
import torch
from omnidreams._src.predict2.networks.selective_activation_checkpoint import (
    CheckpointMode,
    SACConfig,
    attention_output_policy,
    compiled_attention_region,
    is_attention_output_op,
)
from torch.utils.checkpoint import checkpoint


class _NamedOp:
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name


_ATTENTION_CALLS = 0


@torch.library.custom_op("omnidreams_sac_test::flex_attention", mutates_args=())
def _attention_op(value: torch.Tensor) -> torch.Tensor:
    global _ATTENTION_CALLS
    _ATTENTION_CALLS += 1
    return value.sin()


@_attention_op.register_fake
def _attention_op_fake(value: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(value)


def _attention_op_setup_context(ctx, inputs, output) -> None:
    del output
    ctx.save_for_backward(inputs[0])


def _attention_op_backward(ctx, output_gradient: torch.Tensor) -> torch.Tensor:
    (value,) = ctx.saved_tensors
    return output_gradient * value.cos()


_attention_op.register_autograd(_attention_op_backward, setup_context=_attention_op_setup_context)


@pytest.mark.parametrize(
    "op_name",
    [
        "flex_attention",
        "torch.ops.higher_order.flex_attention",
        "flash_attn_3_nv::_flash_attn_forward",
        "flash_attn_4::_flash_attn_forward",
        "aten._scaled_dot_product_flash_attention.default",
        "aten._scaled_dot_product_efficient_attention.default",
        "aten._scaled_dot_product_cudnn_attention.default",
    ],
)
def test_attention_output_policy_saves_fused_attention_ops(op_name: str) -> None:
    op = _NamedOp(op_name)

    assert is_attention_output_op(op)
    assert attention_output_policy(None, op) == torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE


@pytest.mark.parametrize(
    "op_name",
    [
        "aten.mm.default",
        "aten.layer_norm.default",
        "aten.silu.default",
        "aten.add.Tensor",
    ],
)
def test_attention_output_policy_recomputes_non_attention_ops(op_name: str) -> None:
    op = _NamedOp(op_name)

    assert not is_attention_output_op(op)
    assert (
        attention_output_policy(None, op)
        == torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
    )


def test_compiled_region_is_saved_only_inside_attention_scope() -> None:
    compiled_op = _NamedOp("inductor_compiled_code")

    assert not is_attention_output_op(compiled_op)
    with compiled_attention_region():
        assert is_attention_output_op(compiled_op)
        assert (
            attention_output_policy(None, compiled_op)
            == torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE
        )
    assert not is_attention_output_op(compiled_op)


def test_attention_output_mode_builds_selective_checkpoint_contexts() -> None:
    config = SACConfig(mode=CheckpointMode.ATTENTION_OUTPUT)

    forward_context, recompute_context = config.get_context_fn()()

    assert forward_context is not None
    assert recompute_context is not None


def test_minimal_dit_config_inherits_attention_output_mode() -> None:
    from omnidreams._src.predict2.networks.minimal_v4_dit import (
        CheckpointMode as MinimalCheckpointMode,
    )
    from omnidreams._src.predict2.networks.minimal_v4_dit import SACConfig as MinimalSACConfig

    config = MinimalSACConfig(mode=MinimalCheckpointMode.ATTENTION_OUTPUT)

    forward_context, recompute_context = config.get_context_fn()()

    assert forward_context is not None
    assert recompute_context is not None


def test_attention_output_sac_does_not_replay_saved_attention_op() -> None:
    global _ATTENTION_CALLS
    _ATTENTION_CALLS = 0
    value = torch.randn(8, requires_grad=True)
    config = SACConfig(mode=CheckpointMode.ATTENTION_OUTPUT)

    output = checkpoint(
        lambda tensor: _attention_op(tensor).square(),
        value,
        use_reentrant=False,
        context_fn=config.get_context_fn(),
    )
    output.sum().backward()

    assert _ATTENTION_CALLS == 1
    assert value.grad is not None
