# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the sample-side FC2 selective-checkpoint policy."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

import optimized_selective_checkpoint as optimization


@pytest.mark.parametrize(
    ("m", "dtype"),
    [
        (1, torch.float32),
        (37, torch.bfloat16),
        (211, torch.float16),
    ],
)
def test_fc2_match_does_not_depend_on_sequence_or_dtype(
    m: int,
    dtype: torch.dtype,
) -> None:
    lhs = torch.empty(m, 8192, dtype=dtype, device="meta")
    rhs = torch.empty(8192, 2048, dtype=dtype, device="meta")

    assert optimization._is_predict2_2b_mlp_fc2(
        torch.ops.aten.mm.default,
        (lhs, rhs),
    )


@pytest.mark.parametrize(
    ("lhs_shape", "rhs_shape"),
    [
        ((3, 2048), (2048, 8192)),  # MLP FC1
        ((3, 2048), (2048, 2048)),  # attention projection
        ((3, 8192), (8192, 1024)),  # wrong output width
        ((3, 4096), (4096, 2048)),  # wrong hidden width
    ],
)
def test_fc2_match_rejects_other_matrix_multiplications(
    lhs_shape: tuple[int, int],
    rhs_shape: tuple[int, int],
) -> None:
    lhs = torch.empty(lhs_shape, device="meta")
    rhs = torch.empty(rhs_shape, device="meta")

    assert not optimization._is_predict2_2b_mlp_fc2(
        torch.ops.aten.mm.default,
        (lhs, rhs),
    )


def test_fc2_match_rejects_dtype_mismatch() -> None:
    lhs = torch.empty(2, 8192, dtype=torch.bfloat16, device="meta")
    rhs = torch.empty(8192, 2048, dtype=torch.float16, device="meta")

    assert not optimization._is_predict2_2b_mlp_fc2(
        torch.ops.aten.mm.default,
        (lhs, rhs),
    )


class _FC2ExecutionCounter(TorchDispatchMode):
    def __init__(self, hidden_dim: int, model_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model_dim = model_dim
        self.executions = 0

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        del types
        kwargs = {} if kwargs is None else kwargs
        if function == torch.ops.aten.mm.default and len(args) >= 2:
            lhs, rhs = args[:2]
            if (
                lhs.ndim == 2
                and rhs.ndim == 2
                and lhs.shape[1] == self.hidden_dim
                and tuple(rhs.shape) == (self.hidden_dim, self.model_dim)
            ):
                self.executions += 1
        return function(*args, **kwargs)


def _recompute_everything_context_fn():
    def policy_fn(ctx, function, *args, **kwargs):
        del ctx, function, args, kwargs
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy_fn)


def _run_gated_mlp(context_fn, seed: int):
    torch.manual_seed(seed)
    model_dim = optimization._MODEL_DIM
    hidden_dim = optimization._MLP_HIDDEN_DIM
    tensors = [
        torch.randn(3, model_dim, requires_grad=True),
        torch.randn(model_dim, hidden_dim, requires_grad=True),
        torch.randn(hidden_dim, model_dim, requires_grad=True),
        torch.randn(3, model_dim, requires_grad=True),
    ]
    counter = _FC2ExecutionCounter(hidden_dim, model_dim)

    def gated_mlp(x, fc1_weight, fc2_weight, gate):
        hidden = functional.gelu(torch.mm(x, fc1_weight))
        fc2_output = torch.mm(hidden, fc2_weight)
        return x + gate * fc2_output

    with counter:
        output = checkpoint(
            gated_mlp,
            *tensors,
            use_reentrant=False,
            context_fn=context_fn,
            preserve_rng_state=False,
        )
        output.square().mean().backward()
    return output.detach(), [tensor.grad.detach() for tensor in tensors], counter.executions


def test_fc2_save_removes_only_the_fc2_checkpoint_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Exercise real SAC cache/replay behavior with a tiny MLP while preserving
    # the production shape relation used by the policy.
    monkeypatch.setattr(optimization, "_MODEL_DIM", 8)
    monkeypatch.setattr(optimization, "_MLP_HIDDEN_DIM", 16)

    baseline = _run_gated_mlp(_recompute_everything_context_fn, seed=97)
    optimized = _run_gated_mlp(
        optimization.predict2_2b_720_context_fn_save_mlp_fc2,
        seed=97,
    )

    torch.testing.assert_close(optimized[0], baseline[0])
    for optimized_grad, baseline_grad in zip(optimized[1], baseline[1], strict=True):
        torch.testing.assert_close(optimized_grad, baseline_grad)
    # The same dispatcher signature is also used once by the true FC1
    # grad-input GEMM. The optimized run removes exactly the checkpoint replay.
    assert baseline[2] == 3
    assert optimized[2] == 2


def test_installer_replaces_only_the_2b_aggressive_factory() -> None:
    from omnidreams._src.predict2.networks import minimal_v4_dit

    original_aggressive = minimal_v4_dit.predict2_2B_720_context_fn_aggressive
    original_default = minimal_v4_dit.predict2_2B_720_context_fn
    try:
        optimization.install_mlp_fc2_selective_checkpoint()
        assert (
            minimal_v4_dit.predict2_2B_720_context_fn_aggressive
            is optimization.predict2_2b_720_context_fn_save_mlp_fc2
        )
        assert minimal_v4_dit.predict2_2B_720_context_fn is original_default
    finally:
        minimal_v4_dit.predict2_2B_720_context_fn_aggressive = original_aggressive
