# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Training entry point used by reproducible causal-attention A/B profiles."""

from __future__ import annotations

import os

import torch
from omnidreams._src.imaginaire.utils.callback import NVTXCallback

FIRST = int(os.environ.get("OMNI_PROFILE_FIRST", "6"))
LAST = int(os.environ.get("OMNI_PROFILE_LAST", "8"))
CAPTURE = os.environ.get("OMNI_PROFILE_CAPTURE", "0") == "1"
CUSTOM_PREFIX_GRAD_VALUE = os.environ.get("OMNI_FA3_CUSTOM_PREFIX_GRAD", "0")
if CUSTOM_PREFIX_GRAD_VALUE not in {"0", "1"}:
    raise ValueError(
        f"OMNI_FA3_CUSTOM_PREFIX_GRAD must be 0 or 1, got {CUSTOM_PREFIX_GRAD_VALUE!r}"
    )
CUSTOM_PREFIX_GRAD = CUSTOM_PREFIX_GRAD_VALUE == "1"
FA4_EXACT_VALUE = os.environ.get("OMNI_FA4_EXACT_BLOCK_CAUSAL", "0")
if FA4_EXACT_VALUE not in {"0", "1"}:
    raise ValueError(f"OMNI_FA4_EXACT_BLOCK_CAUSAL must be 0 or 1, got {FA4_EXACT_VALUE!r}")
FA4_EXACT = FA4_EXACT_VALUE == "1"
REPEATED_ADALN_VALUE = os.environ.get("OMNI_OPTIMIZE_REPEATED_ADALN", "0")
if REPEATED_ADALN_VALUE not in {"0", "1"}:
    raise ValueError(f"OMNI_OPTIMIZE_REPEATED_ADALN must be 0 or 1, got {REPEATED_ADALN_VALUE!r}")
REPEATED_ADALN = REPEATED_ADALN_VALUE == "1"
if CUSTOM_PREFIX_GRAD and FA4_EXACT:
    raise ValueError(
        "OMNI_FA3_CUSTOM_PREFIX_GRAD and OMNI_FA4_EXACT_BLOCK_CAUSAL "
        "replace the same CP=1 attention binding and are mutually exclusive"
    )
_original_after_backward = NVTXCallback.on_after_backward

if CUSTOM_PREFIX_GRAD:
    from optimized_block_causal_flash_attention import (
        install_optimized_block_causal_flash_attention,
    )

    install_optimized_block_causal_flash_attention()

if FA4_EXACT:
    from fa4_exact_block_causal_attention import (
        install_fa4_exact_block_causal_attention,
    )

    install_fa4_exact_block_causal_attention()

if REPEATED_ADALN:
    from optimized_repeated_adaln import install_repeated_adaln_optimization

    install_repeated_adaln_optimization()


def _range_call(name, function, *args, **kwargs):
    torch.cuda.nvtx.range_push(name)
    try:
        return function(*args, **kwargs)
    finally:
        torch.cuda.nvtx.range_pop()


def _on_train_start(self, model, iteration=0):
    del iteration
    self._encode_call = 0
    original_encode = model.encode
    original_denoise = model.denoise
    original_text = model.inplace_compute_text_embeddings_online

    def wrapped_encode(*args, **kwargs):
        index = self._encode_call
        self._encode_call += 1
        name = "vae_encode_hdmap" if index == 0 else "vae_encode_video"
        return _range_call(name, original_encode, *args, **kwargs)

    def wrapped_denoise(*args, **kwargs):
        return _range_call("net_forward", original_denoise, *args, **kwargs)

    def wrapped_text(*args, **kwargs):
        return _range_call("text_encode", original_text, *args, **kwargs)

    model.encode = wrapped_encode
    model.denoise = wrapped_denoise
    model.inplace_compute_text_embeddings_online = wrapped_text
    # These short profiling runs must not emit a multi-GiB final checkpoint.
    self.trainer.checkpointer.save = lambda *args, **kwargs: None


def _on_training_step_start(self, model, data, iteration=0):
    del model, data
    self._encode_call = 0
    torch.cuda.reset_peak_memory_stats()
    if CAPTURE and iteration == FIRST - 1:
        torch.distributed.barrier()
        torch.cuda.synchronize()
        return_code = torch.cuda.cudart().cudaProfilerStart()
        if return_code != 0:
            raise RuntimeError(f"cudaProfilerStart failed: {return_code}")
        torch.cuda.nvtx.range_push(f"profile_iters_{FIRST}_{LAST}")
    if FIRST - 1 <= iteration < LAST:
        torch.cuda.nvtx.range_push(f"iteration_{iteration + 1}")


def _on_after_backward(self, model_ddp, iteration=0):
    _original_after_backward(self, model_ddp, iteration=iteration)
    if FIRST - 1 <= iteration < LAST:
        torch.cuda.nvtx.range_push("optimizer_full")


def _on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
    del model, data_batch, output_batch, loss
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    print(
        "OMNI_PROFILE_MEMORY"
        f" rank={rank} iteration={iteration}"
        f" allocated_gib={torch.cuda.max_memory_allocated() / 2**30:.3f}"
        f" reserved_gib={torch.cuda.max_memory_reserved() / 2**30:.3f}",
        flush=True,
    )
    if FIRST <= iteration <= LAST:
        torch.cuda.nvtx.range_pop()
        if CAPTURE:
            torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    if CAPTURE and iteration == LAST:
        torch.cuda.nvtx.range_pop()
        return_code = torch.cuda.cudart().cudaProfilerStop()
        if return_code != 0:
            raise RuntimeError(f"cudaProfilerStop failed: {return_code}")


NVTXCallback.on_train_start = _on_train_start
NVTXCallback.on_training_step_start = _on_training_step_start
NVTXCallback.on_after_backward = _on_after_backward
NVTXCallback.on_training_step_end = _on_training_step_end

from cosmos_oss.scripts.train import main  # noqa: E402

main()
