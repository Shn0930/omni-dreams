# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Sample-side fused exact block-causal attention using FA4 CuTeDSL.

The desired token mask is::

    keep(query_pos, key_pos) =
        floor(query_pos / tokens_per_block) >= floor(key_pos / tokens_per_block)

When every logical block is aligned to FA4's 128-token sparse tile, each
128x128 score tile is either entirely present or entirely absent.  Forward
metadata lists the legal K/V tiles for every Q tile; backward metadata stores
the exact transpose.  FA4 therefore evaluates the mask in one fused forward
and one fused backward call, with no token-level masking and no prefix loop.

The custom operator name intentionally contains ``flash_attn``.  OmniDreams'
aggressive selective-checkpoint policy matches that substring and marks this
operator ``MUST_SAVE``.  Keeping the CuTeDSL launch behind the dispatcher
operator is essential: a direct FA4 Python call exposes only ``aten.empty``
and ``aten.expand`` to selective checkpointing and would be recomputed.
"""

from __future__ import annotations

import os
import site
from functools import lru_cache
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import NamedTuple

import torch

_FA4_DISTRIBUTION = "flash-attn-4"
_SUPPORTED_FA4_VERSION = "0.0.1.dev1+g14c377950"
_SPARSE_TILE_SIZE = 128


class SparseBlockMetadata(NamedTuple):
    """FA4 count/index tensors for one direction of sparse traversal."""

    mask_block_cnt: torch.Tensor
    mask_block_idx: torch.Tensor
    full_block_cnt: torch.Tensor
    full_block_idx: torch.Tensor


class ExactBlockCausalMetadata(NamedTuple):
    """Forward sparse rows and their exact backward transpose."""

    forward: SparseBlockMetadata
    backward: SparseBlockMetadata


def _validate_block_geometry(sequence_length: int, tokens_per_block: int) -> None:
    if not isinstance(sequence_length, int) or isinstance(sequence_length, bool):
        raise TypeError(f"sequence_length must be an integer, got {sequence_length!r}")
    if not isinstance(tokens_per_block, int) or isinstance(tokens_per_block, bool):
        raise TypeError(f"tokens_per_block must be an integer, got {tokens_per_block!r}")
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    if tokens_per_block <= 0:
        raise ValueError(f"tokens_per_block must be positive, got {tokens_per_block}")
    if tokens_per_block % _SPARSE_TILE_SIZE != 0:
        raise ValueError(
            "Exact FA4 block-causal attention requires tokens_per_block to be "
            f"aligned to {_SPARSE_TILE_SIZE} tokens, got {tokens_per_block}"
        )
    if sequence_length % tokens_per_block != 0:
        raise ValueError(
            "Exact FA4 block-causal attention does not accept a partial final block: "
            f"sequence_length={sequence_length} is not divisible by "
            f"tokens_per_block={tokens_per_block}"
        )


def build_exact_block_causal_metadata(
    sequence_length: int,
    tokens_per_block: int,
    *,
    device: torch.device | str,
) -> ExactBlockCausalMetadata:
    """Build broadcastable FA4 metadata for any positive number of full blocks.

    Batch and head dimensions are one so FA4 can broadcast the same structural
    mask without duplicating metadata.  ``mask_*`` rows are empty because
    alignment guarantees that all retained sparse tiles are fully valid.
    """

    _validate_block_geometry(sequence_length, tokens_per_block)
    device = torch.device(device)
    num_sparse_tiles = sequence_length // _SPARSE_TILE_SIZE
    sparse_tiles_per_block = tokens_per_block // _SPARSE_TILE_SIZE

    tile_ids = torch.arange(num_sparse_tiles, dtype=torch.int32, device=device)
    logical_block_ids = torch.div(
        tile_ids,
        sparse_tiles_per_block,
        rounding_mode="floor",
    )

    empty_count = torch.zeros(
        1,
        1,
        num_sparse_tiles,
        dtype=torch.int32,
        device=device,
    )
    # FA4 requires mask_block_idx even when every mask count is zero.
    empty_index = torch.zeros(
        1,
        1,
        num_sparse_tiles,
        1,
        dtype=torch.int32,
        device=device,
    )

    forward_count = ((logical_block_ids + 1) * sparse_tiles_per_block).view(1, 1, -1)
    # Every forward row is a prefix.  The count selects the valid portion.
    forward_index = tile_ids.view(1, 1, 1, -1).expand(
        1,
        1,
        num_sparse_tiles,
        num_sparse_tiles,
    )

    first_query_tile = logical_block_ids * sparse_tiles_per_block
    backward_count = (num_sparse_tiles - first_query_tile).view(1, 1, -1)
    # Backward is K/V-tile-centric.  Row n visits every Q tile in the same or a
    # later logical block.  Padding is clamped but never read past row count.
    backward_index = (first_query_tile[:, None] + tile_ids[None, :]).clamp_max(num_sparse_tiles - 1)
    backward_index = backward_index.view(
        1,
        1,
        num_sparse_tiles,
        num_sparse_tiles,
    )

    return ExactBlockCausalMetadata(
        forward=SparseBlockMetadata(
            mask_block_cnt=empty_count,
            mask_block_idx=empty_index,
            full_block_cnt=forward_count,
            full_block_idx=forward_index,
        ),
        backward=SparseBlockMetadata(
            mask_block_cnt=empty_count,
            mask_block_idx=empty_index,
            full_block_cnt=backward_count,
            full_block_idx=backward_index,
        ),
    )


def _fa4_overlay_root() -> Path:
    overlay_value = os.environ.get("OMNI_FA4_OVERLAY")
    if not overlay_value:
        raise RuntimeError(
            "Set OMNI_FA4_OVERLAY to the isolated FA4 --target directory; "
            "source fa4_env.sh before enabling this backend."
        )
    overlay = Path(overlay_value).resolve()
    if not overlay.is_dir():
        raise RuntimeError(f"OMNI_FA4_OVERLAY is not a directory: {overlay}")
    return overlay


@lru_cache(maxsize=1)
def _validate_fa4_install() -> None:
    overlay = _fa4_overlay_root()
    try:
        installed_version = version(_FA4_DISTRIBUTION)
    except PackageNotFoundError as error:
        raise RuntimeError(
            "The fused exact block-causal backend requires the flash-attn-4 CuTeDSL package."
        ) from error
    if installed_version != _SUPPORTED_FA4_VERSION:
        raise RuntimeError(
            "The sample-side FA4 private API has only been validated with "
            f"{_SUPPORTED_FA4_VERSION!r}; got {installed_version!r}."
        )
    distribution_root = Path(distribution(_FA4_DISTRIBUTION).locate_file("")).resolve()
    if not distribution_root.is_relative_to(overlay):
        raise RuntimeError(
            f"{_FA4_DISTRIBUTION} resolved from {distribution_root}, not OMNI_FA4_OVERLAY {overlay}"
        )


@lru_cache(maxsize=1)
def _fa4_symbols():
    """Load the commit-pinned private FA4 API only when the backend is used."""

    _validate_fa4_install()
    overlay = _fa4_overlay_root()

    # ``nvidia_cutlass_dsl.pth`` exposes the nested ``cutlass`` package.
    # Arbitrary PYTHONPATH entries do not process .pth files automatically.
    site.addsitedir(str(overlay))

    # FA2 is a regular ``flash_attn`` package while the FA4 wheel contributes
    # only ``flash_attn/cute``. PYTHONPATH alone cannot merge that child
    # namespace, so explicitly add the overlay package directory before the
    # first FA4 import.
    import flash_attn

    overlay_flash_attn = overlay / "flash_attn"
    if not overlay_flash_attn.is_dir():
        raise RuntimeError(f"FA4 overlay is missing {overlay_flash_attn}")
    overlay_package_path = str(overlay_flash_attn)
    if overlay_package_path not in flash_attn.__path__:
        flash_attn.__path__.insert(0, overlay_package_path)

    block_sparsity = import_module("flash_attn.cute.block_sparsity")
    interface = import_module("flash_attn.cute.interface")
    cutlass = import_module("cutlass")
    for module in (block_sparsity, interface, cutlass):
        module_path = Path(module.__file__).resolve()
        if not module_path.is_relative_to(overlay):
            raise RuntimeError(
                f"{module.__name__} resolved from {module_path}, not OMNI_FA4_OVERLAY {overlay}"
            )

    return (
        block_sparsity.BlockSparseTensorsTorch,
        interface._flash_attn_fwd,
        interface._flash_attn_bwd,
    )


def _as_fa4_metadata(metadata: SparseBlockMetadata):
    block_sparse_tensors, _, _ = _fa4_symbols()
    return block_sparse_tensors(
        mask_block_cnt=metadata.mask_block_cnt,
        mask_block_idx=metadata.mask_block_idx,
        full_block_cnt=metadata.full_block_cnt,
        full_block_idx=metadata.full_block_idx,
        block_size=(_SPARSE_TILE_SIZE, _SPARSE_TILE_SIZE),
    )


def _run_fa4_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    metadata: SparseBlockMetadata,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, flash_attn_forward, _ = _fa4_symbols()
    output, softmax_lse, _, _ = flash_attn_forward(
        query,
        key,
        value,
        softmax_scale=softmax_scale,
        causal=False,
        pack_gqa=False,
        block_sparse_tensors=_as_fa4_metadata(metadata),
        return_lse=True,
    )
    return output, softmax_lse


def _run_fa4_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    softmax_lse: torch.Tensor,
    metadata: SparseBlockMetadata,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, _, flash_attn_backward = _fa4_symbols()
    return flash_attn_backward(
        query,
        key,
        value,
        output,
        grad_output,
        softmax_lse,
        softmax_scale=softmax_scale,
        causal=False,
        block_sparse_tensors=_as_fa4_metadata(metadata),
    )


@torch.library.custom_op(
    "omnidreams_samples::flash_attn4_exact_block_causal",
    mutates_args=(),
)
def _flash_attn4_exact_block_causal_op(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    forward_mask_count: torch.Tensor,
    forward_mask_index: torch.Tensor,
    forward_full_count: torch.Tensor,
    forward_full_index: torch.Tensor,
    backward_mask_count: torch.Tensor,
    backward_mask_index: torch.Tensor,
    backward_full_count: torch.Tensor,
    backward_full_index: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del (
        backward_mask_count,
        backward_mask_index,
        backward_full_count,
        backward_full_index,
    )
    return _run_fa4_forward(
        query,
        key,
        value,
        SparseBlockMetadata(
            forward_mask_count,
            forward_mask_index,
            forward_full_count,
            forward_full_index,
        ),
        softmax_scale,
    )


@_flash_attn4_exact_block_causal_op.register_fake
def _flash_attn4_exact_block_causal_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    forward_mask_count: torch.Tensor,
    forward_mask_index: torch.Tensor,
    forward_full_count: torch.Tensor,
    forward_full_index: torch.Tensor,
    backward_mask_count: torch.Tensor,
    backward_mask_index: torch.Tensor,
    backward_full_count: torch.Tensor,
    backward_full_index: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del (
        key,
        value,
        forward_mask_count,
        forward_mask_index,
        forward_full_count,
        forward_full_index,
        backward_mask_count,
        backward_mask_index,
        backward_full_count,
        backward_full_index,
        softmax_scale,
    )
    return torch.empty_like(query), torch.empty(
        query.shape[0],
        query.shape[2],
        query.shape[1],
        dtype=torch.float32,
        device=query.device,
    )


def _setup_flash_attn4_autograd_context(ctx, inputs, output) -> None:
    (
        query,
        key,
        value,
        _forward_mask_count,
        _forward_mask_index,
        _forward_full_count,
        _forward_full_index,
        backward_mask_count,
        backward_mask_index,
        backward_full_count,
        backward_full_index,
        softmax_scale,
    ) = inputs
    attention_output, softmax_lse = output
    ctx.save_for_backward(
        query,
        key,
        value,
        attention_output,
        softmax_lse,
        backward_mask_count,
        backward_mask_index,
        backward_full_count,
        backward_full_index,
    )
    ctx.softmax_scale = softmax_scale


def _flash_attn4_autograd_backward(ctx, grad_output, grad_softmax_lse):
    del grad_softmax_lse
    (
        query,
        key,
        value,
        attention_output,
        softmax_lse,
        backward_mask_count,
        backward_mask_index,
        backward_full_count,
        backward_full_index,
    ) = ctx.saved_tensors
    grad_query, grad_key, grad_value = _run_fa4_backward(
        query,
        key,
        value,
        attention_output,
        grad_output,
        softmax_lse,
        SparseBlockMetadata(
            backward_mask_count,
            backward_mask_index,
            backward_full_count,
            backward_full_index,
        ),
        ctx.softmax_scale,
    )
    return grad_query, grad_key, grad_value, *((None,) * 9)


torch.library.register_autograd(
    _flash_attn4_exact_block_causal_op,
    _flash_attn4_autograd_backward,
    setup_context=_setup_flash_attn4_autograd_context,
)


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    tokens_per_block: int,
) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must use [batch, sequence, heads, head_dim] layout")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError(
            "query, key, and value shapes must match, got "
            f"{query.shape}, {key.shape}, and {value.shape}"
        )
    if query.device != key.device or query.device != value.device:
        raise ValueError(
            "query, key, and value devices must match, got "
            f"{query.device}, {key.device}, and {value.device}"
        )
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError(
            "query, key, and value dtypes must match, got "
            f"{query.dtype}, {key.dtype}, and {value.dtype}"
        )
    _validate_block_geometry(query.shape[1], tokens_per_block)
    if query.device.type != "cuda":
        raise ValueError(f"FA4 exact block-causal attention requires CUDA, got {query.device}")
    if query.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError(
            f"FA4 exact block-causal attention requires float16 or bfloat16, got {query.dtype}"
        )
    capability = torch.cuda.get_device_capability(query.device)
    if capability != (9, 0):
        raise RuntimeError(
            "This sample-side FA4 backend has only been validated on Hopper SM90, "
            f"got compute capability {capability}"
        )


@lru_cache(maxsize=32)
def _cached_metadata(
    sequence_length: int,
    tokens_per_block: int,
    device_type: str,
    device_index: int | None,
) -> ExactBlockCausalMetadata:
    return build_exact_block_causal_metadata(
        sequence_length,
        tokens_per_block,
        device=torch.device(device_type, device_index),
    )


def fa4_exact_block_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    tokens_per_block: int,
) -> torch.Tensor:
    """Run one exact block-sparse FA4 forward/backward pair for a full sequence."""

    _validate_inputs(query, key, value, tokens_per_block)
    _validate_fa4_install()
    device_index = query.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    metadata = _cached_metadata(
        query.shape[1],
        tokens_per_block,
        query.device.type,
        device_index,
    )
    output, _ = _flash_attn4_exact_block_causal_op(
        query,
        key,
        value,
        *metadata.forward,
        *metadata.backward,
        query.shape[-1] ** -0.5,
    )
    return output


def install_fa4_exact_block_causal_attention() -> None:
    """Patch only the CP=1 network binding; leave every CP backend untouched."""

    _validate_fa4_install()
    from omnidreams._src.omnidreams.networks import causal_cosmos

    causal_cosmos.block_causal_flash_attention = fa4_exact_block_causal_attention
