# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
from functools import lru_cache
from typing import Literal, Optional

try:
    import megatron.core.parallel_state as parallel_state

    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False

import torch
from torch import Tensor
from torch.distributed import ProcessGroup, all_gather, broadcast_object_list, get_process_group_ranks, get_world_size
from torch.distributed.utils import _verify_param_shape_across_processes

from omnidreams._src.imaginaire.utils import distributed


ContextParallelLayout = Literal["contiguous", "zigzag"]


def _validate_cp_layout(layout: ContextParallelLayout) -> None:
    if layout not in {"contiguous", "zigzag"}:
        raise ValueError(f"Unknown context-parallel layout {layout!r}; expected 'contiguous' or 'zigzag'")


@lru_cache(maxsize=128)
def _zigzag_restore_index(cp_size: int, device_type: str, device_index: Optional[int]) -> Tensor:
    """Return a reusable logical-order index without repeated host-to-device copies."""
    num_chunks = 2 * cp_size
    logical_to_physical = tuple(
        2 * logical_idx if logical_idx < cp_size else 2 * (num_chunks - logical_idx - 1) + 1
        for logical_idx in range(num_chunks)
    )
    device = torch.device(device_type) if device_index is None else torch.device(device_type, device_index)
    return torch.tensor(logical_to_physical, dtype=torch.long, device=device)


def restore_zigzag_sequence_order(x: Tensor, seq_dim: int, cp_size: int) -> Tensor:
    """Restore chronological order after gathering zigzag CP shards.

    Zigzag rank ``r`` owns global micro-chunks ``r`` and
    ``2 * cp_size - r - 1``.  A rank-order all-gather therefore produces the
    physical chunk order ``[0, 2P-1, 1, 2P-2, ...]``; this helper applies the
    inverse permutation.
    """
    num_chunks = 2 * cp_size
    if x.shape[seq_dim] % num_chunks != 0:
        raise ValueError(
            "A zigzag CP gather requires the sequence length to be divisible by "
            f"2 * cp_size ({num_chunks}), got {x.shape[seq_dim]}"
        )

    chunk_size = x.shape[seq_dim] // num_chunks
    chunked = x.reshape(
        *x.shape[:seq_dim],
        num_chunks,
        chunk_size,
        *x.shape[(seq_dim + 1) :],
    )
    index = _zigzag_restore_index(cp_size, x.device.type, x.device.index)
    ordered = chunked.index_select(seq_dim, index)
    return ordered.reshape(*x.shape[:seq_dim], -1, *x.shape[(seq_dim + 1) :])


def split_inputs_cp(
    x: Tensor,
    seq_dim: int,
    cp_group: ProcessGroup,
    layout: ContextParallelLayout = "contiguous",
) -> Tensor:
    """
    Split input tensor along the sequence dimension for checkpoint parallelism.

    This function divides the input tensor into equal parts along the specified
    sequence dimension, based on the number of ranks in the checkpoint parallelism group.
    It then selects the part corresponding to the current rank.

    Args:
        x: Input tensor to be split.
        seq_dim: The dimension along which to split the input (sequence dimension).
        cp_group: The process group for context parallelism.
        layout: ``contiguous`` assigns one consecutive shard to each rank.
            ``zigzag`` divides the sequence into ``2 * cp_size`` equal
            micro-chunks and assigns the symmetric pair
            ``(rank, 2 * cp_size - rank - 1)`` to each rank.

    Returns:
        A slice of the input tensor corresponding to the current rank.

    Raises:
        AssertionError: If the sequence dimension is not divisible by the number of ranks.
    """
    _validate_cp_layout(layout)
    cp_ranks = get_process_group_ranks(cp_group)
    cp_size = len(cp_ranks)

    num_chunks = cp_size if layout == "contiguous" else 2 * cp_size
    assert x.shape[seq_dim] % num_chunks == 0, f"{x.shape[seq_dim]} cannot divide {num_chunks} CP chunks"
    x = x.view(*x.shape[:seq_dim], num_chunks, x.shape[seq_dim] // num_chunks, *x.shape[(seq_dim + 1) :])
    rank = cp_group.rank()
    if layout == "contiguous":
        selected_chunks = [rank]
    else:
        selected_chunks = [rank, num_chunks - rank - 1]
    seq_idx = torch.tensor(selected_chunks, dtype=torch.long, device=x.device)
    x = x.index_select(seq_dim, seq_idx)
    # Note that the new sequence length is the original sequence length / cp_size
    x = x.view(*x.shape[:seq_dim], -1, *x.shape[(seq_dim + 2) :])
    return x


@torch.compiler.disable
def cat_outputs_cp(
    x: Tensor,
    seq_dim: int,
    cp_group: ProcessGroup,
    layout: ContextParallelLayout = "contiguous",
) -> Tensor:
    """
    Concatenate outputs from different ranks in the checkpoint parallelism group.

    This function gathers tensors from all ranks in the checkpoint parallelism group
    and concatenates them along the specified sequence dimension.

    The function is decorated with @torch.compiler.disable because it contains distributed
    operations and dynamic tensor creation based on runtime rank information that seem to be
    incompatible with torch.compile's static graph compilation.

    Args:
        x: Input tensor to be concatenated.
        seq_dim: The dimension along which to concatenate the tensors (sequence dimension).
        cp_group: The process group for context parallelism.
        layout: Layout used when the inputs were split.

    Returns:
        A tensor that is the concatenation of tensors from all ranks in the cp_group.

    Raises:
        RuntimeError: If the gather operation fails.
    """
    _validate_cp_layout(layout)
    # Get the world size (number of processes in the group)
    world_size = get_world_size(cp_group)

    # Create a list to store tensors from all ranks
    gathered_tensors = [torch.zeros_like(x) for _ in range(world_size)]

    # Gather tensors from all ranks
    try:
        all_gather(gathered_tensors, x, group=cp_group)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to gather tensors: {e}")

    # Concatenate the gathered tensors along the specified dimension.
    output = torch.cat(gathered_tensors, dim=seq_dim)
    if layout == "zigzag":
        output = restore_zigzag_sequence_order(output, seq_dim=seq_dim, cp_size=world_size)
    return output


def cat_outputs_cp_with_grad(
    x: Tensor,
    seq_dim: int,
    cp_group: ProcessGroup,
    layout: ContextParallelLayout = "contiguous",
) -> Tensor:
    """
    Concatenate outputs from different ranks in the context parallelism group.

    This function gathers tensors from all ranks in the checkpoint parallelism group
    and concatenates them along the specified sequence dimension.

    It retains computational graph locally for each rank by replacing the portion of the tensor with original output.

    Args:
        x: Input tensor to be concatenated.
        seq_dim: The dimension along which to concatenate the tensors (sequence dimension).
        cp_group: The process group for context parallelism.
        layout: Layout used when the inputs were split.

    Returns:
        A tensor that is the concatenation of tensors from all ranks in the cp_group.

    Raises:
        RuntimeError: If the gather operation fails.
    """
    _validate_cp_layout(layout)
    # Get the world size (number of processes in the group)
    cp_size = cp_group.size()
    assert cp_size > 0, "cp_size should be greater than 0"

    # Create a list to store tensors from all ranks
    gathered_tensors = [torch.zeros_like(x) for _ in range(cp_size)]

    # Gather tensors from all ranks
    try:
        all_gather(gathered_tensors, x, group=cp_group)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to gather tensors: {e}")

    rank = cp_group.rank()
    gathered_tensors[rank] = x
    # Concatenate the gathered tensors along the specified dimension.
    output = torch.cat(gathered_tensors, dim=seq_dim)
    if layout == "zigzag":
        output = restore_zigzag_sequence_order(output, seq_dim=seq_dim, cp_size=cp_size)
    return output


@torch.compiler.disable
def robust_broadcast(tensor: torch.Tensor, src: int, pg: ProcessGroup, is_check_shape: bool = False) -> torch.Tensor:
    """
    Perform a robust broadcast operation that works regardless of tensor shapes on different ranks.

    The function is decorated with @torch.compiler.disable because it contains distributed
    operations and dynamic tensor creation based on runtime rank information that seem to be
    incompatible with torch.compile's static graph compilation.

    Args:
        tensor (torch.Tensor): The tensor to broadcast (on src rank) or receive (on other ranks).
        src (int): The source rank for the broadcast. Defaults to 0.

    Returns:
        torch.Tensor: The broadcasted tensor on all ranks.
    """
    # First, broadcast the shape of the tensor
    if distributed.get_rank() == src:
        shape = torch.tensor(tensor.shape, dtype=torch.long).cuda()
    else:
        shape = torch.empty(tensor.dim(), dtype=torch.long).cuda()
    if is_check_shape:
        _verify_param_shape_across_processes(pg, [shape])
    torch.distributed.broadcast(shape, src, group=pg)

    # Resize the tensor on non-src ranks if necessary
    if distributed.get_rank() != src:
        tensor = tensor.new_empty(shape.tolist()).type_as(tensor)

    # Now broadcast the tensor data
    torch.distributed.broadcast(tensor, src, group=pg)

    return tensor


def broadcast(
    item: torch.Tensor | str | None, process_group: Optional[ProcessGroup] = None
) -> torch.Tensor | str | None:
    """
    Broadcast the item from the minimum rank in the specified group(s).
    """
    if process_group is None:
        return item

    min_rank = min(get_process_group_ranks(process_group))
    if isinstance(item, torch.Tensor):  # assume the device is cuda
        item = robust_broadcast(item, min_rank, process_group)
    elif item is not None:
        broadcastable_list = [item]
        broadcast_object_list(broadcastable_list, min_rank, group=process_group)
        item = broadcastable_list[0]
    return item


def broadcast_split_tensor(
    tensor: torch.Tensor,
    seq_dim: int,
    process_group: Optional[ProcessGroup] = None,
) -> torch.Tensor:
    """
    Broadcast the tensor from the minimum rank in the specified group(s).
    """
    if tensor is None:
        return tensor
    min_rank = min(get_process_group_ranks(process_group))
    tensor = robust_broadcast(tensor, min_rank, process_group)
    return split_inputs_cp(tensor, seq_dim, process_group)


def find_split(
    shape_tensor: torch.Size, cp_size: int, patch_values: tuple[int, int, int] = (1, 2, 2), view_factor: int = 1
) -> torch.Size:
    """
    Find the shape of input tensor for post-CP split, taking into account both temporal and spatial split, as well as patching values.
    The split by width is not possible currently, due to memory stride issues, which break quality. This is checked
    by an assert.

    The spatial split is achieved by flattening the input video into a single dimension before CP split is performed,
    and rearranging it back into [T, H, W] format after the CP split, since the input passed to the model must still be in [T, H, W] format.

    Args:
        shape_tensor (torch.Size): The shape of the Tensor that we want to split. Needs to be in [B, C, T, H, W] format.
        cp_size (int): The Context Parallelism size that we want to use.
        patch_values (tuple[int, int, int], optional): The patch values that are applied inside the Diffusion model.
            First element of the tuple is temporal patch size. Two next elements are the spatial patch sizes.
            The default value is (1, 2, 2)
        view_factor (int, optional): The number of views that are present in the temporal dimension. Default value is 1.

    Returns:
        The torch.Size of how the post-split tensor should look like in [T, H, W] dimensions.

    """
    if not USE_MEGATRON:
        raise ImportError("No megatron.core package found, which is required for Context Parallelism usage.")
    B, C, T, H, W = shape_tensor
    ret = []
    assert T % view_factor == 0
    T = T // view_factor
    cp_size_t = 1
    for i, size in enumerate([T, H, W]):
        if i == 2 and cp_size > 1:
            raise ValueError(
                f"Split by width dimension is not currently supported due to quality issues. Width dimension would be split by a factor of {cp_size}. Lower the CP size to avoid splitting by width."
            )
        patch_size = patch_values[i]
        gcd = math.gcd(size // patch_size, cp_size)
        cp_size = cp_size // gcd
        if i == 0:
            cp_size_t = gcd
        ret.append(size // gcd)
    # Saving the CP size in the temporal dimension for VideoPositionEmb embeddings calculation
    parallel_state.cp_size_t = cp_size_t
    return torch.Size(ret)
