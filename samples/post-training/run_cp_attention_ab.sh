#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Configurable single-node causal-attention A/B launcher.
#
# NPROC is inferred from CUDA_VISIBLE_DEVICES unless set explicitly. CP_SIZE
# and FSDP_SIZE default to NPROC and can be chosen independently, provided
# each divides NPROC.
#
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 CP_SIZE=4 FSDP_SIZE=4 \
#     bash samples/post-training/run_cp_attention_ab.sh fa3-zigzag
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC=8 CP_SIZE=2 FSDP_SIZE=8 NSYS=1 \
#     bash samples/post-training/run_cp_attention_ab.sh fa3-ulysses

set -euo pipefail

MODE="${1:-}"
case "$MODE" in
  flex-contiguous)
    ATTENTION_BACKEND=flex
    CP_STRATEGY=contiguous
    SAC_MODE=block_wise
    ;;
  flex-zigzag)
    ATTENTION_BACKEND=flex
    CP_STRATEGY=zigzag
    SAC_MODE=block_wise
    ;;
  fa3-contiguous)
    ATTENTION_BACKEND=flash_attn_3
    CP_STRATEGY=contiguous
    SAC_MODE=predict2_2b_720_aggressive
    ;;
  fa3-zigzag)
    ATTENTION_BACKEND=flash_attn_3
    CP_STRATEGY=zigzag
    SAC_MODE=predict2_2b_720_aggressive
    ;;
  fa3-ulysses)
    ATTENTION_BACKEND=flash_attn_3
    CP_STRATEGY=ulysses
    SAC_MODE=predict2_2b_720_aggressive
    ;;
  *)
    echo "Usage: $0 {flex-contiguous|flex-zigzag|fa3-contiguous|fa3-zigzag|fa3-ulysses} [Hydra overrides ...]" >&2
    exit 2
    ;;
esac
shift
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REL="$REPO_ROOT/post-training"
export REL

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the GPUs for this single-node run}"
if [[ "$CUDA_VISIBLE_DEVICES" == ,* || "$CUDA_VISIBLE_DEVICES" == *, || "$CUDA_VISIBLE_DEVICES" == *,,* ]]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES contains an empty entry." >&2
  exit 2
fi
IFS=',' read -r -a VISIBLE_GPUS <<<"$CUDA_VISIBLE_DEVICES"
VISIBLE_GPU_COUNT="${#VISIBLE_GPUS[@]}"
declare -A SEEN_GPUS=()
for gpu in "${VISIBLE_GPUS[@]}"; do
  if [[ -z "$gpu" || "$gpu" =~ [[:space:]] ]]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES entries must be non-empty and contain no whitespace." >&2
    exit 2
  fi
  if [[ -v "SEEN_GPUS[$gpu]" ]]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES contains duplicate entry '$gpu'." >&2
    exit 2
  fi
  SEEN_GPUS["$gpu"]=1
done

NPROC="${NPROC:-$VISIBLE_GPU_COUNT}"
CP_SIZE="${CP_SIZE:-$NPROC}"
FSDP_SIZE="${FSDP_SIZE:-$NPROC}"
for name in NPROC CP_SIZE FSDP_SIZE; do
  value="${!name}"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: $name must be a positive integer, got '$value'." >&2
    exit 2
  fi
done
if (( NPROC > VISIBLE_GPU_COUNT )); then
  echo "ERROR: NPROC=$NPROC exceeds the $VISIBLE_GPU_COUNT entries in CUDA_VISIBLE_DEVICES." >&2
  exit 2
fi
if (( NPROC % CP_SIZE != 0 )); then
  echo "ERROR: CP_SIZE=$CP_SIZE must divide NPROC=$NPROC." >&2
  exit 2
fi
if (( NPROC % FSDP_SIZE != 0 )); then
  echo "ERROR: FSDP_SIZE=$FSDP_SIZE must divide NPROC=$NPROC." >&2
  exit 2
fi

MAX_ITER="${MAX_ITER:-10}"
# Iteration 5 lazily compiles tokenizer.encode in the release config.  The
# default capture window starts later so one-time compilation is not profiled.
PROFILE_FIRST="${PROFILE_FIRST:-7}"
PROFILE_LAST="${PROFILE_LAST:-$MAX_ITER}"
NSYS="${NSYS:-0}"
MASTER_PORT="${MASTER_PORT:-12460}"
JOB_NAME="${JOB_NAME:-df93_t210_${NPROC}gpu_cp${CP_SIZE}_fsdp${FSDP_SIZE}_${MODE}_$(date +%Y%m%d_%H%M%S)}"

if ! [[ "$MAX_ITER" =~ ^[1-9][0-9]*$ && "$PROFILE_FIRST" =~ ^[1-9][0-9]*$ && "$PROFILE_LAST" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: MAX_ITER, PROFILE_FIRST, and PROFILE_LAST must be positive integers." >&2
  exit 2
fi
if (( PROFILE_FIRST > PROFILE_LAST || PROFILE_LAST > MAX_ITER )); then
  echo "ERROR: require PROFILE_FIRST <= PROFILE_LAST <= MAX_ITER." >&2
  exit 2
fi

: "${OMNI_CACHE_DIR:=${XDG_CACHE_HOME:-$HOME/.cache}}"
: "${TMPDIR:=$OMNI_CACHE_DIR/tmp}"
PROFILE_DATA_ROOT="${PROFILE_DATA_ROOT:-$REPO_ROOT/post-training/data}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$OMNI_CACHE_DIR/nsys/omnidreams_df93_${NPROC}gpu/cp_attention/$JOB_NAME}"
mkdir -p "$ARTIFACT_DIR"

for required_subdir in video hdmap caption; do
  if [[ ! -d "$PROFILE_DATA_ROOT/$required_subdir" ]]; then
    echo "ERROR: dataset directory is missing: $PROFILE_DATA_ROOT/$required_subdir" >&2
    echo "Run samples/post-training/setup_env.sh or set PROFILE_DATA_ROOT." >&2
    exit 2
  fi
done

export OMNI_CACHE_DIR TMPDIR PROFILE_DATA_ROOT
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"
export NCCL_TUNER_PLUGIN="${NCCL_TUNER_PLUGIN:-none}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRITON_CACHE_BASE="${TRITON_CACHE_BASE:-$OMNI_CACHE_DIR/triton/$JOB_NAME}"
export OMNI_PROFILE_FIRST="$PROFILE_FIRST"
export OMNI_PROFILE_LAST="$PROFILE_LAST"
export OMNI_PROFILE_CAPTURE="$NSYS"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/fa3_env.sh"

WRAPPER="$SCRIPT_DIR/triton_per_rank_wrap.sh"
ENTRY="$SCRIPT_DIR/fa3_profile_entry.py"
PYTHON="$OMNI_FA3_PYTHON"
NSYS_BIN="${NSYS_BIN:-}"
if [[ "$NSYS" == "1" ]]; then
  if [[ -z "$NSYS_BIN" ]]; then
    NSYS_BIN="$(command -v nsys || true)"
  elif [[ ! -x "$NSYS_BIN" ]]; then
    NSYS_BIN="$(command -v "$NSYS_BIN" || true)"
  fi
  if [[ -z "$NSYS_BIN" || ! -x "$NSYS_BIN" ]]; then
    echo "ERROR: NSYS=1 requires nsys on PATH or an executable NSYS_BIN." >&2
    exit 2
  fi
fi

{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "mode=$MODE"
  echo "attention_backend=$ATTENTION_BACKEND"
  echo "cp_strategy=$CP_STRATEGY"
  echo "sac_mode=$SAC_MODE"
  echo "nproc_per_node=$NPROC"
  echo "visible_gpu_count=$VISIBLE_GPU_COUNT"
  echo "fsdp_size=$FSDP_SIZE"
  echo "cp_size=$CP_SIZE"
  echo "max_iter=$MAX_ITER"
  echo "profile_first=$PROFILE_FIRST"
  echo "profile_last=$PROFILE_LAST"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "profile_data_root=$PROFILE_DATA_ROOT"
  find "$PROFILE_DATA_ROOT" -type l -printf 'dataset_link=%p -> %l\n' | sort
  while IFS= read -r -d '' dataset_link; do
    resolved_path="$(readlink -f "$dataset_link")"
    if [[ -f "$resolved_path" ]]; then
      echo "dataset_sha256=$(sha256sum "$resolved_path")"
    fi
  done < <(find "$PROFILE_DATA_ROOT" -type l -print0 | sort -z)
  echo "git_branch=$(git -C "$REPO_ROOT" branch --show-current)"
  echo "git_head=$(git -C "$REPO_ROOT" rev-parse HEAD)"
  echo "nsys=$NSYS"
  if [[ "$NSYS" == "1" ]]; then
    "$NSYS_BIN" --version | head -1
  fi
  "$PYTHON" - <<'PY'
from importlib.metadata import version

import torch

print(f"python={__import__('sys').version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"transformer_engine={version('transformer-engine')}")
print(f"flash_attn={version('flash-attn')}")
print(f"flash_attn_3_nv={version('flash-attn-3-nv')}")
print(f"natten={version('natten')}")
PY
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
    --query-gpu=index,uuid,name,driver_version --format=csv,noheader
  sha256sum \
    "$SCRIPT_DIR/fa3_env.sh" \
    "$SCRIPT_DIR/fa3_profile_entry.py" \
    "$SCRIPT_DIR/run_cp_attention_ab.sh" \
    "$REPO_ROOT/post-training/omnidreams/_src/imaginaire/utils/context_parallel.py" \
    "$REPO_ROOT/post-training/omnidreams/_src/omnidreams/modules/block_causal_flash_attention.py" \
    "$REPO_ROOT/post-training/omnidreams/_src/omnidreams/modules/ulysses_attention.py" \
    "$REPO_ROOT/post-training/omnidreams/_src/omnidreams/networks/causal_cosmos.py" \
    "$REPO_ROOT/post-training/omnidreams/_src/omnidreams/networks/causal_cosmos_hdmap.py"
} >"$ARTIFACT_DIR/metadata.txt"
git -C "$REPO_ROOT" status --short >"$ARTIFACT_DIR/git-status.txt"
git -C "$REPO_ROOT" diff HEAD >"$ARTIFACT_DIR/tracked-changes.patch"

WORKER=("$PYTHON" "$ENTRY")
if [[ "$NSYS" == "1" ]]; then
  WORKER=(
    "$NSYS_BIN" profile
    --trace=cuda,nvtx
    --sample=none
    --cpuctxsw=none
    --backtrace=none
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    --kill=none
    --wait=all
    --stats=false
    --force-overwrite=true
    -o "$ARTIFACT_DIR/timing_rank%q{LOCAL_RANK}"
    "$PYTHON" "$ENTRY"
  )
fi

nvidia-smi \
  -i "$CUDA_VISIBLE_DEVICES" \
  --query-gpu=timestamp,index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits \
  -lms 200 >"$ARTIFACT_DIR/nvml.csv" &
NVML_PID=$!
cleanup() {
  kill "$NVML_PID" 2>/dev/null || true
}
trap cleanup EXIT

cd "$REL"
"$PYTHON" -m torch.distributed.run \
  --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" \
  --no-python "$WRAPPER" \
  "${WORKER[@]}" \
  --config=omnidreams/_src/omnidreams/configs/causal_cosmos2/config.py \
  -- \
  experiment=causal_cosmos2_2B_single_view_chunk2_t24_hdmap_vae \
  "job.name=$JOB_NAME" \
  job.wandb_mode=disabled \
  model.config.noise_scheme=diffusion_forcing \
  model.config.num_frame_per_block=2 \
  model.config.state_t=24 \
  model.config.max_latent_frames_per_gpu=24 \
  model.config.split_cp_in_model=false \
  "model.config.fsdp_shard_size=$FSDP_SIZE" \
  "model_parallel.context_parallel_size=$CP_SIZE" \
  "model.config.net.training_attention_backend=$ATTENTION_BACKEND" \
  "model.config.net.training_context_parallel_strategy=$CP_STRATEGY" \
  "model.config.net.sac_config.mode=$SAC_MODE" \
  model.config.net.sac_config.every_n_blocks=1 \
  "dataloader_train.data_root=$PROFILE_DATA_ROOT" \
  "dataloader_val.data_root=$PROFILE_DATA_ROOT" \
  dataloader_train.augmentation_config.num_video_frames=93 \
  dataloader_val.augmentation_config.num_video_frames=93 \
  dataloader_train.repeat_factor=200 \
  dataloader_train.val_holdout_frac=0.0 \
  "trainer.max_iter=$MAX_ITER" \
  trainer.run_validation=false \
  checkpoint.save_iter=1000 \
  '+trainer.callbacks.nsys={_target_:omnidreams._src.imaginaire.utils.callback.NVTXCallback,synchronize:false}' \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$ARTIFACT_DIR/train.log"
