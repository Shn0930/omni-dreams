#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Source this file to run OmniDreams with the CUDA 12.8 FA3 stack provisioned
# from the Cosmos Framework lock.  The environment stays separate from the
# release torch-2.7 venv; OmniDreams source packages are selected through
# PYTHONPATH.
#
#   OMNI_FA3_VENV=/path/to/cosmos-cu128-torch210-venv \
#     source samples/post-training/fa3_env.sh
#
# Create the environment from a Cosmos Framework checkout with:
#
#   git clone https://github.com/NVIDIA/cosmos-framework.git /path/to/cosmos-framework
#   git -C /path/to/cosmos-framework checkout \
#     117c7d21f04b374a57a81a6e6a50416718b4e191
#   UV_PROJECT_ENVIRONMENT=/path/to/cosmos-cu128-torch210-venv \
#     uv sync --project /path/to/cosmos-framework --locked \
#       --no-default-groups --group=cu128-train --python 3.13
#
# Then install the three local OmniDreams projects without CUDA extras:
#
#   uv pip install --python /path/to/cosmos-cu128-torch210-venv/bin/python \
#     -e post-training/packages/cosmos-cuda \
#     -e post-training/packages/cosmos-oss -e post-training \
#     pytz==2026.2 decord==0.6.0

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "ERROR: source this file instead of executing it." >&2
  exit 2
fi

_FA3_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_FA3_REPO_ROOT="$(cd "$_FA3_ENV_DIR/../.." && pwd)"

if [[ -z "${OMNI_FA3_VENV:-}" ]]; then
  _FA3_CACHE_ROOT="${OMNI_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}}"
  if [[ -d "$_FA3_CACHE_ROOT/omnidreams-venvs/fa3-cu128-torch210" ]]; then
    OMNI_FA3_VENV="$_FA3_CACHE_ROOT/omnidreams-venvs/fa3-cu128-torch210"
  elif [[ -n "${COSMOS_FRAMEWORK_ROOT:-}" && -d "$COSMOS_FRAMEWORK_ROOT/.venv-cu128-t210" ]]; then
    OMNI_FA3_VENV="$COSMOS_FRAMEWORK_ROOT/.venv-cu128-t210"
  else
    echo "ERROR: set OMNI_FA3_VENV to the synced Cosmos CUDA 12.8 / torch 2.10 environment." >&2
    return 2
  fi
fi

OMNI_FA3_PYTHON="${OMNI_FA3_PYTHON:-$OMNI_FA3_VENV/bin/python}"
if [[ ! -x "$OMNI_FA3_PYTHON" ]]; then
  echo "ERROR: FA3 Python is not executable: $OMNI_FA3_PYTHON" >&2
  return 2
fi

_FA3_SITE_PACKAGES="$("$OMNI_FA3_PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
_FA3_NVIDIA_LIBS="$(find "$_FA3_SITE_PACKAGES/nvidia" -maxdepth 3 -type d -name lib -printf '%p\n' 2>/dev/null | paste -sd: -)"

export OMNI_FA3_VENV OMNI_FA3_PYTHON
# Deliberately do not append the caller's Python 3.10 PYTHONPATH or CUDA 13
# library paths: mixing either into this CPython 3.13 / CUDA 12.8 stack can
# produce late native-extension failures.
export PYTHONPATH="$_FA3_REPO_ROOT/post-training:$_FA3_REPO_ROOT/post-training/packages/cosmos-oss:$_FA3_REPO_ROOT/post-training/packages/cosmos-cuda:$_FA3_REPO_ROOT/samples/post-training"
export LD_LIBRARY_PATH="$_FA3_NVIDIA_LIBS"
export CUDA_HOME="$_FA3_SITE_PACKAGES/nvidia"
# The host exposes CUDA 13 through ldconfig in addition to the venv's CUDA 12
# runtime. cuDNN frontend 1.18 rejects that dual-runtime visibility before it
# launches a kernel. Keep Transformer Engine cross-attention on its FA2 path;
# the causal self-attention selected by this experiment still uses FA3-NV.
export NVTE_FUSED_ATTN=0

"$OMNI_FA3_PYTHON" - <<'PY'
import sys

import flash_attn
import flash_attn_3_nv
import natten
import torch
import transformer_engine

expected = {
    "python": "3.13",
    "torch": "2.10.",
    "cuda": "12.8",
    "transformer_engine": "2.12",
    "flash_attn": "2.7.4.post1",
    "flash_attn_3_nv": "1.0.3",
    "natten": "0.21.6",
}
actual = {
    "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "transformer_engine": transformer_engine.__version__,
    "flash_attn": flash_attn.__version__,
    "flash_attn_3_nv": flash_attn_3_nv.__version__,
    "natten": natten.__version__,
}
for key, prefix in expected.items():
    if actual[key] is None or not actual[key].startswith(prefix):
        raise RuntimeError(f"FA3 environment mismatch for {key}: expected {prefix!r}, got {actual[key]!r}")
print(
    "OmniDreams FA3 environment:"
    f" Python {actual['python']}, torch {actual['torch']}, CUDA {actual['cuda']},"
    f" TE {actual['transformer_engine']}, FA3 {actual['flash_attn_3_nv']}"
)
PY

unset _FA3_ENV_DIR _FA3_REPO_ROOT _FA3_CACHE_ROOT _FA3_SITE_PACKAGES _FA3_NVIDIA_LIBS
