#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Backward-compatible name for the original four-GPU CP=4/FSDP=4 examples.
# Every default can be overridden; new callers should use run_cp_attention_ab.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CP_SIZE="${CP_SIZE:-4}"

exec "$SCRIPT_DIR/run_cp_attention_ab.sh" "$@"
