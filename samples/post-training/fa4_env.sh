#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Validate and configure the commit-pinned FA4 CuTeDSL overlay after fa3_env.sh.
#
# Keep this research-only package set outside OMNI_FA3_VENV.  CUTLASS DSL
# 4.6.0.dev0 declares protobuf<7, while OmniDreams requires protobuf>=7.35,<8.
# The tested overlay deliberately uses the base environment's protobuf 7 via
# --no-deps; callers must explicitly acknowledge that unsupported metadata
# combination.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "ERROR: source this file instead of executing it." >&2
  exit 2
fi

if [[ -z "${OMNI_FA3_PYTHON:-}" || ! -x "$OMNI_FA3_PYTHON" ]]; then
  echo "ERROR: source fa3_env.sh before fa4_env.sh." >&2
  return 2
fi

if [[ -z "${OMNI_FA4_OVERLAY:-}" || ! -d "$OMNI_FA4_OVERLAY" ]]; then
  echo "ERROR: set OMNI_FA4_OVERLAY to the separate FA4 --target directory." >&2
  echo "Do not install FA4/CUTLASS DSL into OMNI_FA3_VENV; see samples/post-training/README.md." >&2
  return 2
fi

_FA4_OVERLAY_REAL="$(realpath "$OMNI_FA4_OVERLAY")"
_FA3_VENV_REAL="$(realpath "$OMNI_FA3_VENV")"
case "$_FA4_OVERLAY_REAL/" in
  "$_FA3_VENV_REAL/"*)
    echo "ERROR: OMNI_FA4_OVERLAY must be outside OMNI_FA3_VENV." >&2
    return 2
    ;;
esac

if [[ "${OMNI_FA4_ACCEPT_UNSUPPORTED_PROTOBUF7:-0}" != "1" ]]; then
  echo "ERROR: CUTLASS DSL 4.6.0.dev0 declares protobuf<7, but OmniDreams requires protobuf>=7.35." >&2
  echo "Use an isolated --target overlay, review the known dependency conflict, then set" >&2
  echo "OMNI_FA4_ACCEPT_UNSUPPORTED_PROTOBUF7=1 for this research-only experiment." >&2
  return 2
fi

OMNI_FA4_OVERLAY="$_FA4_OVERLAY_REAL"
export OMNI_FA4_OVERLAY
export PYTHONPATH="$OMNI_FA4_OVERLAY:$PYTHONPATH"

: "${FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED:=1}"
: "${FLASH_ATTENTION_CUTE_DSL_CACHE_DIR:=${OMNI_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}}/flash-attn-cute-dsl}"
mkdir -p "$FLASH_ATTENTION_CUTE_DSL_CACHE_DIR"
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED FLASH_ATTENTION_CUTE_DSL_CACHE_DIR

"$OMNI_FA3_PYTHON" - <<'PY'
import os
import site
from importlib.metadata import PackageNotFoundError, distribution, distributions, version
from pathlib import Path

expected = {
    "flash-attn-4": "0.0.1.dev1+g14c377950",
    "nvidia-cutlass-dsl": "4.6.0.dev0",
    "nvidia-cutlass-dsl-libs-base": "4.6.0.dev0",
    "apache-tvm-ffi": "0.1.13rc2",
    "torch-c-dlpack-ext": "0.1.5",
    "quack-kernels": "0.5.3",
    "cuda-python": "12.9.4",
    "cuda-bindings": "12.9.4",
    "cuda-pathfinder": "1.4.0",
}
overlay = Path(os.environ["OMNI_FA4_OVERLAY"]).resolve()
for distribution_name, required in expected.items():
    try:
        actual = version(distribution_name)
    except PackageNotFoundError as error:
        raise SystemExit(
            f"ERROR: missing {distribution_name}=={required}; follow the FA4 "
            "installation commands in samples/post-training/README.md"
        ) from error
    if actual != required:
        raise SystemExit(
            f"ERROR: FA4 environment mismatch for {distribution_name}: "
            f"expected {required!r}, got {actual!r}"
        )
    location = Path(distribution(distribution_name).locate_file("")).resolve()
    if not location.is_relative_to(overlay):
        raise SystemExit(
            f"ERROR: {distribution_name} resolved from {location}, not the isolated "
            f"OMNI_FA4_OVERLAY {overlay}"
        )

base_site = Path(site.getsitepackages()[0]).resolve()
base_distribution_names = {
    (item.metadata.get("Name") or "").lower().replace("_", "-")
    for item in distributions(path=[str(base_site)])
}
for forbidden in ("flash-attn-4", "nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-base"):
    if forbidden in base_distribution_names:
        raise SystemExit(
            f"ERROR: {forbidden} is installed in shared OMNI_FA3_VENV; "
            "remove it and keep the package in OMNI_FA4_OVERLAY only"
        )
if (base_site / "flash_attn" / "cute").exists():
    raise SystemExit(
        "ERROR: shared OMNI_FA3_VENV contains flash_attn/cute; remove the "
        "contaminating FA4 files before using the isolated overlay"
    )

from google.protobuf import __version__ as protobuf_version

protobuf_parts = protobuf_version.split(".")
if len(protobuf_parts) < 2 or not all(part.isdigit() for part in protobuf_parts[:2]):
    raise SystemExit(f"ERROR: cannot validate protobuf version {protobuf_version!r}")
protobuf_major, protobuf_minor = map(int, protobuf_parts[:2])
if protobuf_major != 7 or protobuf_minor < 35:
    raise SystemExit(
        "ERROR: the FA4 experiment must retain OmniDreams protobuf>=7.35,<8; "
        f"got {protobuf_version!r}"
    )

# Process the overlay's .pth files. In particular, CUTLASS DSL stores the
# importable ``cutlass`` package below nvidia_cutlass_dsl/python_packages.
site.addsitedir(str(overlay))

import cutlass
import flash_attn

overlay_flash_attn = overlay / "flash_attn"
if not overlay_flash_attn.is_dir():
    raise SystemExit(f"ERROR: FA4 overlay is missing {overlay_flash_attn}")
flash_attn.__path__.insert(0, str(overlay_flash_attn))

import flash_attn.cute
import flash_attn.cute.block_sparsity
import flash_attn.cute.interface
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

for module in (
    cutlass,
    flash_attn.cute,
    flash_attn.cute.block_sparsity,
    flash_attn.cute.interface,
):
    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(overlay):
        raise SystemExit(
            f"ERROR: {module.__name__} resolved from {module_path}, not "
            f"OMNI_FA4_OVERLAY {overlay}"
        )

if not callable(_flash_attn_fwd) or not callable(_flash_attn_bwd):
    raise RuntimeError("The commit-pinned FA4 private forward/backward API is unavailable")
print(
    "OmniDreams FA4 environment:"
    f" flash-attn-4 {expected['flash-attn-4']},"
    f" nvidia-cutlass-dsl {expected['nvidia-cutlass-dsl']},"
    f" protobuf {protobuf_version} (unsupported upstream metadata override)"
)
PY

unset _FA4_OVERLAY_REAL _FA3_VENV_REAL
