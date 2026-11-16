#!/usr/bin/env bash
set -euo pipefail
FLOWHMR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$FLOWHMR_ROOT"
# ROS and other active environments can otherwise leak plugins/imports into the venv.
unset PYTHONPATH PYTHONHOME
export HF_HOME="$FLOWHMR_ROOT/.cache/huggingface"
export TORCH_HOME="$FLOWHMR_ROOT/.cache/torch"
export XDG_CACHE_HOME="$FLOWHMR_ROOT/.cache"
export UV_CACHE_DIR="$FLOWHMR_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$FLOWHMR_ROOT/.runtime/python"
export UV_LINK_MODE=copy
export TMPDIR="$FLOWHMR_ROOT/.cache/tmp"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
# Match the Windows launcher: bound attention memory for long motion sequences.
export FLOWHMR_ATTENTION_BACKEND="${FLOWHMR_ATTENTION_BACKEND:-query_batched}"
export FLOWHMR_ATTENTION_QUERY_CHUNK_SIZE="${FLOWHMR_ATTENTION_QUERY_CHUNK_SIZE:-256}"
mkdir -p "$TMPDIR" "$FLOWHMR_ROOT/logs"
FLOWHMR_PYTHON="$FLOWHMR_ROOT/.venv/bin/python"
