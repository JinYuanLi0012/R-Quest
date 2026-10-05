#!/usr/bin/env bash
# Source from any directory after activating the Python environment.
RQUEST_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export STORAGE_PATH=${STORAGE_PATH:-${RQUEST_ROOT}/runs}
export HF_HOME=${HF_HOME:-${STORAGE_PATH}/cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export WANDB_DIR=${WANDB_DIR:-${STORAGE_PATH}/wandb}
export TMPDIR=${TMPDIR:-${STORAGE_PATH}/tmp}
export PYTHONPATH="${RQUEST_ROOT}:${PYTHONPATH:-}"
export VLLM_DISABLE_COMPILE_CACHE=1
mkdir -p "$STORAGE_PATH" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$WANDB_DIR" "$TMPDIR" "$STORAGE_PATH/temp_results" "$STORAGE_PATH/generated_question"
