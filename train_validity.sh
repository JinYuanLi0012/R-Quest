#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
source ./configs/rquest.sh
export VALIDITY_RZERO_ENABLED=0
case "${1:-}" in
    --dry-run) export VALIDITY_DRY_RUN=1 ;;
    --resume) export VALIDITY_RESUME=1 ;;
    "") ;;
    *) echo 'Usage: bash train_validity.sh [--dry-run | --resume]' >&2; exit 2 ;;
esac
if [ "$#" -gt 1 ]; then
    echo 'Expected at most one option.' >&2; exit 2
fi
if [ "${VALIDITY_DRY_RUN:-0}" != 1 ]; then
    source ./scripts/prepare_validity_data.sh
    python3 scripts/preflight.py --stage validity
fi
bash methods/validity_rl/train_validity_grpo.sh
if [ "${VALIDITY_DRY_RUN:-0}" != 1 ]; then
    python3 scripts/model_merger.py --local_dir "${VALIDITY_SAVE_PATH:-${STORAGE_PATH}/models/${VALIDITY_EXPERIMENT_NAME}}/global_step_15/actor"
    python3 scripts/validate_hf_checkpoint.py "${VALIDITY_SAVE_PATH:-${STORAGE_PATH}/models/${VALIDITY_EXPERIMENT_NAME}}/global_step_15/actor/huggingface"
fi
