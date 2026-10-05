#!/usr/bin/env bash
# Source after env.sh and configs/rquest.sh. Preserve each split's metadata schema.
if [ -d "$VALIDITY_DATASET" ] && [ -f "$VALIDITY_DATASET/train.parquet" ] && [ -f "$VALIDITY_DATASET/validation.parquet" ]; then
    python3 methods/validity_rl/prepare_dataset.py --dataset "$VALIDITY_DATASET"
else
    RQUEST_PREPARED_DATA="${STORAGE_PATH}/data/validity_clean_v1"
    python3 methods/validity_rl/prepare_dataset.py --dataset "$VALIDITY_DATASET" --output-dir "$RQUEST_PREPARED_DATA"
    export VALIDITY_DATASET="$RQUEST_PREPARED_DATA"
fi
export TERRA_REPLAY_DATASET="$VALIDITY_DATASET"
