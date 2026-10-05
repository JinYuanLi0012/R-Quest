#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
source ./configs/rquest.sh
if [ "${1:-}" = "--dry-run" ]; then
    python3 scripts/preflight.py --stage rquest --config-only
    exit 0
fi
for arg in "$@"; do
    if [ "$arg" != "--resume" ]; then
        echo 'Usage: bash train_rquest.sh [--dry-run | --resume]' >&2; exit 2
    fi
done
source ./scripts/prepare_validity_data.sh
# Download the complete frozen Base snapshot, not just generation_config.json.
python3 scripts/prepare_models.py --base "$BASE_MODEL" --solver "$VALIDITY_RZERO_INITIAL_SOLVER"
python3 scripts/preflight.py --stage rquest
bash scripts/main.sh "$@" "$BASE_MODEL" "$MODEL_ABBR"
