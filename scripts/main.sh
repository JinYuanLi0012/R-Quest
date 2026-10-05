#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/main.sh [--resume] [--rounds N] BASE_MODEL MODEL_ABBR

Examples:
  bash scripts/main.sh Qwen/Qwen3-4B-Base rquest_qwen3_4b_k8
  bash scripts/main.sh --resume Qwen/Qwen3-4B-Base rquest_qwen3_4b_k8

The first invocation creates an atomic pipeline state. Re-run with --resume to
skip completed stages and continue an interrupted GRPO stage from the latest
complete per-step checkpoint.
EOF
}

RESUME=0
NUM_ROUNDS=${RZERO_NUM_ROUNDS:-5}
POSITIONAL=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --resume) RESUME=1; shift ;;
        --rounds) NUM_ROUNDS=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done

if [ "${#POSITIONAL[@]}" -ne 2 ]; then
    usage >&2
    exit 2
fi

BASE_MODEL=${POSITIONAL[0]}
MODEL_ABBR=${POSITIONAL[1]}
VALIDITY_RZERO_ENABLED=1
if [ "$VALIDITY_RZERO_ENABLED" = "1" ]; then
    : "${VALIDITY_RZERO_INITIAL_SOLVER:?set VALIDITY_RZERO_INITIAL_SOLVER}"
    : "${TERRA_REPLAY_DATASET:?set TERRA_REPLAY_DATASET}"
    : "${TERRA_REPLAY_RATIO:?set TERRA_REPLAY_RATIO}"
    VALIDITY_RZERO_DIVERSITY_MODE=semantic_novelty_gate
    if [ "$VALIDITY_RZERO_DIVERSITY_MODE" = "semantic_novelty_gate" ]; then
        VALIDITY_RZERO_SEMANTIC_WORKER_BATCH_SIZE=${VALIDITY_RZERO_SEMANTIC_WORKER_BATCH_SIZE:-8192}
        if ! [[ "$VALIDITY_RZERO_SEMANTIC_WORKER_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
            echo "VALIDITY_RZERO_SEMANTIC_WORKER_BATCH_SIZE must be a positive integer" >&2
            exit 2
        fi
        export VALIDITY_RZERO_SEMANTIC_WORKER_BATCH_SIZE
    fi
    if [ "$VALIDITY_RZERO_DIVERSITY_MODE" = "semantic_novelty_gate" ]; then
        VALIDITY_RZERO_NOVELTY_K=${VALIDITY_RZERO_NOVELTY_K:-8}
        VALIDITY_RZERO_NOVELTY_SEED=${VALIDITY_RZERO_NOVELTY_SEED:-43}
        if ! [[ "$VALIDITY_RZERO_NOVELTY_K" =~ ^[1-9][0-9]*$ ]]; then
            echo "VALIDITY_RZERO_NOVELTY_K must be a positive integer" >&2
            exit 2
        fi
        if ! [[ "$VALIDITY_RZERO_NOVELTY_SEED" =~ ^[0-9]+$ ]]; then
            echo "VALIDITY_RZERO_NOVELTY_SEED must be a nonnegative integer" >&2
            exit 2
        fi
        export VALIDITY_RZERO_NOVELTY_K VALIDITY_RZERO_NOVELTY_SEED
    fi
    export VALIDITY_RZERO_DIVERSITY_MODE
fi
if ! [[ "$NUM_ROUNDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--rounds must be a positive integer" >&2
    exit 2
fi
: "${STORAGE_PATH:?source env.sh or export STORAGE_PATH first}"

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [ "$VALIDITY_RZERO_ENABLED" = "1" ]; then
    python3 scripts/validate_hf_checkpoint.py "$VALIDITY_RZERO_INITIAL_SOLVER" >/dev/null
    python3 - "$TERRA_REPLAY_RATIO" <<'PY'
import sys
value = float(sys.argv[1])
if not 0.0 < value < 1.0:
    raise SystemExit("TERRA_REPLAY_RATIO must be in (0, 1)")
PY
fi

# Reproducible RQuest runtime defaults. All remain environment-overridable, but
# the fingerprint prevents changing them while resuming the same MODEL_ABBR.
export QUESTIONER_TRAIN_GPU_IDS=${QUESTIONER_TRAIN_GPU_IDS:-0,1}
export VLLM_GPU_IDS=${VLLM_GPU_IDS:-2,3}
export VLLM_PORT_BASE=${VLLM_PORT_BASE:-5000}
export QUESTION_GPU_IDS=${QUESTION_GPU_IDS:-0,1,2,3}
if [ "$VALIDITY_RZERO_ENABLED" = "1" ] && [ "$VALIDITY_RZERO_DIVERSITY_MODE" = "semantic_novelty_gate" ]; then
    export VALIDITY_RZERO_SEMANTIC_GPU_IDS=${VALIDITY_RZERO_SEMANTIC_GPU_IDS:-${QUESTIONER_TRAIN_GPU_IDS},${VLLM_GPU_IDS}}
    export VALIDITY_RZERO_SEMANTIC_GPU_MEMORY_UTILIZATION=${VALIDITY_RZERO_SEMANTIC_GPU_MEMORY_UTILIZATION:-0.80}
    python3 - "$VALIDITY_RZERO_SEMANTIC_GPU_IDS" "$VALIDITY_RZERO_SEMANTIC_GPU_MEMORY_UTILIZATION" <<'PY'
import sys
gpu_ids = [value.strip() for value in sys.argv[1].split(",") if value.strip()]
if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
    raise SystemExit("VALIDITY_RZERO_SEMANTIC_GPU_IDS must contain unique GPU IDs")
utilization = float(sys.argv[2])
if not 0.0 < utilization < 1.0:
    raise SystemExit("VALIDITY_RZERO_SEMANTIC_GPU_MEMORY_UTILIZATION must be in (0, 1)")
PY
fi
export QUESTIONER_MAX_STEPS=${QUESTIONER_MAX_STEPS:-5}
export QUESTIONER_MERGE_STEP=${QUESTIONER_MERGE_STEP:-5}
export QUESTIONER_SAVE_FREQ=${QUESTIONER_SAVE_FREQ:-1}
export QUESTIONER_SAVE_LIMIT=${QUESTIONER_SAVE_LIMIT:-1}
export QUESTIONER_ROLLOUT_BATCH_SIZE=${QUESTIONER_ROLLOUT_BATCH_SIZE:-512}
export QUESTIONER_ROLLOUT_N=${QUESTIONER_ROLLOUT_N:-4}
export QUESTIONER_GLOBAL_BATCH_SIZE=${QUESTIONER_GLOBAL_BATCH_SIZE:-4}
export QUESTIONER_MAX_RESPONSE_LENGTH=${QUESTIONER_MAX_RESPONSE_LENGTH:-4096}
export SOLVER_GENERATE_SAMPLES=${SOLVER_GENERATE_SAMPLES:-1000}
export QUESTION_EVAL_TIMEOUT_SECONDS=${QUESTION_EVAL_TIMEOUT_SECONDS:-14400}
export SOLVER_MAX_STEPS=${SOLVER_MAX_STEPS:-15}
export SOLVER_MERGE_STEP=${SOLVER_MERGE_STEP:-15}
export SOLVER_SAVE_FREQ=${SOLVER_SAVE_FREQ:-1}
export SOLVER_SAVE_LIMIT=${SOLVER_SAVE_LIMIT:-1}
export SOLVER_MAX_RESPONSE_LENGTH=${SOLVER_MAX_RESPONSE_LENGTH:-4096}
export SOLVER_TOTAL_EPOCHS=${SOLVER_TOTAL_EPOCHS:-100}
export SOLVER_ROLLOUT_BATCH_SIZE=${SOLVER_ROLLOUT_BATCH_SIZE:-512}
export SOLVER_VAL_FREQ=${SOLVER_VAL_FREQ:-4}
export SOLVER_SKIP_MERGE=0
export QUESTIONER_SKIP_MERGE=0

if [ "$QUESTIONER_MAX_STEPS" -ne "$QUESTIONER_MERGE_STEP" ]; then
    echo "Resume-safe base pipeline requires QUESTIONER_MAX_STEPS=QUESTIONER_MERGE_STEP" >&2
    exit 2
fi
if [ "$SOLVER_MAX_STEPS" -ne "$SOLVER_MERGE_STEP" ]; then
    echo "Resume-safe base pipeline requires SOLVER_MAX_STEPS=SOLVER_MERGE_STEP" >&2
    exit 2
fi

QUESTIONER_GPU_COUNT=$(awk -F',' '{print NF}' <<< "$QUESTIONER_TRAIN_GPU_IDS")
SOLVER_GPU_COUNT=$(awk -F',' '{print NF}' <<< "$QUESTION_GPU_IDS")
RUN_ROOT=${RZERO_RUN_ROOT:-$STORAGE_PATH/rzero_runs/$MODEL_ABBR}
if [ "$VALIDITY_RZERO_ENABLED" = "1" ]; then
    export VALIDITY_RZERO_ARTIFACT_DIR=${VALIDITY_RZERO_ARTIFACT_DIR:-$RUN_ROOT/artifacts}
fi
STATE_DIR=$RUN_ROOT/state
STATE_FILE=$STATE_DIR/run_state.json
SUMMARY_FILE=$RUN_ROOT/summary.json

FINGERPRINT_EXTRA=()
if [ "$VALIDITY_RZERO_ENABLED" = "1" ]; then
    FINGERPRINT_EXTRA+=(
        --field "validity_rzero_enabled=1"
        --field "initial_solver=${VALIDITY_RZERO_INITIAL_SOLVER}"
        --field "terra_replay_dataset=${TERRA_REPLAY_DATASET}"
        --field "terra_replay_config=${TERRA_REPLAY_CONFIG:-default}"
        --field "terra_replay_ratio=${TERRA_REPLAY_RATIO}"
        --field "terra_replay_seed=${TERRA_REPLAY_SEED:-1}"
        --field "solver_rollout_batch_size=${SOLVER_ROLLOUT_BATCH_SIZE}"
        --field "validity_diversity_mode=${VALIDITY_RZERO_DIVERSITY_MODE}"
    )
    if [ "$VALIDITY_RZERO_DIVERSITY_MODE" = "semantic_novelty_gate" ]; then
        FINGERPRINT_EXTRA+=(
            --field "semantic_model=${VALIDITY_RZERO_SEMANTIC_MODEL:-Qwen/Qwen3-4B-Base}"
            --field "semantic_novelty_k=${VALIDITY_RZERO_NOVELTY_K}"
            --field "semantic_novelty_seed=${VALIDITY_RZERO_NOVELTY_SEED}"
            --field "semantic_prompt_protocol=rquest-novelty-v1"
            --field "semantic_sampling_protocol=qwen3-novelty-max1024-seed42"
            --field "semantic_pair_orientation=candidate_then_reference_v1"
            --field "semantic_worker_batch_size=${VALIDITY_RZERO_SEMANTIC_WORKER_BATCH_SIZE}"
            --field "semantic_gpu_ids=${VALIDITY_RZERO_SEMANTIC_GPU_IDS}"
            --field "semantic_gpu_memory_utilization=${VALIDITY_RZERO_SEMANTIC_GPU_MEMORY_UTILIZATION}"
            --field "semantic_questioner_gpu_barrier=old_ref_values_complete_v1"
            --field "semantic_retry_policy=deferred_one_retry_v1"
            --field "semantic_prefix_cache=enabled"
            --field "semantic_novelty_treatment=any_same_hard_gate_v1"
        )
    fi
fi

if [ -e "$STATE_FILE" ] && [ "$RESUME" != "1" ]; then
    echo "Run already exists at $RUN_ROOT. Re-run with --resume or choose a new MODEL_ABBR." >&2
    exit 2
fi
mkdir -p "$STATE_DIR"

FINGERPRINT=$(python3 scripts/rzero_pipeline_state.py init \
    --state "$STATE_FILE" \
    --field "pipeline_version=1" \
    --field "base_model=$BASE_MODEL" \
    "${FINGERPRINT_EXTRA[@]}" \
    --field "model_abbr=$MODEL_ABBR" \
    --field "num_rounds=$NUM_ROUNDS" \
    --field "questioner_train_gpu_ids=$QUESTIONER_TRAIN_GPU_IDS" \
    --field "solver_feedback_gpu_ids=$VLLM_GPU_IDS" \
    --field "question_gpu_ids=$QUESTION_GPU_IDS" \
    --field "questioner_max_steps=$QUESTIONER_MAX_STEPS" \
    --field "questioner_merge_step=$QUESTIONER_MERGE_STEP" \
    --field "questioner_save_freq=$QUESTIONER_SAVE_FREQ" \
    --field "questioner_rollout_batch_size=$QUESTIONER_ROLLOUT_BATCH_SIZE" \
    --field "questioner_rollout_n=$QUESTIONER_ROLLOUT_N" \
    --field "questioner_global_batch_size=$QUESTIONER_GLOBAL_BATCH_SIZE" \
    --field "questioner_max_response_length=$QUESTIONER_MAX_RESPONSE_LENGTH" \
    --field "solver_generate_samples=$SOLVER_GENERATE_SAMPLES" \
    --field "solver_max_steps=$SOLVER_MAX_STEPS" \
    --field "solver_merge_step=$SOLVER_MERGE_STEP" \
    --field "solver_save_freq=$SOLVER_SAVE_FREQ" \
    --field "solver_max_response_length=$SOLVER_MAX_RESPONSE_LENGTH" \
    --field "solver_total_epochs=$SOLVER_TOTAL_EPOCHS" \
    --field "solver_val_freq=$SOLVER_VAL_FREQ" \
    --field "dataset_score_range=${SOLVER_UPLOAD_MIN_SCORE:-0.3}:${SOLVER_UPLOAD_MAX_SCORE:-0.8}")

marker() {
    printf '%s/stages/%s/_SUCCESS.json' "$STATE_DIR" "$1"
}

stage_done() {
    local stage=$1
    shift
    local args=(check --marker "$(marker "$stage")" --fingerprint "$FINGERPRINT")
    local required
    for required in "$@"; do
        args+=(--require "$required")
    done
    python3 scripts/rzero_pipeline_state.py "${args[@]}"
}

complete_stage() {
    local stage=$1
    shift
    local args=(complete --state "$STATE_FILE" --marker "$(marker "$stage")" \
        --stage "$stage" --fingerprint "$FINGERPRINT")
    local artifact
    for artifact in "$@"; do
        args+=(--artifact "$artifact")
    done
    python3 scripts/rzero_pipeline_state.py "${args[@]}"
}

resolve_latest_checkpoint() {
    local root=$1 world_size=$2 status
    LATEST_CHECKPOINT=
    set +e
    LATEST_CHECKPOINT=$(python3 scripts/find_resume_checkpoint.py --root "$root" --world-size "$world_size")
    status=$?
    set -e
    case "$status" in
        0) return 0 ;;
        1) LATEST_CHECKPOINT=; return 0 ;;
        *) echo "Refusing to resume from an invalid checkpoint under $root" >&2; return 2 ;;
    esac
}

checkpoint_valid() {
    python3 scripts/validate_hf_checkpoint.py "$1" >/dev/null 2>&1
}

dataset_receipt_valid() {
    python3 - "$1" "$2" <<'PY' >/dev/null 2>&1
import json
import os
import sys
from pathlib import Path

receipt = Path(sys.argv[1])
payload = json.loads(receipt.read_text(encoding="utf-8"))
if payload.get("dataset_path") != sys.argv[2] or not Path(sys.argv[2]).is_file() or int(payload.get("filtered_count", 0)) <= 0:
    raise SystemExit(1)
if os.environ.get("VALIDITY_RZERO_ENABLED") == "1":
    required = {"rzero_sample_count", "terra_replay_sample_count", "actual_replay_ratio", "phase_b_audit"}
    if required.difference(payload) or not Path(payload["phase_b_audit"]).is_file():
        raise SystemExit(1)
PY
}

CURRENT_QUESTIONER=$BASE_MODEL
CURRENT_SOLVER=$VALIDITY_RZERO_INITIAL_SOLVER

echo "R-Quest: run=$MODEL_ABBR rounds=$NUM_ROUNDS initial_solver=$CURRENT_SOLVER candidates_per_round=$((SOLVER_GENERATE_SAMPLES * SOLVER_GPU_COUNT))"

for ((round=1; round<=NUM_ROUNDS; round++)); do
    echo "================ round $round / $NUM_ROUNDS ================"

    QUESTIONER_NAME=${MODEL_ABBR}_questioner_v${round}
    QUESTIONER_DIR=$STORAGE_PATH/models/$QUESTIONER_NAME
    QUESTIONER_HF=$QUESTIONER_DIR/global_step_${QUESTIONER_MERGE_STEP}/actor/huggingface
    QUESTIONER_STAGE=round_${round}/questioner
    if stage_done "$QUESTIONER_STAGE" "$QUESTIONER_HF"; then
        echo "[resume] skip completed stage $QUESTIONER_STAGE"
    else
        if [ "$RESUME" = "1" ] && checkpoint_valid "$QUESTIONER_HF"; then
            echo "[resume] recovered merged artifact for $QUESTIONER_STAGE"
        else
            QUESTIONER_LOAD_CHECKPOINT=
            if [ "$RESUME" = "1" ]; then
                resolve_latest_checkpoint "$QUESTIONER_DIR" "$QUESTIONER_GPU_COUNT"
                QUESTIONER_LOAD_CHECKPOINT=$LATEST_CHECKPOINT
            elif [ -e "$QUESTIONER_DIR" ]; then
                echo "Existing untracked output at $QUESTIONER_DIR. Use --resume or a new MODEL_ABBR." >&2
                exit 2
            fi

            if [ -n "$QUESTIONER_LOAD_CHECKPOINT" ]; then
                QUESTIONER_RESUME_STEP=${QUESTIONER_LOAD_CHECKPOINT##*global_step_}
                echo "[resume] questioner round $round from completed step $QUESTIONER_RESUME_STEP"
                if [ "$QUESTIONER_RESUME_STEP" -ge "$QUESTIONER_MAX_STEPS" ]; then
                    python3 scripts/model_merger.py --local_dir "$QUESTIONER_LOAD_CHECKPOINT/actor"
                else
                    export QUESTIONER_LOAD_CHECKPOINT
                    QUESTIONER_OUTPUT_DIR="$QUESTIONER_DIR" bash scripts/questioner_train_penalty.sh \
                        "$CURRENT_SOLVER" "$CURRENT_QUESTIONER" "$QUESTIONER_NAME"
                    unset QUESTIONER_LOAD_CHECKPOINT
                fi
            else
                QUESTIONER_OUTPUT_DIR="$QUESTIONER_DIR" bash scripts/questioner_train_penalty.sh \
                    "$CURRENT_SOLVER" "$CURRENT_QUESTIONER" "$QUESTIONER_NAME"
            fi
            python3 scripts/validate_hf_checkpoint.py "$QUESTIONER_HF" >/dev/null
        fi
        complete_stage "$QUESTIONER_STAGE" "$QUESTIONER_HF"
    fi
    CURRENT_QUESTIONER=$QUESTIONER_HF

    SOLVER_NAME=${MODEL_ABBR}_solver_v${round}
    DATASET_STAGE=round_${round}/dataset
    DATASET_RECEIPT=$RUN_ROOT/datasets/round_${round}.json
    DATASET_ARTIFACTS=("$DATASET_RECEIPT" "${DATASET_RECEIPT%.json}.parquet")
    if [ "$VALIDITY_RZERO_ENABLED" = "1" ]; then
        DATASET_ARTIFACTS+=("${DATASET_RECEIPT%.json}_phase_b.jsonl")
    fi
    if stage_done "$DATASET_STAGE" "${DATASET_ARTIFACTS[@]}"; then
        echo "[resume] skip completed stage $DATASET_STAGE"
    else
        if [ "$RESUME" = "1" ] && dataset_receipt_valid "$DATASET_RECEIPT" "${DATASET_RECEIPT%.json}.parquet"; then
            echo "[resume] recovered local dataset receipt for $DATASET_STAGE"
        elif [ "$RESUME" != "1" ] && [ -e "$(marker "$DATASET_STAGE")" ]; then
            echo "Existing dataset state for $DATASET_STAGE; use --resume" >&2
            exit 2
        else
            SOLVER_DATASET_RECEIPT="$DATASET_RECEIPT" \
            SOLVER_PREPARE_ONLY=1 SOLVER_DATASET_READY=0 SOLVER_SKIP_MERGE=1 \
                bash scripts/solver_train.sh "$CURRENT_SOLVER" "$CURRENT_QUESTIONER" "$SOLVER_NAME"
        fi
        complete_stage "$DATASET_STAGE" "${DATASET_ARTIFACTS[@]}"
    fi

    export RQUEST_SOLVER_DATASET="${DATASET_RECEIPT%.json}.parquet"
    SOLVER_DIR=$STORAGE_PATH/models/$SOLVER_NAME
    SOLVER_HF=$SOLVER_DIR/global_step_${SOLVER_MERGE_STEP}/actor/huggingface
    SOLVER_STAGE=round_${round}/solver
    if stage_done "$SOLVER_STAGE" "$SOLVER_HF"; then
        echo "[resume] skip completed stage $SOLVER_STAGE"
    else
        if [ "$RESUME" = "1" ] && checkpoint_valid "$SOLVER_HF"; then
            echo "[resume] recovered merged artifact for $SOLVER_STAGE"
        else
            SOLVER_LOAD_CHECKPOINT=
            if [ "$RESUME" = "1" ]; then
                resolve_latest_checkpoint "$SOLVER_DIR" "$SOLVER_GPU_COUNT"
                SOLVER_LOAD_CHECKPOINT=$LATEST_CHECKPOINT
            elif [ -e "$SOLVER_DIR" ]; then
                echo "Existing untracked output at $SOLVER_DIR. Use --resume or a new MODEL_ABBR." >&2
                exit 2
            fi

            if [ -n "$SOLVER_LOAD_CHECKPOINT" ]; then
                SOLVER_RESUME_STEP=${SOLVER_LOAD_CHECKPOINT##*global_step_}
                echo "[resume] solver round $round from completed step $SOLVER_RESUME_STEP"
                if [ "$SOLVER_RESUME_STEP" -ge "$SOLVER_MAX_STEPS" ]; then
                    python3 scripts/model_merger.py --local_dir "$SOLVER_LOAD_CHECKPOINT/actor"
                else
                    export SOLVER_LOAD_CHECKPOINT
                    SOLVER_DATASET_READY=1 bash scripts/solver_train.sh \
                        "$CURRENT_SOLVER" "$CURRENT_QUESTIONER" "$SOLVER_NAME"
                    unset SOLVER_LOAD_CHECKPOINT
                fi
            else
                SOLVER_DATASET_READY=1 bash scripts/solver_train.sh \
                    "$CURRENT_SOLVER" "$CURRENT_QUESTIONER" "$SOLVER_NAME"
            fi
            python3 scripts/validate_hf_checkpoint.py "$SOLVER_HF" >/dev/null
        fi
        complete_stage "$SOLVER_STAGE" "$SOLVER_HF"
    fi
    CURRENT_SOLVER=$SOLVER_HF

done

python3 - "$SUMMARY_FILE" "$BASE_MODEL" "$CURRENT_QUESTIONER" "$CURRENT_SOLVER" "$NUM_ROUNDS" "$VALIDITY_RZERO_ENABLED" "${VALIDITY_RZERO_INITIAL_SOLVER:-$BASE_MODEL}" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
payload = {
    "base_model": sys.argv[2],
    "final_questioner": sys.argv[3],
    "final_solver": sys.argv[4],
    "rounds": int(sys.argv[5]),
}
if sys.argv[6] == "1":
    payload["initial_solver"] = sys.argv[7]
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY

echo "Completed RQuest: $RUN_ROOT"
echo "Final solver: $CURRENT_SOLVER"
