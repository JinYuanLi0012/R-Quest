#!/bin/bash
set -e

solver_model_path=$1
questioner_model_path=$2
experiment_name=$3

mkdir -p logs
if [ "${VALIDITY_RZERO_ENABLED:-0}" = "1" ] && [ -n "${VALIDITY_RZERO_ARTIFACT_DIR:-}" ]; then
    SOLVER_ARTIFACT_DIR=${VALIDITY_RZERO_ARTIFACT_DIR}/${experiment_name}
    mkdir -p "$SOLVER_ARTIFACT_DIR"
    SOLVER_LOG_FILE=${SOLVER_LOG_FILE:-${SOLVER_ARTIFACT_DIR}/solver_$(date +%Y%m%d_%H%M%S).log}
fi
SOLVER_LOG_FILE=${SOLVER_LOG_FILE:-logs/solver_${experiment_name}_$(date +%Y%m%d_%H%M%S).log}
exec > >(tee -a "$SOLVER_LOG_FILE") 2>&1
echo "logging to $SOLVER_LOG_FILE"
echo "save_path: $experiment_name"

echo $STORAGE_PATH

echo "start train solver $experiment_name $solver_model_path $questioner_model_path" 

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export VLLM_DISABLE_COMPILE_CACHE=1
export QUESTION_GPU_IDS=${QUESTION_GPU_IDS:-0,1,2,3}
QUESTION_NUM_SHARDS=$(echo "$QUESTION_GPU_IDS" | awk -F',' '{print NF}')
export QUESTION_NUM_SHARDS
echo "using QUESTION_GPU_IDS=$QUESTION_GPU_IDS, QUESTION_NUM_SHARDS=$QUESTION_NUM_SHARDS"
SOLVER_GENERATE_SAMPLES=${SOLVER_GENERATE_SAMPLES:-1000}
QUESTION_EVAL_TIMEOUT_SECONDS=${QUESTION_EVAL_TIMEOUT_SECONDS:-14400}
export QUESTION_EVAL_TIMEOUT_SECONDS
SOLVER_MAX_RESPONSE_LENGTH=${SOLVER_MAX_RESPONSE_LENGTH:-4096}
SOLVER_TOTAL_EPOCHS=${SOLVER_TOTAL_EPOCHS:-100}
SOLVER_MAX_STEPS=${SOLVER_MAX_STEPS:-20}
SOLVER_VAL_FREQ=${SOLVER_VAL_FREQ:-4}
SOLVER_MERGE_STEP=${SOLVER_MERGE_STEP:-15}
SOLVER_UPLOAD_MAX_SCORE=${SOLVER_UPLOAD_MAX_SCORE:-0.8}
SOLVER_UPLOAD_MIN_SCORE=${SOLVER_UPLOAD_MIN_SCORE:-0.3}
SOLVER_SKIP_MERGE=${SOLVER_SKIP_MERGE:-0}
SOLVER_PREPARE_ONLY=${SOLVER_PREPARE_ONLY:-0}
SOLVER_DATASET_READY=${SOLVER_DATASET_READY:-0}
SOLVER_SAVE_FREQ=${SOLVER_SAVE_FREQ:-5}
SOLVER_SAVE_LIMIT=${SOLVER_SAVE_LIMIT:-3}
SOLVER_KEEP_LATEST_RESUME_STATE_ONLY=${SOLVER_KEEP_LATEST_RESUME_STATE_ONLY:-false}
SOLVER_LOAD_CHECKPOINT=${SOLVER_LOAD_CHECKPOINT:-}
SOLVER_DATASET_RECEIPT=${SOLVER_DATASET_RECEIPT:-}
SOLVER_ROLLOUT_BATCH_SIZE=${SOLVER_ROLLOUT_BATCH_SIZE:-512}
VALIDITY_RZERO_ENABLED=${VALIDITY_RZERO_ENABLED:-0}
echo "solver config: samples=$SOLVER_GENERATE_SAMPLES label_timeout=${QUESTION_EVAL_TIMEOUT_SECONDS}s max_steps=$SOLVER_MAX_STEPS val_freq=$SOLVER_VAL_FREQ merge_step=$SOLVER_MERGE_STEP"
if [ "$SOLVER_DATASET_READY" != "1" ]; then
    echo 'start generate question'
    bash question_generate/question_generate.bash $questioner_model_path $SOLVER_GENERATE_SAMPLES $experiment_name
    echo 'start evaluate generated question'
    bash question_evaluate/evaluate.sh $solver_model_path $experiment_name
    if [ "$VALIDITY_RZERO_ENABLED" = "1" ]; then
        : "${TERRA_REPLAY_DATASET:?set TERRA_REPLAY_DATASET for validity-RZero}"
        : "${TERRA_REPLAY_RATIO:?set TERRA_REPLAY_RATIO for validity-RZero}"
        : "${SOLVER_DATASET_RECEIPT:?set SOLVER_DATASET_RECEIPT for validity-RZero}"
        echo 'start validity-RZero dataset mixing and local export'
        python methods/rquest/dataset.py \
            --experiment-name "${experiment_name}" \
            --num-shards "${QUESTION_NUM_SHARDS}" \
            --max-score "${SOLVER_UPLOAD_MAX_SCORE}" \
            --min-score "${SOLVER_UPLOAD_MIN_SCORE}" \
            --terra-dataset "${TERRA_REPLAY_DATASET}" \
            --terra-config "${TERRA_REPLAY_CONFIG:-default}" \
            --replay-ratio "${TERRA_REPLAY_RATIO}" \
            --seed "${TERRA_REPLAY_SEED:-1}" \
            --min-train-rows "${SOLVER_ROLLOUT_BATCH_SIZE}" \
            --receipt "${SOLVER_DATASET_RECEIPT}"
    fi
else
    echo "dataset already prepared: ${RQUEST_SOLVER_DATASET:-local parquet}"
fi

if [ "$SOLVER_PREPARE_ONLY" = "1" ]; then
    echo "solver dataset preparation finished"
    exit 0
fi

echo 'start train'

RESUME_ARGS=()
if [ -n "$SOLVER_LOAD_CHECKPOINT" ]; then
    echo "resuming solver training from $SOLVER_LOAD_CHECKPOINT"
    RESUME_ARGS+=(trainer.load_checkpoint_path="$SOLVER_LOAD_CHECKPOINT")
fi

EXTRA_TRAIN_ARGS=()
if [ "$VALIDITY_RZERO_ENABLED" = "1" ]; then
    EXTRA_TRAIN_ARGS+=(
        data.format_prompt_source_key=source
        data.default_source=rzero
        data.format_prompt_by_source.rzero=./examples/format_prompt/solver.jinja
        data.format_prompt_by_source.terra=./methods/validity_rl/validity_solver.jinja
        data.rollout_batch_size=${SOLVER_ROLLOUT_BATCH_SIZE}
        worker.reward.reward_function=./methods/rquest/reward.py:compute_score
        'worker.reward.reward_function_data_keys=[source]'
    )
fi

CUDA_VISIBLE_DEVICES=${QUESTION_GPU_IDS} python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=${SOLVER_MAX_RESPONSE_LENGTH} \
    worker.actor.model.model_path=$solver_model_path \
    trainer.experiment_name=${experiment_name} \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/${experiment_name}/ \
    "data.train_files=${RQUEST_SOLVER_DATASET:?local Solver dataset path is required}" \
    trainer.total_epochs=${SOLVER_TOTAL_EPOCHS} \
    trainer.max_steps=${SOLVER_MAX_STEPS} \
    trainer.save_freq=${SOLVER_SAVE_FREQ} \
    trainer.save_limit=${SOLVER_SAVE_LIMIT} \
    trainer.keep_latest_resume_state_only=${SOLVER_KEEP_LATEST_RESUME_STATE_ONLY} \
    "${RESUME_ARGS[@]}" \
    data.format_prompt=./examples/format_prompt/solver.jinja \
    "${EXTRA_TRAIN_ARGS[@]}" \
    trainer.val_freq=${SOLVER_VAL_FREQ} \
    trainer.n_gpus_per_node=${QUESTION_NUM_SHARDS} \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1

if [ "$SOLVER_SKIP_MERGE" != "1" ]; then
    echo "merging model"
    python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/${experiment_name}/global_step_${SOLVER_MERGE_STEP}/actor
fi

sleep 10

echo "solver training finished"

