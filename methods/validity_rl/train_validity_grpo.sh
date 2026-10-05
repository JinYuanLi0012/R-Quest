#!/usr/bin/env bash
set -euo pipefail

METHOD_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${METHOD_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export VLLM_DISABLE_COMPILE_CACHE=1

MODEL_PATH=${VALIDITY_MODEL_PATH:-Qwen/Qwen3-4B-Base}
DATASET_NAME=${VALIDITY_DATASET:-jinyuan222/rzero-validity-rl-terra-v1-clean-v1}
TRAIN_FILES=${VALIDITY_TRAIN_FILES:-${DATASET_NAME}@train}
VAL_FILES=${VALIDITY_VAL_FILES:-${DATASET_NAME}@validation}
if [ -d "$DATASET_NAME" ]; then
    TRAIN_FILES=${VALIDITY_TRAIN_FILES:-${DATASET_NAME}/train.parquet}
    VAL_FILES=${VALIDITY_VAL_FILES:-${DATASET_NAME}/validation.parquet}
fi
EXPERIMENT_NAME=${VALIDITY_EXPERIMENT_NAME:-qwen3_4b_validity_rl_terra_clean_v1}

if [[ -z "${VALIDITY_SAVE_PATH:-}" ]]; then
    if [[ -z "${STORAGE_PATH:-}" ]]; then
        echo "Set STORAGE_PATH or VALIDITY_SAVE_PATH before training." >&2
        exit 2
    fi
    VALIDITY_SAVE_PATH="${STORAGE_PATH}/models/${EXPERIMENT_NAME}"
fi

GPU_IDS=${VALIDITY_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NUM_GPUS=${VALIDITY_NUM_GPUS:-${#GPU_ARRAY[@]}}

ROLLOUT_BATCH_SIZE=${VALIDITY_ROLLOUT_BATCH_SIZE:-512}
ROLLOUT_N=${VALIDITY_ROLLOUT_N:-8}
MAX_PROMPT_LENGTH=${VALIDITY_MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${VALIDITY_MAX_RESPONSE_LENGTH:-4096}
TOTAL_EPOCHS=${VALIDITY_TOTAL_EPOCHS:-100}
MAX_STEPS=${VALIDITY_MAX_STEPS:-15}
SAVE_FREQ=${VALIDITY_SAVE_FREQ:-5}
SAVE_LIMIT=${VALIDITY_SAVE_LIMIT:-3}
ACTOR_GLOBAL_BATCH_SIZE=${VALIDITY_ACTOR_GLOBAL_BATCH_SIZE:-128}
UPDATE_MICRO_BATCH_SIZE=${VALIDITY_UPDATE_MICRO_BATCH_SIZE:-1}
EXPERIENCE_MICRO_BATCH_SIZE=${VALIDITY_EXPERIENCE_MICRO_BATCH_SIZE:-1}
LOGGER=${VALIDITY_LOGGER:-'["console"]'}
VAL_GENERATIONS_TO_LOG=${VALIDITY_VAL_GENERATIONS_TO_LOG:-0}

mkdir -p "${VALIDITY_SAVE_PATH}"
LOG_DIR=${VALIDITY_LOG_DIR:-${VALIDITY_SAVE_PATH}/logs}
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

COMMAND=(
    python3 -m verl.trainer.main
    config=examples/config.yaml
    "data.train_files=${TRAIN_FILES}"
    "data.val_files=${VAL_FILES}"
    data.prompt_key=question
    data.answer_key=validity_rl_target
    "data.format_prompt=${METHOD_DIR}/validity_solver.jinja"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.rollout_batch_size=${ROLLOUT_BATCH_SIZE}"
    data.shuffle=true
    data.seed=1
    algorithm.adv_estimator=grpo
    algorithm.disable_kl=false
    algorithm.use_kl_loss=true
    algorithm.kl_penalty=low_var_kl
    algorithm.kl_coef=1.0e-2
    "worker.actor.model.model_path=${MODEL_PATH}"
    "worker.actor.global_batch_size=${ACTOR_GLOBAL_BATCH_SIZE}"
    "worker.actor.micro_batch_size_per_device_for_update=${UPDATE_MICRO_BATCH_SIZE}"
    "worker.actor.micro_batch_size_per_device_for_experience=${EXPERIENCE_MICRO_BATCH_SIZE}"
    worker.actor.optim.lr=1.0e-6
    "worker.rollout.n=${ROLLOUT_N}"
    worker.rollout.temperature=1.0
    worker.rollout.top_p=0.99
    "worker.reward.reward_function=${METHOD_DIR}/validity_reward.py:compute_score"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    "trainer.n_gpus_per_node=${NUM_GPUS}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.max_steps=${MAX_STEPS}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.save_limit=${SAVE_LIMIT}"
    "trainer.save_checkpoint_path=${VALIDITY_SAVE_PATH}"
    "trainer.logger=${LOGGER}"
    trainer.val_before_train=false
    trainer.val_freq=-1
    "trainer.val_generations_to_log=${VAL_GENERATIONS_TO_LOG}"
)

echo "Validity-RL experiment: ${EXPERIMENT_NAME}"
echo "Train data: ${TRAIN_FILES}"
echo "Validation data (reward-only, never used for gradients): ${VAL_FILES}"
echo "Checkpoints every ${SAVE_FREQ} steps through step ${MAX_STEPS}: ${VALIDITY_SAVE_PATH}"
echo "Logging to ${LOG_FILE}"

if [[ "${VALIDITY_DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "CUDA_VISIBLE_DEVICES=${GPU_IDS}" "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

# Keep the exact warm-start command stable across restart.
python3 - "${VALIDITY_SAVE_PATH}" "${VALIDITY_RESUME:-0}" "${GPU_IDS}" "${COMMAND[@]}" <<'PYCODE'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
resume = sys.argv[2] == "1"
manifest = root / "training_command.json"
expected = {"gpu_ids": sys.argv[3], "command": sys.argv[4:]}
if manifest.exists():
    if not resume:
        raise SystemExit("Existing validity training run. Use --resume or a new VALIDITY_SAVE_PATH.")
    if json.loads(manifest.read_text()) != expected:
        raise SystemExit("Validity resume configuration differs from the recorded command.")
else:
    if resume or any(root.glob("global_step_*")):
        raise SystemExit("No matching training manifest; refusing to overwrite/import an unrelated run.")
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps(expected, indent=2) + "\n")
    temporary.replace(manifest)
PYCODE
if [ "${VALIDITY_RESUME:-0}" = 1 ]; then
    status=0
    checkpoint=$(python3 scripts/find_resume_checkpoint.py --root "$VALIDITY_SAVE_PATH" --world-size "$NUM_GPUS") || status=$?
    if [ "$status" = 0 ]; then
        completed_step=${checkpoint##*global_step_}
        if [ "$completed_step" -ge "$MAX_STEPS" ]; then
            echo "Validity training already reached step $completed_step; proceeding to merge."
            exit 0
        fi
        COMMAND+=("trainer.load_checkpoint_path=${checkpoint}")
    elif [ "$status" = 1 ]; then
        echo 'No complete checkpoint yet; restarting this recorded run from Base.'
    else
        exit "$status"
    fi
fi
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${COMMAND[@]}" 2>&1 | tee -a "${LOG_FILE}"
