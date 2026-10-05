#!/bin/bash
set -e

solver_model_path=$1
questioner_model_path=$2
save_path=$3
QUESTIONER_OUTPUT_DIR=${QUESTIONER_OUTPUT_DIR:-${STORAGE_PATH}/models/$save_path}
mkdir -p logs
if [ "${VALIDITY_RZERO_ENABLED:-0}" = "1" ] && [ -n "${VALIDITY_RZERO_ARTIFACT_DIR:-}" ]; then
    QUESTIONER_ARTIFACT_DIR=${VALIDITY_RZERO_ARTIFACT_DIR}/${save_path}
    mkdir -p "$QUESTIONER_ARTIFACT_DIR"
    QUESTIONER_LOG_FILE=${QUESTIONER_LOG_FILE:-${QUESTIONER_ARTIFACT_DIR}/questioner_$(date +%Y%m%d_%H%M%S).log}
    VLLM_LOG_DIR=${VLLM_LOG_DIR:-${QUESTIONER_ARTIFACT_DIR}/vllm}
fi
QUESTIONER_LOG_FILE=${QUESTIONER_LOG_FILE:-logs/questioner_${save_path}_$(date +%Y%m%d_%H%M%S).log}
exec > >(tee -a "$QUESTIONER_LOG_FILE") 2>&1
echo "logging to $QUESTIONER_LOG_FILE"
echo "save_path: $save_path"
echo "questioner output directory: $QUESTIONER_OUTPUT_DIR"
# 生成唯一 RUN_ID
RUN_ID=$(date +%s%N)
export RUN_ID

echo "RUN_ID=$RUN_ID"

# 启动 vllm 服务（默认 GPU 2,3；可通过 VLLM_GPU_IDS 覆盖）
export QUESTIONER_TRAIN_GPU_IDS=${QUESTIONER_TRAIN_GPU_IDS:-0,1}
export VLLM_GPU_IDS=${VLLM_GPU_IDS:-2,3}
export VLLM_PORT_BASE=${VLLM_PORT_BASE:-5000}
VLLM_SERVICE_COUNT=$(echo "$VLLM_GPU_IDS" | awk -F',' '{print NF}')
QUESTIONER_TRAIN_GPU_COUNT=$(echo "$QUESTIONER_TRAIN_GPU_IDS" | awk -F',' '{print NF}')
export VLLM_SERVICE_COUNT
export QUESTIONER_VLLM_PID_FILE=${QUESTIONER_VLLM_PID_FILE:-${STORAGE_PATH}/temp_results/questioner_vllm_${RUN_ID}.pids}
export VLLM_LOG_DIR=${VLLM_LOG_DIR:-logs}
export VALIDITY_RZERO_DIVERSITY_MODE=${VALIDITY_RZERO_DIVERSITY_MODE:-semantic_novelty_gate}
if [ "${VALIDITY_RZERO_ENABLED:-0}" = "1" ] && [ "$VALIDITY_RZERO_DIVERSITY_MODE" = "semantic_novelty_gate" ]; then
    export VALIDITY_RZERO_REPO_ROOT
    VALIDITY_RZERO_REPO_ROOT=$(pwd)
    export VALIDITY_RZERO_SOLVER_MODEL_PATH=$solver_model_path
    export VALIDITY_RZERO_SOLVER_RUN_ID=$RUN_ID
    export VALIDITY_RZERO_SEMANTIC_MODEL=${VALIDITY_RZERO_SEMANTIC_MODEL:-Qwen/Qwen3-4B-Base}
    export VALIDITY_RZERO_SEMANTIC_GPU_IDS=${VALIDITY_RZERO_SEMANTIC_GPU_IDS:-${QUESTIONER_TRAIN_GPU_IDS},${VLLM_GPU_IDS}}
    export VALIDITY_RZERO_SEMANTIC_GPU_MEMORY_UTILIZATION=${VALIDITY_RZERO_SEMANTIC_GPU_MEMORY_UTILIZATION:-0.80}
    export VALIDITY_RZERO_SEMANTIC_WORKER_BATCH_SIZE=${VALIDITY_RZERO_SEMANTIC_WORKER_BATCH_SIZE:-8192}
    export VALIDITY_RZERO_SEMANTIC_PID_FILE=${VALIDITY_RZERO_SEMANTIC_PID_FILE:-${STORAGE_PATH}/temp_results/questioner_semantic_${RUN_ID}.pids}
    echo "semantic diversity enabled: mode=$VALIDITY_RZERO_DIVERSITY_MODE, novelty prompt, batch=$VALIDITY_RZERO_SEMANTIC_WORKER_BATCH_SIZE, deferred retry, prefix cache"
    echo "Solver GPUs: $VLLM_GPU_IDS; temporary frozen-judge GPUs: $VALIDITY_RZERO_SEMANTIC_GPU_IDS; memory utilization: $VALIDITY_RZERO_SEMANTIC_GPU_MEMORY_UTILIZATION"
fi
bash vllm_service_init/start.sh $solver_model_path $RUN_ID
echo "vLLM services started with RUN_ID=$RUN_ID on GPUs $VLLM_GPU_IDS and ports starting at $VLLM_PORT_BASE"
echo "Questioner training will use GPUs $QUESTIONER_TRAIN_GPU_IDS"
QUESTIONER_LOGGER=${QUESTIONER_LOGGER:-'["console"]'}
QUESTIONER_SAVE_FREQ=${QUESTIONER_SAVE_FREQ:-5}
QUESTIONER_SAVE_LIMIT=${QUESTIONER_SAVE_LIMIT:-3}
QUESTIONER_KEEP_LATEST_RESUME_STATE_ONLY=${QUESTIONER_KEEP_LATEST_RESUME_STATE_ONLY:-false}
QUESTIONER_LOAD_CHECKPOINT=${QUESTIONER_LOAD_CHECKPOINT:-}

RESUME_ARGS=()
REWARD_DATA_ARGS=()
if [ -n "$QUESTIONER_LOAD_CHECKPOINT" ]; then
    echo "resuming questioner training from $QUESTIONER_LOAD_CHECKPOINT"
    RESUME_ARGS+=(trainer.load_checkpoint_path="$QUESTIONER_LOAD_CHECKPOINT")
fi
if [ "${VALIDITY_RZERO_ENABLED:-0}" = "1" ] && [ "$VALIDITY_RZERO_DIVERSITY_MODE" = "semantic_novelty_gate" ]; then
    REWARD_DATA_ARGS+=("worker.reward.reward_function_optional_data_keys=[validity_rzero_semantic_gpu_ready_file,uid]")
fi

cleanup_pid_file() {
    local pid_file=$1
    if [ -f "$pid_file" ]; then
        while read -r pid; do
            if kill -0 "$pid" 2>/dev/null; then
                kill -- "-$pid" 2>/dev/null || true
                kill "$pid" 2>/dev/null || true
            fi
        done < "$pid_file"
        sleep 3
        while read -r pid; do
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 -- "-$pid" 2>/dev/null || true
                kill -9 "$pid" 2>/dev/null || true
            fi
        done < "$pid_file"
    fi
}
cleanup_vllm() {
    if [ -n "${VALIDITY_RZERO_SEMANTIC_PID_FILE:-}" ]; then
        cleanup_pid_file "$VALIDITY_RZERO_SEMANTIC_PID_FILE"
    fi
    cleanup_pid_file "$QUESTIONER_VLLM_PID_FILE"
}
trap cleanup_vllm EXIT

echo "Waiting for vLLM services to become healthy..."
for i in $(seq 0 $((VLLM_SERVICE_COUNT - 1))); do
    port=$((VLLM_PORT_BASE + i))
    expected_pid=$(sed -n "$((i + 1))p" "$QUESTIONER_VLLM_PID_FILE")
    ok=0
    for _ in $(seq 1 240); do
        if [ -n "$expected_pid" ] && ! kill -0 "$expected_pid" 2>/dev/null; then
            echo "Solver process pid=$expected_pid for port=$port exited before becoming healthy." >&2
            break
        fi
        if python3 -c 'import json,sys,urllib.request; d=json.load(urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=2)); assert d.get("status") == "ok" and str(d.get("run_id")) == sys.argv[2] and int(d.get("pid", -1)) == int(sys.argv[3])' "$port" "$RUN_ID" "$expected_pid" >/dev/null 2>&1; then
            ok=1
            break
        fi
        sleep 1
    done
    if [ "$ok" != "1" ]; then
        echo "vLLM service on port ${port} did not become healthy. Check ${VLLM_LOG_DIR}/vllm_solver_${RUN_ID}_*port${port}.log"
        exit 1
    fi
done
echo "All vLLM services are healthy."

# 开始训练 Questioner
echo "Start training questioner: $questioner_model_path -> $save_path"

CUDA_VISIBLE_DEVICES=${QUESTIONER_TRAIN_GPU_IDS} python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=${QUESTIONER_MAX_RESPONSE_LENGTH:-4096} \
    data.rollout_batch_size=${QUESTIONER_ROLLOUT_BATCH_SIZE:-512} \
    worker.actor.model.model_path=$questioner_model_path \
    trainer.experiment_name=$save_path \
    trainer.logger="$QUESTIONER_LOGGER" \
    trainer.save_checkpoint_path=$QUESTIONER_OUTPUT_DIR \
    trainer.total_epochs=1000 \
    worker.reward.reward_function=./examples/reward_function/caller_penalty.py:compute_score \
    worker.reward.reward_function_kwargs.num_services=$VLLM_SERVICE_COUNT \
    worker.reward.reward_function_kwargs.port_base=$VLLM_PORT_BASE \
    "${REWARD_DATA_ARGS[@]}" \
    trainer.val_freq=-1 \
    trainer.val_before_train=${QUESTIONER_VAL_BEFORE_TRAIN:-false} \
    trainer.n_gpus_per_node=${QUESTIONER_TRAIN_GPU_COUNT} \
    data.format_prompt=./examples/format_prompt/questioner.jinja \
    worker.rollout.n=${QUESTIONER_ROLLOUT_N:-4} \
    worker.actor.global_batch_size=${QUESTIONER_GLOBAL_BATCH_SIZE:-4} \
    worker.actor.micro_batch_size_per_device_for_update=${QUESTIONER_MICRO_BATCH_UPDATE:-2} \
    worker.actor.micro_batch_size_per_device_for_experience=${QUESTIONER_MICRO_BATCH_EXPERIENCE:-8} \
    trainer.max_steps=${QUESTIONER_MAX_STEPS:-6} \
    trainer.save_freq=${QUESTIONER_SAVE_FREQ} \
    trainer.save_limit=${QUESTIONER_SAVE_LIMIT} \
    trainer.keep_latest_resume_state_only=${QUESTIONER_KEEP_LATEST_RESUME_STATE_ONLY} \
    "${RESUME_ARGS[@]}"

sleep 5

if [ "${QUESTIONER_SKIP_MERGE:-0}" != "1" ]; then
    # 合并模型
    echo "merging model"
    python scripts/model_merger.py --local_dir ${QUESTIONER_OUTPUT_DIR}/global_step_${QUESTIONER_MERGE_STEP:-5}/actor
fi

sleep 10

echo "questioner training finished"
