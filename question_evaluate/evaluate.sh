#!/bin/bash
set -e

model_name=$1
save_name=$2

QUESTION_GPU_IDS=${QUESTION_GPU_IDS:-0,1,2,3}
QUESTION_EVAL_TIMEOUT_SECONDS=${QUESTION_EVAL_TIMEOUT_SECONDS:-14400}
if ! [[ "$QUESTION_EVAL_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "QUESTION_EVAL_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
fi
IFS=',' read -ra GPU_IDS <<< "$QUESTION_GPU_IDS"
pids=()

for i in "${!GPU_IDS[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPU_IDS[$i]} python question_evaluate/evaluate.py --model $model_name --suffix $i --save_name $save_name &
  pids[$i]=$!
done

timeout_duration=$QUESTION_EVAL_TIMEOUT_SECONDS
echo "question evaluation timeout: ${timeout_duration}s"
(
  sleep $timeout_duration
  echo "Timeout reached. Killing remaining tasks..."
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null
    fi
  done
) &
timeout_pid=$!

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "Task $i finished."
  else
    echo "Task $i failed."
    status=1
  fi
done

kill "$timeout_pid" 2>/dev/null || true
wait "$timeout_pid" 2>/dev/null || true
exit "$status"
