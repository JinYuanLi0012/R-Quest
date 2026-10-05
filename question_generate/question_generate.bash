#!/bin/bash
set -e

# load the model name from the command line
model_name=$1
num_samples=$2
save_name=$3
export VLLM_DISABLE_COMPILE_CACHE=1
QUESTION_GPU_IDS=${QUESTION_GPU_IDS:-0,1,2,3}
IFS=',' read -ra GPU_IDS <<< "$QUESTION_GPU_IDS"
pids=()

for i in "${!GPU_IDS[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPU_IDS[$i]} python question_generate/question_generate.py --model $model_name --suffix $i --num_samples $num_samples --save_name $save_name &
  pids[$i]=$!
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
