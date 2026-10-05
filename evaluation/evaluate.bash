#!/usr/bin/env bash
# Evaluation from https://github.com/Chengsong-Huang/R-Zero/tree/main/evaluation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ./env.sh
if [ "$#" -ne 1 ]; then
    echo 'Usage: bash evaluation/evaluate.bash /path/to/model' >&2
    exit 2
fi
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY for GPT-4o answer rechecking.}"
model_name=$1
gpu_list=${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}
IFS=',' read -r -a gpu_ids <<< "$gpu_list"
if [ "${#gpu_ids[@]}" -lt 4 ]; then
    echo 'R-Zero general reasoning evaluation requires four visible GPUs.' >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="$gpu_list"
output_dir="${STORAGE_PATH}/evaluation/${model_name//\//_}"
mkdir -p "$output_dir"
export EVAL_RESULTS_FILE="$output_dir/final_results.jsonl"

tasks=(math gsm8k amc minerva olympiad aime2024 aime2025)
pids=()
failed=0
wait_for_batch() {
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then failed=1; fi
    done
    pids=()
    if [ "$failed" -ne 0 ]; then
        echo 'Mathematical evaluation failed; see the task logs.' >&2
        exit 1
    fi
}

for index in "${!tasks[@]}"; do
    slot=$((index % ${#gpu_ids[@]}))
    task=${tasks[$index]}
    echo "Evaluating $task on GPU ${gpu_ids[$slot]}"
    CUDA_VISIBLE_DEVICES="${gpu_ids[$slot]}" python evaluation/generate.py \
        --model "$model_name" --dataset "$task" > "$output_dir/${task}.log" 2>&1 &
    pids+=("$!")
    if [ "${#pids[@]}" -eq "${#gpu_ids[@]}" ]; then wait_for_batch; fi
done
wait_for_batch

python evaluation/results_recheck.py --model_name "$model_name"
python evaluation/eval_supergpqa.py --model_path "$model_name" --output_file "$output_dir/supergpqa.json"
python evaluation/eval_bbeh.py --model_path "$model_name" --output_file "$output_dir/bbeh.json"
python evaluation/eval_mmlupro.py --model_path "$model_name" --output_file "$output_dir/mmlupro.json"
echo "Results saved to $output_dir"
