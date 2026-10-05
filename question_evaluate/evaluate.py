#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Description:
    This script evaluates generated answers against golden answers for a set of questions.
    It uses vLLM for efficient generation and a robust, timed grading mechanism to score the results.
    The script is designed to run as a batch job, often in parallel across multiple GPUs.

Refactoring Notes:
    - Replaced 'timeout-decorator' with the thread-safe 'stopit' library to provide robust
      timeout protection for the grading function without causing errors.
    - Optimized the answer comparison logic to perform cheap checks first, only calling the
      expensive grading function when necessary.
    - Improved error handling and code structure for better readability and stability.

Setup:
    pip install stopit transformers torch vllm

Example Usage (in a shell script):
    # This would run the script for GPU 0, with a specific model and save name.
    CUDA_VISIBLE_DEVICES=0 python evaluate.py --model "Qwen/Qwen3-4B-Base" --suffix 0 --save_name "my_experiment" &
'''

import json
import vllm
from transformers import AutoTokenizer
import argparse
import re
import os
import stopit  # Use the robust, thread-safe stopit library for timeouts
from jinja2 import Template
from mathruler.grader import extract_boxed_content, grade_answer

validity_rzero_enabled = os.getenv("VALIDITY_RZERO_ENABLED", "0") == "1"
if validity_rzero_enabled:
    from methods.rquest.validity import PHASE_B_MATH_VOTES, evaluate_validity_responses, valid_positions

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Evaluate generated questions using vLLM.")
parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Base", help="Path to the model in Hugging Face format.")
parser.add_argument("--num_samples", type=int, default=9, help="Number of candidate answers to generate per question (n).")
parser.add_argument("--suffix", type=str, default="0", help="A unique suffix for file naming, often the GPU index.")
parser.add_argument("--save_name", type=str, required=True, help="A base name for input and output files.")
args = parser.parse_args()

# --- Constants and Paths ---
STORAGE_PATH = os.environ["STORAGE_PATH"]
INPUT_FILE = f"{STORAGE_PATH}/generated_question/{args.save_name}_{args.suffix}.json"
OUTPUT_FILE = f"{STORAGE_PATH}/generated_question/{args.save_name}_{args.suffix}_results.json"

# --- Timeout-Protected Grading Function ---
@stopit.threading_timeoutable(default='TIMED_OUT')
def grade_answer_with_timeout(res1, res2):
    """
    Wraps the mathruler 'grade_answer' function with a timeout.
    If the function takes too long, it returns 'TIMED_OUT' instead of hanging.
    """
    # The actual timeout value is passed as a keyword argument on each call.
    return grade_answer(res1, res2)

# --- Main Script Logic ---

# 1. Load and Prepare Data
print(f"[{args.suffix}] Loading data from: {INPUT_FILE}")
try:
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)
    # Clean up the input file immediately after loading to save space
    os.remove(INPUT_FILE)
except FileNotFoundError:
    print(f"[{args.suffix}] ERROR: Input file not found. Exiting.")
    exit()

# Filter data into questions that need processing
correct_data = [item for item in data if item.get('score') == 0]
if not correct_data:
    print(f"[{args.suffix}] No new questions to process (score=0). Exiting.")
    # Create an empty results file to signal completion
    with open(OUTPUT_FILE, "w") as f:
        json.dump([], f)
    exit()

questions = [item["question"] for item in correct_data]
answers = [item["answer"] for item in correct_data]
print(f"[{args.suffix}] Found {len(questions)} questions to process.")

# 2. Initialize Model and Tokenizer
print(f"[{args.suffix}] Initializing vLLM for model: {args.model}")
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = vllm.LLM(
    model=args.model,
    tokenizer=args.model,
    gpu_memory_utilization=0.85,
    seed=int(args.suffix),
)
math_sample_params = vllm.SamplingParams(
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    top_k=40,
    stop_token_ids=[tokenizer.eos_token_id],
    n=PHASE_B_MATH_VOTES if validity_rzero_enabled else args.num_samples,
)

# 3. Generate Responses
print(f"[{args.suffix}] Generating {args.num_samples} samples for each question...")
chats = [[{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},{"role": "user", "content": q}] for q in questions]

if tokenizer.chat_template:
    prompts = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, add_special_tokens=True) for chat in chats]
else:
    prompts = ["system: " + chat[0]["content"] + '\n' + "user: " + chat[1]["content"] for chat in chats]

gates = None
math_response_by_index = None
if validity_rzero_enabled:
    prompt_path = os.getenv(
        "VALIDITY_RZERO_PROMPT",
        os.path.join(os.path.dirname(__file__), "..", "methods", "validity_rl", "validity_solver.jinja"),
    )
    with open(prompt_path, encoding="utf-8") as handle:
        validity_template = Template(handle.read().strip())
    validity_chats = [[{
        "role": "user",
        "content": validity_template.render(content=question).strip(),
    }] for question in questions]
    if tokenizer.chat_template:
        validity_prompts = [
            tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True, add_special_tokens=True
            )
            for chat in validity_chats
        ]
    else:
        validity_prompts = [f"user: {chat[0]['content']}" for chat in validity_chats]
    validity_params = vllm.SamplingParams(
        max_tokens=4096,
        temperature=1.0,
        top_p=1.0,
        top_k=40,
        stop_token_ids=[tokenizer.eos_token_id],
        n=9,
    )
    validity_responses = model.generate(validity_prompts, sampling_params=validity_params, use_tqdm=True)
    gates = [
        evaluate_validity_responses([output.text for output in response.outputs])
        for response in validity_responses
    ]
    del validity_responses
    valid_indices = valid_positions(gates)
    math_responses = model.generate(
        [prompts[index] for index in valid_indices], sampling_params=math_sample_params, use_tqdm=True
    ) if valid_indices else []
    math_response_by_index = dict(zip(valid_indices, math_responses))
else:
    responses = model.generate(prompts, sampling_params=math_sample_params, use_tqdm=True)
print(f"[{args.suffix}] Generation complete.")

# 4. Process and Grade Responses
results_all = []
print(f"[{args.suffix}] Grading responses...")
for index, (golden_answer, question) in enumerate(zip(answers, questions)):
    try:
        gate = gates[index] if gates is not None else None
        if gate is not None and gate["validity_decision"] == "INVALID":
            item = {
                "question": question,
                "answer": "",
                "score": None,
                "results": [],
                "discarded_by_validity": True,
                **gate,
            }
            results_all.append(item)
            print(f"[{args.suffix}] [rquest][phase_b] " + json.dumps({
                key: item[key] for key in (
                    "invalid_votes", "total_votes", "validity_decision", "discarded_by_validity"
                )
            }))
            continue

        response = math_response_by_index[index] if math_response_by_index is not None else responses[index]
        # Extract the boxed content from all generated samples
        results = [extract_boxed_content(output.text) for output in response.outputs]
        results = [res for res in results if res] # Filter out None/empty results

        if not results:
            print(f"[{args.suffix}] WARNING: No valid boxed answers found for question: '{question[:50]}...'")
            continue

        answer_counts = {}
        for result in results:
            matched = False
            for existing_answer in answer_counts:
                # OPTIMIZATION: Perform cheap string comparisons first.
                if result == existing_answer or ('no ' in result.lower() and 'no ' in existing_answer.lower()):
                    answer_counts[existing_answer] += 1
                    matched = True
                    break
                
                # If cheap checks fail, use the expensive, timed grader.
                # Check both directions (A vs B and B vs A).
                match_1 = grade_answer_with_timeout(result, existing_answer, timeout=10)
                if match_1 == 'TIMED_OUT':
                    print(f"[{args.suffix}] GRADER TIMEOUT on: '{result[:30]}...' vs '{existing_answer[:30]}...'")
                    continue # Skip to the next existing_answer
                
                if match_1:
                    answer_counts[existing_answer] += 1
                    matched = True
                    break

                match_2 = grade_answer_with_timeout(existing_answer, result, timeout=10)
                if match_2 == 'TIMED_OUT':
                    print(f"[{args.suffix}] GRADER TIMEOUT on: '{existing_answer[:30]}...' vs '{result[:30]}...'")
                    continue

                if match_2:
                    answer_counts[existing_answer] += 1
                    matched = True
                    break

            if not matched:
                answer_counts[result] = 1

        if not answer_counts:
            continue

        # Determine the majority answer and its score
        majority_answer = max(answer_counts, key=answer_counts.get)
        max_count = answer_counts[majority_answer]
        score = max_count / len(results)

        # Skip certain question types that are hard to grade automatically
        if "证明" in question or 'box' in question.lower() or 'text' in majority_answer.lower():
            continue

        item = {
            "question": question,
            "answer": majority_answer,
            "score": score,
            'results': results
        }
        if gate is not None:
            item.update({"discarded_by_validity": False, **gate})
            print(f"[{args.suffix}] [rquest][phase_b] " + json.dumps({
                "invalid_votes": item["invalid_votes"],
                "total_votes": item["total_votes"],
                "validity_decision": item["validity_decision"],
                "discarded_by_validity": False,
                "math_majority": majority_answer,
                "original_rzero_score": score,
            }))
        results_all.append(item)

    except Exception as e:
        print(f"[{args.suffix}] CRITICAL ERROR processing question '{question[:50]}...': {e}")
        continue

# 5. Save Final Results
print(f"[{args.suffix}] Processed {len(results_all)} questions. Saving results to: {OUTPUT_FILE}")
with open(OUTPUT_FILE, "w") as f:
    json.dump(results_all, f, indent=4)

print(f"[{args.suffix}] Script finished.")
