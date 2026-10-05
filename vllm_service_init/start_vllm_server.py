#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Refactored Version: This script employs the 'stopit' library to apply fine-grained, thread-safe
timeout control directly to the `grade_answer` function. This approach is more robust than a
global timeout and avoids the 'signal only works in main thread' error common in multi-threaded
Flask applications. The comparison logic is optimized to perform cheap checks first.

Setup Instructions:
    # 1. Install the required library (note the change from previous versions)
    pip install stopit

    # 2. Run the server
    python your_server_file_name.py --port 5000 --model_path Qwen/Qwen3-4B-Base
'''

from flask import Flask, request, jsonify
import vllm
import argparse
import json
import os
import threading
import time
import torch
from transformers import AutoTokenizer
from mathruler.grader import extract_boxed_content, grade_answer
from jinja2 import Template
import stopit  # 1. Import the thread-safe 'stopit' library

VALIDITY_RZERO_ENABLED = os.getenv("VALIDITY_RZERO_ENABLED", "0") == "1"
if VALIDITY_RZERO_ENABLED:
    from methods.rquest.validity import PHASE_A_MATH_VOTES, evaluate_validity_responses, valid_positions

# ------------------------- Command-Line Arguments ------------------------- #
# (This section remains unchanged)
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=str, default='5000')
parser.add_argument('--model_path', type=str, default='Qwen/Qwen3-4B-Base')
parser.add_argument('--run_id', type=str, default='')
parser.add_argument('--gpu_mem_util', type=float, default=0.8,
                    help='The maximum GPU memory utilization fraction for vLLM.')
args = parser.parse_args()

# ------------------------- vLLM Initialization ------------------------ #
# (This section remains unchanged)
print('[init] Loading model...')

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = vllm.LLM(
    model=args.model_path,
    tokenizer=args.model_path,
    gpu_memory_utilization=args.gpu_mem_util,
)

math_sample_params = vllm.SamplingParams(
    max_tokens=int(os.getenv("VLLM_SERVER_MAX_TOKENS", "4096")),
    temperature=1.0,
    top_p=1.0,
    top_k=40,
    stop_token_ids=[tokenizer.eos_token_id],
    n=PHASE_A_MATH_VOTES if VALIDITY_RZERO_ENABLED else int(os.getenv("VLLM_SERVER_N", "10")),
)
validity_sample_params = vllm.SamplingParams(
    max_tokens=int(os.getenv("VLLM_SERVER_MAX_TOKENS", "4096")),
    temperature=1.0,
    top_p=1.0,
    top_k=40,
    stop_token_ids=[tokenizer.eos_token_id],
    n=9,
)
VALIDITY_TEMPLATE = None
if VALIDITY_RZERO_ENABLED:
    template_path = os.getenv(
        "VALIDITY_RZERO_PROMPT",
        os.path.join(os.path.dirname(__file__), "..", "methods", "validity_rl", "validity_solver.jinja"),
    )
    with open(template_path, encoding="utf-8") as handle:
        VALIDITY_TEMPLATE = Template(handle.read().strip())

# ---------------------- GPU Idle Utilization Thread ---------------------- #
# (This section remains unchanged)
stop_event = threading.Event()    # Event to stop the thread globally
pause_event = threading.Event()   # Event to pause the thread during requests

def gpu_idle_worker():
    '''
    This worker occupies the GPU with a continuous matrix multiplication loop when idle,
    preventing potential performance drops from GPU power state changes.
    '''
    print('[idle_worker] GPU idle worker started.')
    running = True
    while not stop_event.is_set():
        if pause_event.is_set():
            if running:
                print('[idle_worker] Paused.')
                running = False
            time.sleep(0.1) # Sleep briefly while paused
            continue
        else:
            if not running:
                print('[idle_worker] Resumed.')
                running = True
        try:
            # A simple but effective way to keep the GPU busy
            a = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            b = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            torch.matmul(a, b)
            torch.cuda.synchronize()
        except RuntimeError as e:
            print(f'[idle_worker] Caught a RuntimeError: {e}. Sleeping for 1s...')
            time.sleep(1)
    print('[idle_worker] GPU idle worker stopped.')

idle_thread = threading.Thread(target=gpu_idle_worker, daemon=True)
idle_thread.start()

# ------------------------ Timeout Utility (Refactored) --------------------------- #
# 2. Use the 'stopit.threading_timeoutable' decorator for thread-safe timeouts.
#    It returns a default value on timeout instead of raising an exception.
@stopit.threading_timeoutable(default='TIMED_OUT')
def grade_answer_with_timeout(res1, res2):
    """
    This wrapper applies a timeout to each individual `grade_answer` call.
    If the function's execution exceeds the specified timeout, it will return 'TIMED_OUT'.
    The timeout duration is passed as a keyword argument during the function call.
    """
    return grade_answer(res1, res2)

# ---------------------------- Flask Application --------------------------- #
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "pid": os.getpid(),
        "run_id": args.run_id,
        "model_path": args.model_path,
        "port": int(args.port),
    })

@app.route('/hello', methods=['GET'])
def hello():
    '''The main processing endpoint: reads a task file, invokes vLLM, consolidates answers, and writes results.'''

    # --- Pause the GPU idle worker to free up resources ---
    pause_event.set()
    torch.cuda.synchronize()

    name = request.args.get('name', 'None')
    print(f'[server] Received request for task file: {name}')

    # ---------- Load Data ----------
    with open(name, 'r') as f:
        data = json.load(f)
    os.remove(name)

    questions = [item.get('question', '') for item in data]
    answers   = [item.get('answer',   '') for item in data]

    # (Data preparation logic remains unchanged)
    valid_indices, valid_questions, valid_answers, valid_chats = [], [], [], []
    for i, (q, a) in enumerate(zip(questions, answers)):
        if q and a:
            valid_indices.append(i)
            valid_questions.append(q)
            valid_answers.append(a)
            valid_chats.append([
                {'role': 'system', 'content': 'Please reason step by step, and put your final answer within \\boxed{}.'},
                {'role': 'user',   'content': q}
            ])
    print('[server] Valid chat prompts have been prepared.')

    def render_prompts(chats):
        if tokenizer.chat_template:
            return [
                tokenizer.apply_chat_template(chat, tokenize=False,
                                              add_generation_prompt=True, add_special_tokens=True)
                for chat in chats
            ]
        return ['\n'.join(f"{message['role']}: {message['content']}" for message in chat) for chat in chats]

    def generate(prompts, sampling_params):
        chunk_size = int(os.getenv("VLLM_SERVER_BATCH_SIZE", "0"))
        if chunk_size > 0:
            generated = []
            for start in range(0, len(prompts), chunk_size):
                generated.extend(model.generate(
                    prompts[start:start + chunk_size], sampling_params=sampling_params, use_tqdm=True
                ))
            return generated
        return model.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

    # ---------- Results Post-Processing (Core Refactoring & Optimization Here) ----------
    def process_single(question, golden_answer, response):
        '''Consolidates and grades vLLM outputs for a single question, returning a result dictionary.'''
        results = [extract_boxed_content(out.text) for out in response.outputs]
        # print(f"[process_single] Processing question: '{question[:70]}...'")

        answer_counts = {}
        for res in results:
            if not res: continue # Skip empty results
            matched = False
            
            for exist_ans in list(answer_counts.keys()):
                # 3. OPTIMIZATION: Perform cheap comparisons first to avoid expensive calls.
                if res == exist_ans or ('no ' in res.lower() and 'no ' in exist_ans.lower()):
                    answer_counts[exist_ans] += 1
                    matched = True
                    break # Match found, break from the inner loop over exist_ans
                
                # 4. If cheap checks fail, proceed to the expensive, timed grade_answer calls.
                try:
                    is_match = False
                    # First direction: res vs exist_ans
                    match_result_1 = grade_answer_with_timeout(res, exist_ans, timeout=10)
                    if match_result_1 == 'TIMED_OUT':
                        print(f"      [grader] TIMEOUT comparing '{res[:30]}...' with '{exist_ans[:30]}...'.")
                    elif match_result_1:
                        is_match = True

                    # Second direction (only if first failed): exist_ans vs res
                    if not is_match:
                        match_result_2 = grade_answer_with_timeout(exist_ans, res, timeout=10)
                        if match_result_2 == 'TIMED_OUT':
                             # Log timeout for the second direction as well
                            print(f"      [grader] TIMEOUT comparing '{exist_ans[:30]}...' with '{res[:30]}...'. Skipping pair.")
                        elif match_result_2:
                            is_match = True
                    
                    if is_match:
                        answer_counts[exist_ans] += 1
                        matched = True
                        break # Match found, break from the inner loop

                except Exception as e:
                    # Catch any other potential errors from the grader function itself.
                    print(f"      [grader] ERROR comparing '{res[:30]}...' with '{exist_ans[:30]}...': {e}. Skipping.")
                    continue # Continue to the next comparison in the inner loop
            
            if not matched:
                answer_counts[res] = 1

        if not answer_counts:
            majority_ans, max_count = '', 0
        else:
            majority_ans = max(answer_counts, key=answer_counts.get)
            max_count = answer_counts[majority_ans]

        score = max_count / len(results) if results else 0.0

        return {
            'question': question,
            'answer':   majority_ans,
            'score':    score,
            'results':  results
        }

    def safe_process_single(question, golden_answer, response):
        try:
            return process_single(question, golden_answer, response)
        except Exception as error:
            print(f'[server] CRITICAL: An unhandled error occurred while processing question: {question}')
            print(f'[server] Error details: {error}')
            return {
                'question': question,
                'answer': golden_answer,
                'score': -1,
                'results': [],
                'error': f'unhandled exception in process_single: {error}',
            }

    results_all = [None] * len(questions)
    if VALIDITY_RZERO_ENABLED and valid_questions:
        validity_chats = [[{
            'role': 'user',
            'content': VALIDITY_TEMPLATE.render(content=question).strip(),
        }] for question in valid_questions]
        validity_responses = generate(render_prompts(validity_chats), validity_sample_params)
        gates = [
            evaluate_validity_responses([output.text for output in response.outputs])
            for response in validity_responses
        ]
        del validity_responses
        math_positions = valid_positions(gates)
        math_responses = generate(
            render_prompts([valid_chats[position] for position in math_positions]), math_sample_params
        ) if math_positions else []
        math_by_position = dict(zip(math_positions, math_responses))
        for position, original_index in enumerate(valid_indices):
            gate = gates[position]
            if gate['validity_decision'] == 'INVALID':
                item = {
                    'question': valid_questions[position],
                    'answer': 'INVALID',
                    'score': gate['validity_penalty'],
                    'results': [],
                    'math_evaluation_performed': False,
                    'math_frontier_score': 0.0,
                    'questioner_base_reward': gate['validity_penalty'],
                    **gate,
                }
            else:
                item = safe_process_single(
                    valid_questions[position], valid_answers[position], math_by_position[position]
                )
                frontier = min(item['score'], 1 - item['score'])
                item.update({
                    'math_evaluation_performed': True,
                    'math_frontier_score': frontier,
                    'questioner_base_reward': frontier,
                    **gate,
                })
            print('[rquest][phase_a] ' + json.dumps({
                key: item[key] for key in (
                    'invalid_votes', 'total_votes', 'validity_decision', 'validity_penalty',
                    'math_evaluation_performed', 'math_frontier_score', 'questioner_base_reward'
                )
            }))
            results_all[original_index] = item
    elif valid_questions:
        responses = generate(render_prompts(valid_chats), math_sample_params)
        for position, original_index in enumerate(valid_indices):
            results_all[original_index] = safe_process_single(
                valid_questions[position], valid_answers[position], responses[position]
            )
    print('[server] Generation completed.')

    for index, (q, a) in enumerate(zip(questions, answers)):
        if results_all[index] is not None:
            continue
        results_all[index] = {'question': q, 'answer': a, 'score': -1, 'results': []}
    print('[server] All results have been processed.')

    out_path = name.replace('.json', '_results.json')
    with open(out_path, 'w') as f:
        json.dump(results_all, f, indent=4)

    # --- Resume the GPU idle worker ---
    pause_event.clear()
    print(f'[server] Processed {name}, results saved to {out_path}. Resuming idle worker.')
    return jsonify({'message': f'Processed {name}, results saved to {out_path}.'})

# ------------------------- Main Application Entrypoint --------------------------- #
# (This section remains unchanged)
if __name__ == '__main__':
    try:
        app.run(host='127.0.0.1', port=int(args.port), threaded=True)
    finally:
        # Gracefully shut down the background thread on exit
        stop_event.set()
        idle_thread.join()
        print('[main] Application shutdown complete.')
