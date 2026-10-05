# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import regex as re
from typing import Dict, List
import json
from mathruler.grader import extract_boxed_content, grade_answer
import os
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

STORAGE_PATH = os.environ["STORAGE_PATH"]

def generate_temp_filename(prefix="temp", suffix=".json"):
    timestamp = int(time.time() * 1000) 
    rand_part = random.randint(0, 99999)
    return f"{STORAGE_PATH}/temp_results/{prefix}_{timestamp}_{rand_part}{suffix}"
def split_list(lst, n=4):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"

def fetch(index,i, port_base=5000):
    response = requests.get(f"http://0.0.0.0:{port_base+index}/hello?name={i}")
    print(response)
    return True

def generate_results(data, num_services=2, port_base=5000):
    datas = split_list(data,num_services)
    random_names = [generate_temp_filename(prefix=f"temp_{i}", suffix=".json") for i in range(num_services)]
    for i in range(num_services):
        with open(random_names[i],'w') as f:
            json.dump(datas[i],f,indent=4)

    final_results = []
    with ThreadPoolExecutor(max_workers=num_services) as executor:
        futures = [executor.submit(fetch, i,random_names[i], port_base) for i in range(num_services)]

        for future in as_completed(futures):
            print(future.result())

    for i in range(num_services):
        with open(random_names[i].replace('.json','_results.json'),'r') as f:
            final_results.extend(json.load(f))
        # os.remove(random_names[i].replace('.json','_results.json'))
    for i in range(num_services):
        os.remove(random_names[i].replace('.json','_results.json'))
    return final_results

def format_reward(predict: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0


def accuracy_reward(predict: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(
    predicts: List[str],
    ground_truths: List[str],
    format_weight: float = 0.1,
    file_path: str = "",
    num_services: int = 2,
    port_base: int = 5000,
    validity_rzero_semantic_gpu_ready_file=None,
    uid=None,
) -> List[Dict[str, float]]:
    results = []
    for i in range(len(predicts)):
        questions = re.findall(r"<question>(.*?)</question>", predicts[i], re.DOTALL)
        answers = extract_boxed_content(predicts[i])
        if questions and answers:
            try:
                question = questions[-1].strip()
                answer = answers[-1].strip()
                results.append({"question": question, "answer": answer})
            except:
                results.append({"question": "", "answer": ""})
        else:
            results.append({"question": "", "answer": ""})

    final_results = generate_results(results, num_services=num_services, port_base=port_base)
    if os.getenv("VALIDITY_RZERO_ENABLED") != "1":
        raise ValueError("RQuest requires validity gating")
    if os.getenv("VALIDITY_RZERO_DIVERSITY_MODE") != "semantic_novelty_gate":
        raise ValueError("This release supports only the K8 novelty gate")
    validity_rzero_enabled = True
    diversity_mode = "semantic_novelty_gate"
    novelty_stats = None
    if validity_rzero_enabled:
        # Importing this module can resolve/load the frozen semantic judge, so the
        # pure R-Zero path must never import it.
        gpu_ready_files = validity_rzero_semantic_gpu_ready_file or []
        if isinstance(gpu_ready_files, str):
            gpu_ready_files = [gpu_ready_files]
        unique_ready_files = set(gpu_ready_files)
        if len(unique_ready_files) > 1:
            raise RuntimeError(f"semantic batch contains multiple GPU barriers: {unique_ready_files}")
        semantic_kwargs = {"gpu_ready_file": next(iter(unique_ready_files))} if unique_ready_files else {}
        from methods.rquest.novelty_reward import compute_online_novelty
        # Novelty consumes only parsed Questioner candidates. Solver outputs
        # are used separately below to compose validity/frontier rewards.
        novelty_stats = compute_online_novelty(
            [result.get("question", "") for result in results], **semantic_kwargs
        )
    scores = []
    for i in range(len(final_results)):
        if validity_rzero_enabled and diversity_mode == "semantic_novelty_gate" and final_results[i].get("question"):
            item = final_results[i]
            stats = novelty_stats[i]
            novelty = int(stats["novelty"])
            if item["validity_decision"] == "INVALID":
                final_score = float(item["questioner_base_reward"])
            else:
                final_score = novelty * float(item["math_frontier_score"])
            print("[rquest][questioner_reward] " + json.dumps({
                "invalid_votes": item["invalid_votes"],
                "total_votes": item["total_votes"],
                "validity_decision": item["validity_decision"],
                "validity_penalty": item["validity_penalty"],
                "math_frontier_score": item["math_frontier_score"],
                "diversity_mode": diversity_mode,
                "novelty": novelty,
                "same_count": stats["same_count"],
                "compared_count": stats["compared_count"],
                "parse_failure_count": stats["parse_failure_count"],
                "final_questioner_reward": final_score,
            }))
            scores.append({
                "overall": final_score,
                "format": 1.0,
                "accuracy": float(novelty),
                "invalid_votes": float(item["invalid_votes"]),
                "validity_invalid": float(item["validity_decision"] == "INVALID"),
                "validity_valid": float(item["validity_decision"] == "VALID"),
                "validity_penalty": float(item["validity_penalty"]),
                "math_frontier_score": float(item["math_frontier_score"]),
                "novelty": float(novelty),
                "same_count": float(stats["same_count"]),
                "compared_count": float(stats["compared_count"]),
                "parse_failure_count": float(stats["parse_failure_count"]),
            })
        else:
            final_score = -0.5
            scores.append({"overall": final_score, "format": 0, "accuracy": 0.0})
    if validity_rzero_enabled and diversity_mode == "semantic_novelty_gate":
        from methods.rquest.novelty import novelty_training_diagnostics
        diagnostics = novelty_training_diagnostics(final_results, novelty_stats, uid)
        for score in scores:
            score.update(diagnostics)
        print("[rquest][semantic_novelty_gate][step_metrics] " + json.dumps(diagnostics))
    return scores
