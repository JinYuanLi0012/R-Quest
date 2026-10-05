import json
from mathruler.grader import extract_boxed_content, grade_answer
import requests
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
args = parser.parse_args()

STORAGE_PATH = os.getenv("STORAGE_PATH")
api_key = os.environ["OPENAI_API_KEY"]
api_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"



def process_example(answer, response):
    try:
        example = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a math answer checker."},
                {"role": "user", "content": f"Hi, there is a answer: {answer}\n\n, and the ground truth answer is: {response}\n\n, please check whether the answer is correct or not, and return the **only** Yes or No."}
            ],
            "temperature": 0.1
        }
        response = requests.post(api_url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json=example, timeout=20)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception:
        raise RuntimeError("GPT-4o answer recheck failed; check the API configuration and retry.") from None
new_results = []
for model_name in [args.model_name]:
    for dataset in [
    "math",
    "gsm8k", 
    "amc",
    "minerva",
    "olympiad",
    "aime2024",
    "aime2025",
    ]:
        with open(f'{STORAGE_PATH}/evaluation/{model_name.replace("/","_")}/results_{dataset}.json', 'r') as f:
            results = json.load(f)

        for i in tqdm(range(len(results)-1)):
                if results[i]['score'] < 0.5:
                    gpt_check = process_example(results[i]['answer'],results[i]['response'])
                    if "yes" in gpt_check.lower():
                        results[i]['score']=1
        new_results.append({
            'model': model_name,
            'dataset': dataset,
            'score': round(sum([result['score'] for result in results[:-1]])/len(results[:-1])*100, 2)
        })
        print(new_results)
        with open(os.getenv('EVAL_RESULTS_FILE', 'final_results.jsonl'), 'a') as f:
            json.dump({
                'model': model_name,
                'dataset': dataset,
                'score': round(sum([result['score'] for result in results[:-1]])/len(results[:-1])*100, 2)
            }, f)
            f.write('\n')





