import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

URL      = "http://localhost:8001/v1/chat/completions"
MODEL    = "Qwen/Qwen3-32B"
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
MAX_TOKENS  = 16384 * 2
K_VALUES    = [1, 2]
MAX_K       = max(K_VALUES)
MAX_WORKERS = 32
CACHE_DIR   = "eval_results/imobench_claude_32k_n-trials2/"
N_SAMPLES = 128


from typing import Optional, Dict, Any
import time
import requests
import warnings
import os

##### --------------- code from imobench[equality template from math500] --------------- #####

EQUALITY_TEMPLATE = r"""
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

    Expression 1: x=5
    Expression 2: 5

Yes

    Expression 1: 60, 100, 8p (for all odd primes p)
    Expression 2: 60, 100, \\text{ and } 8p \\text{ for any odd prime } p

Yes
---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()

def check_equality_judge(
        expr1: str, 
        expr2: str,
        max_tokens: int = 10,
        url: str = None,
        model: str = 'gpt-4o-mini'
    ):
    prompt = EQUALITY_TEMPLATE % {"expression1": expr1, "expression2": expr2}

    token = os.getenv("ELIZA_TOKEN", "EMPTY_TOKEN")
    url = os.getenv("JUDGE_URL", url)
    model = os.getenv("JUDGE_MODEL", model)
    headers = {"Authorization": f"OAuth {token}"}
    messages=[{"role": "user", "content": prompt}]
    payload = {
        "model": model, 
        "messages": messages,
        "max_tokens": max_tokens,
    }

    with warnings.catch_warnings(action="ignore"):
        response_json = requests.post(url, json=payload, headers=headers, verify=False).json()
    print(response_json)
    response_text = response_json['response']['choices'][0]["message"]['content']
    return response_text.lower().strip() == "yes"

def _judge_equivalence(pred: str, gt: str, max_retries: int = 10, sleep_s: float = 0.5) -> bool:
    last_err = None
    for attempt in range(max_retries):
        try:
            return bool(check_equality_judge(pred, gt))
        except Exception as e:
            last_err = e
            time.sleep(sleep_s)
            sleep_s *= 2.0
    raise last_err


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the content inside the LAST occurrence of \\boxed{...}.
    Simple version: grabs until the next '}'.
    (Matches your parse_boxed_answer behavior.)
    """
    token = r"\boxed{"
    last_start = text.rfind(token)
    if last_start == -1:
        return None

    content_start = last_start + len(token)
    content_end = text.find("}", content_start)
    if content_end == -1:
        return None

    return text[content_start:content_end].strip()


def compute_score(
    solution: str,
    ground_truth: str,
) -> dict:
    """
    VERL-style score for IMOBench using an LLM judge for equality.

    Args:
      solution: model output text (may include reasoning); expects final answer in \\boxed{...}
      ground_truth: target answer string (latex or plain)
      extra_info: unused (kept for API compatibility)
      judge_fn: function(pred: str, gt: str) -> bool
               If None, will use global `_judge_equivalence` if available.

    Returns:
      dict with keys consistent with other reward modules.
    """
    pred = extract_boxed_answer(solution)

    if pred is None or pred == "":
        return {
            "score": 0.0,
            "acc": 0.0,
            "pred": None,
            "incorrect_format": 1,
            "feedback": "No \\boxed{...} answer found.",
        }

    ok = bool(_judge_equivalence(pred, ground_truth))
    reward = float(ok)
    return {
        "score": reward,
        "acc": reward,
        "pred": pred,
        "incorrect_format": 0,
        "feedback": "",
    }

##### --------------- end of code from imobench --------------- #####

def call_vllm(url, model, prompt, max_tokens, sampling):
    payload = {"model": model, "messages": prompt, "max_tokens": max_tokens, **sampling}
    resp = requests.post(url, json=payload, timeout=36000)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def cache_path(idx, sample_id, use_hint):
    tag = "hint" if use_hint else "nohint"
    return os.path.join(CACHE_DIR, f"{idx}__{sample_id}__{tag}.json")


def eval_sample_once(row, use_hint, sample_id):
    path = cache_path(row.name, sample_id, use_hint)
    if os.path.exists(path):
        with open(path) as f:
            result = json.load(f)
        if "gt" not in result:
            result["gt"] = row["reward_model"]["ground_truth"]
            with open(path, "w") as f:
                json.dump(result, f)
        return result

    prompt = row["prompt"].tolist()
    if use_hint:
        prompt[0]["content"] = f"{prompt[0]['content']}\n\nHINT:\n{row['hints']}"

    solution = call_vllm(URL, MODEL, prompt, MAX_TOKENS, SAMPLING)
    score    = compute_score(solution, row["reward_model"]["ground_truth"])

    result = {"idx": row.name, "sample_id": sample_id, "use_hint": use_hint,
              "acc": score["acc"], "pred": score["pred"], "solution": solution}
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f)
    return result


def run_eval(df, use_hint):
    tasks = [(row, use_hint, s) for _, row in df.iterrows() for s in range(MAX_K)]
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(eval_sample_once, *t): t for t in tasks}
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"hint={use_hint}"):
            try:
                results.append(f.result())
            except Exception as e:
                _, row, sid = futures[f]
                results.append({"idx": row.name, "sample_id": sid, "acc": 0, "error": str(e)})
    return pd.DataFrame(results).sort_values(["idx", "sample_id"]).reset_index(drop=True)

import numpy as np

def pass_at_k(flat_df, k, n_trials=10):
    rng = np.random.default_rng(42)
    per_problem = [grp["acc"].values for _, grp in flat_df.groupby("idx")]
    
    trial_scores = []
    for _ in range(n_trials):
        scores = [int(rng.choice(accs, size=k, replace=False).sum() > 0) for accs in per_problem]
        trial_scores.append(np.mean(scores))
    
    return float(np.mean(trial_scores)), float(np.std(trial_scores))

df = pd.read_parquet("train.parquet")
df = df.sample(n=min(df.shape[0], N_SAMPLES), random_state=42)

output = {}
flat_dfs = {}
for use_hint in [False, True]:
    label   = "with_hint" if use_hint else "no_hint"
    flat_df = run_eval(df, use_hint)
    flat_dfs[label] = flat_df

    print(f"\n[{label}]")
    metrics = {}
    for k in K_VALUES:
        mean, std = pass_at_k(flat_df, k)
        metrics[f"pass@{k}"] = {"mean": mean, "std": std}
        print(f"  pass@{k} = {mean:.4f} ± {std:.4f}")
    output[label] = {"metrics": metrics, "raw": flat_df.to_dict(orient="records")}
