"""
Download LibrAI/do-not-answer and reshape it to one row per (question, model)
response, with its human action (0-5, see actions.py) and harmfulness labels.

The raw dataset has one row per question and six models' responses side by
side ({model}_response / _action / _harmful); judging is done per response,
regardless of which model wrote it, so the model becomes a column.

Questions are split into dev (prompt/model selection) and test (final
comparison), stratified by risk area. The split is by question text, so all
responses to a question -- and the one question that appears twice in the
dataset (ids 433/434) -- land on the same side.

Output: data/dna_responses.csv with columns
    pair_id, question_id, risk_area, types_of_harm, specific_harms, question,
    model, response, action, action_name, harmful, split

Usage:
    python prepare_data.py [--dev-frac 0.3] [--seed 0]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from actions import ACTIONS

REPO = "LibrAI/do-not-answer"
PARQUET = "data/train-00000-of-00001-6ba0076b818accff.parquet"
MODELS = ["GPT4", "ChatGPT", "Claude", "ChatGLM2", "llama2-7b-chat", "vicuna-7b"]
DATA_DIR = Path(__file__).parent / "data"


def to_long(wide):
    meta = ["id", "risk_area", "types_of_harm", "specific_harms", "question"]
    parts = []
    for model in MODELS:
        part = wide[meta + [f"{model}_response", f"{model}_action", f"{model}_harmful"]]
        part = part.rename(columns={
            "id": "question_id",
            f"{model}_response": "response",
            f"{model}_action": "action",
            f"{model}_harmful": "harmful",
        })
        parts.append(part.assign(model=model))
    long = pd.concat(parts, ignore_index=True)
    long = long.sort_values(["question_id", "model"], kind="stable").reset_index(drop=True)
    long.insert(0, "pair_id", long.index)
    long["action_name"] = long["action"].map(lambda a: ACTIONS[a]["name"])
    return long[["pair_id", "question_id", "risk_area", "types_of_harm", "specific_harms",
                 "question", "model", "response", "action", "action_name", "harmful"]]


def split_questions(wide, dev_frac, seed):
    """Map question text -> "dev"/"test", stratified by risk area."""
    rng = np.random.default_rng(seed)
    questions = wide.drop_duplicates("question")[["question", "risk_area"]]
    split = {}
    for _, group in questions.groupby("risk_area"):
        qs = group["question"].to_numpy()
        rng.shuffle(qs)
        n_dev = round(len(qs) * dev_frac)
        split.update({q: "dev" for q in qs[:n_dev]})
        split.update({q: "test" for q in qs[n_dev:]})
    return split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    path = hf_hub_download(REPO, PARQUET, repo_type="dataset", local_dir=DATA_DIR / "raw")
    wide = pd.read_parquet(path)
    assert wide["id"].is_unique and not wide.isna().any().any()

    long = to_long(wide)
    long["split"] = long["question"].map(split_questions(wide, args.dev_frac, args.seed))
    assert set(long["action"]) <= set(ACTIONS) and long["split"].notna().all()

    out = DATA_DIR / "dna_responses.csv"
    long.to_csv(out, index=False)
    print(f"Wrote {len(long)} responses ({wide.shape[0]} questions x {len(MODELS)} models) to {out}")
    print(pd.crosstab(long["action_name"], long["split"], margins=True))
    print(pd.crosstab(long["harmful"], long["split"], margins=True))


if __name__ == "__main__":
    main()
