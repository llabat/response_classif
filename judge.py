"""
Run one judge model over every candidate template on the Do-Not-Answer
response set (data/dna_responses.csv, built by prepare_data.py), predicting
each response's action category (0-5, see actions.py). One model load serves
all templates; sweep models with one job each (submit_sweep.sh).

Templates live in templates/<name>.txt with {question}, {response} and one
rubric placeholder: {rubric} (6 actions with examples), {rubric_noex}
(without) or {rubric7} (the original DNA evaluator's 7 classes, mapped back
to the 6 released actions). A template may start with a "[system]" section
followed by "[user]". Every template asks for the answer as
<answer>index</answer>; generation stops at "</answer>".

Output: <out_dir>/<judge>/<template>.<split>.jsonl, one line per response:
    {"pair_id", "judge", "template", "split", "pred", "raw"}
pred is the 0-5 action, or null if unparseable. Existing files are skipped,
so a killed job can be resumed.

Usage:
    python judge.py --model google/gemma-4-31B-it [--templates direct,reason] [--split dev]
"""

import argparse
import json
import os
import re
from pathlib import Path

# Greedy decoding doesn't need FlashInfer's top-k/top-p sampler, whose JIT
# build fails when the active nvcc has no curand headers.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import pandas as pd
from transformers import AutoTokenizer

from actions import ACTIONS, DNA7_TO_ACTION, RUBRICS

HERE = Path(__file__).parent
TEMPLATES_DIR = HERE / "templates"
PLACEHOLDER_RE = re.compile(r"\{(question|response|rubric|rubric_noex|rubric7)\}")
ANSWER_RE = re.compile(r"<answer>\s*(\d+)\s*</answer>")


def load_template(name):
    text = (TEMPLATES_DIR / f"{name}.txt").read_text()
    system = None
    if text.startswith("[system]\n"):
        system, text = text[len("[system]\n"):].split("\n[user]\n", 1)
        system = system.strip()
    rubrics = [r for r in RUBRICS if r in text]
    if len(rubrics) != 1 or "{question}" not in text or "{response}" not in text:
        raise ValueError(f"templates/{name}.txt needs {{question}}, {{response}} and exactly one rubric placeholder")
    return {"name": name, "system": system, "user": text.strip(), "seven_way": rubrics[0] == "{rubric7}"}


def fill(template, question, response):
    # single pass, so braces inside a question/response are never re-substituted
    values = {"question": question, "response": response,
              **{k.strip("{}"): v for k, v in RUBRICS.items()}}
    return PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], template["user"])


def parse(raw, seven_way):
    matches = ANSWER_RE.findall(raw)
    if matches:
        label = int(matches[-1])
    elif raw.strip().isdigit():
        label = int(raw.strip())
    else:
        return None
    if seven_way:
        return DNA7_TO_ACTION.get(label)
    return label if label in ACTIONS else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--templates", default="all",
                        help='comma-separated template names, or "all" for every file in templates/')
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--data", default=str(HERE / "data" / "dna_responses.csv"))
    parser.add_argument("--out-dir", default=str(HERE / "runs"))
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--chat-template-kwargs", type=json.loads, default={"enable_thinking": False},
                        help="JSON forwarded to apply_chat_template")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="print one rendered prompt per template, don't load the model")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if args.split != "all":
        df = df[df["split"] == args.split]
    if args.limit:
        df = df.head(args.limit)

    names = (sorted(p.stem for p in TEMPLATES_DIR.glob("*.txt")) if args.templates == "all"
             else args.templates.split(","))
    judge = args.model.rstrip("/").split("/")[-1]
    out_dir = Path(args.out_dir) / judge
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if not (out_dir / f"{n}.{args.split}.jsonl").exists()]
    print(f"{judge}: {len(df)} {args.split} responses; templates to run: {todo} "
          f"(skipping {sorted(set(names) - set(todo))})")
    if not todo:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def render(template, question, response):
        messages = [{"role": "user", "content": fill(template, question, response)}]
        if template["system"]:
            messages.insert(0, {"role": "system", "content": template["system"]})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             **args.chat_template_kwargs)

    if args.dry_run:
        row = df.iloc[0]
        for name in todo:
            print(f"===== {name} =====")
            print(render(load_template(name), row["question"], row["response"]))
        return

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              tensor_parallel_size=args.tensor_parallel_size)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens,
                            stop=["</answer>"], include_stop_str_in_output=True)

    for name in todo:
        template = load_template(name)
        prompts = [render(template, q, r) for q, r in zip(df["question"], df["response"])]
        print(f"[{name}] submitting {len(prompts)} requests to vLLM...")
        outputs = llm.generate(prompts, params)

        path = out_dir / f"{name}.{args.split}.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")  # only a complete file counts as done
        n_failed = 0
        with open(tmp, "w") as f:
            for pair_id, output in zip(df["pair_id"], outputs):
                raw = output.outputs[0].text.strip()
                pred = parse(raw, template["seven_way"])
                n_failed += pred is None
                f.write(json.dumps({"pair_id": int(pair_id), "judge": judge, "template": name,
                                    "split": args.split, "pred": pred, "raw": raw},
                                   ensure_ascii=False) + "\n")
        tmp.rename(path)
        print(f"[{name}] wrote {path} ({n_failed} unparsed)", flush=True)


if __name__ == "__main__":
    main()
