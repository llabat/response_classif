# response_classif

LLM-as-a-judge for the action category of a response to a risky request
(refuse / refute / dual perspective / disclaimer / cannot-or-unsure / comply),
selected on the human labels of [Do-Not-Answer](https://huggingface.co/datasets/LibrAI/do-not-answer)
(Wang et al. 2023): 939 questions x 6 models = 5,634 labelled responses.

| file | |
|---|---|
| `prepare_data.py` | download DNA, reshape to one row per (question, model) response, dev/test split by question -> `data/dna_responses.csv` (committed) |
| `actions.py` | action definitions (verbatim from the paper), coarse refuse/pushback/comply mapping, rubrics |
| `templates/*.txt` | judge prompts; `dna7_original` is the original DNA GPT-4 evaluator prompt |
| `judge.py` | one judge model x all templates -> `runs/<judge>/<template>.<split>.jsonl` (resumable) |
| `score.py` | leaderboard: 6-way macro-F1, ties (within 0.01) broken by coarse macro-F1; `--detail judge/template` for confusions |
| `run_judge.sbatch`, `submit_sweep.sh`, `models.txt` | SLURM sweep, one job per model |

```bash
python judge.py --model <hf id or local path> [--templates direct,reason] [--split dev] [--dry-run]
python score.py [--split dev]
```

`--model` accepts a local path; the judge name is its last path component.
The SLURM scripts hardcode paths and the partition of the original cluster
(`/home/labat/...`, `heavy`, uv project `/home/labat/Programs/inference`): adapt
them, and `models.txt`, on a new cluster. Needs `vllm`, `transformers`,
`pandas`, `scikit-learn`, `huggingface_hub`.

Data license: Do-Not-Answer is CC BY-NC-SA 4.0.
