#!/bin/bash
# Submit one run_judge.sbatch job per model in models.txt, each with the GPU
# count listed there. Extra args are forwarded to judge.py, e.g.
#   ./submit_sweep.sh --split test --templates reason,direct
# Then: python score.py [--split ...]

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

grep -v '^\s*#' "$HERE/models.txt" | while read -r model n_gpus; do
    [ -z "$model" ] && continue
    sbatch --gres=gpu:"$n_gpus" --job-name="dna_judge_${model##*/}" \
        "$HERE/run_judge.sbatch" "$model" "$@"
done
