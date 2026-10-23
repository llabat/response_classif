#!/bin/bash
#SBATCH --job-name=dna_judge
#SBATCH --output=/home/labat/Programs/ft_values/logs/%x_%j.out
#SBATCH --error=/home/labat/Programs/ft_values/logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=heavy

# One judge model, every template (or --templates ...), on the dev split by
# default. Tensor parallel size = number of allocated GPUs, so for a 70B judge
# submit with `sbatch --gres=gpu:2 ...` (submit_sweep.sh does this).
#
# Usage: sbatch [--gres=gpu:N] run_judge.sbatch <hf_model> [judge.py args...]

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch [--gres=gpu:N] run_judge.sbatch <hf_model> [--templates ...] [--split dev|test|all]" >&2
    exit 1
fi
MODEL="$1"; shift

HERE=/home/labat/Programs/ft_values/sft_characterization/response_classif
UV_PROJECT=/home/labat/Programs/inference
N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | grep -c . || true)

# Same vLLM env fixes as generation_exp/run_generate_texts.sbatch: CUDA_HOME
# pointed at the venv's pip-installed toolkit, and spawn instead of fork for
# vLLM workers (needed with tensor parallelism).
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME=$UV_PROJECT/.venv/lib/python3.11/site-packages/nvidia/cu13
export VLLM_WORKER_MULTIPROC_METHOD=spawn

set -x
cd "$HERE"  # actions.py and templates/ are resolved relative to judge.py
uv run --no-sync --project "$UV_PROJECT" python "$HERE/judge.py" \
    --model "$MODEL" --tensor-parallel-size "$N_GPUS" "$@"
