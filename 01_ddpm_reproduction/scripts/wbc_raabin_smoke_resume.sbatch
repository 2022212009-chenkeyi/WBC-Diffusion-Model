#!/usr/bin/env bash
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=wbc-ddpm-resume-check
#SBATCH --output=/projects/keyi005ssd/WBCDiffusion/results/wbc-ddpm/slurm-resume-%j.out
#SBATCH --error=/projects/keyi005ssd/WBCDiffusion/results/wbc-ddpm/slurm-resume-%j.err

set -euo pipefail

cd /home/keyi005/projects/WBCDiffusion/01_ddpm_reproduction
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

checkpoint=/projects/keyi005ssd/WBCDiffusion/checkpoints/wbc-raabin-128-smoke/latest.pt
test -s "$checkpoint"

srun --ntasks=1 --gres=gpu:a6000:1 /projects/keyi005ssd/envs/wbcdiffusion/bin/python -m ddpm.train \
  --config configs/wbc_raabin_128_smoke.toml \
  --max-steps 2 \
  --subset-size 64 \
  --resume

/projects/keyi005ssd/envs/wbcdiffusion/bin/python -c 'import json, pathlib; p=pathlib.Path("/projects/keyi005ssd/WBCDiffusion/results/wbc-raabin-128-smoke/status.json"); s=json.loads(p.read_text()); assert s["status"] == "smoke_only" and s["completed_steps"] == 2, s; print("checkpoint resume path verified")'
