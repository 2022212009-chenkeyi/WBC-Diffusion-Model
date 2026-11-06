#!/usr/bin/env bash
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --signal=TERM@300
#SBATCH --job-name=wbc-ddpm-sample
#SBATCH --output=/projects/keyi005ssd/WBCDiffusion/results/wbc-ddpm/slurm-sample-%j.out
#SBATCH --error=/projects/keyi005ssd/WBCDiffusion/results/wbc-ddpm/slurm-sample-%j.err

set -euo pipefail

cd /home/keyi005/projects/WBCDiffusion/01_ddpm_reproduction
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

/projects/keyi005ssd/envs/wbcdiffusion/bin/python -c 'import json, pathlib; p=pathlib.Path("/projects/keyi005ssd/WBCDiffusion/results/wbc-raabin-256-baseline-v1/status.json"); s=json.loads(p.read_text()); assert s["status"] == "training_complete" and s["completed_steps"] == 200000, s; print("formal training complete; sampling final checkpoint")'

srun --ntasks=1 --gres=gpu:pro6000:1 nvidia-smi
srun --ntasks=1 --gres=gpu:pro6000:1 /projects/keyi005ssd/envs/wbcdiffusion/bin/python -m ddpm.sample \
  --config configs/wbc_raabin_256_baseline_v1.toml \
  --num-samples 2000 \
  --batch-size 16 \
  --seed 2026 \
  --resume
