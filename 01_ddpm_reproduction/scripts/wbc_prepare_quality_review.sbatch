#!/usr/bin/env bash
#SBATCH --partition=cluster02
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=wbc-ddpm-reviewprep
#SBATCH --output=/projects/keyi005ssd/WBCDiffusion/results/wbc-ddpm/slurm-reviewprep-%j.out
#SBATCH --error=/projects/keyi005ssd/WBCDiffusion/results/wbc-ddpm/slurm-reviewprep-%j.err

set -euo pipefail

cd /home/keyi005/projects/WBCDiffusion/01_ddpm_reproduction
sample_file=/projects/keyi005ssd/WBCDiffusion/generated_samples/wbc-raabin-256-baseline-v1/samples_ema_step0200000_n2000_seed2026.npy
test -s "$sample_file"
python_bin=/projects/keyi005ssd/envs/wbcdiffusion/bin/python

"$python_bin" -m ddpm.analyze_wbc_memorization \
  --config configs/wbc_raabin_256_baseline_v1.toml \
  --samples "$sample_file" \
  --threshold 4

"$python_bin" -m ddpm.prepare_wbc_blind_review \
  --config configs/wbc_raabin_256_baseline_v1.toml \
  --samples "$sample_file" \
  --count-per-source 200 \
  --reviewers 3 \
  --seed 20261105
