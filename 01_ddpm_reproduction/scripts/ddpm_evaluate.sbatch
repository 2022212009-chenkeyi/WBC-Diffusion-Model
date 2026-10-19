#!/usr/bin/env bash
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=ddpm-c10-eval
#SBATCH --output=/projects/keyi005ssd/WBCDiffusion/slurm-ddpm-eval-%j.out

set -euo pipefail

cd /home/keyi005/projects/WBCDiffusion/01_ddpm_reproduction
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-not-set}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-not-set}"
nvidia-smi

SAMPLES=/projects/keyi005ssd/WBCDiffusion/generated_samples/ddpm-cifar10-reference/samples_ema_step0800000_n50000_seed2026.npy
if [[ ! -f "$SAMPLES" ]]; then
  echo "complete 50,000-sample array is missing: $SAMPLES" >&2
  exit 2
fi

exec /projects/keyi005ssd/envs/wbcdiffusion/bin/python -u -m ddpm.evaluate \
  --config configs/cifar10.toml \
  --samples "$SAMPLES" \
  --batch-size 128 \
  --num-workers 4 \
  --seed 2026
