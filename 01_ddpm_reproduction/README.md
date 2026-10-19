# DDPM CIFAR-10 reproduction

## Objective

Reproduce the unconditional CIFAR-10 DDPM experiment from Ho, Jain, and Abbeel (2020), then carry the validated training and evaluation workflow forward to WBC data. The first milestone is a faithful CIFAR-10 implementation and complete run record, not a claim of matching the paper's FID from a short run.

## Why this implementation

The authors' reference code uses TensorFlow 1.15 / Python 3.5, TPU v3-8, and Google Cloud Storage. This project has a verified Python 3.11, PyTorch 2.8.0+cu128 environment and an NVIDIA A40, so the primary path is a PyTorch implementation that follows the CIFAR-10 reference configuration. Keep any execution of the historical TensorFlow code in a separate `ddpm-original` environment; do not downgrade or alter `wbcdiffusion` for it.

Reference sources:

- Ho et al., *Denoising Diffusion Probabilistic Models*: https://arxiv.org/abs/2006.11239
- Authors' code and CIFAR-10 settings: https://github.com/hojonathanho/diffusion/blob/master/scripts/run_cifar.py

## Reference configuration to preserve

| Item | Reference setting |
|---|---|
| Dataset | CIFAR-10, train split, 50,000 RGB images, 32 x 32 |
| Data range | Convert uint8 pixels to `[-1, 1]` |
| Augmentation | Random horizontal flip |
| Network | U-Net, base width 128, channel multipliers `(1, 2, 2, 2)`, 2 residual blocks per level, attention at 16 x 16, dropout 0.1 |
| Network details | 32-group normalization; sinusoidal timestep embeddings; fan-average uniform kernels; zero-initialized residual, attention-output, and final convolutions, following the authors' reference U-Net |
| Diffusion | 1,000 steps; linear beta schedule from `1e-4` to `0.02` |
| Prediction / objective | Predict epsilon; unweighted MSE noise objective |
| Reverse variance | Fixed large |
| Optimizer | Adam, learning rate `2e-4`, 5,000-step warmup, gradient norm clip 1.0 |
| Batch size | 128 global batch (single A40 in this run) |
| Seed | 2026 for the first run; record any nondeterministic kernels |

The reference setup uses 50,000 generated samples for its CIFAR-10 Inception/FID evaluation. The authors' evaluator uses TensorFlow-GAN's Inception-v1 graph. Record the metric implementation and sample count because FID varies with preprocessing and feature extractor. Do not interpret a low-step smoke run as a paper reproduction.

## Work stages

1. **Implementation and preflight:** implement the forward/reverse equations, CIFAR-10 U-Net, training loop, checkpoint resume, sample generation, and environment capture. Confirm one batch forward/backward and one optimizer update on the allocated A40 before committing to a long run.
2. **Short pipeline run:** train a small, explicitly non-paper run to verify data loading, loss logging, checkpoint save/resume, and sample output. Label it `smoke`, never report it as the reproduction result.
3. **Paper-configuration run:** train with the table above. Save periodic checkpoints and logs on confirmed SSD storage; retain the latest three periodic checkpoints plus `latest.pt` to control storage, and keep the dataset cache and completed artifacts according to cluster policy.
4. **Evaluation:** sample from the final EMA checkpoint if EMA is enabled, using the same documented CIFAR-10 FID protocol and 50,000 generated images. Also save fixed-seed sample grids and record throughput, elapsed time, and peak VRAM.
5. **Analysis:** compare to the paper's reported CIFAR-10 FID 3.17 and Inception Score 9.46, explain differences, preserve failures, and only then move on to WBC subset analysis and the WBC-DDPM baseline.

`ddpm.evaluate` provides a reproducible PyTorch secondary protocol: TorchMetrics 1.9.0 / torch-fidelity Inception-v3 (2048D), FID against CIFAR-10 train and test splits, KID with 100 subsets of 1,000, and 10-split Inception Score. These values support within-project comparisons but are not numerically interchangeable with the paper's TensorFlow-GAN Inception-v1 metrics. A direct numerical comparison requires also porting the authors' Inception-v1 evaluator.

## Storage and records

- Keep code/configuration in this project directory.
- Keep the canonical CIFAR-10 archive and extracted batches under `/projects/keyi005ssd/WBCDiffusion/datasets/cifar10`; verify the archive checksum and treat these files as immutable. Normalization and random flips happen in memory.
- Put active run artifacts under `/projects/keyi005ssd/WBCDiffusion/checkpoints/ddpm-cifar10-reference/`, logs and metrics under `results/ddpm-cifar10-reference/`, and generated images under `generated_samples/ddpm-cifar10-reference/`. Verify personal quota before a long run; `df` is shared-filesystem capacity, not user quota.
- Keep large outputs out of Git. Preserve the config, exact command, code revision, environment export, Slurm job/GPU details, and run log.
- Record the experiment using [`../experiments/experiment_template.md`](../experiments/experiment_template.md). Proposed first formal ID: `DDPM-C10-001`.

## Current status

- Plan and initial PyTorch implementation are in this directory.
- The user's environment report verified A40 / Python 3.11 / PyTorch 2.8.0+cu128. The Codex process can reach the A40 only while an active Slurm allocation exists.
- The first smoke attempt in Slurm job 195519 ended during the partial dataset download. A later corrected two-step smoke passed inside job 197959, including optimizer updates, checkpoint writing, and sample generation. See [`../experiments/DDPM-C10-SMOKE-20261009.md`](../experiments/DDPM-C10-SMOKE-20261009.md).
- The archive was resumed on the login node and validated by Torchvision against MD5 `c58f30108f718f92721af3b95e74349a`; Torchvision loaded all 50,000 training images at 32 x 32.
- The corrected U-Net has 35,746,307 parameters, consistent with the authors' 35.7M CIFAR-10 model. The two-step A40 smoke passed inside allocation 197959: batch-128 optimizer updates, checkpoint writing, and the four-image reverse-sampling grid all completed. Peak VRAM was 6.77 GiB. See [`../experiments/DDPM-C10-SMOKE-20261009.md`](../experiments/DDPM-C10-SMOKE-20261009.md).
- Formal run `DDPM-C10-001` completed all 800,000 steps in Slurm job 198001 (exit code 0, elapsed 2 days 1:19:48) on `gpu-a40-1`; final loss was 0.0238913 and peak VRAM 6.77 GiB. The 50,000-image EMA sampling job 203843 and evaluation job 203844 both completed successfully. Metrics with TorchMetrics/torch-fidelity Inception-v3: train FID 3.2839, test FID 5.3407, KID 0.0009742 +/- 0.0002321, and IS 9.5411 +/- 0.0873. These are secondary-protocol scores, not numerically identical to the paper's TensorFlow-GAN/Inception-v1 scores. A sample preview is available at [`../generated_samples/ddpm-cifar10-reference/samples_ema_50000_preview.png`](../generated_samples/ddpm-cifar10-reference/samples_ema_50000_preview.png); compact metric and sample metadata are in `../results/ddpm-cifar10-reference/`. The full 50,000-image NumPy array is stored on project SSD. Smoke artifacts remain isolated under their own run names.
- Full training is complete. The queued 50,000-sample generation job and remaining metric evaluation are tracked in [`../experiments/DDPM-C10-001.md`](../experiments/DDPM-C10-001.md).

## Command interface

Invoke the entrypoint inside an active GPU allocation, from this directory, using the full SSD environment path. Do not rely on an ambiguous Conda environment name because both Home and SSD register `wbcdiffusion`.

```bash
/projects/keyi005ssd/envs/wbcdiffusion/bin/python -m ddpm.train --config configs/cifar10.toml --max-steps 2
```

The two-step smoke and full formal run are complete. Slurm jobs 198001 (training), 203843 (50,000-sample generation), and 203844 (evaluation) all completed successfully. The evaluator records TorchMetrics/Inception-v3 values as secondary to the original TF-GAN/Inception-v1 protocol. The next stage is WBC subset analysis; establish data splits and preprocessing before starting that experiment.

Once the sample metadata reports `complete`, submit `scripts/ddpm_evaluate.sbatch` to compute the documented TorchMetrics/Inception-v3 secondary metrics. Do not submit evaluation against a `.partial.npy` file.
