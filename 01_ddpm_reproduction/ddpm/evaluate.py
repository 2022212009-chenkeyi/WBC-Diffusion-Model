"""Compute documented CIFAR-10 FID, KID, and Inception Score protocols."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import tomllib

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    from torchmetrics.image.kid import KernelInceptionDistance
except ImportError as error:
    raise SystemExit(
        "Install the pinned evaluation dependencies from requirements-eval.txt in the active environment."
    ) from error


def update_real(metric, loader: DataLoader, device: torch.device, real: bool = True) -> None:
    for images, _ in loader:
        metric.update(images.to(device, non_blocking=True), real=real)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True, help="Completed uint8 NHWC .npy file from ddpm.sample.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible. Start this command inside an allocated GPU job.")
    config = tomllib.loads(args.config.read_text())
    data_cfg, eval_cfg = config["data"], config["evaluation"]
    samples = np.load(args.samples, mmap_mode="r")
    expected = eval_cfg["sample_count"]
    if samples.dtype != np.uint8 or samples.shape != (expected, 32, 32, 3):
        raise ValueError(f"expected uint8 ({expected},32,32,3), found {samples.dtype} {samples.shape}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    fid_train = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    fid_test = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid = KernelInceptionDistance(
        feature=2048,
        subsets=eval_cfg["kid_subsets"],
        subset_size=eval_cfg["kid_subset_size"],
        normalize=False,
    ).to(device)
    inception_score = InceptionScore(splits=10, normalize=False).to(device)

    transform = transforms.PILToTensor()
    train_set = datasets.CIFAR10(root=data_cfg["root"], train=True, download=False, transform=transform)
    test_set = datasets.CIFAR10(root=data_cfg["root"], train=False, download=False, transform=transform)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    started = datetime.now(timezone.utc)
    update_real(fid_train, train_loader, device)
    update_real(kid, train_loader, device)
    update_real(fid_test, test_loader, device)
    for start in range(0, expected, args.batch_size):
        images = torch.from_numpy(np.array(samples[start : start + args.batch_size], copy=True))
        images = images.permute(0, 3, 1, 2).contiguous().to(device)
        fid_train.update(images, real=False)
        fid_test.update(images, real=False)
        kid.update(images, real=False)
        inception_score.update(images)
        if (start + len(images)) % 5000 == 0 or start + len(images) == expected:
            print(f"evaluated_generated={start + len(images)}/{expected}", flush=True)

    fid_train_value = float(fid_train.compute().cpu())
    fid_test_value = float(fid_test.compute().cpu())
    kid_mean, kid_std = kid.compute()
    inception_mean, inception_std = inception_score.compute()
    results = {
        "timestamp_utc": started.isoformat(),
        "generated_samples": expected,
        "real_train_samples": len(train_set),
        "real_test_samples": len(test_set),
        "sample_file": str(args.samples),
        "protocol": "TorchMetrics 1.9.0 with torch-fidelity Inception-v3 features (2048D); not numerically identical to the original TensorFlow-GAN Inception-v1 protocol",
        "fid_train": fid_train_value,
        "fid_test": fid_test_value,
        "kid_mean": float(kid_mean.cpu()),
        "kid_std": float(kid_std.cpu()),
        "kid_subsets": eval_cfg["kid_subsets"],
        "kid_subset_size": eval_cfg["kid_subset_size"],
        "inception_score_mean": float(inception_mean.cpu()),
        "inception_score_std": float(inception_std.cpu()),
    }
    output_dir = Path(config["experiment"]["output_dir"]) / "results" / config["experiment"]["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"metrics_step{int(args.samples.stem.split('_step')[-1].split('_')[0]):07d}.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
