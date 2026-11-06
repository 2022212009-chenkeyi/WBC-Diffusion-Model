"""Compute explicitly non-clinical Inception FID/KID references for WBC samples."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import tomllib

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
except ImportError as error:
    raise SystemExit(
        "Install the pinned image evaluation dependencies from requirements-eval.txt."
    ) from error


class ManifestUint8Dataset(Dataset):
    """Load a frozen manifest split as uint8 RGB tensors for TorchMetrics."""

    def __init__(self, manifest_path: Path, image_root: Path, split: str, size: int):
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            self.paths = [
                image_root / row["relative_path"]
                for row in csv.DictReader(stream)
                if row["split"] == split and row["qc_decision"] == "keep"
            ]
        if not self.paths:
            raise ValueError(f"No QC-kept images for split={split!r}")
        missing = [path for path in self.paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} manifest files are missing; first: {missing[0]}")
        self.transform = transforms.Compose(
            [
                transforms.Resize((size, size), interpolation=InterpolationMode.LANCZOS),
                transforms.PILToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def update_real(metric, loader: DataLoader, device: torch.device, real: bool = True) -> None:
    for images in loader:
        metric.update(images.to(device, non_blocking=True), real=real)


def build_metrics(feature: int, kid_subsets: int, kid_subset_size: int, device: torch.device) -> dict:
    return {
        "fid": FrechetInceptionDistance(feature=feature, normalize=False).to(device),
        "kid": KernelInceptionDistance(
            feature=feature,
            subsets=kid_subsets,
            subset_size=kid_subset_size,
            normalize=False,
        ).to(device),
    }


def compute_metrics(metrics: dict) -> dict[str, float]:
    fid_value = float(metrics["fid"].compute().cpu())
    kid_mean, kid_std = metrics["kid"].compute()
    return {"fid": fid_value, "kid_mean": float(kid_mean.cpu()), "kid_std": float(kid_std.cpu())}


def prdc(real_features: torch.Tensor, fake_features: torch.Tensor, nearest_k: int = 5) -> dict[str, float]:
    """Improved precision/recall, density, coverage using k-NN feature manifolds."""
    real_features = real_features.float()
    fake_features = fake_features.float()
    if min(len(real_features), len(fake_features)) <= nearest_k:
        raise ValueError(f"PRDC requires more than k={nearest_k} samples in both sets")
    real_distances = torch.cdist(real_features, real_features)
    fake_distances = torch.cdist(fake_features, fake_features)
    real_distances.fill_diagonal_(float("inf"))
    fake_distances.fill_diagonal_(float("inf"))
    real_radii = real_distances.topk(nearest_k, dim=1, largest=False).values[:, -1]
    fake_radii = fake_distances.topk(nearest_k, dim=1, largest=False).values[:, -1]
    cross_distances = torch.cdist(real_features, fake_features)
    within_real_manifold = cross_distances <= real_radii[:, None]
    within_fake_manifold = cross_distances <= fake_radii[None, :]
    return {
        "precision": float(within_real_manifold.any(dim=0).float().mean().cpu()),
        "recall": float(within_fake_manifold.any(dim=1).float().mean().cpu()),
        "density": float(within_real_manifold.float().sum(dim=0).mean().div(nearest_k).cpu()),
        "coverage": float((cross_distances.min(dim=1).values <= real_radii).float().mean().cpu()),
        "nearest_k": nearest_k,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True, help="Completed uint8 NHWC .npy from ddpm.sample")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, choices=(64, 192, 768, 2048), default=768)
    parser.add_argument("--kid-subsets", type=int, default=100)
    parser.add_argument("--kid-subset-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible. Start this command inside an allocated GPU job.")

    config = tomllib.loads(args.config.read_text())
    exp, data_cfg, model_cfg = config["experiment"], config["data"], config["model"]
    samples = np.load(args.samples, mmap_mode="r")
    expected_shape = (samples.shape[0], model_cfg["image_size"], model_cfg["image_size"], model_cfg["channels"])
    if samples.dtype != np.uint8 or samples.ndim != 4 or samples.shape != expected_shape:
        raise ValueError(f"expected uint8 NHWC samples with shape (*,{expected_shape[1:]}); got {samples.dtype} {samples.shape}")
    if samples.shape[0] < args.kid_subset_size:
        raise ValueError("generated sample count must be at least --kid-subset-size")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    manifest_path = Path(data_cfg["split_manifest"])
    image_root = Path(data_cfg["image_root"])
    size = model_cfg["image_size"]
    train_loader = DataLoader(ManifestUint8Dataset(manifest_path, image_root, "train", size), **loader_kwargs)
    test_loader = DataLoader(ManifestUint8Dataset(manifest_path, image_root, "test", size), **loader_kwargs)

    metrics_train = build_metrics(args.feature_dim, args.kid_subsets, args.kid_subset_size, device)
    metrics_test = build_metrics(args.feature_dim, args.kid_subsets, args.kid_subset_size, device)
    metrics_real_reference = build_metrics(args.feature_dim, args.kid_subsets, args.kid_subset_size, device)
    feature_extractor = FeatureExtractorInceptionV3(
        name="inception-v3-compat", features_list=[str(args.feature_dim)]
    ).to(device).eval()
    update_real(metrics_train["fid"], train_loader, device)
    update_real(metrics_train["kid"], train_loader, device)
    train_feature_batches = []
    with torch.inference_mode():
        for images in train_loader:
            train_feature_batches.append(feature_extractor(images.to(device, non_blocking=True))[0].cpu())
    train_features = torch.cat(train_feature_batches).to(device)
    test_feature_batches = []
    for images in test_loader:
        images_device = images.to(device, non_blocking=True)
        metrics_test["fid"].update(images_device, real=True)
        metrics_test["kid"].update(images_device, real=True)
        test_feature_batches.append(feature_extractor(images_device)[0].cpu())
    test_features = torch.cat(test_feature_batches).to(device)
    update_real(metrics_real_reference["fid"], train_loader, device)
    update_real(metrics_real_reference["kid"], train_loader, device)
    update_real(metrics_real_reference["fid"], test_loader, device, real=False)
    update_real(metrics_real_reference["kid"], test_loader, device, real=False)

    generated_feature_batches = []
    for start in range(0, samples.shape[0], args.batch_size):
        batch = np.array(samples[start : start + args.batch_size], copy=True)
        images = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous().to(device)
        generated_feature_batches.append(feature_extractor(images)[0].cpu())
        for metric_group in (metrics_train, metrics_test):
            metric_group["fid"].update(images, real=False)
            metric_group["kid"].update(images, real=False)
        if min(start + len(batch), samples.shape[0]) % 500 == 0:
            print(f"features_generated={min(start + len(batch), samples.shape[0])}/{samples.shape[0]}", flush=True)

    generated_features = torch.cat(generated_feature_batches).to(device)
    split_generator = torch.Generator().manual_seed(args.seed)
    split_order = torch.randperm(len(test_features), generator=split_generator)
    midpoint = len(test_features) // 2
    real_reference_a = test_features[split_order[:midpoint].to(device)]
    real_reference_b = test_features[split_order[midpoint:].to(device)]

    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": f"{exp['name']}-{exp['run_suffix']}" if exp.get("run_suffix") else exp["name"],
        "checkpoint_step": json.loads(args.samples.with_suffix(".json").read_text()).get("checkpoint_step")
        if args.samples.with_suffix(".json").is_file()
        else None,
        "sample_file": str(args.samples),
        "generated_samples": int(samples.shape[0]),
        "real_train_samples": len(train_loader.dataset),
        "real_test_samples": len(test_loader.dataset),
        "input_size": size,
        "feature_extractor": "TorchMetrics/torch-fidelity Inception-v3",
        "feature_dimension": args.feature_dim,
        "feature_nearest_neighbor_search": "top-5 Euclidean neighbors in the same Inception feature space; candidates require visual review and are not proof of copying",
        "protocol_limitations": [
            "These generic natural-image features are not validated WBC morphology embeddings.",
            "FID/KID are descriptive distribution distances and do not establish medical realism.",
            "Report the real train-vs-test reference alongside generated-vs-real scores.",
        ],
        "generated_vs_train": compute_metrics(metrics_train),
        "generated_vs_test": compute_metrics(metrics_test),
        "real_train_vs_test_reference": compute_metrics(metrics_real_reference),
        "prdc_generated_vs_test": prdc(test_features, generated_features),
        "prdc_real_test_split_reference": prdc(real_reference_a, real_reference_b),
        "kid_subset_count": args.kid_subsets,
        "kid_subset_size": args.kid_subset_size,
        "metric_seed": args.seed,
        "split_manifest": str(manifest_path),
    }
    output_root = Path(exp["output_dir"])
    run_suffix = exp.get("run_suffix", "")
    run_name = f"{exp['name']}-{run_suffix}" if run_suffix else exp["name"]
    output_dir = output_root / "results" / run_name / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    nearest_path = output_dir / f"generated_feature_nearest_train_step{results['checkpoint_step'] or 'unknown'}.csv"
    with nearest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("generated_index", "rank", "train_relative_path", "inception_euclidean_distance"),
        )
        writer.writeheader()
        with torch.inference_mode():
            for start in range(0, len(generated_features), 256):
                distances = torch.cdist(generated_features[start : start + 256], train_features)
                nearest_distances, nearest_indices = distances.topk(k=5, dim=1, largest=False)
                for batch_offset in range(nearest_indices.shape[0]):
                    generated_index = start + batch_offset
                    for rank in range(nearest_indices.shape[1]):
                        train_index = int(nearest_indices[batch_offset, rank].item())
                        writer.writerow(
                            {
                                "generated_index": generated_index,
                                "rank": rank + 1,
                                "train_relative_path": str(train_loader.dataset.paths[train_index]),
                                "inception_euclidean_distance": float(nearest_distances[batch_offset, rank].item()),
                            }
                        )
    output_path = output_dir / f"fid_kid_step{results['checkpoint_step'] or 'unknown'}_n{samples.shape[0]}.json"
    results["generated_feature_nearest_train_csv"] = str(nearest_path)
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
