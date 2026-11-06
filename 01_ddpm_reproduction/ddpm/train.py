"""Train the CIFAR-10 DDPM using a TOML experiment configuration."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import sys
import tomllib

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils
from torchvision.transforms import InterpolationMode

from .diffusion import GaussianDiffusion
from .model import UNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted-or-unavailable"


def write_environment(path: Path) -> None:
    details = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_revision": git_revision(),
    }
    path.write_text(json.dumps(details, indent=2) + "\n")


def save_checkpoint(path: Path, payload: dict, keep_steps: int) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)
    if path.name == "latest.pt":
        step_path = path.parent / f"step_{payload['step']:07d}.pt"
        step_temporary = step_path.with_suffix(step_path.suffix + ".tmp")
        torch.save(payload, step_temporary)
        os.replace(step_temporary, step_path)
        old_checkpoints = sorted(path.parent.glob("step_*.pt"))
        for old_path in old_checkpoints[:-keep_steps]:
            old_path.unlink()


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([gpu_state.cpu() for gpu_state in state["cuda"]])


def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        model_params = dict(model.named_parameters())
        for name, ema_param in ema.named_parameters():
            ema_param.lerp_(model_params[name].detach(), 1.0 - decay)
        model_buffers = dict(model.named_buffers())
        for name, ema_buffer in ema.named_buffers():
            ema_buffer.copy_(model_buffers[name])


@torch.no_grad()
def evaluate_noise_loss(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    device: torch.device,
    seed: int,
) -> float:
    """Measure validation epsilon loss with fixed randomness without advancing train RNG."""
    rng_state = capture_rng_state()
    was_training = model.training
    model.eval()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    weighted_loss = 0.0
    image_count = 0
    try:
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            loss = diffusion.loss(model, images)
            weighted_loss += float(loss) * images.shape[0]
            image_count += images.shape[0]
    finally:
        restore_rng_state(rng_state)
        if was_training:
            model.train()
    if image_count == 0:
        raise ValueError("Validation loader is empty")
    return weighted_loss / image_count


class ManifestImageDataset(Dataset):
    """Read RGB images named by a frozen split manifest; labels are not returned."""

    def __init__(
        self,
        manifest_path: Path,
        image_root: Path,
        split: str,
        image_size: int,
        interpolation: str,
        subset_size: int | None,
        seed: int,
    ) -> None:
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            rows = csv.DictReader(stream)
            self.paths = [
                image_root / row["relative_path"]
                for row in rows
                if row["split"] == split and row["qc_decision"] == "keep"
            ]
        if not self.paths:
            raise ValueError(f"No QC-kept images found for split={split!r} in {manifest_path}")
        missing = [path for path in self.paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} manifest images are missing; first: {missing[0]}")
        if subset_size is not None:
            if subset_size < 1:
                raise ValueError("subset_size must be positive")
            if subset_size < len(self.paths):
                self.paths = random.Random(seed).sample(self.paths, subset_size)
        interpolation_modes = {
            "lanczos": InterpolationMode.LANCZOS,
            "bicubic": InterpolationMode.BICUBIC,
            "bilinear": InterpolationMode.BILINEAR,
        }
        if interpolation not in interpolation_modes:
            raise ValueError(f"Unsupported resize interpolation: {interpolation}")
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=interpolation_modes[interpolation]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, help="Short pipeline run; not a paper-reproduction result.")
    parser.add_argument("--subset-size", type=int, help="Deterministic image subset for smoke runs.")
    parser.add_argument("--resume", action="store_true", help="Resume from the run's latest.pt checkpoint.")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())

    exp, data_cfg, model_cfg, diffusion_cfg, train_cfg = (
        config["experiment"], config["data"], config["model"], config["diffusion"], config["training"]
    )
    output_root = Path(exp["output_dir"])
    is_smoke = args.max_steps is not None
    run_suffix = exp.get("run_suffix", "smoke" if is_smoke else "")
    run_name = f"{exp['name']}-{run_suffix}" if run_suffix else exp["name"]
    run_kind = exp.get("run_kind", "smoke" if is_smoke else "formal")
    run_dir = output_root / "results" / run_name
    checkpoint_dir = output_root / "checkpoints" / run_name
    samples_dir = output_root / "generated_samples" / run_name
    for directory in (run_dir, checkpoint_dir, samples_dir):
        directory.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.toml").write_text(args.config.read_text())
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    environment_name = f"environment_job_{slurm_job_id}.json" if slurm_job_id else "environment.json"
    write_environment(run_dir / environment_name)
    (run_dir / "run_params.json").write_text(
        json.dumps({"max_steps": args.max_steps, "subset_size": args.subset_size, "run_kind": run_kind}, indent=2) + "\n"
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible. Start this command inside an allocated GPU job.")
    device = torch.device("cuda")
    seed_everything(exp["seed"])

    dataset_type = data_cfg.get("type", "cifar10")
    validation_loader = None
    if dataset_type == "cifar10":
        transform_list = [
            transforms.RandomHorizontalFlip() if data_cfg["horizontal_flip"] else transforms.Lambda(lambda x: x)
        ]
        transform_list.extend((transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)))
        dataset = datasets.CIFAR10(
            root=data_cfg["root"], train=True, download=True, transform=transforms.Compose(transform_list)
        )
    elif dataset_type == "image_manifest":
        dataset = ManifestImageDataset(
            manifest_path=Path(data_cfg["split_manifest"]),
            image_root=Path(data_cfg["image_root"]),
            split=data_cfg.get("split", "train"),
            image_size=model_cfg["image_size"],
            interpolation=data_cfg.get("resize_interpolation", "lanczos"),
            subset_size=args.subset_size,
            seed=exp["seed"],
        )
        print(f"dataset=manifest split={data_cfg.get('split', 'train')} images={len(dataset):,}", flush=True)
        validation_dataset = ManifestImageDataset(
            manifest_path=Path(data_cfg["split_manifest"]),
            image_root=Path(data_cfg["image_root"]),
            split="validation",
            image_size=model_cfg["image_size"],
            interpolation=data_cfg.get("resize_interpolation", "lanczos"),
            subset_size=None,
            seed=exp["seed"],
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=data_cfg["batch_size"],
            shuffle=False,
            num_workers=data_cfg["num_workers"],
            pin_memory=True,
            drop_last=False,
            persistent_workers=data_cfg["num_workers"] > 0,
        )
        print(f"validation_images={len(validation_dataset):,}", flush=True)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    loader = DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=data_cfg["num_workers"] > 0,
    )

    model = UNet(
        image_channels=model_cfg["channels"],
        base_channels=model_cfg["base_channels"],
        channel_multipliers=tuple(model_cfg["channel_multipliers"]),
        residual_blocks=model_cfg["residual_blocks"],
        attention_resolution=model_cfg["attention_resolution"],
        dropout=model_cfg["dropout"],
        image_size=model_cfg["image_size"],
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    (run_dir / "model.json").write_text(json.dumps({"trainable_parameters": parameter_count}, indent=2) + "\n")
    print(f"trainable_parameters={parameter_count:,}", flush=True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    diffusion = GaussianDiffusion(
        timesteps=diffusion_cfg["timesteps"],
        beta_start=diffusion_cfg["beta_start"],
        beta_end=diffusion_cfg["beta_end"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    step = 0
    latest_path = checkpoint_dir / "latest.pt"
    if args.resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"--resume requested but checkpoint does not exist: {latest_path}")
        state = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        if "rng_state" in state:
            restore_rng_state(state["rng_state"])
        print(f"resumed checkpoint={latest_path} step={step}", flush=True)

    target_steps = args.max_steps if args.max_steps is not None else train_cfg["steps"]
    log_path = run_dir / "train.csv"
    needs_header = not log_path.exists() or not args.resume
    log_file = log_path.open("a" if args.resume else "w", newline="")
    logger = csv.writer(log_file)
    if needs_header:
        logger.writerow(("step", "loss", "learning_rate", "elapsed_seconds", "peak_vram_mib"))
        log_file.flush()

    validation_file = None
    validation_logger = None
    if validation_loader is not None:
        validation_path = run_dir / "validation.csv"
        validation_needs_header = not validation_path.exists() or not args.resume
        validation_file = validation_path.open("a" if args.resume else "w", newline="")
        validation_logger = csv.writer(validation_file)
        if validation_needs_header:
            validation_logger.writerow(("step", "epsilon_mse", "validation_images", "seed"))
            validation_file.flush()

    model.train()
    iterator = iter(loader)
    started = time.monotonic()
    stop_requested = {"signal": None}

    def request_stop(signum, _frame):
        stop_requested["signal"] = signum
        print(f"received signal {signum}; will checkpoint after the current optimizer step", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        while step < target_steps:
            try:
                images, _ = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                images, _ = next(iterator)
            images = images.to(device, non_blocking=True)
            next_step = step + 1
            lr = train_cfg["learning_rate"] * min(1.0, next_step / train_cfg["warmup_steps"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss = diffusion.loss(model, images)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip"])
            optimizer.step()
            update_ema(ema, model, train_cfg["ema_decay"])
            step = next_step

            if step == 1 or step % 100 == 0 or step == target_steps:
                peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
                logger.writerow((step, f"{loss.item():.7f}", f"{lr:.8g}", f"{time.monotonic()-started:.2f}", f"{peak_memory:.1f}"))
                log_file.flush()
                print(f"step={step}/{target_steps} loss={loss.item():.6f} lr={lr:.3g} peak_vram={peak_memory:.0f}MiB", flush=True)

            should_stop = stop_requested["signal"] is not None
            validation_every = int(train_cfg.get("validation_every", train_cfg["checkpoint_every"]))
            if validation_loader is not None and (
                step % validation_every == 0 or step == target_steps or should_stop
            ):
                validation_seed = int(exp.get("validation_seed", exp["seed"] + 1))
                validation_loss = evaluate_noise_loss(
                    ema, diffusion, validation_loader, device, validation_seed
                )
                validation_logger.writerow(
                    (step, f"{validation_loss:.7f}", len(validation_loader.dataset), validation_seed)
                )
                validation_file.flush()
                print(f"validation step={step} epsilon_mse={validation_loss:.6f}", flush=True)

            if step % train_cfg["checkpoint_every"] == 0 or step == target_steps or should_stop:
                payload = {
                    "step": step,
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "seed": exp["seed"],
                    "rng_state": capture_rng_state(),
                }
                save_checkpoint(latest_path, payload, int(train_cfg["keep_checkpoints"]))

            if (step % train_cfg["sample_every"] == 0 or (is_smoke and step == target_steps)):
                sample_count = min(train_cfg["sample_count"], 4) if is_smoke else train_cfg["sample_count"]
                sample_seed = int(exp.get("sample_seed", exp["seed"]))
                sample_generator = torch.Generator(device=device).manual_seed(sample_seed)
                samples = diffusion.sample(
                    ema,
                    (sample_count, model_cfg["channels"], model_cfg["image_size"], model_cfg["image_size"]),
                    device,
                    generator=sample_generator,
                )
                grid = ((samples + 1.0) / 2.0).clamp(0, 1)
                utils.save_image(grid, samples_dir / f"step_{step:07d}.png", nrow=int(sample_count ** 0.5))
                model.train()
            if should_stop:
                break
    finally:
        log_file.close()
        if validation_file is not None:
            validation_file.close()

    status = {
        "status": "interrupted" if stop_requested["signal"] is not None else f"{run_kind}_only" if is_smoke else "training_complete",
        "completed_steps": step,
        "target_steps": target_steps,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
