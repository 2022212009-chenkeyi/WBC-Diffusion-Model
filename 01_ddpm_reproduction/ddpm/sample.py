"""Generate reproducible CIFAR-10 samples from an EMA checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path
import tomllib

import numpy as np
import torch

from .diffusion import GaussianDiffusion
from .model import UNet


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Defaults to the configured run's latest.pt.")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted .partial.npy generation.")
    args = parser.parse_args()
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("num-samples and batch-size must be positive")

    config = tomllib.loads(args.config.read_text())
    exp, model_cfg, diffusion_cfg = config["experiment"], config["model"], config["diffusion"]
    output_root = Path(exp["output_dir"])
    run_suffix = exp.get("run_suffix", "")
    run_name = f"{exp['name']}-{run_suffix}" if run_suffix else exp["name"]
    checkpoint = args.checkpoint or output_root / "checkpoints" / run_name / "latest.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible. Start this command inside an allocated GPU job.")
    device = torch.device("cuda")

    model = UNet(
        image_channels=model_cfg["channels"],
        base_channels=model_cfg["base_channels"],
        channel_multipliers=tuple(model_cfg["channel_multipliers"]),
        residual_blocks=model_cfg["residual_blocks"],
        attention_resolution=model_cfg["attention_resolution"],
        dropout=model_cfg["dropout"],
        image_size=model_cfg["image_size"],
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["ema"])
    model.eval()
    diffusion = GaussianDiffusion(
        timesteps=diffusion_cfg["timesteps"],
        beta_start=diffusion_cfg["beta_start"],
        beta_end=diffusion_cfg["beta_end"],
    ).to(device)

    output_dir = output_root / "generated_samples" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    step = int(state["step"])
    stem = f"samples_ema_step{step:07d}_n{args.num_samples}_seed{args.seed}"
    final_path = output_dir / f"{stem}.npy"
    partial_path = output_dir / f"{stem}.partial.npy"
    metadata_path = output_dir / f"{stem}.json"
    if final_path.exists():
        raise FileExistsError(f"refusing to overwrite completed samples: {final_path}")
    if partial_path.exists() and not args.resume:
        raise FileExistsError(f"partial sample file exists; use --resume: {partial_path}")
    resume_existing = args.resume and partial_path.exists()

    shape = (args.num_samples, model_cfg["image_size"], model_cfg["image_size"], model_cfg["channels"])
    samples = np.lib.format.open_memmap(partial_path, mode="r+" if resume_existing else "w+", dtype=np.uint8, shape=shape)
    metadata = {
        "status": "running",
        "num_samples": args.num_samples,
        "completed_samples": 0,
        "sample_batch_size": args.batch_size,
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_step": step,
        "shape": shape,
        "storage_format": "uint8 NHWC NumPy array; do not normalize the saved values in place",
    }
    if resume_existing:
        if not metadata_path.is_file():
            raise FileNotFoundError(f"resume metadata not found: {metadata_path}")
        previous = json.loads(metadata_path.read_text())
        if (
            previous["checkpoint_step"] != step
            or previous["num_samples"] != args.num_samples
            or previous["seed"] != args.seed
            or previous["sample_batch_size"] != args.batch_size
        ):
            raise ValueError("resume parameters do not match the partial sample metadata")
        metadata.update(previous)
    completed = int(metadata["completed_samples"])
    write_json(metadata_path, metadata)
    stop_requested = {"signal": None}

    def request_stop(signum, _frame):
        stop_requested["signal"] = signum
        print(f"received signal {signum}; stopping after current sample batch", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.monotonic()
    try:
        while completed < args.num_samples:
            count = min(args.batch_size, args.num_samples - completed)
            generator = torch.Generator(device=device).manual_seed(args.seed + completed)
            generated = diffusion.sample(
                model,
                (count, model_cfg["channels"], model_cfg["image_size"], model_cfg["image_size"]),
                device,
                generator=generator,
            )
            pixels = ((generated.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
            pixels = pixels.permute(0, 2, 3, 1).cpu().numpy()
            samples[completed : completed + count] = pixels
            completed += count
            samples.flush()
            metadata["completed_samples"] = completed
            metadata["elapsed_seconds"] = time.monotonic() - started
            write_json(metadata_path, metadata)
            if completed == args.num_samples or completed % (args.batch_size * 10) == 0:
                print(f"generated={completed}/{args.num_samples} elapsed={metadata['elapsed_seconds']:.1f}s", flush=True)
            if stop_requested["signal"] is not None:
                break
    finally:
        samples.flush()
        del samples

    if completed == args.num_samples:
        os.replace(partial_path, final_path)
        metadata["status"] = "complete"
    else:
        metadata["status"] = "interrupted"
    metadata["completed_samples"] = completed
    metadata["elapsed_seconds"] = time.monotonic() - started
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    if completed != args.num_samples:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
