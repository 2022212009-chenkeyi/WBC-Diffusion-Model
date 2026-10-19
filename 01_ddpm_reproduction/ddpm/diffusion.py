"""DDPM forward process, epsilon objective, and ancestral sampler."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def extract(values: Tensor, timesteps: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Gather one schedule value per batch item and reshape for broadcasting."""
    gathered = values.gather(0, timesteps)
    return gathered.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64).float()
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat((torch.ones(1), alpha_bars[:-1]))
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

        self.timesteps = timesteps
        for name, value in {
            "betas": betas,
            "alphas": alphas,
            "alpha_bars": alpha_bars,
            "sqrt_alpha_bars": alpha_bars.sqrt(),
            "sqrt_one_minus_alpha_bars": (1.0 - alpha_bars).sqrt(),
            "sqrt_recip_alphas": alphas.rsqrt(),
            "posterior_variance": posterior_variance,
        }.items():
            self.register_buffer(name, value)

    def q_sample(self, x_start: Tensor, timesteps: Tensor, noise: Tensor | None = None) -> tuple[Tensor, Tensor]:
        noise = torch.randn_like(x_start) if noise is None else noise
        x_t = (
            extract(self.sqrt_alpha_bars, timesteps, tuple(x_start.shape)) * x_start
            + extract(self.sqrt_one_minus_alpha_bars, timesteps, tuple(x_start.shape)) * noise
        )
        return x_t, noise

    def loss(self, model: nn.Module, x_start: Tensor) -> Tensor:
        timesteps = torch.randint(0, self.timesteps, (x_start.shape[0],), device=x_start.device)
        x_t, noise = self.q_sample(x_start, timesteps)
        prediction = model(x_t, timesteps)
        return torch.nn.functional.mse_loss(prediction, noise)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        model_was_training = model.training
        model.eval()
        image = torch.randn(shape, device=device, generator=generator)
        for index in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), index, device=device, dtype=torch.long)
            predicted_noise = model(image, t)
            beta_t = extract(self.betas, t, shape)
            sqrt_one_minus_alpha_bar_t = extract(self.sqrt_one_minus_alpha_bars, t, shape)
            sqrt_recip_alpha_t = extract(self.sqrt_recip_alphas, t, shape)
            mean = sqrt_recip_alpha_t * (
                image - beta_t * predicted_noise / sqrt_one_minus_alpha_bar_t
            )
            if index > 0:
                # The paper's CIFAR-10 reference setting uses fixed-large variance: sigma_t^2 = beta_t.
                image = mean + beta_t.sqrt() * torch.randn(
                    image.shape, device=device, dtype=image.dtype, generator=generator
                )
            else:
                image = mean
        if model_was_training:
            model.train()
        return image.clamp(-1.0, 1.0)
