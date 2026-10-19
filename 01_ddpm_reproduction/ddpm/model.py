"""Time-conditioned U-Net following the authors' CIFAR-10 model dimensions."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def timestep_embedding(timesteps: Tensor, dimension: int, max_period: int = 10000) -> Tensor:
    half = dimension // 2
    frequencies = torch.exp(-math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / (half - 1))
    angles = timesteps.float()[:, None] * frequencies[None]
    # Match the sinusoid order and frequency spacing in the authors' reference implementation.
    embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class AttentionBlock(nn.Module):
    """Spatial self-attention over all pixels at a fixed feature resolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.projection = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        normalized = self.norm(x)
        q = self.query(normalized).reshape(batch, channels, height * width)
        k = self.key(normalized).reshape(batch, channels, height * width)
        v = self.value(normalized).reshape(batch, channels, height * width)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.transpose(1, 2).reshape(batch, channels, height, width)
        return x + self.projection(attended)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_channels, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        # The original DDPM initializes the second residual convolution to zero.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: Tensor, time_embedding: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_projection(F.silu(time_embedding))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.skip(x) + h


def _tf_fan_avg_uniform_(tensor: Tensor, scale: float = 1.0) -> None:
    """Approximate TensorFlow variance_scaling(fan_avg, uniform) initialization."""
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    variance = scale * 2.0 / (fan_in + fan_out)
    bound = math.sqrt(3.0 * variance)
    with torch.no_grad():
        tensor.uniform_(-bound, bound)


def initialize_reference_weights(module: nn.Module) -> None:
    """Use the authors' fan-average uniform kernel initialization and zero biases."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        _tf_fan_avg_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class UNet(nn.Module):
    """CIFAR-10 U-Net: base 128, multipliers (1,2,2,2), two blocks per level."""

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: tuple[int, ...] = (1, 2, 2, 2),
        residual_blocks: int = 2,
        attention_resolution: int = 16,
        dropout: float = 0.1,
        image_size: int = 32,
    ):
        super().__init__()
        time_channels = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )
        self.input = nn.Conv2d(image_channels, base_channels, 3, padding=1)

        channels = [base_channels]
        down_levels = nn.ModuleList()
        current_channels = base_channels
        current_resolution = image_size
        for level_index, multiplier in enumerate(channel_multipliers):
            output_channels = base_channels * multiplier
            blocks = nn.ModuleList()
            attentions = nn.ModuleList()
            for _ in range(residual_blocks):
                blocks.append(ResidualBlock(current_channels, output_channels, time_channels, dropout))
                current_channels = output_channels
                channels.append(current_channels)
                attentions.append(AttentionBlock(current_channels) if current_resolution == attention_resolution else nn.Identity())
            downsample = (
                nn.Conv2d(current_channels, current_channels, 3, stride=2, padding=1)
                if level_index < len(channel_multipliers) - 1
                else None
            )
            if downsample is not None:
                current_resolution //= 2
                channels.append(current_channels)
            down_levels.append(nn.ModuleDict({"blocks": blocks, "attentions": attentions, "downsample": downsample or nn.Identity()}))
        self.down_levels = down_levels

        self.middle1 = ResidualBlock(current_channels, current_channels, time_channels, dropout)
        self.middle_attention = AttentionBlock(current_channels)
        self.middle2 = ResidualBlock(current_channels, current_channels, time_channels, dropout)

        up_levels = nn.ModuleList()
        current_resolution = image_size // (2 ** (len(channel_multipliers) - 1))
        for level in reversed(range(len(channel_multipliers))):
            output_channels = base_channels * channel_multipliers[level]
            blocks = nn.ModuleList()
            attentions = nn.ModuleList()
            for _ in range(residual_blocks + 1):
                skip_channels = channels.pop()
                blocks.append(ResidualBlock(current_channels + skip_channels, output_channels, time_channels, dropout))
                current_channels = output_channels
                attentions.append(AttentionBlock(current_channels) if current_resolution == attention_resolution else nn.Identity())
            upsample = (
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(current_channels, current_channels, 3, padding=1),
                )
                if level > 0
                else nn.Identity()
            )
            up_levels.append(nn.ModuleDict({"blocks": blocks, "attentions": attentions, "upsample": upsample}))
            if level > 0:
                current_resolution *= 2
        if channels:
            raise ValueError(f"U-Net skip construction error: {len(channels)} unconsumed skip entries")

        self.up_levels = up_levels
        self.output_norm = nn.GroupNorm(32, base_channels, eps=1e-6)
        self.output = nn.Conv2d(base_channels, image_channels, 3, padding=1)
        self.apply(initialize_reference_weights)
        # Keep zero-initialized layers zero after applying the reference initializer.
        for level in self.down_levels:
            for block in level["blocks"]:
                nn.init.zeros_(block.conv2.weight)
                nn.init.zeros_(block.conv2.bias)
            for attention in level["attentions"]:
                if isinstance(attention, AttentionBlock):
                    nn.init.zeros_(attention.projection.weight)
                    nn.init.zeros_(attention.projection.bias)
        for level in self.up_levels:
            for block in level["blocks"]:
                nn.init.zeros_(block.conv2.weight)
                nn.init.zeros_(block.conv2.bias)
            for attention in level["attentions"]:
                if isinstance(attention, AttentionBlock):
                    nn.init.zeros_(attention.projection.weight)
                    nn.init.zeros_(attention.projection.bias)
        nn.init.zeros_(self.middle1.conv2.weight)
        nn.init.zeros_(self.middle1.conv2.bias)
        nn.init.zeros_(self.middle2.conv2.weight)
        nn.init.zeros_(self.middle2.conv2.bias)
        nn.init.zeros_(self.middle_attention.projection.weight)
        nn.init.zeros_(self.middle_attention.projection.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: Tensor, timesteps: Tensor) -> Tensor:
        time = self.time_mlp(timestep_embedding(timesteps, self.time_mlp[0].in_features))
        h = self.input(x)
        skips = [h]
        for level in self.down_levels:
            for block, attention in zip(level["blocks"], level["attentions"]):
                h = attention(block(h, time))
                skips.append(h)
            if not isinstance(level["downsample"], nn.Identity):
                h = level["downsample"](h)
                skips.append(h)

        h = self.middle2(self.middle_attention(self.middle1(h, time)), time)
        for level in self.up_levels:
            for block, attention in zip(level["blocks"], level["attentions"]):
                h = torch.cat((h, skips.pop()), dim=1)
                h = attention(block(h, time))
            h = level["upsample"](h)
        if skips:
            raise RuntimeError(f"U-Net forward left {len(skips)} skip tensors")
        return self.output(F.silu(self.output_norm(h)))
