"""Helpers for FramePack first/last-frame anchoring."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image


def load_image(path: str | Path, size_hw: Optional[Tuple[int, int]] = None) -> Image.Image:
    """Load an RGB image and optionally resize it to ``(height, width)``."""

    image = Image.open(path).convert("RGB")
    if size_hw is not None:
        height, width = size_hw
        if height <= 0 or width <= 0:
            raise ValueError("size_hw must contain positive integers")
        image = image.resize((width, height), Image.LANCZOS)
    return image


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0
    return tensor


def _ensure_temporal_dim(vae, tensor: torch.Tensor) -> torch.Tensor:
    """Expand ``tensor`` with a temporal dimension for video VAEs."""

    if tensor.ndim != 4:
        return tensor

    config = getattr(vae, "config", None)
    sample_size = None
    if config is not None:
        for attr in ("sample_size", "spatial_sample_size"):
            sample_size = getattr(config, attr, None)
            if sample_size is not None:
                break

    if sample_size is None:
        return tensor

    if isinstance(sample_size, int):
        dims = (sample_size,)
    else:
        try:
            dims = tuple(sample_size)
        except TypeError:
            return tensor

    if len(dims) == 3:
        tensor = tensor.unsqueeze(2)
    return tensor


def encode_vae(vae, image: Image.Image, device: torch.device, scale: float) -> torch.Tensor:
    """Encode an image with a VAE and return a latent scaled by ``scale``."""

    tensor = _pil_to_tensor(image).to(device=device, dtype=vae.dtype)
    tensor = _ensure_temporal_dim(vae, tensor)
    encoded = vae.encode(tensor).latent_dist
    if hasattr(encoded, "mode"):
        latent = encoded.mode()
    else:
        latent = encoded.sample()
    latent = latent * scale
    return latent.to(device=device, dtype=torch.float32)


def make_schedule(n: int, kind: str = "cosine") -> torch.Tensor:
    """Return a 1D tensor of ``n`` weights following ``kind`` schedule."""

    n = int(n)
    if n <= 0:
        return torch.zeros(0)
    if n == 1:
        return torch.ones(1)

    steps = torch.linspace(0.0, 1.0, steps=n)
    if kind == "linear":
        weights = 1.0 - steps
    elif kind == "cosine":
        weights = 0.5 * (1.0 + torch.cos(math.pi * steps))
    else:
        raise ValueError(f"Unknown endpoint schedule: {kind}")
    return weights.clamp(min=0.0, max=1.0)


def _ensure_sigma_shape(sigma: torch.Tensor, target_rank: int) -> torch.Tensor:
    sigma = sigma.to(dtype=torch.float32)
    while sigma.ndim < target_rank:
        sigma = sigma.view(*sigma.shape, 1)
    return sigma


def noise_latent_to_sigma(
    pipeline,
    z0: torch.Tensor,
    sigma: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Project ``z0`` to the diffusion state at ``sigma`` using ``noise``."""

    noise = torch.zeros_like(z0) if noise is None else noise
    sigma = _ensure_sigma_shape(sigma, z0.ndim)

    scheduler = getattr(pipeline, "scheduler", None)
    if scheduler is not None and hasattr(scheduler, "add_noise"):
        return scheduler.add_noise(z0, noise, sigma)
    return z0 + noise * sigma


def anchor_x0(x0_pred: torch.Tensor, z_ref: torch.Tensor, alpha: torch.Tensor | float) -> torch.Tensor:
    """Blend ``x0_pred`` with ``z_ref`` using ``alpha`` (0 → keep, 1 → replace)."""

    return x0_pred * (1.0 - alpha) + z_ref * alpha


