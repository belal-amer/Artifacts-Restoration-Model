from __future__ import annotations

import math

import cv2
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


def srgb_to_linear(image: Tensor) -> Tensor:
    image = image.clamp(0.0, 1.0)
    return torch.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055).pow(2.4))


def linear_to_srgb(image: Tensor) -> Tensor:
    image = image.clamp(0.0, 1.0)
    return torch.where(image <= 0.0031308, image * 12.92, 1.055 * image.clamp_min(1e-5).pow(1.0 / 2.4) - 0.055)


def ensure_mask(mask: Tensor) -> Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[1] != 1:
        raise ValueError("mask must have shape [B, 1, H, W] or [B, H, W]")
    return (mask > 0.5).to(dtype=torch.float32)


def blueprint_normalize(image_linear: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Normalize linear RGB to [-1, 1] using valid-pixel statistics only.

    For batched inputs `[B, C, H, W]`, compute one shared `(mu, sigma)` per
    channel across the full batch. For unbatched inputs `[C, H, W]`, compute
    `(mu, sigma)` from that image alone. In both cases, masked pixels are
    excluded from the statistics and the normalized tensor is clipped to
    `[-3, 3]` then rescaled into `[-1, 1]`.
    """

    if image_linear.ndim not in {3, 4}:
        raise ValueError("image_linear must have shape [C, H, W] or [B, C, H, W]")

    original_ndim = image_linear.ndim
    if original_ndim == 3:
        image_b = image_linear.unsqueeze(0)
        mask_b = ensure_mask(mask.unsqueeze(0) if mask.ndim == 2 else mask)
    else:
        image_b = image_linear
        mask_b = ensure_mask(mask)

    mask_b = mask_b.to(device=image_b.device, dtype=image_b.dtype)
    if mask_b.shape[0] != image_b.shape[0] or mask_b.shape[-2:] != image_b.shape[-2:]:
        raise ValueError("mask shape must align with image_linear spatial and batch dimensions")

    batch, channels, _, _ = image_b.shape
    valid_map = (1.0 - mask_b).expand(batch, channels, image_b.shape[-2], image_b.shape[-1])
    count_c = valid_map.sum(dim=(0, 2, 3)).clamp_min(1.0)
    valid_vals = image_b * valid_map
    mu_c = valid_vals.sum(dim=(0, 2, 3)) / count_c
    diff_sq = ((image_b - mu_c.view(1, channels, 1, 1)) ** 2) * valid_map
    sigma_c = (diff_sq.sum(dim=(0, 2, 3)) / count_c).sqrt()
    image_z = (image_b - mu_c.view(1, channels, 1, 1)) / (sigma_c.view(1, channels, 1, 1) + 1e-5)
    image_norm = image_z.clamp(-3.0, 3.0) / 3.0

    if original_ndim == 3:
        return image_norm.squeeze(0), mu_c, sigma_c
    return image_norm, mu_c, sigma_c


def blueprint_denormalize(image_norm: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    """Inverse of blueprint_normalize."""

    if image_norm.ndim not in {3, 4}:
        raise ValueError("image_norm must have shape [C, H, W] or [B, C, H, W]")

    mu = mu.to(device=image_norm.device, dtype=image_norm.dtype)
    sigma = sigma.to(device=image_norm.device, dtype=image_norm.dtype)

    if image_norm.ndim == 3:
        if mu.ndim != 1 or sigma.ndim != 1:
            raise ValueError("mu and sigma must have shape [C] for unbatched inputs")
        mu_view = mu.view(-1, 1, 1)
        sigma_view = sigma.view(-1, 1, 1)
    else:
        if mu.ndim == 1 and sigma.ndim == 1:
            mu_view = mu.view(1, -1, 1, 1)
            sigma_view = sigma.view(1, -1, 1, 1)
        elif mu.ndim == 2 and sigma.ndim == 2:
            mu_view = mu.view(mu.shape[0], mu.shape[1], 1, 1)
            sigma_view = sigma.view(sigma.shape[0], sigma.shape[1], 1, 1)
        elif mu.ndim == 4 and sigma.ndim == 4:
            mu_view = mu
            sigma_view = sigma
        else:
            raise ValueError("mu and sigma must have shape [C], [B, C], or [B, C, 1, 1] for batched inputs")

    return ((image_norm * 3.0) * sigma_view + mu_view).clamp(0.0, 1.0)


def compute_valid_stats(image_linear: Tensor, mask: Tensor, eps: float = 1e-5) -> tuple[Tensor, Tensor]:
    _, mean, std = blueprint_normalize(image_linear, mask)
    return mean, std.clamp_min(eps)


def normalize_by_valid_pixels(image_linear: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return blueprint_normalize(image_linear, mask)


def compute_batch_valid_stats(image_linear: Tensor, mask: Tensor, eps: float = 1e-5) -> tuple[Tensor, Tensor]:
    mean, std = compute_valid_stats(image_linear, mask, eps=eps)
    return mean, std


def normalize_batch_by_valid_pixels(image_linear: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return blueprint_normalize(image_linear, mask)


def denormalize_to_linear(image_norm: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return blueprint_denormalize(image_norm, mean, std)


def encode_masked_pixels(image_norm: Tensor, mask: Tensor) -> Tensor:
    mask = ensure_mask(mask).to(device=image_norm.device, dtype=image_norm.dtype)
    return image_norm * (1.0 - mask)


def apply_valid_pixel_lock(network_output: Tensor, original_input: Tensor, mask: Tensor) -> Tensor:
    """Copy every valid pixel from the original input.

    Mask convention: `1.0 = masked`, `0.0 = valid`.
    """

    mask = ensure_mask(mask).to(device=network_output.device, dtype=network_output.dtype)
    return network_output * mask + original_input * (1.0 - mask)


def compute_background_mask(image_linear: Tensor, threshold: float = 0.04) -> Tensor:
    """Pure-black background detector for the hard tile silhouette constraint."""

    return (image_linear.max(dim=1, keepdim=True).values < threshold).to(dtype=image_linear.dtype)


def constrain_mask_to_silhouette(mask: Tensor, background_mask: Tensor) -> Tensor:
    mask = ensure_mask(mask).to(device=background_mask.device, dtype=background_mask.dtype)
    background_mask = ensure_mask(background_mask).to(device=mask.device, dtype=mask.dtype)
    return mask * (1.0 - background_mask)


def gaussian_kernel2d(sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d.view(1, 1, *kernel_2d.shape)
    return kernel_2d.repeat(channels, 1, 1, 1)


def gaussian_blur(image: Tensor, sigma: float) -> Tensor:
    if sigma <= 0:
        return image
    channels = image.shape[1]
    kernel = gaussian_kernel2d(sigma, channels, image.device, image.dtype)
    pad_h = kernel.shape[-2] // 2
    pad_w = kernel.shape[-1] // 2
    padded = F.pad(image, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
    return F.conv2d(padded, kernel, groups=channels)


def antialias_resize(image: Tensor, size: int | tuple[int, int], sigma: float = 0.5) -> Tensor:
    blurred = gaussian_blur(image, sigma=sigma)
    if isinstance(size, int):
        size = (size, size)
    return F.interpolate(blurred, size=size, mode="bilinear", align_corners=False)


def resize_mask(mask: Tensor, size: int | tuple[int, int]) -> Tensor:
    if isinstance(size, int):
        size = (size, size)
    mask = ensure_mask(mask).to(device=mask.device)
    return (F.interpolate(mask, size=size, mode="nearest") > 0.5).to(dtype=mask.dtype)


def dilate_mask(mask: Tensor, radius: int = 4) -> Tensor:
    mask = ensure_mask(mask).to(device=mask.device)
    out = mask
    for _ in range(radius):
        out = F.max_pool2d(out, kernel_size=3, stride=1, padding=1)
    return (out > 0.5).to(dtype=mask.dtype)


def _to_edge_space(image: Tensor) -> Tensor:
    if float(image.detach().amin()) < 0.0 or float(image.detach().amax()) > 1.0:
        image = (image + 1.0) * 0.5
    return image.clamp(0.0, 1.0)


def compute_canny_edges(image: Tensor, mask: Tensor, low: float = 0.05, high: float = 0.15) -> Tensor:
    """Canny edges from valid linear RGB pixels, softened with sigma=1.0."""

    mask = ensure_mask(mask).to(device=image.device, dtype=image.dtype)
    image_01 = _to_edge_space(image) * (1.0 - mask)
    gray = (
        0.2126 * image_01[:, 0:1]
        + 0.7152 * image_01[:, 1:2]
        + 0.0722 * image_01[:, 2:3]
    )
    edges: list[Tensor] = []
    low_u8 = int(round(low * 255.0))
    high_u8 = int(round(high * 255.0))
    for idx in range(gray.shape[0]):
        gray_u8 = (gray[idx, 0].detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
        edge = cv2.Canny(gray_u8, threshold1=low_u8, threshold2=high_u8)
        edge = cv2.GaussianBlur(edge.astype(np.float32) / 255.0, (0, 0), sigmaX=1.0, sigmaY=1.0)
        edge_tensor = torch.from_numpy(edge).to(device=image.device, dtype=image.dtype).unsqueeze(0).unsqueeze(0)
        edges.append(edge_tensor)
    return torch.cat(edges, dim=0)


def _masked_corr(a: Tensor, b: Tensor, valid_a: Tensor, valid_b: Tensor) -> Tensor:
    common = (valid_a * valid_b).expand_as(a)
    count = common.sum(dim=(1, 2, 3)).clamp_min(1.0)
    mean_a = (a * common).sum(dim=(1, 2, 3), keepdim=True) / count.view(-1, 1, 1, 1)
    mean_b = (b * common).sum(dim=(1, 2, 3), keepdim=True) / count.view(-1, 1, 1, 1)
    da = (a - mean_a) * common
    db = (b - mean_b) * common
    denom = (da.square().sum(dim=(1, 2, 3)) * db.square().sum(dim=(1, 2, 3))).sqrt().clamp_min(1e-8)
    corr = (da * db).sum(dim=(1, 2, 3)) / denom
    coverage = (common[:, 0:1].sum(dim=(1, 2, 3)) / valid_a.sum(dim=(1, 2, 3)).clamp_min(1.0)).clamp(0.0, 1.0)
    return corr.clamp(-1.0, 1.0) * coverage


def compute_symmetry_prior(
    image: Tensor,
    mask: Tensor,
    threshold: float = 0.6,
    return_confidence: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    """Autocorrelation-style binary symmetry prior.

    Channel order is fixed to five blueprint channels:
    two-fold rotation, four-fold rotation, horizontal-axis mirror,
    vertical-axis mirror, and main diagonal reflection.
    """

    mask = ensure_mask(mask).to(device=image.device, dtype=image.dtype)
    valid = 1.0 - mask
    transforms = {
        "two_fold": (torch.rot90(image, 2, dims=(-2, -1)), torch.rot90(valid, 2, dims=(-2, -1))),
        "four_fold": (torch.rot90(image, 1, dims=(-2, -1)), torch.rot90(valid, 1, dims=(-2, -1))),
        "horizontal": (torch.flip(image, dims=(-2,)), torch.flip(valid, dims=(-2,))),
        "vertical": (torch.flip(image, dims=(-1,)), torch.flip(valid, dims=(-1,))),
        "diagonal": (_diagonal_reflect(image), _diagonal_reflect(valid)),
    }
    channels = []
    confidences: dict[str, Tensor] = {}
    for name, (img_t, valid_t) in transforms.items():
        conf = _masked_corr(image, img_t, valid, valid_t)
        if name == "four_fold":
            img_t2 = torch.rot90(image, 3, dims=(-2, -1))
            valid_t2 = torch.rot90(valid, 3, dims=(-2, -1))
            conf = torch.minimum(conf, _masked_corr(image, img_t2, valid, valid_t2))
        confidences[name] = conf
        channel = (conf >= threshold).to(dtype=image.dtype).view(-1, 1, 1, 1)
        channels.append(channel.expand(-1, 1, image.shape[-2], image.shape[-1]))
    prior = torch.cat(channels, dim=1)
    if return_confidence:
        return prior, confidences
    return prior


def _diagonal_reflect(x: Tensor) -> Tensor:
    reflected = x.transpose(-2, -1)
    if reflected.shape[-2:] != x.shape[-2:]:
        reflected = F.interpolate(reflected, size=x.shape[-2:], mode="nearest")
    return reflected


def palette_histogram_match(output_norm: Tensor, reference_norm: Tensor, mask: Tensor) -> Tensor:
    """Match synthesized masked pixels to visible tile pixels per channel."""

    mask = ensure_mask(mask).to(device=output_norm.device)
    result = output_norm.clone()
    for b_idx in range(output_norm.shape[0]):
        masked = mask[b_idx, 0] > 0.5
        valid = ~masked
        if masked.sum() == 0 or valid.sum() == 0:
            continue
        for c_idx in range(output_norm.shape[1]):
            synth_vals = result[b_idx, c_idx][masked]
            valid_vals = reference_norm[b_idx, c_idx][valid]
            if synth_vals.numel() == 0 or valid_vals.numel() == 0:
                continue
            _, order = torch.sort(synth_vals)
            valid_sorted = torch.sort(valid_vals).values
            quantile_idx = torch.linspace(
                0,
                valid_sorted.numel() - 1,
                steps=synth_vals.numel(),
                device=output_norm.device,
            ).round().long()
            mapped_sorted = valid_sorted[quantile_idx]
            mapped = torch.empty_like(synth_vals)
            mapped[order] = mapped_sorted
            result[b_idx, c_idx][masked] = mapped
    return result


def make_uncertainty_report(
    confidence_full: Tensor,
    mask_full: Tensor,
    *,
    threshold: float = 0.4,
    review_fraction_threshold: float = 0.15,
) -> dict[str, Tensor | float | bool]:
    mask_full = ensure_mask(mask_full).to(device=confidence_full.device, dtype=confidence_full.dtype)
    low_conf_mask = ((confidence_full < threshold).to(dtype=confidence_full.dtype) * mask_full)
    masked_count = mask_full.sum().clamp_min(1.0)
    low_conf_fraction = (low_conf_mask.sum() / masked_count).item()
    return {
        "low_confidence_pixel_fraction": low_conf_fraction,
        "low_confidence_mask": low_conf_mask,
        "requires_expert_review": low_conf_fraction > review_fraction_threshold,
    }
