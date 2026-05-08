from __future__ import annotations

import torch
from torch import Tensor

from .losses import sobel_gradients
from .preprocessing import ensure_mask


def masked_psnr(pred: Tensor, target: Tensor, mask: Tensor, peak: float | None = 2.0) -> Tensor:
    """Mean masked-region PSNR, computed per image and then averaged.
    
    Default peak=2.0 corresponds to the full normalized range [-1, 1].
    This makes the metric comparable across different masks and samples.
    """
    mask = ensure_mask(mask).to(device=pred.device, dtype=pred.dtype)
    psnrs: list[Tensor] = []
    for idx in range(pred.shape[0]):
        sample_mask = mask[idx]
        n_masked = sample_mask.sum()
        if n_masked < 1:
            continue
        err = (pred[idx] - target[idx]).square()
        mse = (err * sample_mask).sum() / (n_masked * pred.shape[1])
        if (mse < 1e-10).item():
            psnrs.append(torch.tensor(60.0, device=pred.device, dtype=pred.dtype))
            continue
        if peak is None:
            masked_values = target[idx].masked_select(sample_mask.expand_as(target[idx]) > 0.5)
            signal_peak = (masked_values.max() - masked_values.min()).clamp_min(0.1)
        else:
            signal_peak = torch.tensor(float(peak), device=pred.device, dtype=pred.dtype)
        psnrs.append(10.0 * torch.log10(signal_peak.square() / mse.clamp_min(1e-12)))
    if not psnrs:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    return torch.stack(psnrs).mean()


def boundary_gradient_alignment(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    mask = ensure_mask(mask).to(device=pred.device, dtype=pred.dtype)
    boundary = _mask_boundary(mask)
    pred_x, pred_y = sobel_gradients(pred)
    target_x, target_y = sobel_gradients(target)
    pred_vec = torch.cat([pred_x, pred_y], dim=1)
    target_vec = torch.cat([target_x, target_y], dim=1)
    dot = (pred_vec * target_vec).sum(dim=1, keepdim=True)
    denom = pred_vec.square().sum(dim=1, keepdim=True).sqrt() * target_vec.square().sum(dim=1, keepdim=True).sqrt()
    cosine = dot / denom.clamp_min(1e-8)
    return (cosine * boundary).sum() / boundary.sum().clamp_min(1.0)


def _mask_boundary(mask: Tensor) -> Tensor:
    dilated = torch.nn.functional.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - torch.nn.functional.max_pool2d(1.0 - mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)
