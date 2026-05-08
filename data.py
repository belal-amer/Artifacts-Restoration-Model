from __future__ import annotations

import csv
from dataclasses import dataclass
from io import BytesIO
import json
import math
import random
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import BatchSampler, Dataset, WeightedRandomSampler

from .config import CollectionSpec, TRAINING_COLLECTIONS
from .preprocessing import (
    blueprint_denormalize,
    blueprint_normalize,
    compute_canny_edges,
    compute_symmetry_prior,
    dilate_mask,
    encode_masked_pixels,
    srgb_to_linear,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MANIFEST_NAME = "manifest.csv"
LATE_CURRICULUM_ITERATION = 15_000
VALID_SYMMETRY_TYPES = {"none", "horizontal_only", "4fold", "radial"}


@dataclass(frozen=True)
class PartialLossMaskResult:
    mask: Tensor | None
    issue: str


def discover_image_paths(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def load_rgb_tensor(path: str | Path) -> Tensor:
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        try:
            array = tifffile.imread(path)
            if array.ndim == 2:
                array = np.repeat(array[:, :, None], 3, axis=2)
            if array.shape[2] > 3:
                array = array[:, :, :3]
            original_dtype = array.dtype
            array = array.astype(np.float32)
            if array.max(initial=0.0) > 1.0:
                scale = float(np.iinfo(original_dtype).max) if np.issubdtype(original_dtype, np.integer) else 255.0
                array = array / scale
            array = np.clip(array, 0.0, 1.0)
        except ValueError:
            image = Image.open(path).convert("RGB")
            array = np.asarray(image).astype(np.float32) / 255.0
    else:
        image = Image.open(path).convert("RGB")
        array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def jpeg_roundtrip_srgb(srgb: Tensor, quality: int) -> Tensor:
    image = srgb.detach().cpu().clamp(0.0, 1.0)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    decoded = Image.open(buffer).convert("RGB")
    out = np.asarray(decoded).astype(np.float32) / 255.0
    return torch.from_numpy(out).permute(2, 0, 1)


def apply_synthetic_degradation(srgb: Tensor) -> Tensor:
    quality = random.randint(87, 95)
    degraded = jpeg_roundtrip_srgb(srgb, quality)
    sigma = random.uniform(0.001, 0.008)
    degraded = degraded + torch.randn_like(degraded) * sigma
    return degraded.clamp(0.0, 1.0)


def _gaussian_blur_srgb(srgb: Tensor, sigma: float) -> Tensor:
    if sigma <= 0.0:
        return srgb
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=srgb.dtype, device=srgb.device)
    kernel_1d = torch.exp(-(x.square()) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-8)
    channels = srgb.shape[0]
    kernel_x = kernel_1d.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    kernel_y = kernel_1d.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    image = srgb.unsqueeze(0)
    image = F.conv2d(image, kernel_x, padding=(0, radius), groups=channels)
    image = F.conv2d(image, kernel_y, padding=(radius, 0), groups=channels)
    return image.squeeze(0).clamp(0.0, 1.0)


def apply_safe_tiny_data_augmentation(srgb: Tensor) -> Tensor:
    brightness = random.uniform(0.75, 1.25)
    srgb = (srgb * brightness).clamp(0.0, 1.0)
    contrast = random.uniform(0.75, 1.25)
    mean_val = srgb.mean()
    srgb = ((srgb - mean_val) * contrast + mean_val).clamp(0.0, 1.0)
    gamma = random.uniform(0.80, 1.20)
    srgb = srgb.clamp(0.0, 1.0).pow(gamma)
    srgb = jpeg_roundtrip_srgb(srgb, random.randint(70, 97))
    blur_sigma = random.uniform(0.0, 0.8)
    srgb = _gaussian_blur_srgb(srgb, blur_sigma)
    noise_sigma = random.uniform(0.001, 0.020)
    return (srgb + torch.randn_like(srgb) * noise_sigma).clamp(0.0, 1.0)


def random_erase_augmentation(srgb: Tensor, mask: Tensor, max_patches: int = 3, probability: float = 0.30) -> Tensor:
    """Erase only valid pixels so train-time erasing never changes the synthesis target."""

    if random.random() >= probability:
        return srgb
    erased = srgb.clone()
    valid = mask.squeeze(0) <= 0.5
    _, height, width = srgb.shape
    n_patches = random.randint(1, max_patches)
    for _ in range(n_patches):
        patch_h = random.randint(max(1, height // 32), max(2, height // 10))
        patch_w = random.randint(max(1, width // 32), max(2, width // 10))
        y0 = random.randint(0, max(0, height - patch_h))
        x0 = random.randint(0, max(0, width - patch_w))
        patch_valid = valid[y0 : y0 + patch_h, x0 : x0 + patch_w]
        if not patch_valid.any():
            continue
        fill = torch.empty((srgb.shape[0], 1, 1), dtype=srgb.dtype, device=srgb.device).uniform_(0.0, 1.0)
        patch = erased[:, y0 : y0 + patch_h, x0 : x0 + patch_w]
        patch = torch.where(patch_valid.unsqueeze(0), fill.expand_as(patch), patch)
        erased[:, y0 : y0 + patch_h, x0 : x0 + patch_w] = patch
    if torch.allclose(erased, srgb) and valid.any():
        ys, xs = valid.nonzero(as_tuple=True)
        y = int(ys[0].item())
        x = int(xs[0].item())
        erased[:, y, x] = 1.0 - erased[:, y, x]
    return erased.clamp(0.0, 1.0)


def random_channel_shuffle(srgb: Tensor, probability: float = 0.10) -> Tensor:
    if random.random() >= probability:
        return srgb
    order = torch.randperm(srgb.shape[0], device=srgb.device)
    return srgb[order]


def safe_geometry_augment(srgb: Tensor, mask: Tensor, symmetry_type: str) -> tuple[Tensor, Tensor]:
    symmetry_type = normalize_symmetry_type(symmetry_type)
    if symmetry_type in {"horizontal_only", "4fold", "radial"} and random.random() < 0.5:
        srgb = torch.flip(srgb, dims=(-1,))
        mask = torch.flip(mask, dims=(-1,))
    if symmetry_type in {"4fold", "radial"}:
        rotations = random.randint(0, 3)
        if rotations:
            srgb = torch.rot90(srgb, rotations, dims=(-2, -1))
            mask = torch.rot90(mask, rotations, dims=(-2, -1))
    return srgb, mask


def mask_area_bucket(area: float) -> tuple[float, float]:
    if area < 0.20:
        return 0.0, 0.20
    if area < 0.40:
        return 0.20, 0.40
    return 0.40, 0.65


def perturb_real_mask_within_bucket(
    mask: Tensor,
    *,
    min_area: float | None = None,
    max_area: float | None = None,
) -> Tensor:
    original_area = float(mask.mean().item())
    bucket_min, bucket_max = mask_area_bucket(original_area)
    if min_area is not None:
        bucket_min = max(bucket_min, float(min_area))
    if max_area is not None:
        bucket_max = min(bucket_max, float(max_area))
    radius = random.randint(0, 2)
    operation = random.choice(["none", "erode", "dilate", "open", "close"])
    if radius == 0 or operation == "none":
        return mask
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    mask_np = (mask.squeeze(0).detach().cpu().numpy() > 0.5).astype(np.uint8)
    if operation == "erode":
        perturbed = cv2.erode(mask_np, kernel, iterations=1)
    elif operation == "dilate":
        perturbed = cv2.dilate(mask_np, kernel, iterations=1)
    elif operation == "open":
        perturbed = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
    elif operation == "close":
        perturbed = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
    else:
        perturbed = mask_np
    area = float(perturbed.mean())
    if bucket_min <= area <= bucket_max:
        return torch.from_numpy(perturbed.astype(np.float32)).unsqueeze(0)
    return mask


class MaskCurriculum:
    def area_bounds(self, iteration: int) -> tuple[float, float]:
        if iteration < 2_000:
            return 0.10, 0.30
        if iteration < 8_000:
            progress = (iteration - 2_000) / 6_000
            return 0.10, 0.30 + 0.35 * progress
        return 0.10, 0.65


class SyntheticMaskGenerator:
    """Synthetic masks mixed as 3 blob, 1 free-form, 1 rectangular."""

    def __init__(self) -> None:
        self.curriculum = MaskCurriculum()

    def __call__(self, height: int, width: int, iteration: int = 0, *, force_large: bool = False, kind: str | None = None) -> Tensor:
        min_area, max_area = self.curriculum.area_bounds(iteration)
        if force_large and iteration >= 8_000:
            min_area = max(min_area, 0.40)
        choice = random.random()
        if kind == "freeform" or (kind is None and choice < 0.2):
            mask = self._stroke_mask(height, width, min_area, max_area)
        elif kind == "blob" or (kind is None and choice < 0.8):
            mask = self._blob_mask(height, width, min_area, max_area)
        elif kind == "rectangular" or kind is None:
            mask = self._rectangle_mask(height, width, min_area, max_area)
        else:
            raise ValueError(f"unknown mask kind: {kind}")
        return torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

    def _stroke_mask(self, height: int, width: int, min_area: float, max_area: float) -> np.ndarray:
        for _ in range(64):
            mask = np.zeros((height, width), dtype=np.uint8)
            n_strokes = random.randint(3, 10)
            for _ in range(n_strokes):
                points = []
                n_points = random.randint(4, 10)
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                points.append((x, y))
                for _ in range(n_points - 1):
                    x = int(np.clip(x + random.randint(-width // 4, width // 4), 0, width - 1))
                    y = int(np.clip(y + random.randint(-height // 4, height // 4), 0, height - 1))
                    points.append((x, y))
                thickness = random.randint(max(4, min(height, width) // 40), max(8, min(height, width) // 12))
                cv2.polylines(mask, [np.array(points, dtype=np.int32)], False, 1, thickness=thickness)
            area = float(mask.mean())
            if min_area <= area <= max_area:
                return mask
        return self._fallback_rectangle(height, width, min_area, max_area)

    def _blob_mask(self, height: int, width: int, min_area: float, max_area: float) -> np.ndarray:
        for _ in range(96):
            mask = np.zeros((height, width), dtype=np.uint8)
            if random.random() < 0.5:
                center = (random.randint(width // 4, 3 * width // 4), random.randint(height // 4, 3 * height // 4))
                axes = (
                    random.randint(max(1, width // 6), max(2, width // 2)),
                    random.randint(max(1, height // 6), max(2, height // 2)),
                )
                angle = random.uniform(0, 180)
                cv2.ellipse(mask, center, axes, angle, 0, 360, 1, thickness=-1)
            else:
                n_vertices = random.randint(3, 8)
                angles = np.sort(np.random.rand(n_vertices) * 2 * math.pi)
                radius_x = random.uniform(width * 0.12, width * 0.45)
                radius_y = random.uniform(height * 0.12, height * 0.45)
                center_x = random.uniform(width * 0.3, width * 0.7)
                center_y = random.uniform(height * 0.3, height * 0.7)
                points = []
                for angle in angles:
                    scale = random.uniform(0.55, 1.15)
                    points.append(
                        [
                            int(np.clip(center_x + math.cos(angle) * radius_x * scale, 0, width - 1)),
                            int(np.clip(center_y + math.sin(angle) * radius_y * scale, 0, height - 1)),
                        ]
                    )
                cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 1)
            blurred = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=5.0, sigmaY=5.0)
            mask = (blurred > 0.35).astype(np.uint8)
            area = float(mask.mean())
            if min_area <= area <= max_area:
                return mask
        return self._fallback_rectangle(height, width, min_area, max_area)

    def _rectangle_mask(self, height: int, width: int, min_area: float, max_area: float) -> np.ndarray:
        for _ in range(64):
            mask = np.zeros((height, width), dtype=np.uint8)
            target = random.uniform(min_area, max_area)
            aspect = random.uniform(0.5, 2.0)
            rect_h = int(math.sqrt(target * height * width / aspect))
            rect_w = int(rect_h * aspect)
            rect_h = int(np.clip(rect_h, 1, height))
            rect_w = int(np.clip(rect_w, 1, width))
            y0 = random.randint(0, max(0, height - rect_h))
            x0 = random.randint(0, max(0, width - rect_w))
            mask[y0 : y0 + rect_h, x0 : x0 + rect_w] = 1
            area = float(mask.mean())
            if min_area <= area <= max_area:
                return mask
        return self._fallback_rectangle(height, width, min_area, max_area)

    @staticmethod
    def _fallback_rectangle(height: int, width: int, min_area: float, max_area: float) -> np.ndarray:
        target = (min_area + max_area) * 0.5
        side = int(math.sqrt(target * height * width))
        side = max(1, min(side, height, width))
        y0 = max(0, (height - side) // 2)
        x0 = max(0, (width - side) // 2)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y0 : y0 + side, x0 : x0 + side] = 1
        return mask


def large_hole_sample_weights(mask_areas: Tensor | list[float], threshold: float = 0.40) -> Tensor:
    areas = torch.as_tensor(mask_areas, dtype=torch.float32)
    return torch.where(areas > threshold, torch.full_like(areas, 3.0), torch.ones_like(areas))


def save_rgb_tensor(path: str | Path, srgb: Tensor) -> None:
    image = srgb.detach().cpu().clamp(0.0, 1.0)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def save_mask_tensor(path: str | Path, mask: Tensor) -> None:
    mask_cpu = mask.detach().cpu().clamp(0.0, 1.0)
    if mask_cpu.ndim == 3:
        mask_cpu = mask_cpu[0]
    array = (mask_cpu.numpy() * 255.0).round().astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="L").save(path)


def load_mask_tensor(path: str | Path) -> Tensor:
    mask = Image.open(path).convert("L")
    array = (np.asarray(mask).astype(np.float32) / 255.0 > 0.5).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0)


def normalize_sample(image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Normalize a single sample with blueprint per-channel valid-pixel stats."""

    return blueprint_normalize(image, mask)


def denormalize(image_norm: Tensor, norm_params: dict[str, float | Tensor]) -> Tensor:
    mean = norm_params["mu"]
    std = norm_params["sigma"]
    if not isinstance(mean, Tensor):
        mean = torch.full((image_norm.shape[-3],), float(mean), device=image_norm.device, dtype=image_norm.dtype)
    else:
        mean = mean.to(device=image_norm.device, dtype=image_norm.dtype)
    if not isinstance(std, Tensor):
        std = torch.full((image_norm.shape[-3],), float(std), device=image_norm.device, dtype=image_norm.dtype)
    else:
        std = std.to(device=image_norm.device, dtype=image_norm.dtype)
    return blueprint_denormalize(image_norm, mean, std)


def _normalize_batch_samples(linear: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor, list[dict[str, float]]]:
    normalized, mean, std = blueprint_normalize(linear, mask)
    norm_params = [{"mu": mean.clone(), "sigma": std.clone()} for _ in range(linear.shape[0])]
    return normalized, mean, std, norm_params


def _apply_blueprint_stats(image_linear: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    mu = mu.to(device=image_linear.device, dtype=image_linear.dtype)
    sigma = sigma.to(device=image_linear.device, dtype=image_linear.dtype)
    if image_linear.ndim == 4:
        mu_view = mu.view(1, mu.shape[0], 1, 1)
        sigma_view = sigma.view(1, sigma.shape[0], 1, 1)
    elif image_linear.ndim == 3:
        mu_view = mu.view(mu.shape[0], 1, 1)
        sigma_view = sigma.view(sigma.shape[0], 1, 1)
    else:
        raise ValueError("image_linear must have shape [C, H, W] or [B, C, H, W]")
    image_z = (image_linear - mu_view) / (sigma_view + 1e-5)
    return image_z.clamp(-3.0, 3.0) / 3.0


def normalize_symmetry_type(value: str | None) -> str:
    if value is None:
        return "none"
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "": "none",
        "unknown": "none",
        "asymmetric": "none",
        "horizontal": "horizontal_only",
        "hflip": "horizontal_only",
        "fourfold": "4fold",
        "4_fold": "4fold",
        "4": "4fold",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in VALID_SYMMETRY_TYPES:
        raise ValueError(f"unknown symmetry type {value!r}; expected one of {sorted(VALID_SYMMETRY_TYPES)}")
    return normalized


def load_symmetry_metadata(path: str | Path | None) -> dict[str, str] | None:
    """Load optional per-tile symmetry metadata from JSON or CSV.

    Keys may be filenames or stems. Values are conservative labels from
    VALID_SYMMETRY_TYPES; missing tiles default to "none".
    """

    if path is None:
        return None
    metadata_path = Path(path)
    if metadata_path.suffix.lower() == ".json":
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("symmetry metadata JSON must be an object mapping filename/stem to label")
        return {str(key): normalize_symmetry_type(str(value)) for key, value in raw.items()}

    if metadata_path.suffix.lower() == ".csv":
        with metadata_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("symmetry metadata CSV must include a header")
            name_field = next((field for field in ["filename", "file", "path", "stem", "image"] if field in reader.fieldnames), None)
            type_field = next((field for field in ["symmetry_type", "symmetry", "label"] if field in reader.fieldnames), None)
            if name_field is None or type_field is None:
                raise ValueError("symmetry metadata CSV needs filename/file/path/stem/image and symmetry_type/symmetry/label columns")
            return {
                row[name_field]: normalize_symmetry_type(row[type_field])
                for row in reader
                if row.get(name_field) and row.get(type_field) is not None
            }

    raise ValueError(f"unsupported symmetry metadata format: {metadata_path.suffix}")


def split_collection_paths(paths: list[Path], spec: CollectionSpec, seed: int = 0) -> dict[str, list[Path]]:
    expected = spec.train + spec.val + spec.test
    if len(paths) != expected:
        raise ValueError(f"{spec.name} requires exactly {expected} images from training.md, found {len(paths)}")
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    return {
        "train": sorted(shuffled[: spec.train]),
        "val": sorted(shuffled[spec.train : spec.train + spec.val]),
        "test": sorted(shuffled[spec.train + spec.val :]),
    }


def canonical_pair_stem(path: Path) -> str:
    return path.stem.removesuffix(" copy")


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _pair_similarity_score(clean_path: Path, partial_path: Path, size: int = 128) -> float:
    clean = resize_training_image(load_rgb_tensor(clean_path), size)
    partial = resize_training_image(load_rgb_tensor(partial_path), size)
    clean_array = clean.permute(1, 2, 0).numpy()
    partial_array = partial.permute(1, 2, 0).numpy()
    white_fill = partial_array.min(axis=2) > 0.88
    foreground = (clean_array.max(axis=2) > 0.04) | (partial_array.max(axis=2) > 0.04)
    valid = (~white_fill) & foreground
    if not np.any(valid):
        return float("inf")
    return float(np.mean((clean_array[valid] - partial_array[valid]) ** 2))


def discover_collection_pairs(project_root: str | Path, spec: CollectionSpec) -> list[tuple[Path, Path | None]]:
    source_dir = Path(project_root) / spec.clean_subdir
    clean_paths = discover_image_paths(source_dir)
    if spec.mask_subdir is None:
        return [(path, None) for path in clean_paths]

    mask_dir = Path(project_root) / spec.mask_subdir
    mask_paths = discover_image_paths(mask_dir)
    masks_by_stem: dict[str, list[Path]] = {}
    for path in mask_paths:
        masks_by_stem.setdefault(canonical_pair_stem(path), []).append(path)

    clean_sizes = {path: _image_size(path) for path in clean_paths}
    mask_sizes = {path: _image_size(path) for path in mask_paths}
    unused_masks = set(mask_paths)
    pairs: list[tuple[Path, Path | None]] = []
    unresolved: list[Path] = []

    for clean_path in clean_paths:
        candidates = [
            mask_path
            for mask_path in masks_by_stem.get(canonical_pair_stem(clean_path), [])
            if mask_path in unused_masks and mask_sizes[mask_path] == clean_sizes[clean_path]
        ]
        if candidates:
            best = candidates[0] if len(candidates) == 1 else min(candidates, key=lambda mask_path: _pair_similarity_score(clean_path, mask_path))
            pairs.append((clean_path, best))
            unused_masks.remove(best)
        else:
            unresolved.append(clean_path)

    for clean_path in unresolved:
        candidates = [mask_path for mask_path in unused_masks if mask_sizes[mask_path] == clean_sizes[clean_path]]
        if not candidates:
            continue
        best = candidates[0] if len(candidates) == 1 else min(candidates, key=lambda mask_path: _pair_similarity_score(clean_path, mask_path))
        pairs.append((clean_path, best))
        unused_masks.remove(best)

    return sorted(pairs, key=lambda pair: pair[0].name)


def split_collection_pairs(pairs: list[tuple[Path, Path | None]], spec: CollectionSpec, seed: int = 0) -> dict[str, list[tuple[Path, Path | None]]]:
    expected = spec.train + spec.val + spec.test
    if len(pairs) != expected:
        raise ValueError(f"{spec.name} requires exactly {expected} paired images for the active non-EDIT training plan, found {len(pairs)}")
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    return {
        "train": sorted(shuffled[: spec.train], key=lambda pair: pair[0].name),
        "val": sorted(shuffled[spec.train : spec.train + spec.val], key=lambda pair: pair[0].name),
        "test": sorted(shuffled[spec.train + spec.val :], key=lambda pair: pair[0].name),
    }


def derive_partial_loss_mask_with_reason(clean_srgb: Tensor, partial_srgb: Tensor, resolution: int) -> PartialLossMaskResult:
    if clean_srgb.shape != partial_srgb.shape:
        return PartialLossMaskResult(None, "shape_mismatch")
    clean_resized = resize_training_image(clean_srgb, resolution)
    partial_resized = resize_training_image(partial_srgb, resolution)
    foreground = clean_resized.max(dim=0, keepdim=True).values > 0.04
    difference = (clean_resized - partial_resized).abs().mean(dim=0, keepdim=True)
    partial_mean = partial_resized.mean(dim=0, keepdim=True)
    clean_mean = clean_resized.mean(dim=0, keepdim=True)
    white_fill = (
        (partial_resized.min(dim=0, keepdim=True).values > 0.88)
        & (partial_mean > clean_mean + 0.08)
        & (difference > 0.08)
        & foreground
    )
    if float(white_fill.float().mean().item()) >= 0.005:
        mask = white_fill.to(dtype=torch.float32)
        derivation = "white_fill"
    else:
        mask = ((difference > 0.05) & foreground).to(dtype=torch.float32)
        derivation = "difference"
    area = float(mask.mean().item())
    if area < 0.005:
        return PartialLossMaskResult(None, f"{derivation}_area_too_small")
    if area > 0.75:
        return PartialLossMaskResult(None, f"{derivation}_area_too_large")
    array = mask.squeeze(0).numpy().astype(np.uint8)
    array = _remove_small_mask_components(array)
    array = cv2.morphologyEx(array, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    array = _remove_small_mask_components(array)
    mask_out = torch.from_numpy(array.astype(np.float32)).unsqueeze(0)
    morphed_area = float(mask_out.mean().item())
    if morphed_area < 0.005:
        return PartialLossMaskResult(None, f"{derivation}_empty_after_morphology")
    if morphed_area > 0.75:
        return PartialLossMaskResult(None, f"{derivation}_area_too_large_after_morphology")
    return PartialLossMaskResult(mask_out, derivation)


def _remove_small_mask_components(mask: np.ndarray, min_fraction: float = 0.001) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask.astype(np.uint8)
    min_pixels = max(16, int(round(mask.shape[0] * mask.shape[1] * min_fraction)))
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for component_id in range(1, count):
        if int(stats[component_id, cv2.CC_STAT_AREA]) >= min_pixels:
            cleaned[labels == component_id] = 1
    if cleaned.sum() == 0:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        cleaned[labels == largest] = 1
    return cleaned


def derive_partial_loss_mask(clean_srgb: Tensor, partial_srgb: Tensor, resolution: int) -> Tensor | None:
    return derive_partial_loss_mask_with_reason(clean_srgb, partial_srgb, resolution).mask


def tile_extent_crop(srgb: Tensor, crop_fraction: float) -> Tensor:
    foreground = srgb.max(dim=0).values > 0.04
    if not foreground.any():
        return srgb
    ys, xs = foreground.nonzero(as_tuple=True)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    h = y1 - y0
    w = x1 - x0
    crop_h = max(1, int(round(h * crop_fraction)))
    crop_w = max(1, int(round(w * crop_fraction)))
    center_y = (y0 + y1) // 2
    center_x = (x0 + x1) // 2
    top = max(0, min(srgb.shape[-2] - crop_h, center_y - crop_h // 2))
    left = max(0, min(srgb.shape[-1] - crop_w, center_x - crop_w // 2))
    return srgb[:, top : top + crop_h, left : left + crop_w]


def paired_random_tile_extent_crop(clean: Tensor, partial: Tensor, crop_fraction: float) -> tuple[Tensor, Tensor]:
    foreground = clean.max(dim=0).values > 0.04
    if not foreground.any():
        return clean, partial
    ys, xs = foreground.nonzero(as_tuple=True)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    h = y1 - y0
    w = x1 - x0
    crop_h = max(1, int(round(h * crop_fraction)))
    crop_w = max(1, int(round(w * crop_fraction)))
    top_min = max(0, y1 - crop_h)
    top_max = min(y0, clean.shape[-2] - crop_h)
    left_min = max(0, x1 - crop_w)
    left_max = min(x0, clean.shape[-1] - crop_w)
    if top_min > top_max:
        top = max(0, min(clean.shape[-2] - crop_h, (y0 + y1) // 2 - crop_h // 2))
    else:
        top = random.randint(top_min, top_max)
    if left_min > left_max:
        left = max(0, min(clean.shape[-1] - crop_w, (x0 + x1) // 2 - crop_w // 2))
    else:
        left = random.randint(left_min, left_max)
    return clean[:, top : top + crop_h, left : left + crop_w], partial[:, top : top + crop_h, left : left + crop_w]


def resize_training_image(srgb: Tensor, resolution: int) -> Tensor:
    return F.interpolate(srgb.unsqueeze(0), size=(resolution, resolution), mode="bilinear", align_corners=False).squeeze(0)


def geometric_variants(srgb: Tensor, *, allow_rotations: bool) -> list[tuple[str, Tensor]]:
    variants: list[tuple[str, Tensor]] = []
    if not allow_rotations:
        return [("identity", srgb), ("hflip", torch.flip(srgb, dims=(-1,)))]
    rotations = [0, 1, 2, 3] if allow_rotations else [0]
    for rot in rotations:
        rotated = torch.rot90(srgb, rot, dims=(-2, -1))
        variants.append((f"rot{rot * 90}_hflip", torch.flip(rotated, dims=(-1,))))
        variants.append((f"rot{rot * 90}_vflip", torch.flip(rotated, dims=(-2,))))
    return variants


def paired_geometric_variants(clean: Tensor, partial: Tensor, *, allow_rotations: bool) -> list[tuple[str, Tensor, Tensor]]:
    clean_variants = geometric_variants(clean, allow_rotations=allow_rotations)
    partial_variants = geometric_variants(partial, allow_rotations=allow_rotations)
    return [
        (clean_name, clean_tensor, partial_tensor)
        for (clean_name, clean_tensor), (partial_name, partial_tensor) in zip(clean_variants, partial_variants)
        if clean_name == partial_name
    ]


def _metadata_symmetry(path: Path, symmetry_metadata: dict[str, str] | None) -> str:
    if not symmetry_metadata:
        return "none"
    return normalize_symmetry_type(symmetry_metadata.get(path.name, symmetry_metadata.get(path.stem, "none")))


def _allow_rotation_augmentation(path: Path, symmetry_metadata: dict[str, str] | None) -> bool:
    return _metadata_symmetry(path, symmetry_metadata) in {"4fold", "radial"}


def prepare_fixed_augmented_dataset(
    project_root: str | Path,
    output_root: str | Path,
    *,
    resolution: int = 512,
    seed: int = 0,
    symmetry_metadata: dict[str, str] | None = None,
    allow_synthetic_eval_masks: bool = False,
    crops_per_variant: int = 3,
    collections: tuple[CollectionSpec, ...] = TRAINING_COLLECTIONS,
) -> Path:
    """Write the fixed training.md augmented dataset and return manifest path."""

    random.seed(seed)
    np.random.seed(seed)
    project_root = Path(project_root)
    output_root = Path(output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    image_dir = output_root / "images"
    mask_dir = output_root / "masks"
    rows: list[dict[str, str | float]] = []
    skipped_rows: list[dict[str, str]] = []
    mask_generator = SyntheticMaskGenerator()

    for spec in collections:
        pairs = discover_collection_pairs(project_root, spec)
        splits = split_collection_pairs(pairs, spec, seed=seed)
        for split, split_pairs in splits.items():
            for src, source_mask in split_pairs:
                if source_mask is None:
                    raise ValueError(f"{src} has no paired partial mask image")
                clean_base = resize_training_image(tile_extent_crop(load_rgb_tensor(src), 1.0), resolution)
                partial_base = resize_training_image(tile_extent_crop(load_rgb_tensor(source_mask), 1.0), resolution)
                if split == "train":
                    allow_rotations = _allow_rotation_augmentation(src, symmetry_metadata)
                    variants = paired_geometric_variants(clean_base, partial_base, allow_rotations=allow_rotations)
                    for geom_idx, (geom_name, clean_variant, partial_variant) in enumerate(variants):
                        for crop_idx in range(max(1, crops_per_variant)):
                            if crop_idx == 0:
                                clean_crop = clean_variant
                                partial_crop = partial_variant
                                crop_name = "full"
                            else:
                                crop_fraction = random.uniform(0.85, 1.00)
                                clean_crop, partial_crop = paired_random_tile_extent_crop(
                                    clean_variant,
                                    partial_variant,
                                    crop_fraction,
                                )
                                clean_crop = resize_training_image(clean_crop, resolution)
                                partial_crop = resize_training_image(partial_crop, resolution)
                                crop_name = f"crop{crop_idx}"
                            mask_result = derive_partial_loss_mask_with_reason(clean_crop, partial_crop, resolution)
                            if mask_result.mask is None:
                                skipped_rows.append(
                                    {
                                        "split": split,
                                        "collection": spec.name,
                                        "source_path": str(src),
                                        "source_mask_path": str(source_mask),
                                        "geometry": geom_name,
                                        "crop": crop_name,
                                        "reason": mask_result.issue,
                                    }
                                )
                                continue
                            image_name = f"{spec.name}_{src.stem}_{geom_idx:02d}_{geom_name}_{crop_name}.png"
                            image_path = image_dir / split / image_name
                            save_rgb_tensor(image_path, clean_crop)
                            mask_name = f"{spec.name}_{src.stem}_{geom_idx:02d}_{geom_name}_{crop_name}_real_partial.png"
                            mask_path = mask_dir / split / mask_name
                            save_mask_tensor(mask_path, mask_result.mask)
                            rows.append(
                                {
                                    "split": split,
                                    "collection": spec.name,
                                    "image_path": str(image_path),
                                    "mask_path": str(mask_path),
                                    "source_path": str(src),
                                    "source_mask_path": str(source_mask),
                                    "mask_area": float(mask_result.mask.mean().item()),
                                    "mask_kind": "real_partial",
                                    "mask_issue": mask_result.issue,
                                    "symmetry_type": _metadata_symmetry(src, symmetry_metadata),
                                }
                            )
                else:
                    image_name = f"{spec.name}_{src.stem}.png"
                    image_path = image_dir / split / image_name
                    save_rgb_tensor(image_path, clean_base)
                    if allow_synthetic_eval_masks:
                        mask = mask_generator(resolution, resolution, LATE_CURRICULUM_ITERATION, kind="blob")
                        mask_kind = "blob_synthetic_eval"
                        mask_issue = "synthetic_eval_fair_comparison"
                    else:
                        mask_result = derive_partial_loss_mask_with_reason(clean_base, partial_base, resolution)
                        if mask_result.mask is None:
                            skipped_rows.append(
                                {
                                    "split": split,
                                    "collection": spec.name,
                                    "source_path": str(src),
                                    "source_mask_path": str(source_mask),
                                    "geometry": "identity",
                                    "reason": mask_result.issue,
                                }
                            )
                            continue
                        mask = mask_result.mask
                        mask_kind = "real_partial"
                        mask_issue = mask_result.issue
                    mask_path = mask_dir / split / f"{spec.name}_{src.stem}_{mask_kind}.png"
                    save_mask_tensor(mask_path, mask)
                    rows.append(
                        {
                            "split": split,
                            "collection": spec.name,
                            "image_path": str(image_path),
                            "mask_path": str(mask_path),
                            "source_path": str(src),
                            "source_mask_path": str(source_mask),
                            "mask_area": float(mask.mean().item()),
                            "mask_kind": mask_kind,
                            "mask_issue": mask_issue,
                            "symmetry_type": _metadata_symmetry(src, symmetry_metadata),
                        }
                    )

    manifest_path = output_root / MANIFEST_NAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "split",
            "collection",
            "image_path",
            "mask_path",
            "source_path",
            "source_mask_path",
            "mask_area",
            "mask_kind",
            "mask_issue",
            "symmetry_type",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if skipped_rows:
        skipped_path = output_root / "skipped_real_masks.csv"
        with skipped_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = ["split", "collection", "source_path", "source_mask_path", "geometry", "crop", "reason"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(skipped_rows)
    return manifest_path


def load_manifest_rows(manifest_path: str | Path, split: str | None = None, collection: str | None = None) -> list[dict[str, str]]:
    with Path(manifest_path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if split is not None:
        rows = [row for row in rows if row["split"] == split]
    if collection is not None:
        rows = [row for row in rows if row["collection"] == collection]
    return rows


class PreparedTileDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str,
        collection: str | None = None,
        min_mask_area: float | None = None,
        max_mask_area: float | None = None,
        synthetic_mask_ratio: float = 0.50,
    ) -> None:
        self.split = split
        self.manifest_path = Path(manifest_path)
        self.min_mask_area = min_mask_area
        self.max_mask_area = max_mask_area
        self.synthetic_mask_ratio = synthetic_mask_ratio
        self.rows = load_manifest_rows(self.manifest_path, split=split, collection=collection)
        if min_mask_area is not None:
            self.rows = [row for row in self.rows if float(row["mask_area"]) >= min_mask_area]
        if max_mask_area is not None:
            self.rows = [row for row in self.rows if float(row["mask_area"]) <= max_mask_area]
        if not self.rows:
            raise ValueError(f"no rows for split={split!r}, collection={collection!r} in {self.manifest_path}")
        self.mask_generator = SyntheticMaskGenerator()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Tensor | str | float]:
        row = self.rows[index]
        srgb = load_rgb_tensor(row["image_path"])
        mask = load_mask_tensor(row["mask_path"])

        if self.split == "train":
            symmetry_type = row.get("symmetry_type", "none")
            srgb, mask = safe_geometry_augment(srgb, mask, symmetry_type)
            mask = perturb_real_mask_within_bucket(
                mask,
                min_area=self.min_mask_area,
                max_area=self.max_mask_area,
            )
            real_area = float(mask.mean().item())
            if random.random() < self.synthetic_mask_ratio:
                target_min = max(0.005, real_area - 0.10)
                target_max = min(0.75, real_area + 0.10)
                if self.min_mask_area is not None:
                    target_min = max(target_min, float(self.min_mask_area))
                if self.max_mask_area is not None:
                    target_max = min(target_max, float(self.max_mask_area))
                kind = random.choice(["blob", "blob", "blob", "freeform", "rectangular"])
                synthetic_mask = self.mask_generator(
                    mask.shape[-2],
                    mask.shape[-1],
                    LATE_CURRICULUM_ITERATION,
                    kind=kind,
                )
                synthetic_area = float(synthetic_mask.mean().item())
                if target_min <= synthetic_area <= target_max:
                    mask = synthetic_mask
            srgb_clean = srgb.clone()
            srgb = random_erase_augmentation(srgb, mask)
            srgb = random_channel_shuffle(srgb)
            srgb = apply_safe_tiny_data_augmentation(srgb)
        else:
            srgb_clean = srgb

        linear = srgb_to_linear(srgb.unsqueeze(0)).squeeze(0)
        gt_linear = srgb_to_linear(srgb_clean.unsqueeze(0)).squeeze(0)
        symmetry_prior = compute_symmetry_prior(linear.unsqueeze(0), mask).squeeze(0)
        
        mask_area = float(mask.mean().item())
        
        return {
            "linear": linear,
            "image_linear": linear,
            "gt_linear": gt_linear,
            "mask": mask,
            "mask_area": mask_area,
            "symmetry_prior": symmetry_prior,
            "path": row["image_path"],
            "mask_path": row["mask_path"],
            "source_path": row.get("source_path", row["image_path"]),
            "source_mask_path": row.get("source_mask_path", row["mask_path"]),
            "collection": row["collection"],
            "split": row.get("split", self.split),
            "mask_kind": row.get("mask_kind", ""),
            "diagnostic_bucket": row.get("diagnostic_bucket", ""),
        }


class MuseumStratifiedBatchSampler(BatchSampler):
    """Batch sampler that keeps Coptic/Egyptian/Graeco-Roman present when possible."""

    def __init__(
        self,
        dataset: PreparedTileDataset,
        batch_size: int,
        *,
        seed: int = 0,
        drop_last: bool = False,
        large_hole_weighting: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.by_collection: dict[str, list[int]] = {}
        for idx, row in enumerate(dataset.rows):
            repeats = 3 if large_hole_weighting and float(row["mask_area"]) > 0.40 else 1
            self.by_collection.setdefault(row["collection"], []).extend([idx] * repeats)

    def __iter__(self):
        rng = random.Random(self.seed)
        pools = {name: indices[:] for name, indices in self.by_collection.items()}
        for indices in pools.values():
            rng.shuffle(indices)
        all_indices = [idx for indices in pools.values() for idx in indices]
        rng.shuffle(all_indices)
        batch: list[int] = []
        while all_indices:
            for collection in sorted(pools):
                if len(batch) >= self.batch_size:
                    break
                while pools[collection]:
                    idx = pools[collection].pop()
                    if idx in all_indices:
                        all_indices.remove(idx)
                        batch.append(idx)
                        break
            while len(batch) < self.batch_size and all_indices:
                batch.append(all_indices.pop())
            if len(batch) == self.batch_size or (batch and not self.drop_last):
                yield batch
            batch = []

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)


class SourceBalancedBatchSampler(BatchSampler):
    """Collection-balanced sampler that treats each source artifact as one unit.

    Row-level variants are sampled only after a source has been selected. Large
    masks receive source-level, not row-level, emphasis so one artifact with
    multiple large diagnostic rows cannot dominate the stream.
    """

    def __init__(
        self,
        dataset: PreparedTileDataset,
        batch_size: int,
        *,
        seed: int = 0,
        drop_last: bool = False,
        large_hole_weighting: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.large_hole_weighting = large_hole_weighting
        self._epoch = 0
        grouped: dict[str, dict[str, list[int]]] = {}
        for idx, row in enumerate(dataset.rows):
            collection = row.get("collection", "")
            source = row.get("source_path") or row.get("image_path") or str(idx)
            grouped.setdefault(collection, {}).setdefault(source, []).append(idx)
        self.by_collection = grouped
        self.sources_by_collection = {
            collection: sorted(sources)
            for collection, sources in grouped.items()
            if sources
        }

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        collections = sorted(self.sources_by_collection)
        if not collections:
            return
        weighted_sources: dict[str, list[str]] = {}
        for collection, sources in self.sources_by_collection.items():
            pool: list[str] = []
            for source in sources:
                indices = self.by_collection[collection][source]
                has_large = any(float(self.dataset.rows[idx]["mask_area"]) > 0.40 for idx in indices)
                repeats = 3 if self.large_hole_weighting and has_large else 1
                pool.extend([source] * repeats)
            weighted_sources[collection] = pool
        collection_cursor = 0
        for _ in range(len(self)):
            batch: list[int] = []
            while len(batch) < self.batch_size:
                collection = collections[collection_cursor % len(collections)]
                collection_cursor += 1
                source = rng.choice(weighted_sources[collection])
                row_indices = self.by_collection[collection][source]
                batch.append(rng.choice(row_indices))
            if len(batch) == self.batch_size or (batch and not self.drop_last):
                yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)


class CleanTileDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        resolution: int,
        iteration: int = 0,
        paths: list[str | Path] | None = None,
        augment: bool = True,
        degrade: bool = True,
    ) -> None:
        self.root = Path(root)
        self.paths = [Path(path) for path in paths] if paths is not None else discover_image_paths(self.root)
        if not self.paths:
            raise ValueError(f"no image files found under {self.root}")
        self.resolution = resolution
        self.iteration = iteration
        self.augment = augment
        self.degrade = degrade
        self.mask_generator = SyntheticMaskGenerator()
        self._force_large_by_index: set[int] = set()
        self.last_plan_areas = torch.ones(len(self.paths), dtype=torch.float32) * 0.10

    def set_iteration(self, iteration: int) -> None:
        self.iteration = iteration

    def refresh_sampling_plan(self, iteration: int) -> Tensor:
        self.set_iteration(iteration)
        areas = []
        force_large: set[int] = set()
        for idx in range(len(self.paths)):
            mask = self.mask_generator(self.resolution, self.resolution, iteration)
            area = float(mask.mean().item())
            areas.append(area)
            if area > 0.40:
                force_large.add(idx)
        self._force_large_by_index = force_large
        self.last_plan_areas = torch.tensor(areas, dtype=torch.float32)
        return self.last_plan_areas

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Tensor | str | float]:
        srgb = load_rgb_tensor(self.paths[index])
        srgb = self._augment_and_crop(srgb) if self.augment else self._resize_or_crop_center(srgb)
        if self.degrade:
            srgb = apply_synthetic_degradation(srgb)
        linear = srgb_to_linear(srgb.unsqueeze(0)).squeeze(0)
        force_large = index in self._force_large_by_index
        mask_raw = self.mask_generator(self.resolution, self.resolution, self.iteration, force_large=force_large)
        mask_area = float(mask_raw.mean().item())
        symmetry_prior = compute_symmetry_prior(linear.unsqueeze(0), mask_raw).squeeze(0)
        return {
            "linear": linear,
            "image_linear": linear,
            "gt_linear": linear,
            "mask": mask_raw,
            "mask_area": mask_area,
            "symmetry_prior": symmetry_prior,
            "path": str(self.paths[index]),
        }

    def _augment_and_crop(self, srgb: Tensor) -> Tensor:
        if random.random() < 0.5:
            srgb = torch.flip(srgb, dims=(-1,))
        rot_pick = random.random()
        if rot_pick < 0.3:
            srgb = torch.rot90(srgb, 1, dims=(-2, -1))
        elif rot_pick < 0.6:
            srgb = torch.rot90(srgb, 2, dims=(-2, -1))
        elif rot_pick < 0.9:
            srgb = torch.rot90(srgb, 3, dims=(-2, -1))
        return self._resize_or_random_crop(srgb)

    def _resize_or_random_crop(self, srgb: Tensor) -> Tensor:
        _, height, width = srgb.shape
        if height >= self.resolution and width >= self.resolution:
            y0 = random.randint(0, height - self.resolution)
            x0 = random.randint(0, width - self.resolution)
            return srgb[:, y0 : y0 + self.resolution, x0 : x0 + self.resolution]
        return F.interpolate(
            srgb.unsqueeze(0),
            size=(self.resolution, self.resolution),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def _resize_or_crop_center(self, srgb: Tensor) -> Tensor:
        _, height, width = srgb.shape
        if height >= self.resolution and width >= self.resolution:
            y0 = (height - self.resolution) // 2
            x0 = (width - self.resolution) // 2
            return srgb[:, y0 : y0 + self.resolution, x0 : x0 + self.resolution]
        return F.interpolate(
            srgb.unsqueeze(0),
            size=(self.resolution, self.resolution),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)


def collate_training_batch(batch: list[dict[str, Tensor | str | float]], mask_dilation: int = 4) -> dict[str, Tensor | list[str]]:
    linear = torch.stack([
        sample["image_linear"] if "image_linear" in sample else sample["linear"] for sample in batch
    ])
    gt_linear = torch.stack([
        sample["gt_linear"] if "gt_linear" in sample else sample["linear"] for sample in batch
    ])
    mask = torch.stack([sample["mask"] for sample in batch])

    image_norm_items = []
    gt_norm_items = []
    mu_items = []
    sigma_items = []
    for idx in range(linear.shape[0]):
        sample_norm, sample_mu, sample_sigma = blueprint_normalize(linear[idx], mask[idx])
        image_norm_items.append(sample_norm)
        gt_norm_items.append(_apply_blueprint_stats(gt_linear[idx], sample_mu, sample_sigma))
        mu_items.append(sample_mu)
        sigma_items.append(sample_sigma)
    image_norm = torch.stack(image_norm_items)
    gt_norm = torch.stack(gt_norm_items)
    norm_mu = torch.stack(mu_items)
    norm_sigma = torch.stack(sigma_items)
    gt_norm = (gt_norm + torch.randn_like(gt_norm) * 0.02).clamp(-1.0, 1.0)
    norm_params = [{"mu": norm_mu[idx].clone(), "sigma": norm_sigma[idx].clone()} for idx in range(linear.shape[0])]
    mask_network = dilate_mask(mask, radius=mask_dilation)
    masked = encode_masked_pixels(image_norm, mask_network)
    edge = compute_canny_edges(linear, mask)
    symmetry = torch.stack([
        sample["symmetry_prior"]
        if "symmetry_prior" in sample
        else compute_symmetry_prior(
            (sample["image_linear"] if "image_linear" in sample else sample["linear"]).unsqueeze(0),
            sample["mask"],
        ).squeeze(0)
        for sample in batch
    ])
    mask_area = torch.tensor([float(sample["mask_area"]) for sample in batch], dtype=torch.float32)
    return {
        "image": image_norm,
        "gt": gt_norm,
        "masked": masked,
        "mask": mask,
        "mask_network": mask_network,
        "edge": edge,
        "symmetry": symmetry,
        "mask_area": mask_area,
        "mean": norm_mu,
        "std": norm_sigma,
        "norm_mu": norm_mu,
        "norm_sigma": norm_sigma,
        "norm_params": norm_params,
        "path": [str(sample["path"]) for sample in batch],
        "mask_path": [str(sample.get("mask_path", "")) for sample in batch],
        "source_path": [str(sample.get("source_path", "")) for sample in batch],
        "source_mask_path": [str(sample.get("source_mask_path", "")) for sample in batch],
        "collection": [str(sample.get("collection", "")) for sample in batch],
        "split": [str(sample.get("split", "")) for sample in batch],
        "mask_kind": [str(sample.get("mask_kind", "")) for sample in batch],
        "diagnostic_bucket": [str(sample.get("diagnostic_bucket", "")) for sample in batch],
    }


def collate_evaluation_batch(batch: list[dict[str, Tensor | str | float]], mask_dilation: int = 4) -> dict[str, Tensor | list[str]]:
    """Evaluation collation using per-image valid-pixel normalization."""
    linear = torch.stack([
        sample["image_linear"] if "image_linear" in sample else sample["linear"] for sample in batch
    ])
    gt_linear = torch.stack([
        sample["gt_linear"] if "gt_linear" in sample else sample["linear"] for sample in batch
    ])
    mask = torch.stack([sample["mask"] for sample in batch])

    image_norm_items = []
    gt_norm_items = []
    mu_items = []
    sigma_items = []
    for idx in range(linear.shape[0]):
        sample_norm, sample_mu, sample_sigma = blueprint_normalize(linear[idx], mask[idx])
        image_norm_items.append(sample_norm)
        gt_norm_items.append(_apply_blueprint_stats(gt_linear[idx], sample_mu, sample_sigma))
        mu_items.append(sample_mu)
        sigma_items.append(sample_sigma)
    image_norm = torch.stack(image_norm_items)
    gt_norm = torch.stack(gt_norm_items)
    norm_mu = torch.stack(mu_items)
    norm_sigma = torch.stack(sigma_items)
    norm_params = [{"mu": norm_mu[idx].clone(), "sigma": norm_sigma[idx].clone()} for idx in range(linear.shape[0])]

    mask_network = dilate_mask(mask, radius=mask_dilation)
    masked = encode_masked_pixels(image_norm, mask_network)
    edge = compute_canny_edges(linear, mask)
    symmetry = torch.stack([
        sample["symmetry_prior"]
        if "symmetry_prior" in sample
        else compute_symmetry_prior(
            (sample["image_linear"] if "image_linear" in sample else sample["linear"]).unsqueeze(0),
            sample["mask"],
        ).squeeze(0)
        for sample in batch
    ])
    mask_area = torch.tensor([float(sample["mask_area"]) for sample in batch], dtype=torch.float32)

    return {
        "image": image_norm,
        "gt": gt_norm,
        "masked": masked,
        "mask": mask,
        "mask_network": mask_network,
        "edge": edge,
        "symmetry": symmetry,
        "mask_area": mask_area,
        "mean": norm_mu,
        "std": norm_sigma,
        "norm_mu": norm_mu,
        "norm_sigma": norm_sigma,
        "norm_params": norm_params,
        "path": [str(sample["path"]) for sample in batch],
        "mask_path": [str(sample.get("mask_path", "")) for sample in batch],
        "source_path": [str(sample.get("source_path", "")) for sample in batch],
        "source_mask_path": [str(sample.get("source_mask_path", "")) for sample in batch],
        "collection": [str(sample.get("collection", "")) for sample in batch],
        "split": [str(sample.get("split", "")) for sample in batch],
        "mask_kind": [str(sample.get("mask_kind", "")) for sample in batch],
        "diagnostic_bucket": [str(sample.get("diagnostic_bucket", "")) for sample in batch],
    }


def build_curriculum_sampler(dataset: CleanTileDataset, iteration: int) -> WeightedRandomSampler | None:
    areas = dataset.refresh_sampling_plan(iteration)
    if iteration < 8_000:
        return None
    weights = large_hole_sample_weights(areas)
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
