"""Train unet_pseudo_v1 against the classical pseudo-labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from sem.qc.config import load_config
from sem.qc.io import load_stem, valid_mask
from sem.qc.models.unet import build_unet, prepare_input
from sem.qc.partial_data import list_complete_stems


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross entropy with ignored pixels plus Dice over present target classes."""
    keep = target != 255
    if not torch.any(keep):
        return logits.sum() * 0.0
    cross_entropy = F.cross_entropy(logits, target.long(), ignore_index=255)
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target.clamp(0, logits.shape[1] - 1).long(), logits.shape[1])
    one_hot = one_hot.permute(0, 3, 1, 2).to(dtype=probabilities.dtype)
    keep = keep[:, None]
    probabilities = probabilities * keep
    one_hot = one_hot * keep
    present = torch.unique(target[keep[:, 0]])
    present = present[present != 255]
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    return cross_entropy + (1.0 - dice[present].mean())


def _read_manifest(root: Path, config: dict) -> dict[str, str]:
    path = root / "data" / "splits" / "manifest.csv"
    try:
        from sem.qc.split import load_manifest
    except ImportError:
        if not path.exists():
            raise FileNotFoundError(
                f"Frozen split manifest is required at {path}; test stems are never "
                "used for training or model selection"
            )
        with path.open(newline="", encoding="utf-8") as manifest_file:
            return {
                row["stem"]: row["split"]
                for row in csv.DictReader(manifest_file)
            }
    rows = load_manifest(path, config.get("split", {}).get("frozen_manifest_sha256"))
    return {str(row["stem"]): str(row["split"]) for row in rows}


def _pseudo_target(
    semantic: np.ndarray,
    uncertainty: np.ndarray,
    valid: np.ndarray,
    uncertainty_threshold: float,
) -> np.ndarray:
    target = semantic.copy()
    target[(uncertainty / 255.0) > uncertainty_threshold] = 255
    target[~valid] = 255
    target[target == 255] = 255
    return target


def _ignored_fraction_breakdown(
    semantic: np.ndarray,
    uncertainty: np.ndarray,
    valid: np.ndarray,
    uncertainty_threshold: float,
) -> dict[str, float]:
    base_ignored = (~valid) | (semantic == 255)
    uncertainty_ignored = (
        (uncertainty / 255.0 > uncertainty_threshold) & ~base_ignored
    )
    return {
        "invalid_or_classical_255": float(np.mean(base_ignored)),
        "uncertainty_gt_threshold": float(np.mean(uncertainty_ignored)),
        "total": float(np.mean(base_ignored | uncertainty_ignored)),
    }


def _load_stem_data(
    record,
    work_root: Path,
    uncertainty_threshold: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, float]]:
    from PIL import Image

    views = load_stem(record)
    semantic_path = work_root / "preds" / "classical_v1" / f"{record.stem}_semantic.png"
    uncertainty_path = (
        work_root / "preds" / "classical_v1" / f"{record.stem}_uncertainty.png"
    )
    semantic = np.asarray(Image.open(semantic_path).convert("L"), dtype=np.uint8)
    uncertainty = np.asarray(Image.open(uncertainty_path).convert("L"), dtype=np.uint8)
    valid = valid_mask(record)
    if semantic.shape != valid.shape or uncertainty.shape != valid.shape:
        raise ValueError(f"Classical maps have wrong shape for {record.stem}")
    target = _pseudo_target(semantic, uncertainty, valid, uncertainty_threshold)
    ignored_fractions = _ignored_fraction_breakdown(
        semantic, uncertainty, valid, uncertainty_threshold
    )
    return views, target, ignored_fractions


def _precompute_class_coordinates(
    target: np.ndarray,
    rng: np.random.Generator,
    max_per_class: int = 200_000,
) -> dict[int, np.ndarray]:
    height, width = target.shape
    coordinates = {}
    for class_id in np.unique(target[target != 255]):
        linear = np.flatnonzero(target.ravel() == class_id)
        if linear.size > max_per_class:
            linear = rng.choice(linear, size=max_per_class, replace=False)
        coordinates[int(class_id)] = np.column_stack(
            (linear // width, linear % width)
        ).astype(np.int32, copy=False)
    return coordinates


def _budget_aware_learning_rate(
    base_learning_rate: float,
    completed_iters: int,
    planned_iters: int,
    elapsed_s: float,
    max_seconds: float,
) -> float:
    if max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    progress = max(completed_iters / max(1, planned_iters), elapsed_s / max_seconds)
    progress = float(np.clip(progress, 0.0, 1.0))
    return base_learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def _sample_crop(
    item: tuple[dict[str, np.ndarray], np.ndarray],
    crop_size: int,
    rng: np.random.Generator,
    class_aware: bool,
    class_coordinates: dict[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    views, target = item
    height, width = target.shape
    crop_h, crop_w = min(crop_size, height), min(crop_size, width)
    if class_aware and class_coordinates is None:
        class_coordinates = _precompute_class_coordinates(target, rng)
    classes = tuple(class_coordinates or ())
    for _ in range(100):
        if class_aware and len(classes):
            class_id = int(rng.choice(classes))
            coordinates = class_coordinates[class_id]
            index = int(rng.integers(len(coordinates)))
            center_y, center_x = (int(value) for value in coordinates[index])
            y0 = int(np.clip(center_y - crop_h // 2, 0, height - crop_h))
            x0 = int(np.clip(center_x - crop_w // 2, 0, width - crop_w))
        else:
            y0 = int(rng.integers(max(1, height - crop_h + 1)))
            x0 = int(rng.integers(max(1, width - crop_w + 1)))
        y1, x1 = y0 + crop_h, x0 + crop_w
        image = prepare_input(
            {name: plane[y0:y1, x0:x1] for name, plane in views.items()}
        )
        label = target[y0:y1, x0:x1].copy()
        if crop_h < crop_size or crop_w < crop_size:
            pad_h, pad_w = crop_size - crop_h, crop_size - crop_w
            image = np.pad(
                image,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode="reflect" if min(crop_h, crop_w) > 1 else "edge",
            )
            label = np.pad(
                label,
                ((0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=255,
            )
        if np.mean(label != 255) >= 0.3:
            return image, label
    raise ValueError("Unable to sample a crop with at least 30% usable pixels")


def _augment(
    image: np.ndarray, target: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        image, target = image[:, ::-1, :], target[::-1, :]
    if rng.random() < 0.5:
        image, target = image[:, :, ::-1], target[:, ::-1]
    turns = int(rng.integers(4))
    image = np.rot90(image, turns, axes=(1, 2))
    target = np.rot90(target, turns)
    for channel in range(image.shape[0]):
        contrast = float(rng.uniform(0.9, 1.1))
        brightness = float(rng.uniform(-0.1, 0.1))
        mean = image[channel].mean()
        image[channel] = np.clip(
            (image[channel] - mean) * contrast + mean + brightness, -2.0, 2.0
        )
    return np.ascontiguousarray(image), np.ascontiguousarray(target)


def _validation_metrics(
    model: torch.nn.Module,
    validation_crops: list[tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> dict[str, object]:
    classes = 8
    intersections = np.zeros(classes, dtype=np.int64)
    unions = np.zeros(classes, dtype=np.int64)
    target_counts = np.zeros(classes, dtype=np.int64)
    matched = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(validation_crops), 8):
            batch = validation_crops[start : start + 8]
            inputs = torch.from_numpy(np.stack([item[0] for item in batch])).to(device)
            targets = np.stack([item[1] for item in batch])
            predicted = model(inputs).argmax(dim=1).cpu().numpy()
            keep = targets != 255
            matched += int(np.count_nonzero((predicted == targets) & keep))
            total += int(keep.sum())
            for class_id in range(classes):
                truth = targets == class_id
                guess = (predicted == class_id) & keep
                intersections[class_id] += int(np.count_nonzero(truth & guess))
                unions[class_id] += int(np.count_nonzero(truth | guess))
                target_counts[class_id] += int(np.count_nonzero(truth))
    present = np.flatnonzero(target_counts)
    ious = {
        str(int(class_id)): float(intersections[class_id] / unions[class_id])
        for class_id in present
    }
    return {
        "agreement_with_classical_v1_pixel": float(matched / max(1, total)),
        "agreement_with_classical_v1_miou": float(
            np.mean([intersections[c] / unions[c] for c in present])
        ) if len(present) else 0.0,
        "agreement_with_classical_v1_iou_by_class": ious,
        "agreement_with_classical_v1_classes_present": [int(value) for value in present],
    }


def _save_curves(path: Path, epochs: list[dict[str, object]]) -> None:
    width, height = 960, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 16), "Training loss and agreement with classical_v1", fill="black")
    if epochs:
        left, top, plot_w, plot_h = 60, 60, width - 100, height - 110
        draw.rectangle((left, top, left + plot_w, top + plot_h), outline="#777777")
        losses = [float(row["train_loss"]) for row in epochs]
        agreements = [
            float(row["agreement_with_classical_v1_miou"]) for row in epochs
        ]
        for values, color in ((losses, "#2369bd"), (agreements, "#cf4436")):
            low, high = min(values), max(values)
            if high == low:
                high = low + 1.0
            points = [
                (
                    left + round(i * plot_w / max(1, len(values) - 1)),
                    top + round((1.0 - (value - low) / (high - low)) * plot_h),
                )
                for i, value in enumerate(values)
            ]
            if len(points) > 1:
                draw.line(points, fill=color, width=3)
            else:
                x, y = points[0]
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        draw.text((left, height - 35), "Blue: loss   Red: val agreement mIoU", fill="black")
    image.save(path)


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def train(args: argparse.Namespace) -> dict[str, object]:
    config = load_config(args.config)
    training_config = config.get("unet_pseudo_v1", {})
    arch = args.arch or training_config.get("arch", "smp_resnet18")
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(8)
    rng = np.random.default_rng(seed)
    work_root = Path(config["paths"]["work_root"])
    project_root = Path(__file__).resolve().parents[1]
    manifest = _read_manifest(project_root, config)
    manifest_path = project_root / "data" / "splits" / "manifest.csv"
    records, _ = list_complete_stems(
        config["paths"]["data_root"], work_root, manifest_path
    )
    record_by_stem = {record.stem: record for record in records}
    requested_train_stems = sorted(
        stem for stem, split in manifest.items() if split == "train"
    )
    requested_val_stems = sorted(
        stem for stem, split in manifest.items() if split == "val"
    )
    test_stems = {
        stem for stem, split in manifest.items() if split == "test"
    }
    if set(requested_train_stems + requested_val_stems) & test_stems:
        raise AssertionError("Test stems cannot be used for training or selection")
    classical_root = work_root / "preds" / "classical_v1"

    def is_available(stem: str) -> bool:
        return stem in record_by_stem and all(
            (classical_root / f"{stem}_{suffix}.png").is_file()
            for suffix in ("semantic", "uncertainty")
        )

    missing_train_stems = [
        stem for stem in requested_train_stems if not is_available(stem)
    ]
    missing_val_stems = [
        stem for stem in requested_val_stems if not is_available(stem)
    ]
    train_stems = [
        stem for stem in requested_train_stems if stem not in missing_train_stems
    ]
    val_stems = [
        stem for stem in requested_val_stems if stem not in missing_val_stems
    ]
    print(f"missing_train_stems={missing_train_stems}", flush=True)
    print(f"missing_val_stems={missing_val_stems}", flush=True)
    if not train_stems or not val_stems:
        raise ValueError(
            "Training requires at least one available train stem and one available val stem"
        )

    threshold = float(training_config.get("ignore_if_classical_uncertainty_gt", 0.5))
    cache: dict[str, tuple[dict[str, np.ndarray], np.ndarray]] = {}
    for stem in train_stems + val_stems:
        record = record_by_stem[stem]
        views, target, ignored_fractions = _load_stem_data(
            record, work_root, threshold
        )
        pair = (views, target)
        cache[stem] = pair
        histogram_counts = np.bincount(target[target != 255], minlength=8)[:8]
        histogram = {
            class_id: int(count)
            for class_id, count in enumerate(histogram_counts)
            if count
        }
        print(
            f"{stem}: ignored_invalid_or_classical_255_fraction="
            f"{ignored_fractions['invalid_or_classical_255']:.6f} "
            f"ignored_uncertainty_gt_threshold_fraction="
            f"{ignored_fractions['uncertainty_gt_threshold']:.6f} "
            f"ignored_total_fraction={ignored_fractions['total']:.6f} "
            f"pseudo_label_class_histogram={dict(sorted(histogram.items()))}",
            flush=True,
        )

    crop_size = int(training_config.get("crop_size_px", 256))
    fixed_rng = np.random.default_rng(seed)
    train_class_coordinates = {
        stem: _precompute_class_coordinates(cache[stem][1], rng)
        for stem in train_stems
    }
    validation_crops = [
        _sample_crop(cache[str(fixed_rng.choice(val_stems))], 512, fixed_rng, False)
        for _ in range(64)
    ]
    model = build_unet(
        arch=arch,
        encoder_weights=training_config.get("encoder_weights"),
    ).to("cpu")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )
    iters_per_epoch = int(training_config.get("iters_per_epoch", 200))
    batch_size = int(training_config.get("batch_size", 8))
    max_epochs = int(training_config.get("max_epochs", 100))
    patience = int(training_config.get("early_stop_patience", 4))
    max_seconds = float(training_config.get("max_train_minutes", 100)) * 60.0
    if max_seconds <= 0:
        raise ValueError("max_train_minutes must be positive")
    base_learning_rate = float(training_config.get("learning_rate", 1e-3))
    planned_iters = max(1, iters_per_epoch * max_epochs)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_miou = -1.0
    best_epoch = 0
    stale_epochs = 0
    best_epoch_pixel_agreement = 0.0
    epochs: list[dict[str, object]] = []
    started = time.monotonic()
    total_iters_done = 0
    stop_reason = "max_epochs"
    for epoch in range(1, max_epochs + 1):
        if time.monotonic() - started >= max_seconds:
            stop_reason = "max_train_minutes"
            break
        model.train()
        epoch_loss = 0.0
        steps = 0
        epoch_lrs = []
        epoch_started = time.monotonic()
        for _ in range(iters_per_epoch):
            elapsed = time.monotonic() - started
            if elapsed >= max_seconds:
                stop_reason = "max_train_minutes"
                break
            learning_rate = _budget_aware_learning_rate(
                base_learning_rate,
                total_iters_done,
                planned_iters,
                elapsed,
                max_seconds,
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate
            epoch_lrs.append(learning_rate)
            inputs = []
            targets = []
            for _ in range(batch_size):
                chosen = str(rng.choice(train_stems))
                image, target = _sample_crop(
                    cache[chosen],
                    crop_size,
                    rng,
                    rng.random() < 0.5,
                    train_class_coordinates[chosen],
                )
                image, target = _augment(image, target, rng)
                inputs.append(image)
                targets.append(target)
            input_tensor = torch.from_numpy(np.stack(inputs))
            target_tensor = torch.from_numpy(np.stack(targets)).long()
            optimizer.zero_grad(set_to_none=True)
            loss = segmentation_loss(model(input_tensor), target_tensor)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            steps += 1
            total_iters_done += 1
        if not steps:
            break
        training_wall_s = time.monotonic() - epoch_started
        val_metrics = _validation_metrics(model, validation_crops, torch.device("cpu"))
        epoch_wall_s = time.monotonic() - epoch_started
        wall_s = time.monotonic() - started
        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / steps,
            "lr": epoch_lrs[-1],
            "wall_s": epoch_wall_s,
            "total_wall_s": wall_s,
            "iters_done": steps,
            "total_iters_done": total_iters_done,
            "seconds_per_iter": training_wall_s / steps,
            **val_metrics,
        }
        epochs.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        score = float(val_metrics["agreement_with_classical_v1_miou"])
        if score > best_miou:
            best_miou = score
            best_epoch = epoch
            best_epoch_pixel_agreement = float(
                val_metrics["agreement_with_classical_v1_pixel"]
            )
            stale_epochs = 0
            torch.save(model.state_dict(), output_dir / "weights.pt")
        else:
            stale_epochs += 1
        if wall_s >= max_seconds:
            stop_reason = "max_train_minutes"
            break
        if stop_reason == "max_train_minutes":
            break
        if stale_epochs >= patience:
            stop_reason = "early_stopping"
            break

    wall_s = time.monotonic() - started
    if best_epoch == 0:
        raise RuntimeError("Training ended before a validation checkpoint was saved")
    weights_path = output_dir / "weights.pt"
    weights_sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    metrics = {
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_agreement_with_classical_v1_miou": best_miou,
        "best_epoch_agreement_with_classical_v1_pixel": best_epoch_pixel_agreement,
        "stop_reason": stop_reason,
        "wall_s": wall_s,
        "note": (
            "agreement with classical_v1 pseudo-labels, NOT accuracy; "
            "classical_v1 is not ground truth"
        ),
        "weights_sha256": weights_sha,
    }
    run_config = {
        "method": "unet_pseudo_v1",
        "arch": arch,
        "seed": seed,
        "train_stems": train_stems,
        "val_stems": val_stems,
        "missing_train_stems": missing_train_stems,
        "missing_val_stems": missing_val_stems,
        "encoder_weights": training_config.get("encoder_weights"),
        "hyperparameters": training_config,
        "torch_version": torch.__version__,
        "smp_version": _package_version("segmentation-models-pytorch"),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "weights_sha256": weights_sha,
    }
    (output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (output_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as log:
        fields = sorted({key for row in epochs for key in row})
        writer = csv.DictWriter(log, fieldnames=fields)
        writer.writeheader()
        writer.writerows(epochs)
    _save_curves(output_dir / "training_curves.png", epochs)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qc.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arch", choices=("smp_resnet18", "small"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
