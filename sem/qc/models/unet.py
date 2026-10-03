"""Semantic U-Net model and tiled prediction helpers."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from scipy import ndimage as ndi
from skimage import measure
from torch import nn
from torch.nn import functional as F

from sem.qc.config import load_config
from sem.qc.schema import Instance, Prediction


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _SmallUNet(nn.Module):
    def __init__(self, in_channels: int, n_classes: int) -> None:
        super().__init__()
        self.enc0 = _ConvBlock(in_channels, 32)
        self.enc1 = _ConvBlock(32, 64)
        self.enc2 = _ConvBlock(64, 128)
        self.enc3 = _ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bridge = _ConvBlock(256, 128)
        self.dec3 = _ConvBlock(128 + 256, 128)
        self.dec2 = _ConvBlock(128 + 128, 64)
        self.dec1 = _ConvBlock(64 + 64, 32)
        self.dec0 = _ConvBlock(32 + 32, 32)
        self.classifier = nn.Conv2d(32, n_classes, kernel_size=1)

    @staticmethod
    def _up(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bridge = self.bridge(self.pool(e3))
        d3 = self.dec3(torch.cat((self._up(bridge, e3), e3), dim=1))
        d2 = self.dec2(torch.cat((self._up(d3, e2), e2), dim=1))
        d1 = self.dec1(torch.cat((self._up(d2, e1), e1), dim=1))
        d0 = self.dec0(torch.cat((self._up(d1, e0), e0), dim=1))
        return self.classifier(d0)


def build_unet(
    arch: str = "smp_resnet18",
    in_channels: int = 2,
    n_classes: int = 8,
    encoder_weights: str | None = None,
) -> nn.Module:
    """Build the configured U-Net, with a dependency-free compact fallback."""
    if arch == "small":
        return _SmallUNet(in_channels, n_classes)
    if arch == "smp_resnet18":
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                "arch='smp_resnet18' requires segmentation-models-pytorch"
            ) from exc
        return smp.Unet(
            encoder_name="resnet18",
            in_channels=in_channels,
            classes=n_classes,
            encoder_weights=encoder_weights,
        )
    raise ValueError(f"Unknown U-Net architecture: {arch}")


def prepare_input(views: dict[str, np.ndarray]) -> np.ndarray:
    """Stack fixed-scale BSE and Inlens channels as normalized float32."""
    if "BSE" not in views or "Inlens" not in views:
        raise ValueError("U-Net input requires BSE and Inlens views")
    bse = np.asarray(views["BSE"])
    inlens = np.asarray(views["Inlens"])
    if bse.ndim != 2 or inlens.shape != bse.shape:
        raise ValueError("BSE and Inlens must be co-registered HxW arrays")
    scaled = np.stack((bse, inlens)).astype(np.float32) / 255.0
    return (scaled - 0.5) / 0.25


def _tile_starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    final = length - tile
    if starts[-1] != final:
        starts.append(final)
    return starts


def sliding_window_probs(
    model: nn.Module,
    x: np.ndarray,
    tile: int = 256,
    stride: int = 192,
    batch_size: int = 16,
) -> np.ndarray:
    """Infer overlapping tiles using clipped-Hann weighted probability blending."""
    if x.ndim != 3 or not np.issubdtype(x.dtype, np.floating):
        raise ValueError("x must be a floating-point CxHxW array")
    if tile <= 0 or stride <= 0 or batch_size <= 0:
        raise ValueError("tile, stride, and batch_size must be positive")
    _, original_h, original_w = x.shape
    pad_h = max(0, tile - original_h)
    pad_w = max(0, tile - original_w)
    top, left = pad_h // 2, pad_w // 2
    pad_mode = "reflect" if min(original_h, original_w) > 1 else "edge"
    padded = np.pad(
        x,
        ((0, 0), (top, pad_h - top), (left, pad_w - left)),
        mode=pad_mode,
    )
    _, height, width = padded.shape
    y_starts = _tile_starts(height, tile, stride)
    x_starts = _tile_starts(width, tile, stride)
    locations = [(y, x0) for y in y_starts for x0 in x_starts]
    hann = np.hanning(tile).astype(np.float32)
    window = np.maximum(np.outer(hann, hann), 1e-3)
    device = next(model.parameters(), torch.empty(0)).device
    probabilities = None
    weights = np.zeros((height, width), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(locations), batch_size):
            batch_locations = locations[offset : offset + batch_size]
            tiles = np.stack(
                [padded[:, y : y + tile, x0 : x0 + tile] for y, x0 in batch_locations]
            )
            logits = model(torch.from_numpy(tiles).to(device=device, dtype=torch.float32))
            batch_probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
            if probabilities is None:
                probabilities = np.zeros(
                    (batch_probs.shape[1], height, width), dtype=np.float32
                )
            for probs, (y, x0) in zip(batch_probs, batch_locations):
                probabilities[:, y : y + tile, x0 : x0 + tile] += probs * window
                weights[y : y + tile, x0 : x0 + tile] += window
    if probabilities is None:
        raise RuntimeError("No tiles were generated")
    probabilities /= np.maximum(weights[None, ...], 1e-12)
    return probabilities[:, top : top + original_h, left : left + original_w]


def _component_polygon(component: np.ndarray, x0: int, y0: int) -> list[list[float]]:
    rows, cols = np.nonzero(component)
    if not len(rows):
        return []
    local = component[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    padded = np.pad(local.astype(np.uint8), 1)
    contours = measure.find_contours(padded, 0.5)
    if not contours:
        return []
    contour = max(contours, key=len)
    polygon = measure.approximate_polygon(contour, tolerance=1.0)
    return [
        [float(col - 1 + x0), float(row - 1 + y0)]
        for row, col in polygon
    ]


def _classical_agglomerate_instances_hook(
    bright_particle_mask: np.ndarray,
    classical_config: dict,
    source: str,
) -> list[Instance]:
    from sem.qc.classical import ClassicalV1

    particle_components, _ = ClassicalV1._large_components(
        bright_particle_mask,
        int(classical_config["bright_min_area_px"]),
    )
    instances = ClassicalV1(classical_config)._agglomerates(particle_components)
    for instance in instances:
        instance.source = source
    return instances


def instances_from_semantic(
    semantic: np.ndarray,
    uncertainty: np.ndarray,
    min_area: int,
    source: str,
) -> list[Instance]:
    semantic = np.asarray(semantic)
    uncertainty = np.asarray(uncertainty)
    if semantic.ndim != 2 or uncertainty.shape != semantic.shape:
        raise ValueError("semantic and uncertainty must be co-registered HxW arrays")
    class_names = {
        2: "bright_particle",
        3: "pore",
        4: "subsurface_uncertain",
        5: "crack_intraparticle",
        6: "interparticle_gap",
        7: "artifact",
    }
    structure = np.ones((3, 3), dtype=bool)
    instances = []
    for class_id, class_name in class_names.items():
        labels, _ = ndi.label(semantic == class_id, structure=structure)
        for label_id, region in enumerate(ndi.find_objects(labels), start=1):
            if region is None:
                continue
            component = labels[region] == label_id
            if int(np.count_nonzero(component)) < min_area:
                continue
            y_slice, x_slice = region
            y0, x0 = int(y_slice.start), int(x_slice.start)
            y1, x1 = int(y_slice.stop), int(x_slice.stop)
            uncertainty_region = uncertainty[region]
            instances.append(
                Instance(
                    class_name=class_name,
                    subtype="other" if class_id == 7 else None,
                    bbox=[x0, y0, x1, y1],
                    polygon=_component_polygon(component, x0, y0),
                    score=float(np.mean(1.0 - uncertainty_region[component])),
                    source=source,
                )
            )
    return instances


class UNetPseudoV1:
    name = "unet_pseudo_v1"

    def __init__(
        self,
        weights_path: str | Path,
        arch: str = "small",
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = build_unet(arch=arch).to(self.device)
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint) if isinstance(
            checkpoint, dict
        ) else checkpoint
        self.model.load_state_dict(state)
        self.model.eval()
        config = load_config()
        self.config = config.get("unet_pseudo_v1", {})
        self.classical_config = config.get("classical_v1", {})

    @classmethod
    def from_run_dir(cls, path: str | Path, device: str = "cpu") -> "UNetPseudoV1":
        import json

        run_dir = Path(path)
        with (run_dir / "config.json").open(encoding="utf-8") as config_file:
            run_config = json.load(config_file)
        return cls(
            run_dir / "weights.pt",
            arch=run_config.get("arch", "small"),
            device=device,
        )

    def predict(self, views: dict[str, np.ndarray], valid: np.ndarray) -> Prediction:
        x = prepare_input(views)
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != x.shape[1:]:
            raise ValueError("valid mask must have the same HxW shape as the views")
        probs = sliding_window_probs(self.model, x)
        semantic = np.argmax(probs, axis=0).astype(np.uint8)
        uncertainty = np.clip(1.0 - np.max(probs, axis=0), 0.0, 1.0).astype(
            np.float32
        )
        semantic[~valid] = 255
        uncertainty[~valid] = 0.0

        minimum_area = int(self.config.get("instance_min_area_px", 16))
        instances = instances_from_semantic(
            semantic, uncertainty, minimum_area, self.name
        )
        instances.extend(
            _classical_agglomerate_instances_hook(
                semantic == 2, self.classical_config, self.name
            )
        )
        return Prediction(semantic=semantic, instances=instances, uncertainty=uncertainty)
