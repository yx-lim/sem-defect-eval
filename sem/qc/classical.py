"""Deterministic CPU-only classical_v1 baseline."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, QhullError
from skimage.filters import threshold_multiotsu
from skimage.morphology import black_tophat, disk, skeletonize

from sem.qc.config import load_config
from sem.qc.schema import CLASS_IDS, IGNORE_LABEL, Instance, Prediction


class ClassicalV1:
    """Four-class multi-Otsu baseline with geometric defect post-processing.

    Uncertainty is ``max(clip(1 - d_otsu / margin, 0, 1),
    clip(1 - d_local_std / margin, 0, 1))``. ``d_otsu`` is the distance to
    the closest multi-Otsu threshold. ``d_local_std`` is the distance to the
    pore split in darkest pixels or the graphite split in middle pixels;
    pixels outside those local-standard-deviation decisions have zero split
    uncertainty.
    """

    name = "classical_v1"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_config()["classical_v1"]

    @staticmethod
    def _large_components(
        mask: np.ndarray, minimum_area: int
    ) -> tuple[np.ndarray, np.ndarray]:
        components, count = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
        sizes = np.bincount(components.ravel(), minlength=count + 1)
        components[sizes[components] < minimum_area] = 0
        return components, components > 0

    @staticmethod
    def _aspect_ratio(component: np.ndarray) -> float:
        rows, cols = np.nonzero(component)
        if len(rows) < 2:
            return 1.0
        stride = max(1, int(math.ceil(len(rows) / 100_000)))
        coordinates = np.column_stack((rows[::stride], cols[::stride])).astype(
            np.float32
        )
        eigenvalues = np.linalg.eigvalsh(np.cov(coordinates, rowvar=False))
        minor = max(float(eigenvalues[0]), 0.0)
        major = max(float(eigenvalues[1]), 0.0)
        return math.inf if minor == 0 else math.sqrt(major / minor)

    def _mark_thin_defects(
        self, gray: np.ndarray, valid: np.ndarray, semantic: np.ndarray
    ) -> None:
        radius = int(self.config["black_tophat_radius_px"])
        response = black_tophat(gray, footprint=disk(radius))
        candidates = valid & (
            response > float(self.config["black_tophat_threshold"])
        )
        components, count = ndi.label(
            candidates, structure=np.ones((3, 3), dtype=bool)
        )
        particles = (semantic == 1) | (semantic == 2)
        particle_components, _ = ndi.label(
            particles, structure=np.ones((3, 3), dtype=bool)
        )
        sample_distance = int(self.config["crack_enclosure_radius_px"]) + 1
        minimum_length = int(self.config["thin_component_min_skeleton_px"])
        minimum_aspect = float(self.config["thin_component_min_aspect_ratio"])

        for component_id, region in enumerate(ndi.find_objects(components), start=1):
            if region is None:
                continue
            component = components[region] == component_id
            if int(np.count_nonzero(skeletonize(component))) < minimum_length:
                continue
            if self._aspect_ratio(component) < minimum_aspect:
                continue

            y0, x0 = region[0].start, region[1].start
            y1, x1 = region[0].stop, region[1].stop
            particle_fraction = float(
                np.count_nonzero(component & particles[region])
            ) / max(1, int(np.count_nonzero(component)))
            rows, cols = np.nonzero(component)
            stride = max(1, int(math.ceil(len(rows) / 100_000)))
            coordinates = np.column_stack((rows[::stride], cols[::stride])).astype(
                np.float32
            )
            covariance = np.cov(coordinates, rowvar=False)
            _, eigenvectors = np.linalg.eigh(covariance)
            normal = eigenvectors[:, 0]
            delta = np.rint(normal * sample_distance).astype(int)
            if not np.any(delta):
                delta[np.argmin(np.abs(normal))] = 1
            global_rows = rows[::stride] + y0
            global_cols = cols[::stride] + x0
            side_ids = []
            side_classes = []
            for sign in (-1, 1):
                sample_rows = global_rows + sign * delta[0]
                sample_cols = global_cols + sign * delta[1]
                in_bounds = (
                    (sample_rows >= 0)
                    & (sample_rows < semantic.shape[0])
                    & (sample_cols >= 0)
                    & (sample_cols < semantic.shape[1])
                )
                ids = particle_components[
                    sample_rows[in_bounds], sample_cols[in_bounds]
                ]
                ids = ids[ids > 0]
                if len(ids) / max(1, len(global_rows)) >= 0.5 and ids.size:
                    side_ids.append(int(np.bincount(ids).argmax()))
                else:
                    side_ids.append(0)
                values = semantic[sample_rows[in_bounds], sample_cols[in_bounds]]
                values = values[values != IGNORE_LABEL]
                side_classes.append(
                    int(np.bincount(values).argmax()) if values.size else IGNORE_LABEL
                )
            if particle_fraction >= 0.5 or (
                side_ids[0] > 0 and side_ids[0] == side_ids[1]
            ):
                class_id = CLASS_IDS["crack_intraparticle"]
            elif (
                side_ids[0] > 0
                or side_ids[1] > 0
                or (
                    side_classes[0] in (0, 1, 2)
                    and side_classes[1] in (3, 4)
                )
                or (
                    side_classes[1] in (0, 1, 2)
                    and side_classes[0] in (3, 4)
                )
            ):
                class_id = CLASS_IDS["interparticle_gap"]
            else:
                continue
            target = semantic[region]
            target[component] = class_id
            semantic[region] = target

    @staticmethod
    def _convex_polygon(mask: np.ndarray, x0: int, y0: int) -> list[list[float]]:
        boundary = mask & ~ndi.binary_erosion(mask)
        rows, cols = np.nonzero(boundary)
        if len(rows) > 100_000:
            step = int(math.ceil(len(rows) / 100_000))
            rows, cols = rows[::step], cols[::step]
        points = np.column_stack((cols, rows))
        if len(points) >= 3:
            try:
                points = points[ConvexHull(points).vertices]
            except QhullError:
                pass
        return [[float(x + x0), float(y + y0)] for x, y in points]

    def _agglomerates(self, particle_components: np.ndarray) -> list[Instance]:
        particle_mask = particle_components > 0
        if not particle_mask.any():
            return []
        link_px = int(self.config["agglomerate_link_distance_px"])
        dilate_radius = int(math.ceil(link_px / 2))
        expanded = ndi.binary_dilation(
            particle_mask, structure=disk(dilate_radius)
        )
        groups, group_count = ndi.label(expanded, structure=np.ones((3, 3), bool))
        regions = ndi.find_objects(groups)
        minimum_particles = int(self.config["agglomerate_min_particles"])
        instances = []
        for group_id, region in enumerate(regions, start=1):
            if region is None:
                continue
            group = groups[region] == group_id
            components = np.unique(particle_components[region][group])
            components = components[components != 0]
            if len(components) < minimum_particles:
                continue
            member_mask = np.isin(particle_components[region], components)
            rows, cols = np.nonzero(member_mask)
            y0, x0 = region[0].start, region[1].start
            y1, x1 = region[0].stop, region[1].stop
            polygon = self._convex_polygon(member_mask, x0, y0)
            instances.append(
                Instance(
                    class_name="agglomerate",
                    bbox=[float(x0), float(y0), float(x1), float(y1)],
                    polygon=polygon,
                    score=1.0,
                    source=self.name,
                )
            )
        return instances

    def _scan_streaks(
        self, gray: np.ndarray, valid: np.ndarray, semantic: np.ndarray
    ) -> list[Instance]:
        highpass_sigma = float(self.config["scan_streak_highpass_sigma"])
        highpass = gray - ndi.gaussian_filter(gray, sigma=highpass_sigma)
        row_counts = valid.sum(axis=1)
        row_sums = np.where(valid, highpass, 0.0).sum(axis=1)
        row_means = np.divide(
            row_sums,
            row_counts,
            out=np.zeros_like(row_sums, dtype=np.float32),
            where=row_counts > 0,
        )
        usable_means = row_means[row_counts > 0]
        if usable_means.size == 0:
            return []
        center = float(np.median(usable_means))
        mad = float(np.median(np.abs(usable_means - center)))
        scale = 1.4826 * mad
        if scale == 0:
            scale = float(np.std(usable_means))
        if scale == 0:
            return []
        threshold = float(self.config["scan_streak_z_threshold"])
        outliers = (row_counts > 0) & (
            np.abs(row_means - center) > threshold * scale
        )
        if not outliers.any():
            return []
        semantic[outliers, :] = np.where(
            valid[outliers, :], CLASS_IDS["artifact"], semantic[outliers, :]
        )
        runs, run_count = ndi.label(outliers)
        instances = []
        for region in ndi.find_objects(runs):
            if region is None:
                continue
            y0, y1 = region[0].start, region[0].stop
            x1 = semantic.shape[1]
            instances.append(
                Instance(
                    class_name="artifact",
                    subtype="scan_streak",
                    bbox=[0.0, float(y0), float(x1), float(y1)],
                    polygon=[
                        [0.0, float(y0)],
                        [float(x1), float(y0)],
                        [float(x1), float(y1)],
                        [0.0, float(y1)],
                    ],
                    score=1.0,
                    source=self.name,
                )
            )
        return instances

    def predict(
        self, views: dict[str, np.ndarray], valid: np.ndarray
    ) -> Prediction:
        if "BSE" not in views:
            raise ValueError("classical_v1 requires the BSE view")
        bse = views["BSE"]
        if bse.dtype != np.uint8 or bse.ndim != 2:
            raise ValueError("BSE must be a 2D uint8 array")
        if valid.shape != bse.shape:
            raise ValueError("valid mask must match BSE shape")
        valid = valid.astype(bool, copy=False)
        if not valid.any():
            raise ValueError("BSE has no valid pixels")

        sigma = float(self.config["gaussian_sigma"])
        gray = ndi.gaussian_filter(bse.astype(np.float32), sigma=sigma)
        histogram, _ = np.histogram(gray[valid], bins=256, range=(0, 256))
        thresholds = threshold_multiotsu(
            classes=int(self.config["multi_otsu_classes"]),
            hist=(histogram, np.arange(256, dtype=float)),
        )
        bins = np.digitize(gray, thresholds)
        local_window = int(self.config["local_std_window_px"])
        local_mean = ndi.uniform_filter(gray, size=local_window)
        local_mean_square = ndi.uniform_filter(gray * gray, size=local_window)
        local_std = np.sqrt(np.maximum(local_mean_square - local_mean**2, 0.0))

        semantic = np.full(bse.shape, IGNORE_LABEL, dtype=np.uint8)
        semantic[valid] = CLASS_IDS["matrix_other"]
        darkest = valid & (bins == 0)
        pore_std = float(self.config["pore_local_std_max"])
        semantic[darkest & (local_std <= pore_std)] = CLASS_IDS["pore"]
        semantic[darkest & (local_std > pore_std)] = CLASS_IDS[
            "subsurface_uncertain"
        ]
        middle = valid & ((bins == 1) | (bins == 2))
        graphite_std = float(self.config["graphite_local_std_max"])
        semantic[middle & (local_std <= graphite_std)] = CLASS_IDS[
            "graphite_particle"
        ]

        brightest = valid & (bins == int(self.config["multi_otsu_classes"]) - 1)
        opening_radius = int(self.config["bright_opening_radius_px"])
        if opening_radius:
            brightest = ndi.binary_opening(
                brightest, structure=disk(opening_radius)
            )
        particle_components, bright_mask = self._large_components(
            brightest, int(self.config["bright_min_area_px"])
        )
        semantic[bright_mask] = CLASS_IDS["bright_particle"]

        self._mark_thin_defects(gray, valid, semantic)
        instances = self._agglomerates(particle_components)
        instances.extend(self._scan_streaks(gray, valid, semantic))

        otsu_distance = np.full(bse.shape, np.inf, dtype=np.float32)
        for threshold in thresholds:
            np.minimum(
                otsu_distance,
                np.abs(gray - float(threshold)),
                out=otsu_distance,
            )
        otsu_margin = float(self.config["uncertainty_otsu_margin"])
        uncertainty = np.clip(1.0 - otsu_distance / otsu_margin, 0.0, 1.0)
        local_margin = np.full(bse.shape, np.inf, dtype=np.float32)
        local_margin[darkest] = np.abs(local_std[darkest] - pore_std)
        graphite_split = middle
        local_margin[graphite_split] = np.abs(
            local_std[graphite_split] - graphite_std
        )
        split_margin = float(self.config["uncertainty_local_std_margin"])
        split_uncertainty = np.clip(1.0 - local_margin / split_margin, 0.0, 1.0)
        uncertainty = np.maximum(uncertainty, split_uncertainty).astype(np.float32)
        uncertainty[~valid] = 0.0
        return Prediction(
            semantic=semantic,
            instances=instances,
            uncertainty=uncertainty,
        )
