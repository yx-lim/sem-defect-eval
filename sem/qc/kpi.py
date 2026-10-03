"""Single-source QC measurements for image tiles and stems."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import polygon as polygon_pixels
from skimage.measure import label
from skimage.morphology import skeletonize

from sem.qc.schema import CLASS_IDS, CLASS_NAMES, IGNORE_LABEL, Instance


def _polygon_mask(points: list[list[float]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if len(points) < 3:
        return mask
    xy = np.asarray(points, dtype=float)
    rows, cols = polygon_pixels(xy[:, 1], xy[:, 0], shape=shape)
    mask[rows, cols] = True
    return mask


def compute_kpis(
    semantic: np.ndarray,
    instances: list[Instance],
    valid_mask: np.ndarray,
    pixel_size_nm: float,
) -> dict[str, Any]:
    """Compute QC metrics with the definitions in the shared evaluation spec."""
    if semantic.ndim != 2 or semantic.dtype != np.uint8:
        raise ValueError("semantic must be a 2D uint8 label map")
    if valid_mask.shape != semantic.shape:
        raise ValueError("valid_mask must match the semantic map shape")
    if pixel_size_nm <= 0:
        raise ValueError("pixel_size_nm must be positive")

    usable = valid_mask & (semantic != IGNORE_LABEL)
    area_mask = usable & (semantic != CLASS_IDS["artifact"])
    area_px = int(np.count_nonzero(area_mask))
    if area_px == 0:
        raise ValueError("QC metrics require positive valid non-artifact area")
    pixel_size_um = pixel_size_nm / 1000.0
    area_mm2 = area_px * pixel_size_um**2 / 1_000_000.0

    class_counts = {
        class_id: int(np.count_nonzero(usable & (semantic == class_id)))
        for class_id in range(8)
    }
    pore = class_counts[CLASS_IDS["pore"]]
    gap = class_counts[CLASS_IDS["interparticle_gap"]]
    uncertain = class_counts[CLASS_IDS["subsurface_uncertain"]]

    particle_labels, particle_count = ndi.label(
        usable & (semantic == CLASS_IDS["bright_particle"]),
        structure=np.ones((3, 3), dtype=bool),
    )
    diameters_um = []
    invalid_pixels = semantic == IGNORE_LABEL
    for component_id, region in enumerate(
        ndi.find_objects(particle_labels), start=1
    ):
        if region is None:
            continue
        component = particle_labels[region] == component_id
        component_area = int(np.count_nonzero(component))
        if component_area < 16:
            continue
        y_slice, x_slice = region
        y0, y1 = y_slice.start, y_slice.stop
        x0, x1 = x_slice.start, x_slice.stop
        if y0 == 0 or x0 == 0 or y1 == semantic.shape[0] or x1 == semantic.shape[1]:
            continue
        expanded_region = (
            slice(max(0, y0 - 1), min(semantic.shape[0], y1 + 1)),
            slice(max(0, x0 - 1), min(semantic.shape[1], x1 + 1)),
        )
        component_in_region = np.zeros(
            (
                expanded_region[0].stop - expanded_region[0].start,
                expanded_region[1].stop - expanded_region[1].start,
            ),
            dtype=bool,
        )
        component_in_region[
            y0 - expanded_region[0].start : y1 - expanded_region[0].start,
            x0 - expanded_region[1].start : x1 - expanded_region[1].start,
        ] = component
        expanded = ndi.binary_dilation(
            component_in_region, structure=np.ones((3, 3), bool)
        )
        if np.any(expanded & invalid_pixels[expanded_region]):
            continue
        diameters_um.append(2.0 * math.sqrt(component_area / math.pi) * pixel_size_um)
    diameters_um_array = np.asarray(diameters_um, dtype=float)

    crack_skeleton_px = int(
        np.count_nonzero(
            skeletonize(usable & (semantic == CLASS_IDS["crack_intraparticle"]))
        )
    )
    gap_skeleton_px = int(
        np.count_nonzero(
            skeletonize(usable & (semantic == CLASS_IDS["interparticle_gap"]))
        )
    )

    agglomerate_count = 0
    agglomerate_area_px = 0
    for instance in instances:
        if instance.class_name != "agglomerate":
            continue
        polygon_mask = _polygon_mask(instance.polygon, semantic.shape)
        ys, xs = np.nonzero(polygon_mask)
        if len(xs) == 0:
            continue
        centroid_x = int(round(float(xs.mean())))
        centroid_y = int(round(float(ys.mean())))
        if (
            0 <= centroid_y < valid_mask.shape[0]
            and 0 <= centroid_x < valid_mask.shape[1]
            and valid_mask[centroid_y, centroid_x]
        ):
            agglomerate_count += 1
        agglomerate_area_px += int(np.count_nonzero(polygon_mask & valid_mask))

    results: dict[str, Any] = {
        "valid_area_px": area_px,
        "void_fraction": (pore + gap) / area_px,
        "void_fraction_incl_uncertain": (pore + uncertain + gap) / area_px,
        "bright_particle_ecd_count": int(diameters_um_array.size),
        "bright_particle_ecd_median_um": (
            float(np.median(diameters_um_array)) if diameters_um_array.size else None
        ),
        "bright_particle_ecd_p10_um": (
            float(np.percentile(diameters_um_array, 10))
            if diameters_um_array.size
            else None
        ),
        "bright_particle_ecd_p90_um": (
            float(np.percentile(diameters_um_array, 90))
            if diameters_um_array.size
            else None
        ),
        "graphite_particle_area_frac": class_counts[1] / area_px,
        "crack_density_um_per_mm2": (
            crack_skeleton_px * pixel_size_um / area_mm2
        ),
        "gap_density_um_per_mm2": gap_skeleton_px * pixel_size_um / area_mm2,
        "agglomerate_per_mm2": agglomerate_count / area_mm2,
        "agglomerate_area_frac": agglomerate_area_px / area_px,
    }
    for class_id, class_name in CLASS_NAMES.items():
        results[f"area_frac_{class_name}"] = class_counts[class_id] / area_px
    return results
