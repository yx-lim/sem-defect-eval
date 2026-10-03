import math

import numpy as np
import pytest

from sem.qc.kpi import compute_kpis
from sem.qc.schema import Instance


def test_known_disc_line_void_fraction_and_artifact_exclusion():
    semantic = np.zeros((16, 16), dtype=np.uint8)
    semantic[2:6, 2:6] = 2
    semantic[2, 9:13] = 3
    semantic[4, 9:11] = 4
    semantic[6, 9:12] = 6
    semantic[9, 2:8] = 5
    semantic[11, 2:9] = 6
    semantic[0, 0:2] = 7
    semantic[15, 0:3] = 255
    valid = np.ones(semantic.shape, dtype=bool)

    values = compute_kpis(semantic, [], valid, 25.0)
    area = 16 * 16 - 2 - 3
    expected_ecd = 2 * math.sqrt(16 / math.pi) * 0.025
    crack_skeleton_length = 6
    pixel_um = 0.025
    area_mm2 = area * pixel_um**2 / 1_000_000

    assert values["valid_area_px"] == area
    assert values["void_fraction"] == pytest.approx((4 + 3 + 7) / area)
    assert values["void_fraction_incl_uncertain"] == pytest.approx((4 + 2 + 3 + 7) / area)
    assert values["bright_particle_ecd_count"] == 1
    assert values["bright_particle_ecd_median_um"] == pytest.approx(expected_ecd)
    assert values["crack_density_um_per_mm2"] == pytest.approx(
        crack_skeleton_length * pixel_um / area_mm2
    )
    assert values["area_frac_artifact"] == pytest.approx(2 / area)


def test_agglomerate_kpis_use_polygon_area_and_valid_centroid():
    semantic = np.zeros((20, 20), dtype=np.uint8)
    valid = np.ones(semantic.shape, dtype=bool)
    instance = Instance(
        class_name="agglomerate",
        bbox=[2, 2, 7, 7],
        polygon=[[2, 2], [7, 2], [7, 7], [2, 7]],
    )

    values = compute_kpis(semantic, [instance], valid, 25.0)

    assert values["agglomerate_per_mm2"] == pytest.approx(1 / (400 * 0.025**2 / 1e6))
    assert values["agglomerate_area_frac"] == pytest.approx(36 / 400)
