from pathlib import Path

import numpy as np
import pytest
import tifffile

from sem.qc.io import StemRecord, load_stem, parse_pixel_size_nm, valid_mask


def _write_rgb(path: Path, image: np.ndarray) -> None:
    tifffile.imwrite(
        path,
        image,
        photometric="rgb",
        resolution=(1_016_000, 1_016_000),
        resolutionunit="INCH",
    )


def test_parse_pixel_size_from_tiff(tmp_path):
    path = tmp_path / "tiny.tif"
    _write_rgb(path, np.zeros((24, 30, 3), dtype=np.uint8))

    assert parse_pixel_size_nm(path) == pytest.approx(25.0)


def test_parse_pixel_size_asserts_configured_bounds(tmp_path):
    path = tmp_path / "wrong_scale.tif"
    tifffile.imwrite(
        path,
        np.zeros((5, 6, 3), dtype=np.uint8),
        photometric="rgb",
        resolution=(1_000_000, 1_000_000),
        resolutionunit="INCH",
    )

    with pytest.raises(AssertionError, match="outside"):
        parse_pixel_size_nm(path)


def test_load_stem_channel_zero_and_valid_mask_color_dilation(tmp_path):
    image = np.zeros((24, 30, 3), dtype=np.uint8)
    image[12, 15] = [100, 101, 100]
    path = tmp_path / "tiny.tif"
    _write_rgb(path, image)
    record = StemRecord(
        stem="tiny",
        batch="Batch_1",
        detector_set="ETD",
        views={"BSE": path, "Inlens": path},
        height=24,
        width=30,
        pixel_size_nm=25.0,
    )

    loaded = load_stem(record)
    mask = valid_mask(record)
    assert loaded["BSE"].dtype == np.uint8
    assert loaded["BSE"][12, 15] == 100
    assert not mask[12, 15]
    assert not mask[12, 17]
    assert mask[12, 18]
    assert not mask[7, 20]
    assert mask[8, 20]
