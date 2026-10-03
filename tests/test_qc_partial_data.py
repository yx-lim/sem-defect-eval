import numpy as np
import tifffile

from sem.qc.partial_data import list_complete_stems


def _write_view(path):
    tifffile.imwrite(
        path,
        np.full((8, 10), 50, dtype=np.uint8),
        resolution=(1_016_000, 1_016_000),
        resolutionunit="INCH",
    )


def test_list_complete_stems_filters_missing_and_ambiguous_detectors(tmp_path):
    data_root = tmp_path / "data"
    batch = data_root / "Batch_1"
    batch.mkdir(parents=True)
    for view in ("BSE", "Inlens", "SE"):
        _write_view(batch / f"img_complete_{view}.tif")
    for view in ("BSE", "Inlens"):
        _write_view(batch / f"img_incomplete_{view}.tif")
    for view in ("BSE", "Inlens", "ETD", "SE"):
        _write_view(batch / f"img_ambiguous_{view}.tif")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "stem,batch\ncomplete,Batch_1\nincomplete,Batch_1\nambiguous,Batch_1\n",
        encoding="utf-8",
    )

    records, missing = list_complete_stems(data_root, tmp_path / "work", manifest)

    assert [record.stem for record in records] == ["complete"]
    assert missing == ["ambiguous", "incomplete"]
    assert (
        tmp_path
        / "work"
        / "data_complete"
        / "Batch_1"
        / "img_complete_BSE.tif"
    ).is_symlink()
