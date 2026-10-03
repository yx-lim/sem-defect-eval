import numpy as np

from scripts.stem_similarity import (
    _overlap_slices,
    _phase_correlation_shift,
    _refined_ncc,
)


def test_zero_padded_phase_correlation_and_overlap_refinement():
    rng = np.random.default_rng(42)
    reference = rng.normal(size=(80, 90)).astype(np.float32)
    moving = np.zeros_like(reference)
    moving[4:, :-6] = reference[:-4, 6:]
    valid = np.ones(reference.shape, dtype=bool)

    shift = _phase_correlation_shift(reference, moving, valid, valid)
    ncc, overlap = _refined_ncc(reference, moving, valid, valid, shift)

    assert shift == (-4, 6)
    assert ncc > 0.99
    assert 0 < overlap < 1


def test_overlap_slices_apply_shift_to_moving_image():
    ref, moving = _overlap_slices((8, 10), (7, 11), (2, -3))

    assert ref == (slice(2, 8), slice(0, 8))
    assert moving == (slice(0, 6), slice(3, 11))
