import numpy as np

from sem.qc.adapters.pipeline_v0 import (
    CLASS_MAP,
    CLASSES,
    EXTRA_LABELS,
    SOURCES,
    SOURCE_MAP,
    convert_proposals,
    decode_mask_rle,
)


def _encode_counts(mask):
    flat = np.asarray(mask, dtype=bool).ravel(order="F")
    counts = []
    value = False
    run = 0
    for pixel in flat:
        if bool(pixel) == value:
            run += 1
        else:
            counts.append(run)
            run = 1
            value = bool(pixel)
    counts.append(run)
    return counts


def _encode_compressed(counts):
    encoded = []
    for index, count in enumerate(counts):
        value = count - counts[index - 2] if index > 2 else count
        more = True
        while more:
            code = value & 0x1F
            value >>= 5
            more = (code & 0x10 and value != -1) or (
                not (code & 0x10) and value != 0
            )
            if more:
                code |= 0x20
            encoded.append(chr(code + 48))
    return "".join(encoded)


def test_all_external_labels_and_sources_have_explicit_mappings():
    assert set(CLASS_MAP) == set(CLASSES) | set(EXTRA_LABELS)
    assert set(SOURCE_MAP) == SOURCES
    for source in SOURCES:
        assert len(SOURCE_MAP[source]) == 2


def test_rle_decoder_supports_list_and_compressed_counts():
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 2:4] = True
    counts = _encode_counts(mask)
    decoded = decode_mask_rle({"size": list(mask.shape), "counts": counts})
    np.testing.assert_array_equal(decoded, mask)
    compressed = decode_mask_rle(
        {"size": list(mask.shape), "counts": _encode_compressed(counts)}
    )
    np.testing.assert_array_equal(compressed, mask)


def test_proposals_create_mapped_instances_and_mask_semantics():
    mask = np.zeros((3, 4), dtype=bool)
    mask[1, 2] = True
    proposals = [
        {
            "proposal_id": "one",
            "group_id": "sample",
            "source": "dark_void",
            "score": 8.0,
            "bbox": [1, 1, 5, 4],
            "mask_rle": {"size": list(mask.shape), "counts": _encode_counts(mask)},
        },
        {
            "proposal_id": "two",
            "group_id": "sample",
            "source": "tophat_crack",
            "score": 2.0,
            "bbox": [0, 0, 2, 2],
            "mask_rle": None,
        },
        {
            "proposal_id": "three",
            "group_id": "sample",
            "source": "random",
            "score": 0.0,
            "bbox": [3, 2, 5, 4],
            "mask_rle": None,
        },
    ]
    predictions = convert_proposals(proposals, {"sample": (8, 8)})
    prediction = predictions["sample"]
    assert prediction.semantic[2, 3] == 3
    assert np.count_nonzero(prediction.semantic != 255) == 1
    assert prediction.uncertainty is None
    assert [item.class_name for item in prediction.instances] == [
        "pore",
        "crack_intraparticle",
        "unmapped",
    ]
    assert prediction.instances[0].bbox == [1, 1, 5, 4]
    assert prediction.instances[2].source == "pipeline_v0:random"
    assert all(0.0 <= item.score <= 1.0 for item in prediction.instances)


def test_unmapped_sources_keep_proposal_instances_and_blank_semantics():
    proposal = {
        "proposal_id": "candidate",
        "group_id": "unknown",
        "source": "anomaly_peak",
        "score": 4.5,
        "bbox": [0, 0, 2, 2],
        "mask_rle": None,
    }
    result = convert_proposals([proposal], {"unknown": (4, 4)})["unknown"]
    assert result.instances[0].class_name == "unmapped"
    assert result.instances[0].source == "pipeline_v0:anomaly_peak"
    assert np.all(result.semantic == 255)


def test_semantic_pixels_respect_supplied_valid_mask():
    mask = np.ones((2, 2), dtype=bool)
    valid = np.array(
        [[True, False, True], [True, True, True], [True, True, True]]
    )
    proposal = {
        "group_id": "stem",
        "source": "dark_void",
        "score": 1.0,
        "bbox": [0, 0, 2, 2],
        "mask_rle": {"size": [2, 2], "counts": _encode_counts(mask)},
    }
    result = convert_proposals([proposal], {"stem": (3, 3)}, {"stem": valid})[
        "stem"
    ]
    assert result.semantic[0, 0] == 3
    assert result.semantic[0, 1] == 255
