import numpy as np
from skimage.draw import disk, line

from sem.qc.classical import ClassicalV1
from sem.qc.schema import CLASS_IDS


def _image():
    rng = np.random.default_rng(23)
    image = np.clip(rng.normal(200, 12, (256, 256)), 0, 255).astype(np.uint8)
    image[20:44, 20:44] = 20
    rr, cc = disk((220, 220), 9, shape=image.shape)
    image[rr, cc] = 245
    return image


def _predict(image):
    valid = np.ones(image.shape, dtype=bool)
    return ClassicalV1().predict({"BSE": image}, valid)


def test_bright_disc_is_a_bright_particle():
    image = _image()
    rr, cc = disk((100, 100), 18, shape=image.shape)
    image[rr, cc] = 245

    result = _predict(image)

    assert np.count_nonzero(result.semantic[rr, cc] == CLASS_IDS["bright_particle"]) > 500
    assert result.uncertainty.dtype == np.float32


def test_dark_smooth_hole_is_a_pore():
    image = _image()
    rr, cc = disk((120, 120), 18, shape=image.shape)
    image[rr, cc] = 5

    result = _predict(image)

    assert result.semantic[120, 120] == CLASS_IDS["pore"]


def test_thin_dark_line_inside_graphite_is_a_crack():
    image = _image()
    image[80:176, 70:186] = 150
    rr, cc = line(92, 128, 164, 128)
    image[rr, cc] = 5

    result = _predict(image)

    assert result.semantic[128, 128] == CLASS_IDS["crack_intraparticle"]


def test_three_nearby_bright_particles_form_one_agglomerate():
    image = _image()
    for center in ((80, 80), (80, 99), (98, 89)):
        rr, cc = disk(center, 7, shape=image.shape)
        image[rr, cc] = 245

    result = _predict(image)

    agglomerates = [item for item in result.instances if item.class_name == "agglomerate"]
    assert len(agglomerates) == 1
    assert len(agglomerates[0].polygon) >= 3
