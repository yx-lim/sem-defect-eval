import time

import numpy as np
import pytest
import torch
from torch import nn

from sem.qc.models.unet import (
    UNetPseudoV1,
    _classical_agglomerate_instances_hook,
    build_unet,
    instances_from_semantic,
    prepare_input,
    sliding_window_probs,
)
from scripts.train_unet import (
    _budget_aware_learning_rate,
    _ignored_fraction_breakdown,
    _precompute_class_coordinates,
    _sample_crop,
    segmentation_loss,
)


def test_small_unet_forward_and_parameter_budget():
    model = build_unet("small")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert 1_500_000 <= parameters <= 2_500_000
    assert model(torch.zeros(1, 2, 64, 64)).shape == (1, 8, 64, 64)


def test_smp_unet_forward_when_dependency_is_installed():
    pytest.importorskip("segmentation_models_pytorch")
    model = build_unet("smp_resnet18", encoder_weights=None)
    model.eval()
    with torch.no_grad():
        assert model(torch.zeros(1, 2, 64, 64)).shape == (1, 8, 64, 64)


def test_smp_unet_missing_dependency_has_clear_error(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def import_without_smp(name, *args, **kwargs):
        if name == "segmentation_models_pytorch":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_smp)
    with pytest.raises(ImportError, match="requires segmentation-models-pytorch"):
        build_unet("smp_resnet18")


class _Pointwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 8, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


@pytest.mark.parametrize("shape", [(300, 517), (100, 90)])
def test_sliding_window_matches_full_pointwise_inference(shape):
    torch.manual_seed(4)
    model = _Pointwise()
    x = np.random.default_rng(4).normal(size=(2, *shape)).astype(np.float32)
    actual = sliding_window_probs(model, x, tile=256, stride=192, batch_size=3)
    with torch.no_grad():
        expected = torch.softmax(
            model(torch.from_numpy(x[None])), dim=1
        )[0].numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_prepare_input_uses_fixed_scale():
    image = np.array([[0, 127, 255]], dtype=np.uint8)
    result = prepare_input({"BSE": image, "Inlens": image})
    np.testing.assert_allclose(
        result[0, 0], (image[0].astype(np.float32) / 255.0 - 0.5) / 0.25
    )
    assert result.dtype == np.float32


def test_loss_ignores_255_pixels_and_includes_dice():
    torch.manual_seed(3)
    logits = torch.randn(1, 8, 8, 8, requires_grad=True)
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, 2:6, 2:6] = 2
    target[:, :2] = 255
    loss = segmentation_loss(logits, target)
    changed = logits.detach().clone()
    changed[:, :, :2] = torch.randn_like(changed[:, :, :2]) * 100
    assert torch.allclose(loss.detach(), segmentation_loss(changed, target))
    loss.backward()
    assert logits.grad is not None


def test_tiny_model_overfits_one_batch():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(8)
        inputs = torch.randn(2, 2, 64, 64)
        targets = (inputs[:, 0] > 0).long()
        targets[:, 0, :8] = 255
        model = nn.Sequential(nn.Conv2d(2, 8, kernel_size=1))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        initial = float(segmentation_loss(model(inputs), targets).detach())
        final = initial
        for _ in range(60):
            optimizer.zero_grad()
            final_loss = segmentation_loss(model(inputs), targets)
            final_loss.backward()
            optimizer.step()
            final = float(segmentation_loss(model(inputs), targets).detach())
            if final <= initial * 0.5:
                break
        assert final <= initial * 0.5
    finally:
        torch.set_num_threads(previous_threads)


class _BrightnessModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        logits = torch.zeros((x.shape[0], 8, *x.shape[-2:]), device=x.device)
        logits[:, 2] = x[:, 0] * 100.0
        return logits


def test_predict_masks_invalid_pixels_and_emits_components(tmp_path):
    base = build_unet("small")
    weights = tmp_path / "weights.pt"
    torch.save(base.state_dict(), weights)
    method = UNetPseudoV1(weights, arch="small")
    method.model = _BrightnessModel()
    bse = np.zeros((64, 64), dtype=np.uint8)
    bse[20:30, 20:30] = 255
    valid = np.ones((64, 64), dtype=bool)
    valid[22, 22] = False
    prediction = method.predict(
        {"BSE": bse, "Inlens": np.zeros_like(bse)},
        valid,
    )
    assert prediction.semantic[22, 22] == 255
    assert prediction.uncertainty[22, 22] == 0.0
    assert np.all((prediction.uncertainty >= 0) & (prediction.uncertainty <= 1))
    bright = [item for item in prediction.instances if item.class_name == "bright_particle"]
    assert len(bright) == 1
    assert bright[0].bbox == [20, 20, 30, 30]
    assert all(19.0 <= point[0] <= 30.0 for point in bright[0].polygon)
    assert all(19.0 <= point[1] <= 30.0 for point in bright[0].polygon)


def test_agglomerate_instances_use_classical_rule():
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:15, 10:15] = True
    mask[10:15, 22:27] = True
    mask[10:15, 34:39] = True
    instances = _classical_agglomerate_instances_hook(
        mask,
        {
            "bright_min_area_px": 16,
            "agglomerate_link_distance_px": 10,
            "agglomerate_min_particles": 3,
        },
        "unet_pseudo_v1",
    )
    assert len(instances) == 1
    assert instances[0].class_name == "agglomerate"
    assert instances[0].source == "unet_pseudo_v1"


def test_instance_extraction_scales_to_thousands_of_components():
    semantic = np.zeros((2000, 2000), dtype=np.uint8)
    for y in range(0, 2000, 45):
        for x in range(0, 2000, 45):
            semantic[y : y + 2, x : x + 2] = 2
    started = time.perf_counter()
    instances = instances_from_semantic(
        semantic, np.zeros_like(semantic, dtype=np.float32), 4, "test"
    )
    elapsed = time.perf_counter() - started
    assert len(instances) == 2025
    assert elapsed < 3.0


def test_class_coordinates_are_seeded_and_bounded_per_class():
    target = np.ones((500, 1000), dtype=np.uint8)
    target[:, 500:] = 2
    first = _precompute_class_coordinates(target, np.random.default_rng(12))
    second = _precompute_class_coordinates(target, np.random.default_rng(12))
    assert set(first) == {1, 2}
    assert all(points.shape == (200_000, 2) for points in first.values())
    for class_id in first:
        np.testing.assert_array_equal(first[class_id], second[class_id])
        assert np.all(target[first[class_id][:, 0], first[class_id][:, 1]] == class_id)


def test_class_aware_crop_uses_precomputed_coordinates(monkeypatch):
    import scripts.train_unet as train_unet

    def fail_if_recomputed(*args, **kwargs):
        pytest.fail("class coordinates were recomputed during crop sampling")

    monkeypatch.setattr(
        train_unet, "_precompute_class_coordinates", fail_if_recomputed
    )
    target = np.ones((64, 64), dtype=np.uint8)
    views = {
        "BSE": np.zeros((64, 64), dtype=np.uint8),
        "Inlens": np.zeros((64, 64), dtype=np.uint8),
    }
    image, label = _sample_crop(
        (views, target),
        32,
        np.random.default_rng(5),
        True,
        {1: np.array([[20, 20]], dtype=np.int32)},
    )
    assert image.shape == (2, 32, 32)
    assert np.all(label == 1)


def test_ignored_fractions_separate_invalid_and_uncertain_pixels():
    semantic = np.array([[0, 0, 255, 1], [1, 1, 1, 1]], dtype=np.uint8)
    uncertainty = np.array([[0, 255, 0, 255], [0, 0, 255, 0]], dtype=np.uint8)
    valid = np.array([[1, 1, 1, 1], [0, 1, 1, 1]], dtype=bool)
    fractions = _ignored_fraction_breakdown(semantic, uncertainty, valid, 0.5)
    assert fractions == {
        "invalid_or_classical_255": 0.25,
        "uncertainty_gt_threshold": 0.375,
        "total": 0.625,
    }


def test_budget_aware_learning_rate_decays_with_time_or_iterations():
    base = 1e-3
    by_time = _budget_aware_learning_rate(base, 100, 20_000, 50, 100)
    by_iterations = _budget_aware_learning_rate(base, 10_000, 20_000, 0, 100)
    assert by_time == pytest.approx(base * 0.5)
    assert by_iterations == pytest.approx(base * 0.5)
    assert _budget_aware_learning_rate(base, 20_000, 20_000, 0, 100) == 0.0
