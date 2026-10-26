import numpy as np
import pytest

from eval.localization import area_fraction, energy_inside, iou_at, map_box, pointing_hit


def test_map_box_resize_and_crop():
    # 1024-px frame resized to 256 then centre-cropped to 224: scale 1/4, offset 16
    assert map_box((64, 64, 400, 200), 0.25, 16, 224) == (0, 0, 100, 50)
    assert map_box((400, 400, 400, 400), 0.25, 16, 224) == (84, 84, 184, 184)
    assert map_box((0, 0, 40, 40), 0.25, 16, 224) is None  # entirely in the cropped margin
    assert map_box((1000, 1000, 100, 100), 0.25, 16, 224) is None


def test_pointing_energy_iou():
    cam = np.zeros((8, 8))
    cam[2, 3] = 1.0  # peak at row 2, col 3
    inside = (2, 1, 5, 4)  # x0, y0, x1, y1 covers cols 2..4, rows 1..3
    outside = (5, 5, 8, 8)
    assert pointing_hit(cam, inside) == 1.0 and pointing_hit(cam, outside) == 0.0
    assert energy_inside(cam, inside) == 1.0 and energy_inside(cam, outside) == 0.0
    assert iou_at(cam, inside) == pytest.approx(1 / 9)  # one hot pixel vs a 3x3 box
    assert area_fraction(inside, cam.shape) == pytest.approx(9 / 64)


def test_flat_map_scores_chance():
    cam = np.zeros((8, 8))
    box = (0, 0, 4, 4)
    assert pointing_hit(cam, box) == pytest.approx(0.25)
    assert energy_inside(cam, box) == pytest.approx(0.25)
    assert iou_at(cam, box) == 0.0
    uniform = np.ones((8, 8))
    assert energy_inside(uniform, box) == pytest.approx(0.25)
