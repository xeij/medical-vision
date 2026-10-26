"""Agreement between a class activation map and an expert bounding box, in the network's input
frame. A uniform map scores the box's area fraction on both pointing game and energy, which is
the chance level reported next to each number."""

import numpy as np


def map_box(box, scale, offset, size):
    """(x, y, w, h) in source pixels -> (x0, y0, x1, y1) after resizing by `scale` and centre
    cropping `offset` pixels to a `size` square, clipped; None if nothing of the box remains."""
    x, y, w, h = box
    x0, y0 = x * scale - offset, y * scale - offset
    x1, y1 = x0 + w * scale, y0 + h * scale
    x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(size), x1), min(float(size), y1)
    return None if x1 <= x0 or y1 <= y0 else (x0, y0, x1, y1)


def box_mask(box, shape):
    m = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = box
    m[int(np.floor(y0)) : int(np.ceil(y1)), int(np.floor(x0)) : int(np.ceil(x1))] = True
    return m


def area_fraction(box, shape):
    return float(box_mask(box, shape).mean())


def pointing_hit(cam, box):
    """1.0 if the map's maximum lies inside the box; a flat map scores its chance level."""
    if cam.max() <= 0:
        return area_fraction(box, cam.shape)
    return float(box_mask(box, cam.shape)[np.unravel_index(cam.argmax(), cam.shape)])


def energy_inside(cam, box):
    """Fraction of the map's mass inside the box; a flat map scores its chance level."""
    total = cam.sum()
    if total <= 0:
        return area_fraction(box, cam.shape)
    return float(cam[box_mask(box, cam.shape)].sum() / total)


def iou_at(cam, box, frac=0.5):
    """IoU between the box and the map thresholded at `frac` of its maximum."""
    if cam.max() <= 0:
        return 0.0
    pred, gt = cam >= frac * cam.max(), box_mask(box, cam.shape)
    return float((pred & gt).sum() / (pred | gt).sum())
