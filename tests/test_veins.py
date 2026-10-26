import numpy as np

from leafgen.veins.connect import bridge_loopcut, connectivity
from leafgen.veins.raster import outline_mask


def test_repair_joins_a_detached_vein():
    t = np.linspace(-np.pi / 2, 3 * np.pi / 2, 256, endpoint=False)
    contour = np.column_stack([0.3 * np.cos(t), 0.5 + 0.5 * np.sin(t)])
    mask = outline_mask(contour, 256)
    veins = np.zeros((256, 256), np.float32)
    rows, cols = np.nonzero(mask)
    mid = int(np.median(cols))
    veins[rows.min() + 5 : rows.max() - 5, mid - 1 : mid + 2] = 1.0  # midrib
    veins[120:123, mid + 15 : mid + 50] = 0.6  # a secondary left detached from it
    before = connectivity(veins, mask)
    after = connectivity(bridge_loopcut(veins, mask), mask)
    assert before["lcc_coarse"] < 1.0
    assert after["lcc_coarse"] == 1.0
