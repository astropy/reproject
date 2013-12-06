import numpy as np
from .. import compute_overlap

def test_overlap():
    lon, lat = [0, 1, 1, 0], [0, 0, 1, 1]
    overlap, area_ratio = compute_overlap(lon, lat, lon, lat)
    np.testing.assert_allclose(overlap, np.radians(1) ** 2, rtol=1e-3)
    np.testing.assert_allclose(area_ratio, 1, rtol=1e-3)
