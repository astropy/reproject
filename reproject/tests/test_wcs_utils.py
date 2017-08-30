import numpy as np
from numpy.testing import assert_allclose

from astropy.wcs import WCS
from astropy.coordinates import FK5

from ..wcs_utils import find_optimal_celestial_wcs


class TestSingleImage():

    def setup_method(self, method):

        self.wcs = WCS(naxis=2)
        self.wcs.wcs.ctype = 'RA---TAN', 'DEC--TAN'
        self.wcs.wcs.crpix = 10, 15
        self.wcs.wcs.crval = 43, 23
        self.wcs.wcs.cdelt = -0.1, 0.1
        self.wcs.wcs.equinox = 2000.

        self.array = np.ones((30, 40))

    def test_identity(self):

        wcs, shape = find_optimal_celestial_wcs([(self.array, self.wcs)], frame=FK5(), projection='TAN')
        assert tuple(wcs.wcs.ctype) == ('RA---TAN', 'DEC--TAN')
        assert_allclose(wcs.wcs.crpix, (10, 15))
        assert_allclose(wcs.wcs.crval, (43, 23))
        assert_allclose(wcs.wcs.cdelt, (-0.1, 0.1))
        assert wcs.wcs.equinox == 2000
        assert wcs.wcs.radesys == 'FK5'
        assert shape == (30, 40)
