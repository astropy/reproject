# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import absolute_import, division, print_function

from copy import deepcopy

import pytest

import numpy as np
from numpy.testing import assert_allclose

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, FK5, Galactic
from astropy import units as u

try:
    import shapely  # noqa
except ImportError:
    SHAPELY_INSTALLED = False
else:
    SHAPELY_INSTALLED = True

from ..mosaicking import find_optimal_celestial_wcs


class TestOptimalWCS():

    def setup_method(self, method):

        self.wcs = WCS(naxis=2)
        self.wcs.wcs.ctype = 'RA---TAN', 'DEC--TAN'
        self.wcs.wcs.crpix = 10, 15
        self.wcs.wcs.crval = 43, 23
        self.wcs.wcs.cdelt = -0.1, 0.1
        self.wcs.wcs.equinox = 2000.

        self.array = np.ones((30, 40))

    def test_identity(self):

        wcs, shape = find_optimal_celestial_wcs([(self.array, self.wcs)], frame=FK5())

        assert tuple(wcs.wcs.ctype) == ('RA---TAN', 'DEC--TAN')
        assert_allclose(wcs.wcs.crval, (43, 23))
        assert_allclose(wcs.wcs.cdelt, (-0.1, 0.1))
        assert wcs.wcs.equinox == 2000
        assert wcs.wcs.radesys == 'FK5'

        assert_allclose(wcs.wcs.crpix, (10, 15))
        assert shape == (30, 40)

    def test_frame_projection(self):

        wcs, shape = find_optimal_celestial_wcs([(self.array, self.wcs)], frame=Galactic(),
                                                projection='CAR')

        assert tuple(wcs.wcs.ctype) == ('GLON-CAR', 'GLAT-CAR')
        c = SkyCoord(43, 23, unit=('deg', 'deg'), frame='fk5').galactic
        assert_allclose(wcs.wcs.crval, (c.l.degree, c.b.degree))
        assert_allclose(wcs.wcs.cdelt, (-0.1, 0.1))
        assert np.isnan(wcs.wcs.equinox)
        assert wcs.wcs.radesys == ''

        # The following values are empirical and just to make sure there are no regressions
        assert_allclose(wcs.wcs.crpix, (16.21218937, 28.86119519))
        assert shape == (47, 50)

    def test_resolution(self):
        wcs, shape = find_optimal_celestial_wcs([(self.array, self.wcs)], resolution=3 * u.arcmin)
        assert_allclose(wcs.wcs.cdelt, (-0.05, 0.05))

    @pytest.mark.skipif('not SHAPELY_INSTALLED')
    def test_auto_rotate(self):

        # To test auto_rotate, we set the frame to Galactic and the final image
        # should have the same size as the input image. In this case, the image
        # actually gets rotated 90 degrees, so the values aren't quite the same
        # as the input, but they are round values.

        wcs, shape = find_optimal_celestial_wcs([(self.array, self.wcs)],
                                                frame=Galactic(), auto_rotate=True)

        assert tuple(wcs.wcs.ctype) == ('GLON-TAN', 'GLAT-TAN')
        c = SkyCoord(43, 23, unit=('deg', 'deg'), frame='fk5').galactic
        assert_allclose(wcs.wcs.crval, (c.l.degree, c.b.degree))
        assert_allclose(wcs.wcs.cdelt, (-0.1, 0.1))
        assert np.isnan(wcs.wcs.equinox)
        assert wcs.wcs.radesys == ''

        assert_allclose(wcs.wcs.crpix, (15, 31))
        assert shape == (40, 30)

    def test_multiple_size(self):

        wcs1 = self.wcs

        wcs2 = deepcopy(self.wcs)
        wcs2.wcs.crpix[0] += 10

        wcs3 = deepcopy(self.wcs)
        wcs3.wcs.crpix[1] -= 5

        input_data = [(self.array, wcs1), (self.array, wcs2), (self.array, wcs3)]

        wcs, shape = find_optimal_celestial_wcs(input_data, frame=FK5())

        assert tuple(wcs.wcs.ctype) == ('RA---TAN', 'DEC--TAN')
        assert_allclose(wcs.wcs.crval, (43, 23))
        assert_allclose(wcs.wcs.cdelt, (-0.1, 0.1))
        assert wcs.wcs.equinox == 2000
        assert wcs.wcs.radesys == 'FK5'

        assert_allclose(wcs.wcs.crpix, (20, 15))
        assert shape == (35, 50)

    def test_multiple_resolution(self):

        wcs1 = self.wcs

        wcs2 = deepcopy(self.wcs)
        wcs2.wcs.cdelt = -0.01, 0.02

        wcs3 = deepcopy(self.wcs)
        wcs3.wcs.crpix = -0.2, 0.3

        input_data = [(self.array, wcs1), (self.array, wcs2), (self.array, wcs3)]

        wcs, shape = find_optimal_celestial_wcs(input_data)
        assert_allclose(wcs.wcs.cdelt, (-0.01, 0.01))

    def test_invalid_array_shape(self):

        array = np.ones((30, 20, 10))

        with pytest.raises(ValueError) as exc:
            wcs, shape = find_optimal_celestial_wcs([(array, self.wcs)])
        assert exc.value.args[0] == 'Input data is not 2-dimensional'

    def test_invalid_wcs_shape(self):

        wcs = WCS(naxis=3)
        wcs.wcs.ctype = 'RA---TAN', 'DEC--TAN', 'VELO-LSR'
        wcs.wcs.set()

        with pytest.raises(ValueError) as exc:
            wcs, shape = find_optimal_celestial_wcs([(self.array, wcs)])
        assert exc.value.args[0] == 'Input WCS is not 2-dimensional'

    def test_invalid_not_celestial(self):

        self.wcs.wcs.ctype = 'OFFSETX', 'OFFSETY'

        with pytest.raises(TypeError) as exc:
            wcs, shape = find_optimal_celestial_wcs([(self.array, self.wcs)])
        assert exc.value.args[0] == 'WCS does not have celestial components'
