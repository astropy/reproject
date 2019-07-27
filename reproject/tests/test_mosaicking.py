# Licensed under a 3-clause BSD style license - see LICENSE.rst

from copy import deepcopy

import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.coordinates import SkyCoord, FK5, Galactic
from astropy import units as u

try:
    import shapely  # noqa
except ImportError:
    SHAPELY_INSTALLED = False
else:
    SHAPELY_INSTALLED = True

from .. import reproject_interp
from ..mosaicking import find_optimal_celestial_wcs, reproject_and_coadd


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

        assert_allclose(wcs.wcs.crpix, (10, 15))
        assert shape == (30, 40)

    @pytest.mark.skipif('not SHAPELY_INSTALLED')
    @pytest.mark.parametrize('angle', np.linspace(0, 360, 13))
    def test_auto_rotate_systematic(self, angle):

        # This is a test to make sure for a number of angles that the corners
        # of the image are inside the final WCS but the next pixels outwards are
        # not. We test the full 360 range of angles.

        angle = np.radians(angle)
        pc = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
        self.wcs.wcs.pc = pc

        wcs, shape = find_optimal_celestial_wcs([(self.array, self.wcs)], auto_rotate=True)

        ny, nx = self.array.shape

        xp = np.array([0, 0, nx - 1, nx - 1, -1, -1, nx, nx])
        yp = np.array([0, ny - 1, ny - 1, 0, -1, ny, ny, -1])

        c = pixel_to_skycoord(xp, yp, self.wcs, origin=0)
        xp_final, yp_final = skycoord_to_pixel(c, wcs, origin=0)

        ny_final, nx_final = shape

        inside = ((xp_final >= -0.5) & (xp_final <= nx_final - 0.5) &
                  (yp_final >= -0.5) & (yp_final <= ny_final - 0.5))

        assert_equal(inside, [1, 1, 1, 1, 0, 0, 0, 0])

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


class TestReprojectAndCoAdd():

    def setup_method(self, method):

        self.wcs = WCS(naxis=2)
        self.wcs.wcs.ctype = 'RA---TAN', 'DEC--TAN'
        self.wcs.wcs.crpix = 322, 151
        self.wcs.wcs.crval = 43, 23
        self.wcs.wcs.cdelt = -0.1, 0.1
        self.wcs.wcs.equinox = 2000.

        self.array = np.random.random((423, 344))

    def _get_tiles(self, views):

        # Given a list of views as (imin, imax, jmin, jmax), construct
        #  tiles that can be passed into the co-adding code

        input_data = []

        for (jmin, jmax, imin, imax) in views:
            array = self.array[jmin:jmax, imin:imax]
            wcs = self.wcs.deepcopy()
            wcs.wcs.crpix[0] -= imin
            wcs.wcs.crpix[1] -= jmin
            input_data.append((array, wcs))

        return input_data

    @pytest.mark.parametrize('combine_function', ['mean', 'sum'])
    def test_coadd_no_overlap(self, combine_function):

        # Make sure that if all tiles are exactly non-overlapping, and
        # we use 'sum' or 'mean', we get the exact input array back.

        ie = (0, 122, 233, 245, 344)
        je = (0, 44, 45, 333, 335, 423)

        views = []
        for i in range(4):
            for j in range(5):
                views.append((je[j], je[j+1], ie[i], ie[i+1]))

        input_data = self._get_tiles(views)

        array, footprint = reproject_and_coadd(input_data, self.wcs, shape_out=self.array.shape,
                                               combine_function=combine_function, reproject_function=reproject_interp)

        assert_allclose(array, self.array)
        assert_equal(footprint, 1)

    def test_coadd_with_overlap(self):

        # Here we make the input tiles overlapping. We can only check the
        # mean, not the sum.

        ie = (0, 122, 233, 245, 344)
        je = (0, 44, 45, 333, 335, 423)

        views = []
        for i in range(4):
            for j in range(5):
                views.append((je[j], je[j+1] + 10, ie[i], ie[i+1] + 10))

        input_data = self._get_tiles(views)

        array, footprint = reproject_and_coadd(input_data, self.wcs, shape_out=self.array.shape,
                                               combine_function='mean', reproject_function=reproject_interp)

        assert_allclose(array, self.array)
