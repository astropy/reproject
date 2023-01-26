# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import FK5, Galactic, SkyCoord
from astropy.wcs import WCS
from astropy.wcs.wcsapi import HighLevelWCSWrapper
from numpy.testing import assert_allclose, assert_equal

from ..wcs_helpers import find_optimal_celestial_wcs

try:
    import shapely  # noqa
except ImportError:
    SHAPELY_INSTALLED = False
else:
    SHAPELY_INSTALLED = True


class BaseTestOptimalWCS:
    def setup_method(self, method):
        self.wcs = self.generate_wcs()
        self.array = np.ones((30, 40))

    def test_identity(self):

        wcs, shape = find_optimal_celestial_wcs([(self.array, self.wcs)], frame=FK5())

        assert tuple(wcs.wcs.ctype) == ("RA---TAN", "DEC--TAN")
        assert_allclose(wcs.wcs.crval, (43, 23), atol=self.crval_atol)
        assert_allclose(wcs.wcs.cdelt, (-0.1, 0.1), rtol=self.cdelt_rtol)
        assert wcs.wcs.equinox == 2000
        assert wcs.wcs.radesys == "FK5"

        assert_allclose(wcs.wcs.crpix, self.identity_expected_crpix)
        assert shape == (30, 40)

    def test_args_tuple_wcs(self):
        wcs, shape = find_optimal_celestial_wcs([(self.array.shape, self.wcs)], frame=FK5())

    def test_args_tuple_header(self):
        wcs, shape = find_optimal_celestial_wcs(
            [(self.array.shape, self.wcs.to_header())], frame=FK5()
        )

    def test_frame_projection(self):

        wcs, shape = find_optimal_celestial_wcs(
            [(self.array, self.wcs)], frame=Galactic(), projection="CAR"
        )

        assert tuple(wcs.wcs.ctype) == ("GLON-CAR", "GLAT-CAR")
        c = SkyCoord(43, 23, unit=("deg", "deg"), frame="fk5").galactic
        assert_allclose(wcs.wcs.crval, (c.l.degree, c.b.degree), atol=self.crval_atol)
        assert_allclose(wcs.wcs.cdelt, (-0.1, 0.1), rtol=self.cdelt_rtol)
        assert np.isnan(wcs.wcs.equinox)
        assert wcs.wcs.radesys == ""

        assert_allclose(wcs.wcs.crpix, self.frame_projection_expected_crpix)
        assert shape == self.frame_projection_expected_shape

    def test_frame_str(self):
        wcs, shape = find_optimal_celestial_wcs([(self.array, self.wcs)], frame="galactic")
        assert tuple(wcs.wcs.ctype) == ("GLON-TAN", "GLAT-TAN")

    def test_resolution(self):
        wcs, shape = find_optimal_celestial_wcs([(self.array, self.wcs)], resolution=3 * u.arcmin)
        assert_allclose(wcs.wcs.cdelt, (-0.05, 0.05))

    @pytest.mark.skipif("not SHAPELY_INSTALLED")
    def test_auto_rotate(self):

        # To test auto_rotate, we set the frame to Galactic and the final image
        # should have the same size as the input image. In this case, the image
        # actually gets rotated 90 degrees, so the values aren't quite the same
        # as the input, but they are round values.

        wcs, shape = find_optimal_celestial_wcs(
            [(self.array, self.wcs)], frame=Galactic(), auto_rotate=True
        )

        assert tuple(wcs.wcs.ctype) == ("GLON-TAN", "GLAT-TAN")
        c = SkyCoord(43, 23, unit=("deg", "deg"), frame="fk5").galactic
        assert_allclose(wcs.wcs.crval, (c.l.degree, c.b.degree), atol=self.crval_atol)
        assert_allclose(wcs.wcs.cdelt, (-0.1, 0.1), rtol=self.cdelt_rtol)
        assert np.isnan(wcs.wcs.equinox)
        assert wcs.wcs.radesys == ""

        assert_allclose(wcs.wcs.crpix, self.auto_rotate_expected_crpix)
        assert shape == (30, 40)

    @pytest.mark.skipif("not SHAPELY_INSTALLED")
    @pytest.mark.parametrize("angle", np.linspace(0, 360, 13))
    def test_auto_rotate_systematic(self, angle):

        # This is a test to make sure for a number of angles that the corners
        # of the image are inside the final WCS but the next pixels outwards are
        # not. We test the full 360 range of angles.

        angle = np.radians(angle)
        pc = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        self.generate_wcs(pc=pc)

        wcs, shape = find_optimal_celestial_wcs([(self.array, self.wcs)], auto_rotate=True)

        ny, nx = self.array.shape

        xp = np.array([0, 0, nx - 1, nx - 1, -1, -1, nx, nx])
        yp = np.array([0, ny - 1, ny - 1, 0, -1, ny, ny, -1])

        c = self.wcs.pixel_to_world(xp, yp)
        xp_final, yp_final = wcs.world_to_pixel(c)

        ny_final, nx_final = shape

        inside = (
            (xp_final >= -0.5)
            & (xp_final <= nx_final - 0.5)
            & (yp_final >= -0.5)
            & (yp_final <= ny_final - 0.5)
        )

        assert_equal(inside, [1, 1, 1, 1, 0, 0, 0, 0])

    def test_multiple_size(self):

        wcs1 = self.wcs
        wcs2 = self.generate_wcs(crpix=(20, 15))
        wcs3 = self.generate_wcs(crpix=(10, 10))

        input_data = [(self.array, wcs1), (self.array, wcs2), (self.array, wcs3)]

        wcs, shape = find_optimal_celestial_wcs(input_data, frame=FK5())

        assert tuple(wcs.wcs.ctype) == ("RA---TAN", "DEC--TAN")
        assert_allclose(wcs.wcs.crval, (43, 23), atol=self.crval_atol)
        assert_allclose(wcs.wcs.cdelt, (-0.1, 0.1), rtol=self.cdelt_rtol)
        assert wcs.wcs.equinox == 2000
        assert wcs.wcs.radesys == "FK5"

        assert_allclose(wcs.wcs.crpix, self.multiple_size_expected_crpix)
        assert shape == (35, 50)

    def test_multiple_resolution(self):

        wcs1 = self.wcs
        wcs2 = self.generate_wcs(cdelt=(-0.01, 0.02))
        wcs3 = self.generate_wcs(cdelt=(-0.2, 0.3))

        input_data = [(self.array, wcs1), (self.array, wcs2), (self.array, wcs3)]

        wcs, shape = find_optimal_celestial_wcs(input_data)
        assert_allclose(wcs.wcs.cdelt, (-0.01, 0.01), rtol=self.cdelt_rtol)

    def test_invalid_array_shape(self):

        array = np.ones((30, 20, 10))

        with pytest.raises(ValueError) as exc:
            wcs, shape = find_optimal_celestial_wcs([(array, self.wcs)])
        assert exc.value.args[0] == "Input data is not 2-dimensional (got shape (30, 20, 10))"

    def test_invalid_wcs_shape(self):

        wcs = WCS(naxis=3)
        wcs.wcs.ctype = "RA---TAN", "DEC--TAN", "VELO-LSR"
        wcs.wcs.set()

        with pytest.raises(ValueError) as exc:
            wcs, shape = find_optimal_celestial_wcs([(self.array, wcs)])
        assert exc.value.args[0] == "Input WCS is not 2-dimensional"

    def test_invalid_not_celestial(self):

        self.wcs = self.generate_wcs(celestial=False)

        with pytest.raises(TypeError) as exc:
            wcs, shape = find_optimal_celestial_wcs([(self.array, self.wcs)])
        assert exc.value.args[0] == "WCS does not have celestial components"


class TestOptimalFITSWCS(BaseTestOptimalWCS):
    def generate_wcs(
        self, crpix=(10, 15), crval=(43, 23), cdelt=(-0.1, 0.1), pc=None, celestial=True
    ):
        wcs = WCS(naxis=2)
        if celestial:
            wcs.wcs.ctype = "RA---TAN", "DEC--TAN"
        else:
            wcs.wcs.ctype = "OFFSETX", "OFFSETY"
        wcs.wcs.crpix = crpix
        wcs.wcs.crval = crval
        wcs.wcs.cdelt = cdelt
        wcs.wcs.equinox = 2000.0
        if pc is not None:
            wcs.wcs.pc = pc
        return wcs

    crval_atol = 1e-8
    crpix_atol = 1e-6
    cdelt_rtol = 1e-8

    identity_expected_crpix = 10, 15
    auto_rotate_expected_crpix = 10, 15
    multiple_size_expected_crpix = 20, 15

    # The following values are empirical and just to make sure there are no regressions
    frame_projection_expected_crpix = 16.212189, 28.861195
    frame_projection_expected_shape = 47, 50


class TestOptimalAPE14WCS(TestOptimalFITSWCS):
    def generate_wcs(
        self, crpix=(10, 15), crval=(43, 23), cdelt=(-0.1, 0.1), pc=None, celestial=True
    ):
        wcs = super().generate_wcs(
            crpix=crpix, crval=crval, cdelt=cdelt, pc=pc, celestial=celestial
        )
        return HighLevelWCSWrapper(wcs)

    def test_args_tuple_header(self):
        pytest.skip()

    crval_atol = 1.5
    crpix_atol = 1e-6
    cdelt_rtol = 1.0e-3

    # The following values are empirical and just to make sure there are no regressions
    identity_expected_crpix = 20.630112, 15.649142
    frame_projection_expected_crpix = 25.381691, 23.668728
    frame_projection_expected_shape = 46, 50
    auto_rotate_expected_crpix = 20.520875, 15.503349
    multiple_size_expected_crpix = 27.279739, 17.29016
