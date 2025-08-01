# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import FK5, Galactic, SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.wcsapi import BaseHighLevelWCS, BaseLowLevelWCS, HighLevelWCSWrapper
from numpy.testing import assert_allclose, assert_equal

from reproject.tests.helpers import assert_wcs_allclose

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
        array = np.ones((30,))

        with pytest.raises(ValueError) as exc:
            wcs, shape = find_optimal_celestial_wcs([(array, self.wcs)])
        assert exc.value.args[0] == "Input data is not 2-dimensional (got shape (30,))"

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
    auto_rotate_expected_crpix = 20.513458, 15.513241
    multiple_size_expected_crpix = 27.279739, 17.29016


# In cases where the input WCS is wrapped in a pure APE-14 WCS, the results
# are a little different - as find_optimal_celestial_wcs takes shortcuts if
# FITS WCSes are passed in.

APE14_HEADER_REF = """
WCSAXES =                    2 / Number of coordinate axes
CRPIX1  =       25.72083769123 / Pixel coordinate of reference point
CRPIX2  =       15.85922213012 / Pixel coordinate of reference point
CDELT1  =   -0.039990388998799 / [deg] Coordinate increment at reference point
CDELT2  =    0.039990388998799 / [deg] Coordinate increment at reference point
CUNIT1  = 'deg'                / Units of coordinate increment and value
CUNIT2  = 'deg'                / Units of coordinate increment and value
CTYPE1  = 'RA---TAN'           / Right ascension, gnomonic projection
CTYPE2  = 'DEC--TAN'           / Declination, gnomonic projection
CRVAL1  =      28.717296496293 / [deg] Coordinate value at reference point
CRVAL2  =      40.532891284598 / [deg] Coordinate value at reference point
LONPOLE =                180.0 / [deg] Native longitude of celestial pole
LATPOLE =      40.532891284598 / [deg] Native latitude of celestial pole
MJDREF  =                  0.0 / [d] MJD of fiducial time
RADESYS = 'FK5'                / Equatorial coordinate system
EQUINOX =               2000.0 / [yr] Equinox of equatorial coordinates
END
""".strip()


@pytest.mark.parametrize("iterable", [False, True])
def test_input_types(valid_celestial_input_shapes, iterable):
    # Test different kinds of inputs and check the result is always the same

    array, wcs, input_value, kwargs = valid_celestial_input_shapes

    wcs_ref, shape_ref = find_optimal_celestial_wcs([(array, wcs)], frame=FK5())

    if (
        not isinstance(input_value, WCS)
        and isinstance(input_value, BaseLowLevelWCS | BaseHighLevelWCS)
    ) or (
        isinstance(input_value, tuple)
        and not isinstance(input_value[1], WCS)
        and isinstance(input_value[1], BaseLowLevelWCS | BaseHighLevelWCS)
    ):
        wcs_ref = WCS(fits.Header.fromstring(APE14_HEADER_REF, sep="\n"))
        shape_ref = (31, 50)

    if iterable:
        input_value = [input_value]

    wcs_test, shape_test = find_optimal_celestial_wcs(input_value, frame=FK5(), **kwargs)
    assert_wcs_allclose(wcs_test, wcs_ref)
    assert shape_test == shape_ref

    if isinstance(input_value, fits.HDUList) and not iterable:
        # Also check case of not passing hdu_in and having all HDUs being included

        wcs_test, shape_test = find_optimal_celestial_wcs(input_value, frame=FK5())

        assert_wcs_allclose(wcs_test, wcs_ref)
        assert shape_test == shape_ref


SOLAR_HEADER = """
CRPIX1  =   -1374.571094981584 / [pix]
CRPIX2  =    2081.629159922445 / [pix]
CRDATE1 = '2017-01-01T00:00:00.000'
CRDATE2 = '2017-01-01T00:00:00.000'
CRVAL1  =   -619.0078311637853
CRVAL2  =    -407.000970936774
CDELT1  =  0.01099999994039536
CDELT2  =  0.01099999994039536
CUNIT1  = 'arcsec  '
CUNIT2  = 'arcsec  '
CTYPE1  = 'HPLN-TAN'
CTYPE2  = 'HPLT-TAN'
PC1_1   =    0.966887196065055
PC1_2   = -0.01087372434907635
PC2_1   =  0.01173971407248916
PC2_2   =   0.9871195868097251
LONPOLE =                180.0 / [deg]
DATEREF = '2022-06-02T17:22:53.220'
OBSGEO-X=   -5466045.256954942 / [m]
OBSGEO-Y=   -2404388.737412784 / [m]
OBSGEO-Z=    2242133.887690042 / [m]
SPECSYS = 'TOPOCENT'
VELOSYS =                  0.0
"""


@pytest.mark.filterwarnings("ignore::astropy.wcs.wcs.FITSFixedWarning")
def test_solar_wcs():
    # Regression test for issues that occurred when trying to find
    # the optimal WCS for a set of solar WCSes

    pytest.importorskip("sunpy", minversion="6.0.1")

    # The following registers the WCS <-> frame for solar coordinates

    import sunpy.coordinates  # noqa

    # Make sure the WCS <-> frame functions are registered

    wcs_ref = WCS(fits.Header.fromstring(SOLAR_HEADER, sep="\n"))

    wcs1 = wcs_ref.deepcopy()
    wcs2 = wcs_ref.deepcopy()
    wcs2.wcs.crpix[0] -= 4096

    wcs, shape = find_optimal_celestial_wcs(
        [((4096, 4096), wcs1), ((4096, 4096), wcs2)], negative_lon_cdelt="auto"
    )

    wcs.wcs.set()

    assert wcs.wcs.ctype[0] == wcs_ref.wcs.ctype[0]
    assert wcs.wcs.ctype[1] == wcs_ref.wcs.ctype[1]
    assert wcs.wcs.cunit[0] == wcs_ref.wcs.cunit[0]
    assert wcs.wcs.cunit[1] == wcs_ref.wcs.cunit[1]

    # Make sure cdelt[0] and cdelt[1] are both positive
    assert np.all(wcs.wcs.cdelt > 0)

    assert shape == (4281, 8237)


@pytest.mark.filterwarnings("ignore::astropy.wcs.wcs.FITSFixedWarning")
def test_negative_lon_cdelt():
    # Regression test for issues that occurred when trying to find
    # the optimal WCS for a set of solar WCSes

    pytest.importorskip("sunpy", minversion="6.0.1")

    # The following registers the WCS <-> frame for solar coordinates

    import sunpy.coordinates  # noqa

    # Make sure the WCS <-> frame functions are registered

    wcs_ref = WCS(fits.Header.fromstring(SOLAR_HEADER, sep="\n"))

    with pytest.warns(DeprecationWarning, match="negative_lon_cdelt is not set"):
        wcs_out, _ = find_optimal_celestial_wcs(((10, 10), wcs_ref))

    assert wcs_out.wcs.cdelt[0] < 0 and wcs_out.wcs.cdelt[1] > 0

    wcs_out, _ = find_optimal_celestial_wcs(((10, 10), wcs_ref), negative_lon_cdelt=True)

    assert wcs_out.wcs.cdelt[0] < 0 and wcs_out.wcs.cdelt[1] > 0

    wcs_out, _ = find_optimal_celestial_wcs(((10, 10), wcs_ref), negative_lon_cdelt=False)

    assert np.all(wcs_out.wcs.cdelt > 0)

    wcs_out, _ = find_optimal_celestial_wcs(((10, 10), wcs_ref), negative_lon_cdelt="auto")

    assert np.all(wcs_out.wcs.cdelt > 0)
