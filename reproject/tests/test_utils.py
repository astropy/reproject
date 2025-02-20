import dask.array as da
import numpy as np
import pytest
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS

from reproject.conftest import set_wcs_array_shape
from reproject.tests.helpers import assert_wcs_allclose
from reproject.utils import (
    hdu_to_numpy_memmap,
    parse_input_data,
    parse_input_shape,
    parse_output_projection,
)
from reproject.wcs_utils import has_celestial


@pytest.mark.filterwarnings("ignore:unclosed file:ResourceWarning")
def test_parse_input_data(tmpdir, valid_celestial_input_data, request):
    array_ref, wcs_ref, input_value, kwargs = valid_celestial_input_data

    data, wcs = parse_input_data(input_value, **kwargs)
    assert isinstance(data, da.Array | np.ndarray)
    np.testing.assert_allclose(data, array_ref)
    assert_wcs_allclose(wcs, wcs_ref)


def test_parse_input_data_invalid():
    data = np.ones((30, 40))

    with pytest.raises(TypeError, match="input_data should either be an HDU object"):
        parse_input_data(data)


def test_parse_input_data_missing_hdu_in():
    hdulist = fits.HDUList(
        [fits.PrimaryHDU(data=np.ones((30, 40))), fits.ImageHDU(data=np.ones((20, 30)))]
    )

    with pytest.raises(ValueError, match="More than one HDU"):
        parse_input_data(hdulist)


def test_parse_input_data_distortion_map():
    # Verify that the file can be successfully loaded and parsed
    fname = get_pkg_data_filename("data/image_with_distortion_map.fits", package="reproject.tests")
    parse_input_data(fname, hdu_in=0)


@pytest.mark.filterwarnings("ignore:unclosed file:ResourceWarning")
def test_parse_input_shape(tmpdir, valid_celestial_input_shapes):
    """
    This should support everything that parse_input_data does, *plus* an
    "array-like" argument that is just a shape rather than a populated array.
    """

    array_ref, wcs_ref, input_value, kwargs = valid_celestial_input_shapes

    shape, wcs = parse_input_shape(input_value, **kwargs)
    assert shape == array_ref.shape
    assert_wcs_allclose(wcs, wcs_ref)


def test_parse_input_shape_invalid():
    data = np.ones((30, 40))

    # Invalid
    with pytest.raises(TypeError) as exc:
        parse_input_shape(data)
    assert exc.value.args[0] == (
        "input_shape should either be an HDU object or a tuple "
        "of (array-or-shape, WCS) or (array-or-shape, Header)"
    )


def test_parse_input_shape_missing_hdu_in():
    hdulist = fits.HDUList(
        [fits.PrimaryHDU(data=np.ones((30, 40))), fits.ImageHDU(data=np.ones((20, 30)))]
    )

    with pytest.raises(ValueError) as exc:
        shape, coordinate_system = parse_input_shape(hdulist)
    assert exc.value.args[0] == (
        "More than one HDU is present, please specify HDU to use with ``hdu_in=`` option"
    )


def test_parse_output_projection(valid_celestial_output_projections):
    wcs_ref, shape_ref, output_value, kwargs = valid_celestial_output_projections

    wcs, shape = parse_output_projection(output_value, **kwargs)

    assert shape == shape_ref
    assert_wcs_allclose(wcs, wcs_ref)


def test_parse_output_projection_invalid_header(simple_celestial_fits_wcs):
    with pytest.raises(ValueError, match="Need to specify shape"):
        parse_output_projection(simple_celestial_fits_wcs.to_header())


def test_parse_output_projection_invalid_wcs(simple_celestial_fits_wcs):
    with pytest.raises(ValueError, match="Need to specify shape"):
        parse_output_projection(simple_celestial_fits_wcs)


def test_parse_output_projection_override_shape_out(simple_celestial_wcs):
    # Regression test for a bug that caused shape_out to be ignored if the
    # WCS object had array_shape set - but shape_out should override the WCS
    # shape.

    wcs_ref = simple_celestial_wcs

    set_wcs_array_shape(wcs_ref, (10, 20))

    if hasattr(wcs_ref, "low_level_wcs"):
        assert wcs_ref.low_level_wcs.array_shape == (10, 20)
    else:
        assert wcs_ref.array_shape == (10, 20)

    wcs, shape = parse_output_projection(wcs_ref, shape_out=(30, 40))

    assert shape == (30, 40)
    assert_wcs_allclose(wcs, wcs_ref)


@pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
@pytest.mark.filterwarnings("ignore::astropy.wcs.wcs.FITSFixedWarning")
def test_has_celestial():
    from .test_high_level import INPUT_HDR

    hdr = fits.Header.fromstring(INPUT_HDR)
    ww = WCS(hdr)
    assert ww.has_celestial
    assert has_celestial(ww)

    from astropy.wcs.wcsapi import HighLevelWCSWrapper, SlicedLowLevelWCS

    wwh = HighLevelWCSWrapper(SlicedLowLevelWCS(ww, Ellipsis))
    assert has_celestial(wwh)

    wwh2 = HighLevelWCSWrapper(SlicedLowLevelWCS(ww, [slice(0, 1), slice(0, 1)]))
    assert has_celestial(wwh2)


class TestHDUToMemmap:

    def test_compressed(self, tmp_path):

        hdu = fits.CompImageHDU(data=np.random.random((128, 128)))
        hdu.writeto(tmp_path / "test.fits")

        mmap = hdu_to_numpy_memmap(hdu)

        np.testing.assert_allclose(hdu.data, mmap)
