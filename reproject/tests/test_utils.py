import numpy as np
import pytest
from astropy.io import fits
from astropy.nddata import NDData
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS

from reproject.tests.helpers import assert_header_allclose
from reproject.utils import parse_input_data, parse_input_shape, parse_output_projection


@pytest.mark.filterwarnings("ignore:unclosed file:ResourceWarning")
def test_parse_input_data(tmpdir, valid_celestial_input_data, request):
    array_ref, wcs_ref, input_value, kwargs = valid_celestial_input_data

    data, wcs = parse_input_data(input_value, **kwargs)
    np.testing.assert_allclose(data, array_ref)
    assert_header_allclose(wcs.to_header(), wcs_ref.to_header())


def test_parse_input_data_invalid():
    data = np.ones((30, 40))

    with pytest.raises(TypeError, match="input_data should either be an HDU object"):
        parse_input_data(data)


def test_parse_input_shape_missing_hdu_in():
    hdulist = fits.HDUList(
        [fits.PrimaryHDU(data=np.ones((30, 40))), fits.ImageHDU(data=np.ones((20, 30)))]
    )

    with pytest.raises(TypeError, match="More than one HDU"):
        parse_input_data(hdulist)


@pytest.mark.filterwarnings("ignore:unclosed file:ResourceWarning")
def test_parse_input_shape(tmpdir, valid_celestial_input_shapes):
    """
    This should support everything that parse_input_data does, *plus* an
    "array-like" argument that is just a shape rather than a populated array.
    """

    array_ref, wcs_ref, input_value, kwargs = valid_celestial_input_shapes

    shape, wcs = parse_input_shape(input_value, **kwargs)
    assert shape == array_ref.shape
    assert_header_allclose(wcs.to_header(), wcs_ref.to_header())


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


def test_parse_output_projection(tmpdir):
    header = fits.Header.fromtextfile(get_pkg_data_filename("data/gc_ga.hdr"))
    wcs = WCS(header)

    # As header

    with pytest.raises(ValueError) as exc:
        parse_output_projection(header)
    assert exc.value.args[0] == (
        "Need to specify shape since output header does not contain complete shape information"
    )

    parse_output_projection(header, shape_out=(200, 200))

    header["NAXIS"] = 2
    header["NAXIS1"] = 200
    header["NAXIS2"] = 300

    parse_output_projection(header)

    # As WCS

    with pytest.raises(ValueError) as exc:
        parse_output_projection(wcs)
    assert exc.value.args[0] == (
        "Need to specify shape_out when specifying output_projection as WCS object"
    )

    parse_output_projection(wcs, shape_out=(200, 200))
