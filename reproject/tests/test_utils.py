import numpy as np
import pytest
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
from astropy.nddata import NDData

from reproject.utils import parse_input_data, parse_input_shape, parse_output_projection


@pytest.mark.filterwarnings('ignore:unclosed file:ResourceWarning')
def test_parse_input_data(tmpdir):

    header = fits.Header.fromtextfile(get_pkg_data_filename('data/gc_ga.hdr'))

    data = np.arange(200).reshape((10, 20))

    hdu = fits.ImageHDU(data)

    # As HDU
    array, coordinate_system = parse_input_data(hdu)
    np.testing.assert_allclose(array, data)

    # As filename
    filename = tmpdir.join('test.fits').strpath
    hdu.writeto(filename)

    with pytest.raises(ValueError) as exc:
        array, coordinate_system = parse_input_data(filename)
    assert exc.value.args[0] == ("More than one HDU is present, please specify "
                                 "HDU to use with ``hdu_in=`` option")

    array, coordinate_system = parse_input_data(filename, hdu_in=1)
    np.testing.assert_allclose(array, data)

    # As array, header
    array, coordinate_system = parse_input_data((data, header))
    np.testing.assert_allclose(array, data)

    # As array, WCS
    wcs = WCS(hdu.header)
    array, coordinate_system = parse_input_data((data, wcs))
    np.testing.assert_allclose(array, data)

    ndd = NDData(data, wcs=wcs)
    array, coordinate_system = parse_input_data(ndd)
    np.testing.assert_allclose(array, data)
    assert coordinate_system is wcs

    # Invalid
    with pytest.raises(TypeError) as exc:
        parse_input_data(data)
    assert exc.value.args[0] == ("input_data should either be an HDU object or "
                                 "a tuple of (array, WCS) or (array, Header)")


@pytest.mark.filterwarnings('ignore:unclosed file:ResourceWarning')
def test_parse_input_shape(tmpdir):
    """
    This should support everything that parse_input_data does, *plus* an
    "array-like" argument that is just a shape rather than a populated array.
    """
    header = fits.Header.fromtextfile(get_pkg_data_filename('data/gc_ga.hdr'))
    in_shape = (10, 20)
    data = np.arange(200).reshape(in_shape)
    hdu = fits.ImageHDU(data)

    # As HDU
    shape, coordinate_system = parse_input_shape(hdu)
    assert shape == in_shape

    # As filename
    filename = tmpdir.join('test.fits').strpath
    hdu.writeto(filename)

    with pytest.raises(ValueError) as exc:
        shape, coordinate_system = parse_input_shape(filename)
    assert exc.value.args[0] == ("More than one HDU is present, please specify "
                                 "HDU to use with ``hdu_in=`` option")

    shape, coordinate_system = parse_input_shape(filename, hdu_in=1)
    assert shape == in_shape

    # As array, header
    shape, coordinate_system = parse_input_shape((data, header))
    assert shape == in_shape

    # As array, WCS
    wcs = WCS(hdu.header)
    shape, coordinate_system = parse_input_shape((data, wcs))
    assert shape == in_shape

    ndd = NDData(data, wcs=wcs)
    shape, coordinate_system = parse_input_shape(ndd)
    assert shape == in_shape
    assert coordinate_system is wcs

    # As shape, header
    shape, coordinate_system = parse_input_shape((data.shape, header))
    assert shape == in_shape

    # As shape, WCS
    shape, coordinate_system = parse_input_shape((data.shape, wcs))
    assert shape == in_shape

    # Invalid
    with pytest.raises(TypeError) as exc:
        parse_input_shape(data)
    assert exc.value.args[0] == ("input_shape should either be an HDU object or a tuple "
                                 "of (array-or-shape, WCS) or (array-or-shape, Header)")


def test_parse_output_projection(tmpdir):

    header = fits.Header.fromtextfile(get_pkg_data_filename('data/gc_ga.hdr'))
    wcs = WCS(header)

    # As header

    with pytest.raises(ValueError) as exc:
        parse_output_projection(header)
    assert exc.value.args[0] == ("Need to specify shape since output header "
                                 "does not contain complete shape information")

    parse_output_projection(header, shape_out=(200, 200))

    header['NAXIS'] = 2
    header['NAXIS1'] = 200
    header['NAXIS2'] = 300

    parse_output_projection(header)

    # As WCS

    with pytest.raises(ValueError) as exc:
        parse_output_projection(wcs)
    assert exc.value.args[0] == ("Need to specify shape_out when specifying "
                                 "output_projection as WCS object")

    parse_output_projection(wcs, shape_out=(200, 200))
