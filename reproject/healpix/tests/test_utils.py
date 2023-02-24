import numpy as np
import pytest
from astropy.coordinates import FK5, Galactic
from astropy.io import fits

from reproject.healpix.utils import parse_coord_system, parse_input_healpix_data


def test_parse_coord_system():
    frame = parse_coord_system(Galactic())
    assert isinstance(frame, Galactic)

    frame = parse_coord_system("fk5")
    assert isinstance(frame, FK5)

    with pytest.raises(ValueError) as exc:
        frame = parse_coord_system("e")
    assert exc.value.args[0] == "Ecliptic coordinate frame not yet supported"

    frame = parse_coord_system("g")
    assert isinstance(frame, Galactic)

    with pytest.raises(ValueError) as exc:
        frame = parse_coord_system("spam")
    assert exc.value.args[0] == "Could not determine frame for system=spam"


@pytest.mark.filterwarnings("ignore:unclosed file:ResourceWarning")
def test_parse_input_healpix_data(tmpdir):
    data = np.arange(3072)

    col = fits.Column(array=data, name="flux", format="E")
    hdu = fits.BinTableHDU.from_columns([col])
    hdu.header["NSIDE"] = 512
    hdu.header["COORDSYS"] = "G"

    # As HDU
    array, coordinate_system, nested = parse_input_healpix_data(hdu)
    np.testing.assert_allclose(array, data)

    # As filename
    filename = tmpdir.join("test.fits").strpath
    hdu.writeto(filename)
    array, coordinate_system, nested = parse_input_healpix_data(filename)
    np.testing.assert_allclose(array, data)

    # As array
    array, coordinate_system, nested = parse_input_healpix_data((data, "galactic"))
    np.testing.assert_allclose(array, data)

    # Invalid
    with pytest.raises(TypeError) as exc:
        parse_input_healpix_data(data)
    assert exc.value.args[0] == (
        "input_data should either be an HDU object or a tuple of (array, frame)"
    )
