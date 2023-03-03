from astropy.io import fits
from numpy.testing import assert_allclose


def array_footprint_to_hdulist(array, footprint, header):
    hdulist = fits.HDUList()
    hdulist.append(fits.PrimaryHDU(array, header))
    hdulist.append(fits.ImageHDU(footprint, header, name="footprint"))
    return hdulist


def assert_header_allclose(header1, header2, **kwargs):
    assert sorted(header1) == sorted(header2)

    for key1, value1 in header1.items():
        if isinstance(value1, str):
            assert value1 == header2[key1]
        else:
            assert_allclose(value1, header2[key1], **kwargs)
