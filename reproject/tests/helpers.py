from astropy.io import fits
from astropy.wcs import WCS
from numpy.testing import assert_allclose

from reproject.conftest import TestLowLevelWCS


def array_footprint_to_hdulist(array, footprint, header):
    hdulist = fits.HDUList()
    hdulist.append(fits.PrimaryHDU(array, header))
    hdulist.append(fits.ImageHDU(footprint, header, name="footprint"))
    return hdulist


def _underlying_wcs(wcs):
    # For testing purposes, try and return an underlying WCS object if equivalent

    if hasattr(wcs, "low_level_wcs"):
        if isinstance(wcs.low_level_wcs, WCS):
            return wcs.low_level_wcs
        elif isinstance(wcs.low_level_wcs, TestLowLevelWCS):
            return wcs.low_level_wcs._low_level_wcs
    elif isinstance(wcs, TestLowLevelWCS):
        return wcs._low_level_wcs

    return wcs


def assert_wcs_allclose(wcs1, wcs2, **kwargs):
    # First check whether the WCSes are actually the same, either directly
    # or through layers

    if wcs1 is wcs2:
        return True

    if _underlying_wcs(wcs1) is _underlying_wcs(wcs2):
        return True

    header1 = wcs1.to_header()
    header2 = wcs2.to_header()

    assert sorted(header1) == sorted(header2)

    for key1, value1 in header1.items():
        if isinstance(value1, str):
            assert value1 == header2[key1]
        else:
            assert_allclose(value1, header2[key1], **kwargs)
