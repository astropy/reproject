# This file is used to configure the behavior of pytest when using the Astropy
# test infrastructure. It needs to live inside the package in order for it to
# get picked up when running the tests inside an interpreter using
# packagename.test

import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.nddata import NDData
from astropy.wcs import WCS

try:
    from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS

    ASTROPY_HEADER = True
except ImportError:
    ASTROPY_HEADER = False

os.environ["MPLBACKEND"] = "Agg"


def pytest_configure(config):
    if ASTROPY_HEADER:
        config.option.astropy_header = True

        PYTEST_HEADER_MODULES.pop("Pandas", None)
        PYTEST_HEADER_MODULES.pop("h5py", None)
        PYTEST_HEADER_MODULES.pop("Matplotlib", None)
        PYTEST_HEADER_MODULES["Astropy"] = "astropy"
        PYTEST_HEADER_MODULES["astropy-healpix"] = "astropy_healpix"
        PYTEST_HEADER_MODULES["Cython"] = "cython"

        from reproject import __version__

        TESTED_VERSIONS["reproject"] = __version__


def valid_celestial_input(tmp_path, request):
    array = np.ones((30, 40))

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "RA---TAN", "DEC--TAN"
    wcs.wcs.crpix = (1, 2)
    wcs.wcs.crval = (30, 40)
    wcs.wcs.cdelt = (-0.05, 0.04)
    wcs.wcs.equinox = 2000.0

    hdulist = fits.HDUList(
        [
            fits.PrimaryHDU(array, wcs.to_header()),
            fits.ImageHDU(array, wcs.to_header()),
            fits.CompImageHDU(array, wcs.to_header()),
        ]
    )

    kwargs = {}

    if request.param in ["filename", "path"]:
        input_value = tmp_path / "test.fits"
        if request.param == "filename":
            input_value = str(input_value)
        hdulist.writeto(input_value)
        kwargs["hdu_in"] = 0
    elif request.param == "hdulist":
        input_value = hdulist
        kwargs["hdu_in"] = 1
    elif request.param == "primary_hdu":
        input_value = hdulist[0]
    elif request.param == "image_hdu":
        input_value = hdulist[1]
    elif request.param == "comp_image_hdu":
        input_value = hdulist[2]
    elif request.param == "shape_wcs_tuple":
        input_value = (array.shape, wcs)
    elif request.param == "data_wcs_tuple":
        input_value = (array, wcs)
    elif request.param == "nddata":
        input_value = NDData(data=array, wcs=wcs)
    elif request.param == "ape14_wcs":
        input_value = wcs
        input_value._naxis = list(array.shape[::-1])
    elif request.param == "shape_wcs_tuple":
        input_value = (array.shape, wcs)
    else:
        raise ValueError(f"Unknown mode: {request.param}")

    return array, wcs, input_value, kwargs


COMMON_PARAMS = [
    "filename",
    "path",
    "hdulist",
    "primary_hdu",
    "image_hdu",
    "comp_image_hdu",
    "data_wcs_tuple",
    "nddata",
]


@pytest.fixture(params=COMMON_PARAMS)
def valid_celestial_input_data(tmp_path, request):
    return valid_celestial_input(tmp_path, request)


@pytest.fixture(
    params=COMMON_PARAMS
    + [
        "ape14_wcs",
        "shape_wcs_tuple",
    ]
)
def valid_celestial_input_shapes(tmp_path, request):
    return valid_celestial_input(tmp_path, request)
