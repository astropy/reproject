# This file is used to configure the behavior of pytest when using the Astropy
# test infrastructure. It needs to live inside the package in order for it to
# get picked up when running the tests inside an interpreter using
# packagename.test

import os

import dask.array as da
import numpy as np
import pytest
from astropy.io import fits
from astropy.nddata import NDData
from astropy.wcs import WCS
from astropy.wcs.wcsapi import HighLevelWCSMixin, SlicedLowLevelWCS

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


class TestLowLevelWCS(SlicedLowLevelWCS):
    # The simplest way to get a 'pure' low level WCS is to call SlicedLowLevelWCS
    # with an ellipsis slice!

    def __init__(self, low_level_wcs):
        self._low_level_wcs = low_level_wcs
        super().__init__(low_level_wcs, Ellipsis)


class TestHighLevelWCS(HighLevelWCSMixin):
    def __init__(self, low_level_wcs):
        self._low_level_wcs = low_level_wcs

    @property
    def low_level_wcs(self):
        return self._low_level_wcs

    # FIXME: due to a bug in astropy we need world_n_dim to be defined here

    @property
    def world_n_dim(self):
        return self.low_level_wcs.world_n_dim

    @property
    def pixel_n_dim(self):
        return self.low_level_wcs.pixel_n_dim


@pytest.fixture
def simple_celestial_fits_wcs():
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "RA---TAN", "DEC--TAN"
    wcs.wcs.crpix = (1, 2)
    wcs.wcs.crval = (30, 40)
    wcs.wcs.cdelt = (-0.05, 0.04)
    wcs.wcs.equinox = 2000.0
    return wcs


@pytest.fixture(params=["fits_wcs", "ape14_low_level_wcs", "ape14_high_level_wcs"])
def simple_celestial_wcs(request, simple_celestial_fits_wcs):
    if request.param == "fits_wcs":
        return simple_celestial_fits_wcs
    elif request.param == "ape14_low_level_wcs":
        return TestLowLevelWCS(simple_celestial_fits_wcs)
    elif request.param == "ape14_high_level_wcs":
        return TestHighLevelWCS(simple_celestial_fits_wcs)


def set_wcs_array_shape(wcs, shape):
    if isinstance(wcs, WCS):
        wcs._naxis = list(shape[::-1])
    elif isinstance(wcs, TestLowLevelWCS):
        wcs._low_level_wcs._naxis = list(shape[::-1])
    elif isinstance(wcs, TestHighLevelWCS):
        wcs.low_level_wcs._naxis = list(shape[::-1])


def valid_celestial_input(tmp_path, request, wcs):
    array = np.ones((30, 40))

    kwargs = {}

    if "hdu" in request.param or request.param in ["filename", "path"]:
        if not isinstance(wcs, WCS):
            pytest.skip()

        hdulist = fits.HDUList(
            [
                fits.PrimaryHDU(array, wcs.to_header()),
                fits.ImageHDU(array, wcs.to_header()),
                fits.CompImageHDU(array, wcs.to_header()),
            ]
        )

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
    elif request.param == "dask_wcs_tuple":
        input_value = (da.from_array(array), wcs)
    elif request.param == "nddata":
        input_value = NDData(data=array, wcs=wcs)
    elif request.param == "nddata_dask":
        input_value = NDData(data=da.from_array(array), wcs=wcs)
    elif request.param == "wcs":
        set_wcs_array_shape(wcs, array.shape)
        input_value = wcs
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
    "dask_wcs_tuple",
    "nddata",
    "nddata_dask",
]


@pytest.fixture(params=COMMON_PARAMS)
def valid_celestial_input_data(tmp_path, request, simple_celestial_wcs):
    return valid_celestial_input(tmp_path, request, simple_celestial_wcs)


@pytest.fixture(
    params=COMMON_PARAMS
    + [
        "wcs",
        "shape_wcs_tuple",
    ]
)
def valid_celestial_input_shapes(tmp_path, request, simple_celestial_wcs):
    return valid_celestial_input(tmp_path, request, simple_celestial_wcs)


@pytest.fixture(
    params=[
        "wcs_shape",
        "header",
        "header_shape",
        "wcs",
    ]
)
def valid_celestial_output_projections(request, simple_celestial_wcs):
    shape = (30, 40)
    wcs = simple_celestial_wcs

    # Rotate the WCS in case this is used for actual reprojection tests

    # wcs.wcs.pc = np.array([[np.cos(0.4), -np.sin(0.4)], [np.sin(0.4), np.cos(0.4)]])

    kwargs = {}

    if request.param == "wcs_shape":
        output_value = wcs
        kwargs["shape_out"] = shape
    elif request.param == "header":
        if not isinstance(wcs, WCS):
            pytest.skip()
        header = wcs.to_header()
        header["NAXIS"] = 2
        header["NAXIS1"] = 40
        header["NAXIS2"] = 30
        output_value = header
    elif request.param == "header_shape":
        if not isinstance(wcs, WCS):
            pytest.skip()
        output_value = wcs.to_header()
        kwargs["shape_out"] = shape
    elif request.param == "wcs":
        set_wcs_array_shape(wcs, (30, 40))
        output_value = wcs

    return wcs, shape, output_value, kwargs
