# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings

import numpy as np
import pytest
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
from numpy.testing import assert_allclose

from ..high_level import reproject_exact


class TestReprojectExact:
    def setup_class(self):
        header_gal = get_pkg_data_filename("../../tests/data/gc_ga.hdr")
        header_equ = get_pkg_data_filename("../../tests/data/gc_eq.hdr")
        self.header_in = fits.Header.fromtextfile(header_gal)
        self.header_out = fits.Header.fromtextfile(header_equ)

        self.header_out["NAXIS"] = 2
        self.header_out["NAXIS1"] = 600
        self.header_out["NAXIS2"] = 550

        self.array_in = np.ones((100, 100))

        self.wcs_in = WCS(self.header_in)
        self.wcs_out = WCS(self.header_out)

    def test_array_wcs(self):
        reproject_exact((self.array_in, self.wcs_in), self.wcs_out, shape_out=(200, 200))

    def test_array_header(self):
        reproject_exact((self.array_in, self.header_in), self.header_out)

    def test_parallel_option(self):
        reproject_exact((self.array_in, self.header_in), self.header_out, parallel=1)

        with pytest.raises(ValueError) as exc:
            reproject_exact((self.array_in, self.header_in), self.header_out, parallel=-1)
        assert exc.value.args[0] == "The number of processors to use must be strictly positive"

    def test_reproject_parallel_consistency(self):
        reproject_exact((self.array_in, self.header_in), self.header_out, parallel=1)

        array1, footprint1 = reproject_exact(
            (self.array_in, self.header_in), self.header_out, parallel=False
        )
        array2, footprint2 = reproject_exact(
            (self.array_in, self.header_in), self.header_out, parallel=4
        )

        np.testing.assert_allclose(array1, array2, rtol=1.0e-5)

        np.testing.assert_allclose(footprint1, footprint2, rtol=3.0e-5)


def test_identity():
    # Reproject an array and WCS to itself

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "RA---TAN", "DEC--TAN"
    wcs.wcs.crpix = 322, 151
    wcs.wcs.crval = 43, 23
    wcs.wcs.cdelt = -0.1, 0.1
    wcs.wcs.equinox = 2000.0

    np.random.seed(1249)

    array_in = np.random.random((423, 344))
    array_out, footprint = reproject_exact((array_in, wcs), wcs, shape_out=array_in.shape)

    assert_allclose(array_out, array_in, atol=1e-10)


def test_reproject_precision_warning():
    for res in [0.1 / 3600, 0.01 / 3600]:
        wcs1 = WCS()
        wcs1.wcs.ctype = "RA---TAN", "DEC--TAN"
        wcs1.wcs.crval = 13, 80
        wcs1.wcs.crpix = 10.0, 10.0
        wcs1.wcs.cdelt = res, res

        wcs2 = WCS()
        wcs2.wcs.ctype = "RA---TAN", "DEC--TAN"
        wcs2.wcs.crval = 13, 80
        wcs2.wcs.crpix = 3, 3
        wcs2.wcs.cdelt = 3 * res, 3 * res

        array = np.zeros((19, 19))
        array[9, 9] = 1

        if res < 0.05 / 3600:
            with pytest.warns(
                UserWarning, match="The reproject_exact function currently has precision"
            ):
                reproject_exact((array, wcs1), wcs2, shape_out=(5, 5))
        else:
            with warnings.catch_warnings(record=True) as w:
                reproject_exact((array, wcs1), wcs2, shape_out=(5, 5))
            assert len(w) == 0


def _setup_for_broadcast_test():
    with fits.open(get_pkg_data_filename("data/galactic_2d.fits", package="reproject.tests")) as pf:
        hdu_in = pf[0]
        header_in = hdu_in.header.copy()
        header_out = hdu_in.header.copy()
        header_out["CTYPE1"] = "RA---TAN"
        header_out["CTYPE2"] = "DEC--TAN"
        header_out["CRVAL1"] = 266.39311
        header_out["CRVAL2"] = -28.939779

        data = hdu_in.data

    image_stack = np.stack((data, data.T, data[::-1], data[:, ::-1]))

    # Build the reference array through un-broadcast reprojections
    array_ref = []
    footprint_ref = []
    for i in range(len(image_stack)):
        array_out, footprint_out = reproject_exact((image_stack[i], header_in), header_out)
        array_ref.append(array_out)
        footprint_ref.append(footprint_out)
    array_ref = np.stack(array_ref)
    footprint_ref = np.stack(footprint_ref)

    return image_stack, array_ref, footprint_ref, header_in, header_out


@pytest.mark.parametrize("input_extra_dims", (1, 2))
@pytest.mark.parametrize("output_shape", (None, "single", "full"))
@pytest.mark.parametrize("input_as_wcs", (True, False))
@pytest.mark.parametrize("output_as_wcs", (True, False))
def test_broadcast_reprojection(input_extra_dims, output_shape, input_as_wcs, output_as_wcs):
    image_stack, array_ref, footprint_ref, header_in, header_out = _setup_for_broadcast_test()
    # Test both single and multiple dimensions being broadcast
    if input_extra_dims == 2:
        image_stack = image_stack.reshape((2, 2, *image_stack.shape[-2:]))
        array_ref.shape = image_stack.shape
        footprint_ref.shape = image_stack.shape

    # Test different ways of providing the output shape
    if output_shape == "single":
        # Have the broadcast dimensions be auto-added to the output shape
        output_shape = image_stack.shape[-2:]
    elif output_shape == "full":
        # Provide the broadcast dimensions as part of the output shape
        output_shape = image_stack.shape

    # Ensure logic works with WCS inputs as well as Header inputs
    if input_as_wcs:
        header_in = WCS(header_in)
    if output_as_wcs:
        header_out = WCS(header_out)
        if output_shape is None:
            # This combination of parameter values is not valid
            return

    array_broadcast, footprint_broadcast = reproject_exact(
        (image_stack, header_in), header_out, output_shape
    )

    np.testing.assert_allclose(footprint_broadcast, footprint_ref)
    np.testing.assert_allclose(array_broadcast, array_ref)


@pytest.mark.parametrize("input_extra_dims", (1, 2))
@pytest.mark.parametrize("output_shape", (None, "single", "full"))
@pytest.mark.parametrize("parallel", (2, False))
def test_broadcast_parallel_reprojection(input_extra_dims, output_shape, parallel):
    image_stack, array_ref, footprint_ref, header_in, header_out = _setup_for_broadcast_test()
    # Test both single and multiple dimensions being broadcast
    if input_extra_dims == 2:
        image_stack = image_stack.reshape((2, 2, *image_stack.shape[-2:]))
        array_ref.shape = image_stack.shape
        footprint_ref.shape = image_stack.shape

    # Test different ways of providing the output shape
    if output_shape == "single":
        # Have the broadcast dimensions be auto-added to the output shape
        output_shape = image_stack.shape[-2:]
    elif output_shape == "full":
        # Provide the broadcast dimensions as part of the output shape
        output_shape = image_stack.shape

    array_broadcast, footprint_broadcast = reproject_exact(
        (image_stack, header_in), header_out, output_shape, parallel=parallel
    )

    np.testing.assert_allclose(footprint_broadcast, footprint_ref)
    np.testing.assert_allclose(array_broadcast, array_ref)


def test_exact_input_output_types(valid_celestial_input_data, valid_celestial_output_projections):
    # Check that all valid input/output types work properly

    array_ref, wcs_in_ref, input_value, kwargs_in = valid_celestial_input_data

    wcs_out_ref, shape_ref, output_value, kwargs_out = valid_celestial_output_projections

    # Compute reference

    output_ref, footprint_ref = reproject_exact(
        (array_ref, wcs_in_ref), wcs_out_ref, shape_out=shape_ref
    )

    # Compute test

    output_test, footprint_test = reproject_exact(
        input_value, output_value, **kwargs_in, **kwargs_out
    )

    assert_allclose(output_ref, output_test)
    assert_allclose(footprint_ref, footprint_test)
