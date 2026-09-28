# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from numpy.testing import assert_allclose
from scipy.ndimage import binary_erosion

pytest.importorskip("drizzle")

from ... import reproject_drizzle, reproject_exact  # noqa: E402


def _interior(expected_footprint):
    # Interior of the region that the reference algorithm reports as fully
    # covered, eroded to stay away from the edge of the input image, where
    # drizzle underweights the outermost input pixels (it cannot form pixel
    # overlap quadrilaterals for them without extrapolating). Note that this
    # is deliberately based on the reference footprint only, so that
    # assertions comparing the drizzle footprint to the reference inside this
    # region are meaningful.
    return binary_erosion(expected_footprint > 0.99, iterations=3)


def _wcs_pair():
    wcs_in = WCS(naxis=2)
    wcs_in.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs_in.wcs.crval = [30.0, 40.0]
    wcs_in.wcs.crpix = [32.5, 32.5]
    wcs_in.wcs.cdelt = [-0.001, 0.001]

    wcs_out = WCS(naxis=2)
    wcs_out.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs_out.wcs.crval = [30.0, 40.0]
    wcs_out.wcs.crpix = [40.5, 40.5]
    wcs_out.wcs.cdelt = [-0.0008, 0.0008]
    wcs_out.wcs.crota = [0, 30.0]

    return wcs_in, wcs_out


def _gaussian_data(n=64):
    y, x = np.mgrid[:n, :n]
    return np.exp(-((x - n / 2) ** 2 + (y - n / 2) ** 2) / (2 * (n / 10) ** 2))


def test_against_exact():
    # With the square kernel and pixfrac=1, drizzle implements the same
    # overlap algorithm as reproject_exact up to the flat-sky approximation,
    # so for a small field of view the results should agree to within the
    # float32 precision that drizzle works at internally.
    wcs_in, wcs_out = _wcs_pair()
    data = _gaussian_data()

    result, footprint = reproject_drizzle((data, wcs_in), wcs_out, shape_out=(80, 80))
    expected, expected_footprint = reproject_exact((data, wcs_in), wcs_out, shape_out=(80, 80))

    interior = _interior(expected_footprint)
    assert interior.sum() > 2000
    assert_allclose(result[interior], expected[interior], atol=5e-6)
    assert_allclose(footprint[interior], expected_footprint[interior], atol=0.01)


def test_frame_change():
    # The input and output WCS have different celestial frames - the drizzle
    # package itself ignores frame differences (its calc_pixmap assumes the
    # frames are the same), so make sure our own pixel mapping converts
    # between them
    wcs_in = WCS(naxis=2)
    wcs_in.wcs.ctype = ["GLON-TAN", "GLAT-TAN"]
    wcs_in.wcs.crval = [30.0, 40.0]
    wcs_in.wcs.crpix = [32.5, 32.5]
    wcs_in.wcs.cdelt = [-0.001, 0.001]

    center = SkyCoord(30.0, 40.0, unit="deg", frame="galactic").icrs
    wcs_out = WCS(naxis=2)
    wcs_out.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs_out.wcs.crval = [center.ra.deg, center.dec.deg]
    wcs_out.wcs.crpix = [40.5, 40.5]
    wcs_out.wcs.cdelt = [-0.0008, 0.0008]

    data = _gaussian_data()

    result, footprint = reproject_drizzle((data, wcs_in), wcs_out, shape_out=(80, 80))
    expected, expected_footprint = reproject_exact((data, wcs_in), wcs_out, shape_out=(80, 80))

    interior = _interior(expected_footprint)
    assert interior.sum() > 2000
    assert_allclose(result[interior], expected[interior], atol=5e-6)
    assert_allclose(footprint[interior], expected_footprint[interior], atol=0.01)


def test_flux_conservation():
    wcs_in, wcs_out = _wcs_pair()
    data = _gaussian_data()

    result, footprint = reproject_drizzle((data, wcs_in), wcs_out, shape_out=(120, 120))

    flux_in = data.sum() * 0.001**2
    flux_out = np.nansum(result) * 0.0008**2
    assert_allclose(flux_out, flux_in, rtol=1e-6)


def test_footprint_normalization():
    # The footprint should be the fraction of each output pixel covered by
    # the input image, so should be 1 everywhere in an output image that lies
    # entirely inside the input image, regardless of the relative pixel scales
    wcs_in, wcs_out = _wcs_pair()
    wcs_out = wcs_out.deepcopy()
    wcs_out.wcs.crpix = [20.5, 20.5]
    data = np.ones((64, 64))

    result, footprint = reproject_drizzle((data, wcs_in), wcs_out, shape_out=(40, 40))

    # The outermost rim of the output is excluded because drizzle skips input
    # pixels whose centers map outside the output grid, even if their
    # footprint extends into it
    assert_allclose(footprint[2:-2, 2:-2], 1, rtol=1e-5)
    assert_allclose(result[2:-2, 2:-2], 1, rtol=1e-6)


@pytest.mark.filterwarnings("ignore:Kernel .* is not a flux-conserving kernel")
@pytest.mark.parametrize("kernel", ["square", "turbo", "gaussian", "lanczos3", "point"])
def test_kernels(kernel):
    wcs_in, wcs_out = _wcs_pair()
    data = _gaussian_data()

    result, footprint = reproject_drizzle(
        (data, wcs_in), wcs_out, shape_out=(80, 80), kernel=kernel
    )
    reference, reference_footprint = reproject_drizzle((data, wcs_in), wcs_out, shape_out=(80, 80))

    # All kernels should place the peak of the gaussian in the same position
    # as the default square kernel
    assert np.unravel_index(np.nanargmax(result), result.shape) == np.unravel_index(
        np.nanargmax(reference), reference.shape
    )

    # The turbo kernel is a fast approximation of the square kernel, so the
    # flux should also be conserved (the other kernels are not
    # flux-conserving, hence the warning they emit)
    if kernel in ("square", "turbo"):
        assert_allclose(np.nansum(result), np.nansum(reference), rtol=1e-4)


def test_pixfrac():
    # With pixfrac < 1 a single image no longer fully covers the output, so
    # the flux integrated over the covered pixels is only statistically
    # conserved, and the weights become less uniform
    wcs_in, wcs_out = _wcs_pair()
    data = _gaussian_data()

    result_1, footprint_1 = reproject_drizzle((data, wcs_in), wcs_out, shape_out=(120, 120))
    result_h, footprint_h = reproject_drizzle(
        (data, wcs_in), wcs_out, shape_out=(120, 120), pixfrac=0.5
    )

    assert_allclose(np.nansum(result_h) * 0.0008**2, np.nansum(result_1) * 0.0008**2, rtol=2e-3)

    covered = (footprint_1 > 0.5) & (footprint_h > 0)
    assert footprint_h[covered].std() > 5 * footprint_1[covered].std()


def test_broadcasting():
    # Extra leading dimensions should be looped over with the pixel mapping
    # computed only once
    wcs_in, wcs_out = _wcs_pair()
    data = _gaussian_data()
    cube = np.stack([data, data * 2, data * 3])

    result_cube, footprint_cube = reproject_drizzle((cube, wcs_in), wcs_out, shape_out=(3, 80, 80))

    for i in range(3):
        result, footprint = reproject_drizzle((cube[i], wcs_in), wcs_out, shape_out=(80, 80))
        assert_allclose(result_cube[i], result, equal_nan=True)
        assert_allclose(footprint_cube[i], footprint)


def test_output_arrays():
    wcs_in, wcs_out = _wcs_pair()
    data = _gaussian_data()

    output_array = np.zeros((80, 80))
    output_footprint = np.zeros((80, 80))
    result, footprint = reproject_drizzle(
        (data, wcs_in),
        wcs_out,
        shape_out=(80, 80),
        output_array=output_array,
        output_footprint=output_footprint,
    )
    assert result is output_array
    assert footprint is output_footprint

    # Check the provided arrays were actually filled with the results
    reference, reference_footprint = reproject_drizzle((data, wcs_in), wcs_out, shape_out=(80, 80))
    assert_allclose(output_array, reference, equal_nan=True)
    assert_allclose(output_footprint, reference_footprint)

    result = reproject_drizzle((data, wcs_in), wcs_out, shape_out=(80, 80), return_footprint=False)
    assert isinstance(result, np.ndarray)
    assert_allclose(result, reference, equal_nan=True)


@pytest.mark.parametrize(
    "kwargs",
    [{"block_size": (10, 10)}, {"block_size": "auto"}, {"parallel": True}, {"return_type": "dask"}],
)
def test_unsupported_blocked_modes(kwargs):
    # Blocks that split the celestial dimensions would silently lose the flux
    # that input pixels contribute across block boundaries, so anything
    # relying on them should raise clearly
    wcs_in, wcs_out = _wcs_pair()
    data = _gaussian_data()

    with pytest.raises(NotImplementedError, match="block"):
        reproject_drizzle((data, wcs_in), wcs_out, shape_out=(80, 80), **kwargs)


def test_blocked_over_broadcast_dims():
    # Blocks that span the full celestial plane and only iterate over extra
    # leading dimensions never split flux across a block boundary, so blocked,
    # parallel and dask-output reprojection are all allowed and should match
    # the unblocked result exactly
    wcs_in, wcs_out = _wcs_pair()
    data = _gaussian_data()
    cube = np.stack([data, data * 2, data * 3])

    reference, reference_footprint = reproject_drizzle(
        (cube, wcs_in), wcs_out, shape_out=(3, 80, 80)
    )

    result, footprint = reproject_drizzle(
        (cube, wcs_in), wcs_out, shape_out=(3, 80, 80), block_size=(1, 80, 80), parallel=2
    )
    assert_allclose(result, reference, equal_nan=True)
    assert_allclose(footprint, reference_footprint)

    result = reproject_drizzle(
        (cube, wcs_in),
        wcs_out,
        shape_out=(3, 80, 80),
        block_size=(1, 80, 80),
        return_type="dask",
        return_footprint=False,
    )
    assert_allclose(np.asarray(result), reference, equal_nan=True)


def test_reproject_and_coadd():
    # reproject_drizzle should be usable as a drop-in reproject_function for
    # mosaicking
    from ...mosaicking import reproject_and_coadd

    wcs_in, wcs_out = _wcs_pair()
    data = _gaussian_data()

    wcs_in2 = wcs_in.deepcopy()
    wcs_in2.wcs.crpix = [10.5, 20.5]

    input_data = [(data, wcs_in), (data, wcs_in2)]

    result, footprint = reproject_and_coadd(
        input_data, wcs_out, shape_out=(120, 120), reproject_function=reproject_drizzle
    )
    expected, expected_footprint = reproject_and_coadd(
        input_data, wcs_out, shape_out=(120, 120), reproject_function=reproject_exact
    )

    interior = _interior(expected_footprint)
    assert interior.sum() > 2000
    assert_allclose(result[interior], expected[interior], atol=5e-6)
    assert_allclose(footprint[interior], expected_footprint[interior], atol=0.01)


def test_reproject_and_coadd_disjoint_input():
    # An input image entirely outside the mosaic, including more than 90
    # degrees away where the pixel map has no valid entries at all, should be
    # ignored rather than derail the mosaicking
    from ...mosaicking import reproject_and_coadd

    wcs_in, wcs_out = _wcs_pair()
    data = _gaussian_data()

    wcs_far = wcs_in.deepcopy()
    wcs_far.wcs.crval = [210.0, -40.0]

    result, footprint = reproject_and_coadd(
        [(data, wcs_in), (data, wcs_far)],
        wcs_out,
        shape_out=(120, 120),
        reproject_function=reproject_drizzle,
    )
    reference, reference_footprint = reproject_and_coadd(
        [(data, wcs_in)], wcs_out, shape_out=(120, 120), reproject_function=reproject_drizzle
    )

    assert_allclose(result, reference, equal_nan=True)
    assert_allclose(footprint, reference_footprint)


@pytest.mark.parametrize("crval", [(33.0, 40.0), (210.0, -40.0)])
def test_no_overlap(crval):
    # Fields that do not overlap on the sky should give an empty result
    # rather than raising, including when the fields are more than 90 degrees
    # apart and the pixel map therefore contains no valid entries at all
    # (drizzle itself raises an error in that case)
    wcs_in, _ = _wcs_pair()
    wcs_out = wcs_in.deepcopy()
    wcs_out.wcs.crval = list(crval)
    data = _gaussian_data()

    result, footprint = reproject_drizzle((data, wcs_in), wcs_out, shape_out=(80, 80))

    assert np.all(np.isnan(result))
    assert_allclose(footprint, 0)


def test_footprint_near_longitude_wrap():
    # The reference pixel around which the pixel scale ratio used for the
    # footprint normalization is estimated sits right on the longitude
    # wrap-around; drizzle's own estimate_pixel_scale_ratio mis-computes the
    # pixel area in this case (which is why we don't use it)
    wcs_in = WCS(naxis=2)
    wcs_in.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs_in.wcs.crval = [0.0, 40.0]
    wcs_in.wcs.crpix = [0.5, 0.5]
    wcs_in.wcs.cdelt = [-0.01, 0.01]

    wcs_out = WCS(naxis=2)
    wcs_out.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs_out.wcs.crval = [359.6, 40.4]
    wcs_out.wcs.crpix = [40.5, 40.5]
    wcs_out.wcs.cdelt = [-0.01, 0.01]

    data = _gaussian_data(80)

    result, footprint = reproject_drizzle((data, wcs_in), wcs_out, shape_out=(80, 80))
    expected, expected_footprint = reproject_exact((data, wcs_in), wcs_out, shape_out=(80, 80))

    interior = _interior(expected_footprint)
    assert interior.sum() > 1000
    assert_allclose(footprint[interior], expected_footprint[interior], atol=0.01)


def test_non_celestial():
    wcs_in = WCS(naxis=2)
    wcs_in.wcs.ctype = ["OFFSET", "WAVE"]

    wcs_out = WCS(naxis=2)
    wcs_out.wcs.ctype = ["OFFSET", "WAVE"]

    data = np.ones((10, 10))

    with pytest.raises(NotImplementedError, match="2-d celestial"):
        reproject_drizzle((data, wcs_in), wcs_out, shape_out=(10, 10))
