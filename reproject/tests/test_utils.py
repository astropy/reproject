import dask.array as da
import numpy as np
import pytest
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS

from reproject._wcs_utils import has_celestial, pixel_to_pixel_chunked
from reproject.conftest import set_wcs_array_shape
from reproject.tests.helpers import assert_wcs_allclose
from reproject.utils import (
    hdu_to_numpy_memmap,
    parse_input_data,
    parse_input_shape,
    parse_output_projection,
)


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


def test_pixel_to_pixel_chunked():
    from astropy.wcs.utils import pixel_to_pixel

    wcs1 = WCS(naxis=2)
    wcs1.wcs.ctype = "RA---TAN", "DEC--TAN"
    wcs1.wcs.crval = 43.0, 23.0
    wcs1.wcs.cdelt = -0.1, 0.1
    wcs1.wcs.crpix = 5.0, 5.0

    wcs2 = WCS(naxis=2)
    wcs2.wcs.ctype = "GLON-CAR", "GLAT-CAR"
    # Center on the galactic coordinates of the wcs1 reference point so the
    # two WCSes overlap
    wcs2.wcs.crval = 155.97, -32.03
    wcs2.wcs.cdelt = -0.1, 0.1
    wcs2.wcs.crpix = 5.0, 5.0

    rng = np.random.default_rng(42)
    x = rng.uniform(0, 10, 1000)
    y = rng.uniform(0, 10, 1000)

    # Chunking should not change the results
    xp, yp = pixel_to_pixel(wcs1, wcs2, x, y)
    xc, yc = pixel_to_pixel_chunked(wcs1, wcs2, x, y, chunk_size=17)
    np.testing.assert_array_equal(xp, xc)
    np.testing.assert_array_equal(yp, yc)

    # Results can be written into arrays of a different shape (matched in
    # flat C order), and broadcast inputs should be handled without being
    # expanded
    output = np.empty((2, 40, 50))
    pixel_to_pixel_chunked(
        wcs1,
        wcs2,
        np.broadcast_to(x, (2, 1000)),
        np.broadcast_to(y, (2, 1000)),
        chunk_size=17,
        output=list(output),
    )
    np.testing.assert_array_equal(output[0].ravel(), np.tile(xc, 2))
    np.testing.assert_array_equal(output[1].ravel(), np.tile(yc, 2))

    # Scalar inputs should be accepted and give the same answer as astropy
    sx, sy = pixel_to_pixel_chunked(wcs1, wcs2, 5.0, 5.0)
    ex, ey = pixel_to_pixel(wcs1, wcs2, 5.0, 5.0)
    np.testing.assert_allclose([np.asarray(sx), np.asarray(sy)], [ex, ey])


def test_pixel_to_pixel_chunked_1d():
    # A one-dimensional WCS makes pixel_to_pixel return a bare array rather
    # than a tuple, which needs special handling on both the forward and the
    # roundtrip pass.
    from astropy.wcs.utils import pixel_to_pixel

    wcs1 = WCS(naxis=1)
    wcs1.wcs.ctype = ("WAVE",)
    wcs1.wcs.cunit = ("m",)
    wcs1.wcs.crval = [1e-6]
    wcs1.wcs.cdelt = [1e-9]
    wcs1.wcs.crpix = [1]

    wcs2 = WCS(naxis=1)
    wcs2.wcs.ctype = ("WAVE",)
    wcs2.wcs.cunit = ("m",)
    wcs2.wcs.crval = [1e-6]
    wcs2.wcs.cdelt = [2e-9]
    wcs2.wcs.crpix = [1]

    x = np.linspace(0, 50, 60)
    ref = pixel_to_pixel(wcs1, wcs2, x)
    (chunked,) = pixel_to_pixel_chunked(wcs1, wcs2, x, roundtrip=True, chunk_size=7)
    np.testing.assert_array_equal(ref, chunked)


def test_pixel_to_pixel_chunked_roundtrip():
    # Reprojecting a full-sky output grid back to a small TAN input produces
    # "ghost" pixels far from the footprint that transform to a finite input
    # pixel but do not round-trip - these should be set to NaN when roundtrip
    # is enabled, while pixels that do round-trip are left untouched.
    from astropy.wcs.utils import pixel_to_pixel

    wcs_in = WCS(naxis=2)
    wcs_in.wcs.ctype = "RA---TAN", "DEC--TAN"
    wcs_in.wcs.crval = 0.0, 0.0
    wcs_in.wcs.crpix = 25.0, 25.0
    wcs_in.wcs.cdelt = -0.1, 0.1
    wcs_in.wcs.set()

    wcs_out = WCS(naxis=2)
    wcs_out.wcs.ctype = "RA---CAR", "DEC--CAR"
    wcs_out.wcs.crval = 0.0, 0.0
    wcs_out.wcs.crpix = 90.0, 45.0
    wcs_out.wcs.cdelt = -2.0, 2.0
    wcs_out.wcs.set()

    y, x = np.mgrid[0:90, 0:180]
    x = x.ravel().astype(float)
    y = y.ravel().astype(float)

    # Reference: forward transform with the round-trip reset applied by hand
    fx, fy = pixel_to_pixel(wcs_out, wcs_in, x, y)
    bx, by = pixel_to_pixel(wcs_in, wcs_out, fx, fy)
    reset = (np.abs(bx - x) > 1) | (np.abs(by - y) > 1)
    ref_x = np.where(reset, np.nan, fx)
    ref_y = np.where(reset, np.nan, fy)

    cx, cy = pixel_to_pixel_chunked(wcs_out, wcs_in, x, y, roundtrip=True, chunk_size=777)
    np.testing.assert_array_equal(cx, ref_x)
    np.testing.assert_array_equal(cy, ref_y)

    # Sanity check that this geometry actually exercises the reset (some
    # pixels transform to a finite input pixel but are still reset to NaN)
    forward_finite = np.isfinite(fx) & np.isfinite(fy)
    assert (reset & forward_finite).any()
    assert np.isfinite(cx).any()


class TestHDUToMemmap:

    def test_compressed(self, tmp_path):

        hdu = fits.CompImageHDU(data=np.random.random((128, 128)))
        hdu.writeto(tmp_path / "test.fits")

        mmap = hdu_to_numpy_memmap(hdu)

        np.testing.assert_allclose(hdu.data, mmap)


@pytest.mark.filterwarnings("ignore:Spatial")
@pytest.mark.parametrize("parse_function", [parse_input_data, parse_input_shape])
def test_parse_avm_image_rescaled(tmp_path, parse_function):
    # The AVM metadata in an image may be defined for a higher-resolution
    # version of that image, in which case the WCS has to be rescaled to the
    # dimensions of the image actually being parsed.
    pytest.importorskip("PIL")
    from PIL import Image
    from pyavm import AVM

    wcs_ref = WCS(naxis=2)
    wcs_ref.wcs.ctype = "RA---TAN", "DEC--TAN"
    wcs_ref.wcs.crpix = 10.5, 15.5
    wcs_ref.wcs.crval = 30.0, 40.0
    wcs_ref.wcs.cdelt = -0.01, 0.01

    avm = AVM.from_wcs(wcs_ref, shape=(30, 20))

    plain = tmp_path / "plain.png"
    full = tmp_path / "full.png"
    small_plain = tmp_path / "small_plain.png"
    small = tmp_path / "small.png"

    image = Image.fromarray(np.zeros((30, 20, 3), dtype=np.uint8))
    image.save(plain)
    avm.embed(str(plain), str(full))

    # Embed the same AVM (defined for the (30, 20) image) in an image half the size
    image.resize((10, 15)).save(small_plain)
    avm.embed(str(small_plain), str(small))

    result_full, wcs_full = parse_function(str(full))
    result_small, wcs_small = parse_function(str(small))

    # The world coordinates of the image corners should agree between the
    # full-size and rescaled WCS to within half a pixel of the rescaled image
    # (pyavm rescales the reference pixel without the FITS half-pixel edge
    # offset, so exact agreement is not currently possible)
    corners_full = wcs_full.pixel_to_world([-0.5, 19.5], [-0.5, 29.5])
    corners_small = wcs_small.pixel_to_world([-0.5, 9.5], [-0.5, 14.5])
    half_pixel_deg = 0.01
    assert (corners_full.separation(corners_small).deg < half_pixel_deg).all()
