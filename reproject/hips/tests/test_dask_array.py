import numpy as np
import pytest
from astropy.io import fits
from astropy.io.fits import Header
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS

from reproject import reproject_interp
from reproject.hips import _dask_array as dask_array_module
from reproject.hips import reproject_to_hips
from reproject.hips._dask_array import HiPSArray, hips_as_dask_array
from reproject.hips._high_level import find_indices


class TestHIPSDaskArray:

    def setup_method(self):
        # We use an all-sky WCS image as input since this will test all parts
        # of the HiPS projection (some issues happen around boundaries for instance)
        hdu = fits.open(get_pkg_data_filename("allsky/allsky_rosat.fits"))[0]
        self.original_wcs = WCS(hdu.header)
        self.original_array = hdu.data.size + np.arange(hdu.data.size).reshape(hdu.data.shape)

        self.original_array_3d = self.original_array.reshape((1,) + self.original_array.shape)
        self.original_array_3d = self.original_array_3d * np.arange(1, 11).reshape((10, 1, 1))
        assert self.original_array_3d.shape == (10, 240, 480)

        self.original_wcs_3d = self.original_wcs.sub([1, 2, 0])
        self.original_wcs_3d.wcs.ctype[2] = "FREQ"
        self.original_wcs_3d.wcs.crval[2] = 1e10
        self.original_wcs_3d.wcs.cdelt[2] = 1e9
        self.original_wcs_3d.wcs.crpix[2] = 1
        self.original_wcs_3d._naxis = list(self.original_array_3d.shape[::-1])

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.parametrize("frame", ("galactic", "equatorial"))
    @pytest.mark.parametrize("level", (0, 1))
    @pytest.mark.remote_data
    def test_roundtrip(self, tmp_path, frame, level):

        output_directory = tmp_path / "roundtrip"

        # Note that we always use level=1 to generate, but use a variable level
        # to construct the dask array - this is deliberate and ensure that the
        # dask array has a proper separation of maximum and current level.
        reproject_to_hips(
            (self.original_array, self.original_wcs),
            coord_system_out=frame,
            level=1,
            level_depth=6,
            reproject_function=reproject_interp,
            output_directory=output_directory,
            tile_size=256,
        )

        # Represent the HiPS as a dask array
        dask_array, wcs = hips_as_dask_array(output_directory, level=level)

        # Reproject back to the original WCS
        final_array, footprint = reproject_interp(
            (dask_array, wcs),
            self.original_wcs,
            shape_out=self.original_array.shape,
        )

        # FIXME: Due to boundary effects and the fact there are NaN values in
        # the whole-map dask array, there are a few NaN pixels in the image in
        # the end. For now, we tolerate a small fraction of NaN pixels, and to
        # fix this we should modify the dask array so that in empty tiles
        # adjacent to non-empty tiles, we set the values to the boundaries of
        # the non-empty neighbouring tiles so that interpolation doesn't run
        # into any issues. In theory there should be around 90500 pixels inside
        # the valid region of the image, so we require at least 90000 valid
        # values.

        valid = ~np.isnan(final_array)
        assert np.sum(valid) > 90000
        np.testing.assert_allclose(final_array[valid], self.original_array[valid], rtol=0.01)

    @pytest.mark.remote_data
    def test_level_validation(self, tmp_path):

        output_directory = tmp_path / "levels"

        reproject_to_hips(
            (self.original_array, self.original_wcs),
            coord_system_out="equatorial",
            level=1,
            reproject_function=reproject_interp,
            output_directory=output_directory,
            tile_size=32,
        )

        dask_array, wcs = hips_as_dask_array(output_directory, level=0)
        assert dask_array.shape == (160, 160)

        dask_array, wcs = hips_as_dask_array(output_directory, level=1)
        assert dask_array.shape == (320, 320)

        dask_array, wcs = hips_as_dask_array(output_directory)
        assert dask_array.shape == (320, 320)

        with pytest.raises(Exception, match=r"does not contain spatial level 2 data"):
            hips_as_dask_array(output_directory, level=2)

        with pytest.raises(Exception, match=r"should be positive"):
            hips_as_dask_array(output_directory, level=-1)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.parametrize("frame", ("galactic", "equatorial"))
    @pytest.mark.parametrize("level", (0, 1))
    @pytest.mark.remote_data
    def test_roundtrip_3d(self, tmp_path, frame, level):

        output_directory = tmp_path / "roundtrip"

        # Note that we always use level=1 to generate, but use a variable level
        # to construct the dask array - this is deliberate and ensure that the
        # dask array has a proper separation of maximum and current level.
        reproject_to_hips(
            (self.original_array_3d, self.original_wcs_3d),
            coord_system_out=frame,
            level=1,
            reproject_function=reproject_interp,
            output_directory=output_directory,
            tile_size=32,
            tile_depth=8,
        )

        # Represent the HiPS as a dask array
        dask_array, wcs = hips_as_dask_array(output_directory, level=level)

        # FIXME: at this point we should be able to do:
        #
        # Reproject back to the original WCS
        # final_array, footprint = reproject_interp(
        #     (dask_array, wcs),
        #     self.original_wcs_3d,
        #     shape_out=self.original_array_3d.shape,
        # )
        #
        # However this does not work properly due to this issue:
        # https://github.com/astropy/astropy/issues/18690
        #
        # For now, we pick a sub-region of the array to check

        subset = (slice(None), slice(50, None), slice(50, None))

        final_array, footprint = reproject_interp(
            (dask_array, wcs),
            self.original_wcs_3d[subset],
            shape_out=self.original_array_3d[subset].shape,
        )

        # NOTE: The two last channels are empty - this is normal and is because
        # of the interpolation on the spectral grid

        valid = ~np.isnan(final_array)[:8]
        assert np.sum(valid) > 450000  # similar to 2D test
        np.testing.assert_allclose(
            final_array[:8][valid],
            self.original_array_3d[subset][:8][valid],
            rtol=0.1 if level == 1 else 0.4,
        )


CUBE_HEADER = """
WCSAXES = 3
CRPIX1  = 16.
CRPIX2  = 16.
CRPIX3  = 1.
CDELT1  = -0.15
CDELT2  = 0.15
CDELT3  = {cdelt3}
CUNIT1  = 'deg'
CUNIT2  = 'deg'
CUNIT3  = 'Hz'
CTYPE1  = 'RA---SIN'
CTYPE2  = 'DEC--SIN'
CTYPE3  = 'FREQ-LOG'
CRVAL1  = 50.
CRVAL2  = 70.
CRVAL3  = 1.e9
SPECSYS = 'LSRK'
"""


def _make_2d_hips(tmp_path):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "RA---TAN", "DEC--TAN"
    wcs.wcs.crpix = 15, 15
    wcs.wcs.crval = 50, 70
    wcs.wcs.cdelt = -0.01, 0.01
    output_directory = tmp_path / "h2d"
    reproject_to_hips(
        (np.ones((30, 40)), wcs),
        coord_system_out="equatorial",
        level=5,
        reproject_function=reproject_interp,
        output_directory=output_directory,
    )
    return output_directory


def _make_3d_hips(tmp_path):
    nz = 8
    cdelt3 = 1e9 * np.log(10**4) / (nz - 1)
    header = Header.fromstring(CUBE_HEADER.format(cdelt3=cdelt3), sep="\n")
    cube = np.arange(nz * 32 * 32).reshape(nz, 32, 32).astype(float)
    output_directory = tmp_path / "h3d"
    reproject_to_hips(
        (cube, WCS(header)),
        coord_system_out="equatorial",
        reproject_function=reproject_interp,
        output_directory=output_directory,
        tile_size=16,
        tile_depth=8,
        level=4,
        level_depth=1,
    )
    return output_directory


def test_dask_array_uses_moc_2d(tmp_path):
    output_directory = _make_2d_hips(tmp_path)

    array = HiPSArray(output_directory)
    assert type(array._moc).__name__ == "MOC"

    covered = sorted(
        find_indices(output_directory=output_directory, ndim=2, spatial_level=5, level_depth=None)
    )
    far = (int(covered[0]) + 6000) % (12 * 4**5)

    assert array._in_coverage(int(covered[0])) is True
    assert array._in_coverage(far) is False

    # An out-of-coverage tile returns the blank (NaN) tile, an in-coverage one has data
    assert np.all(np.isnan(array._get_tile(level=array._level, index=far)))
    assert np.any(np.isfinite(array._get_tile(level=array._level, index=int(covered[0]))))


def test_dask_array_uses_moc_3d(tmp_path):
    output_directory = _make_3d_hips(tmp_path)

    array = HiPSArray(output_directory)
    assert type(array._moc).__name__ == "SFMOC"

    covered = sorted(
        find_indices(output_directory=output_directory, ndim=3, spatial_level=4, level_depth=1)
    )
    far = ((covered[0][0] + 5000) % (12 * 4**4), covered[0][1])

    assert array._in_coverage(covered[0]) is True
    assert array._in_coverage(far) is False
    assert np.all(np.isnan(array._get_tile(level=array._level, index=far)))


def test_dask_array_moc_skips_filesystem(tmp_path, monkeypatch):
    # Out-of-coverage tiles should be skipped without any attempt to read them.
    output_directory = _make_2d_hips(tmp_path)
    array = HiPSArray(output_directory)

    covered = sorted(
        find_indices(output_directory=output_directory, ndim=2, spatial_level=5, level_depth=None)
    )
    far = (int(covered[0]) + 6000) % (12 * 4**5)

    calls = []
    real_getdata = dask_array_module.fits.getdata
    monkeypatch.setattr(
        dask_array_module.fits, "getdata", lambda *a, **k: calls.append(a) or real_getdata(*a, **k)
    )

    # Out-of-coverage: no data is read from disk
    array._get_tile(level=array._level, index=far)
    assert calls == []

    # In-coverage: the tile is actually read
    array._get_tile(level=array._level, index=int(covered[0]))
    assert len(calls) == 1


def test_dask_array_without_moc(tmp_path):
    # If there is no Moc.fits, the reader still works (coverage check disabled).
    output_directory = _make_2d_hips(tmp_path)
    (output_directory / "Moc.fits").unlink()

    array = HiPSArray(output_directory)
    assert array._moc is None
    assert array._in_coverage(0) is True
