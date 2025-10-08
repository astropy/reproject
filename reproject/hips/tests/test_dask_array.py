import numpy as np
import pytest
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS

from reproject import reproject_interp
from reproject.hips import reproject_to_hips
from reproject.hips._dask_array import hips_as_dask_array


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

    @pytest.mark.parametrize("frame", ("galactic", "equatorial"))
    @pytest.mark.parametrize("level", (0, 1))
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

    @pytest.mark.parametrize("frame", ("galactic", "equatorial"))
    @pytest.mark.parametrize("level", (0, 1))
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
