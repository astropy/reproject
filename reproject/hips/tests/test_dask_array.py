import pytest
import numpy as np

from astropy.wcs import WCS
from astropy.io import fits

from reproject import reproject_interp
from reproject.hips import reproject_to_hips
from reproject.hips._dask_array import hips_as_dask_and_wcs
from astropy.utils.data import get_pkg_data_filename

class TestHIPSDaskArray:

    def setup_method(self):

        hdu = fits.open(get_pkg_data_filename('allsky/allsky_rosat.fits'))[0]
        self.original_header = hdu.header
        self.original_wcs = WCS(hdu.header)
        self.original_array = hdu.data.size + np.arange(hdu.data.size).reshape(hdu.data.shape)

    @pytest.mark.parametrize('frame', ('galactic', 'equatorial'))
    @pytest.mark.parametrize('level',  (0, 1))
    def test_roundtrip(self, tmp_path, frame, level):

        self.output_directory = tmp_path / 'roundtrip'

        reproject_to_hips(
            (self.original_array, self.original_wcs),
            coord_system_out=frame,
            level=level,
            reproject_function=reproject_interp,
            output_directory=self.output_directory,
        )

        dask_array, wcs = hips_as_dask_and_wcs(self.output_directory, level=level)

        final_array, footprint = reproject_interp((dask_array, wcs), self.original_wcs, shape_out=self.original_array.shape)

        # FIXME: Due to boundary effects and the fact there are NaN values in
        # the whole-map dask array, there are a few NaN pixels in the image in
        # the end. For now, we tolerate a small fraction of NaN pixels, and to
        # fix this we should modify the dask array so that in empty tiles
        # adjacent to non-empty tiles, we set the values to the boundaries of
        # the non-empty neighbouring tiles so that interpolation doesn't run
        # into any issues. In theory there should be around 90500 pixels inside
        # the valid region of the image, so we require at least 90400 valid
        # values.

        valid = ~np.isnan(final_array)

        assert np.sum(valid) > 90400

        np.testing.assert_allclose(final_array[valid], self.original_array[valid], rtol=0.01)


    # VALIDATE LEVEL
