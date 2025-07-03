import os
import urllib

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy_healpix import HEALPix, level_to_nside

from .utils import is_url, load_properties, tile_filename


class HiPSArray:

    def __init__(self, directory_or_url, level=None):

        self._directory_or_url = directory_or_url

        self._is_url = is_url(directory_or_url)

        self._properties = load_properties(directory_or_url)

        self._tile_size = int(self._properties["hips_tile_width"])
        self._order = int(self._properties["hips_order"])
        self._level = self._order if level is None else level

        self._tile_format = self._properties["hips_tile_format"]

        self._nside = level_to_nside(self._level)
        self._hp = HEALPix(nside=self._nside, frame="icrs", order="nested")
        self._cdelt = 45 / self._tile_size / 2**self._level * 2**0.5

        image_size = 5 * self._nside * self._tile_size

        self.wcs = WCS(naxis=2)
        self.wcs.wcs.ctype = "RA---HPX", "DEC--HPX"
        self.wcs.wcs.crval = 0, 0
        self.wcs.wcs.cdelt = self._cdelt, self._cdelt
        self.wcs.wcs.crpix = image_size / 2, image_size / 2
        self.wcs.wcs.crota = 0, 45
        self.wcs.wcs.set()

        self.shape = (image_size, image_size)
        self.dtype = float
        self.ndim = 2

        self.chunksize = (self._tile_size, self._tile_size)

        self._nan = np.nan * np.ones(self.chunksize, dtype=self.dtype)

    def __getitem__(self, item):

        # For now assume item is a list of slices. Find

        imid = (item[0].start + item[0].stop) // 2
        jmid = (item[1].start + item[1].stop) // 2

        # Convert pixel coordinates to HEALPix indices

        index = self._hp.skycoord_to_healpix(self.wcs.pixel_to_world(imid, jmid))

        if index == -1:
            return self._nan

        return self._get_tile(level=self._level, index=index)

    def _get_tile(self, *, level, index):

        filename = tile_filename(
            level=self._level,
            index=index,
            output_directory=self._directory_or_url,
            extension="fits",
        )

        if self._is_url:
            try:
                return fits.getdata(filename).astype(float)[::-1]
            except urllib.error.HTTPError:
                return self._nan
        else:
            if os.path.exists(filename):
                # FIXME: why flip vertically?
                return fits.getdata(filename).astype(float)[::-1]
            else:
                return self._nan
