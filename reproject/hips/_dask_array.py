import functools
import os
import urllib
import uuid

import numpy as np
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.wcs import WCS
from astropy_healpix import HEALPix, level_to_nside
from dask import array as da

from .high_level import VALID_COORD_SYSTEM
from .utils import is_url, load_properties, map_header, tile_filename

__all__ = ["hips_as_dask_array"]


class HiPSArray:

    def __init__(self, directory_or_url, level=None):

        self._directory_or_url = directory_or_url

        self._is_url = is_url(directory_or_url)

        self._properties = load_properties(directory_or_url)

        self._tile_width = int(self._properties["hips_tile_width"])
        self._order = int(self._properties["hips_order"])
        if level is None:
            self._level = self._order
        else:
            if level > self._order:
                raise ValueError(
                    f"HiPS dataset at {directory_or_url} does not contain level {level} data"
                )
            elif level < 0:
                raise ValueError("level should be positive")
            else:
                self._level = int(level)
        self._level = self._order if level is None else level
        self._tile_format = self._properties["hips_tile_format"]
        self._frame_str = self._properties["hips_frame"]
        self._frame = VALID_COORD_SYSTEM[self._frame_str]

        self._hp = HEALPix(nside=level_to_nside(self._level), frame=self._frame, order="nested")

        self._header = map_header(level=self._level, frame=self._frame, tile_size=self._tile_width)

        self.wcs = WCS(self._header)
        self.shape = self.wcs.array_shape

        self.dtype = float
        self.ndim = 2

        self.chunksize = (self._tile_width, self._tile_width)

        self._nan = np.nan * np.ones(self.chunksize, dtype=self.dtype)

        self._blank = np.broadcast_to(np.nan, self.shape)

    def __getitem__(self, item):

        if item[0].start == item[0].stop or item[1].start == item[1].stop:
            return self._blank[item]

        # We use two points in different parts of the image because in some
        # cases using the exact center or corners can cause issues.

        istart = item[0].start
        irange = item[0].stop - item[0].start
        imid = np.array([istart + 0.25 * irange, istart + 0.75 * irange])

        jstart = item[1].start
        jrange = item[1].stop - item[1].start
        jmid = np.array([jstart + 0.25 * jrange, jstart + 0.75 * jrange])

        # Convert pixel coordinates to HEALPix indices

        coord = self.wcs.pixel_to_world(jmid, imid)

        if self._frame_str == "equatorial":
            lon, lat = coord.ra.deg, coord.dec.deg
        elif self._frame_str == "galactic":
            lon, lat = coord.l.deg, coord.b.deg
        else:
            raise NotImplementedError()

        invalid = np.isnan(lon) | np.isnan(lat)

        if np.all(invalid):
            return self._nan
        elif np.any(invalid):
            coord = coord[~invalid]

        index = self._hp.skycoord_to_healpix(coord)

        if np.all(index == -1):
            return self._nan

        index = np.max(index)

        return self._get_tile(level=self._level, index=index)

    @functools.lru_cache(maxsize=128)  # noqa: B019
    def _get_tile(self, *, level, index):

        filename_or_url = tile_filename(
            level=self._level,
            index=index,
            output_directory=self._directory_or_url,
            extension="fits",
        )

        if self._is_url:
            try:
                filename = download_file(filename_or_url, cache=True)
            except urllib.error.HTTPError:
                return self._nan
        elif not os.path.exists(filename_or_url):
            return self._nan
        else:
            filename = filename_or_url

        with fits.open(filename) as hdulist:
            hdu = hdulist[0]
            data = hdu.data

        return data


def hips_as_dask_array(directory_or_url, *, level=None):
    """
    Return a dask array and WCS that represent a HiPS dataset at a particular level.
    """
    array_wrapper = HiPSArray(directory_or_url, level=level)
    return (
        da.from_array(
            array_wrapper,
            chunks=array_wrapper.chunksize,
            name=str(uuid.uuid4()),
            meta=np.array([], dtype=float),
        ),
        array_wrapper.wcs,
    )
