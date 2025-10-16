import functools
import os
import urllib
import uuid

import numpy as np
from astropy import units as u
from astropy.coordinates import SpectralCoord
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.wcs import WCS
from astropy_healpix import HEALPix, level_to_nside
from dask import array as da

from ._trim_utils import fits_getdata_untrimmed
from .high_level import VALID_COORD_SYSTEM
from .utils import (
    is_url,
    load_properties,
    map_header,
    skycoord_first,
    spectral_coord_to_index,
    tile_filename,
)

__all__ = ["hips_as_dask_array"]


class HiPSArray:

    def __init__(self, directory_or_url, level=None, level_depth=None):

        self._directory_or_url = directory_or_url

        self._is_url = is_url(directory_or_url)

        self._properties = load_properties(directory_or_url)

        if self._properties["dataproduct_type"] == "image":
            self.ndim = 2
        elif self._properties["dataproduct_type"] == "spectral-cube":
            self.ndim = 3
        else:
            raise TypeError(f"HiPS type {self._properties['dataproduct_type']} not recognized")

        self._tile_width = int(self._properties["hips_tile_width"])
        self._order_spatial = int(self._properties["hips_order"])

        if level is None:
            self._level_spatial = self._order_spatial
        else:
            if level > self._order_spatial:
                raise ValueError(
                    f"HiPS dataset at {directory_or_url} does not contain spatial level {level} data"
                )
            elif level < 0:
                raise ValueError("level should be positive")
            else:
                self._level_spatial = int(level)

        if self.ndim == 3:

            # TODO: here need to check consistency, maybe actually don't allow spectral level to be passed in

            self._tile_depth = int(self._properties["hips_tile_depth"])
            self._order_depth = int(self._properties["hips_order_freq"])

            if level_depth is None:
                self._level_depth = self._order_depth - (self._order_spatial - self._level_spatial)
            else:
                if level_depth > self._order_depth:
                    raise ValueError(
                        f"HiPS dataset at {directory_or_url} does not contain spectral level {level_depth} data"
                    )
                elif level_depth < 0:
                    raise ValueError("level_depth should be positive")
                else:
                    self._level_depth = int(level_depth)

            self._level = (self._level_spatial, self._level_depth)
            self._tile_dims = (self._tile_width, self._tile_depth)

        else:

            self._level_depth = None
            self._level = self._level_spatial
            self._tile_dims = self._tile_width

        self._tile_format = self._properties["hips_tile_format"]
        self._frame_str = self._properties["hips_frame"]
        self._frame = VALID_COORD_SYSTEM[self._frame_str]

        self._hp = HEALPix(
            nside=level_to_nside(self._level_spatial), frame=self._frame, order="nested"
        )

        self._header = map_header(level=self._level, frame=self._frame, tile_dims=self._tile_dims)

        self.wcs = WCS(self._header)
        self.shape = self.wcs.array_shape

        # Determine actual spectral range, because we don't actually want to
        # create a dask array with the full possible range of spectral indices
        # since this will be huge and unnecessary

        if self.ndim == 3:

            wav_min = SpectralCoord(float(self._properties["em_min"]), u.m)
            wav_max = SpectralCoord(float(self._properties["em_max"]), u.m)

            index_min = spectral_coord_to_index(self._level_depth, wav_min)
            index_max = spectral_coord_to_index(self._level_depth, wav_max)

            if index_min > index_max:
                index_min, index_max = index_max, index_min

            index_max += 1

            index_min *= self._tile_depth
            index_max *= self._tile_depth

            self.wcs = self.wcs[index_min:index_max]
            self.shape = (index_max - index_min,) + self.shape[1:]

        # FIX following
        self.dtype = float

        if self.ndim == 2:
            self.chunksize = (self._tile_width, self._tile_width)
        else:
            self.chunksize = (self._tile_depth, self._tile_width, self._tile_width)

        self._nan = np.nan * np.ones(self.chunksize, dtype=self.dtype)

        self._blank = np.broadcast_to(np.nan, self.shape)

    def __getitem__(self, item):

        for idx in range(self.ndim):
            if item[idx].start == item[idx].stop:
                return self._blank[item]

        # Determine spatial healpix index - we use two points in different
        # parts of the image because in some cases using the exact center or
        # corners can cause issues.

        istart = item[-2].start
        irange = item[-2].stop - item[-2].start
        imid = np.array([istart + 0.25 * irange, istart + 0.75 * irange])

        jstart = item[-1].start
        jrange = item[-1].stop - item[-1].start
        jmid = np.array([jstart + 0.25 * jrange, jstart + 0.75 * jrange])

        # Convert pixel coordinates to HEALPix indices

        if self.ndim == 2:
            coord = self.wcs.pixel_to_world(jmid, imid)
        else:
            kmid = 0.5 * (item[0].start + item[0].stop)
            coord, spectral_coord = skycoord_first(self.wcs.pixel_to_world(jmid, imid, kmid))

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

        spatial_index = self._hp.skycoord_to_healpix(coord)

        if np.all(spatial_index == -1):
            return self._nan

        spatial_index = np.max(spatial_index)

        # Determine spectral index, if needed
        if self.ndim == 3:
            spectral_index = spectral_coord_to_index(self._level_depth, spectral_coord).max()
            index = (spatial_index, spectral_index)
        else:
            index = spatial_index

        return self._get_tile(level=self._level, index=index).astype(float)

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

        if self.ndim == 2:
            return fits.getdata(filename)
        else:
            return fits_getdata_untrimmed(
                filename,
                tile_size=self._tile_width,
                tile_depth=self._tile_depth,
            )


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
