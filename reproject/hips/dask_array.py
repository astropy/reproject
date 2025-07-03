import os
import struct
import urllib
import uuid

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy_healpix import HEALPix, level_to_nside
from dask import array as da

from .utils import is_url, load_properties, tile_filename, tile_filename_3d

__all__ = ['hips_as_dask', 'hips3d_as_dask']


class HiPSArray:

    def __init__(self, directory_or_url, level=None):

        self._directory_or_url = directory_or_url

        self._is_url = is_url(directory_or_url)

        self._properties = load_properties(directory_or_url)

        self._tile_width = int(self._properties["hips_tile_width"])
        self._order = int(self._properties["hips_order"])
        self._level = self._order if level is None else level

        self._tile_format = self._properties["hips_tile_format"]

        self._nside = level_to_nside(self._level)
        self._hp = HEALPix(nside=self._nside, frame="icrs", order="nested")
        self._cdelt = 45 / self._tile_width / 2**self._level * 2**0.5

        image_size = 5 * self._nside * self._tile_width

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

        self.chunksize = (self._tile_width, self._tile_width)

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


def freq2pix(order, freq):
    hash_value = get_hash(freq)
    return hash_value >> (59 - order)


def get_hash(param_double):
    l1 = struct.unpack(">q", struct.pack(">d", param_double))[0]
    l2 = (l1 & 0x7FF0000000000000) >> 52
    l2 = (l2 - 929) << 52
    return (l1 & 0x800FFFFFFFFFFFFF) | l2


def pix2freq(order, pix):
    delta_order = 59 - order
    nb_pix_in = pow2(delta_order)
    hash_value = (pix << delta_order) + nb_pix_in // 2
    return get_freq(hash_value)


def pow2(exponent):
    return 1 << exponent


def get_freq(hash_value):
    packed = struct.pack(">q", hash_value)
    return struct.unpack(">d", packed)[0]


class HiPS3DArray:

    def __init__(self, directory_or_url, level=None):

        self._directory_or_url = directory_or_url

        self._is_url = is_url(directory_or_url)

        self._properties = load_properties(directory_or_url)

        assert self._properties["dataproduct_type"] == "spectral-cube"

        self._tile_width = int(self._properties["hips_tile_width"])
        self._tile_depth = int(self._properties["hips_tile_depth"])

        self._order = int(self._properties["hips_order"])
        self._order_freq = int(self._properties["hips_order_freq"])

        # FIXME: for now assume minimum order is 0

        self._level = self._order if level is None else level
        self._level_freq = self._level + (self._order_freq - self._order)

        self._tile_format = self._properties["hips_tile_format"]

        self._nside = level_to_nside(self._level)
        self._hp = HEALPix(nside=self._nside, frame="icrs", order="nested")
        self._cdelt = 45 / self._tile_width / 2**self._level * 2**0.5

        image_size = 5 * self._nside * self._tile_width

        # For the image depth we could in principe do whole spectral domain but
        # this would make too many chunks for dask so we have to be more
        # sensible

        # NOTE: em_min is given as wav but is minimum frequency

        self._freq_min = (float(self._properties["em_min"]) * u.m).to_value(u.Hz, u.spectral())
        self._freq_max = (float(self._properties["em_min"]) * u.m).to_value(u.Hz, u.spectral())

        # Now determine what the indices would be for this at the given spectral order

        self._index_min = freq2pix(self._level_freq, self._freq_min)
        self._index_max = freq2pix(self._level_freq, self._freq_max) + 1

        image_depth = (self._index_max - self._index_min) * self._tile_depth

        # FIXME: make WCS 3D

        self.wcs = WCS(naxis=2)
        self.wcs.wcs.ctype = "RA---HPX", "DEC--HPX"
        self.wcs.wcs.crval = 0, 0
        self.wcs.wcs.cdelt = self._cdelt, self._cdelt
        self.wcs.wcs.crpix = image_size / 2, image_size / 2
        self.wcs.wcs.crota = 0, 45
        self.wcs.wcs.set()

        self.shape = (image_depth, image_size, image_size)
        self.dtype = float
        self.ndim = 2

        self.chunksize = (self._tile_depth, self._tile_width, self._tile_width)

        self._nan = np.nan * np.ones(self.chunksize, dtype=self.dtype)

    def __getitem__(self, item):

        # For now assume item is a list of slices. Find

        ispec = (item[0].start + item[0].stop) // 2
        imid = (item[1].start + item[1].stop) // 2
        jmid = (item[2].start + item[2].stop) // 2

        # Convert pixel coordinates to HEALPix indices

        spatial_index = self._hp.skycoord_to_healpix(self.wcs.pixel_to_world(imid, jmid))

        if spatial_index == -1:
            return self._nan

        # Get spectral index

        spectral_index = ispec // self._tile_depth + self._index_min

        return self._get_tile(
            level=self._level, spatial_index=spatial_index, spectral_index=spectral_index
        )

    def _get_tile(self, *, level, spatial_index, spectral_index):

        filename_or_url = tile_filename_3d(
            spatial_level=self._level,
            spectral_level=self._level_freq,
            spatial_index=spatial_index,
            spectral_index=spectral_index,
            output_directory=self._directory_or_url,
            extension="fits",
        )

        if self._is_url:
            try:
                filename, _ = urllib.request.urlretrieve(filename_or_url)
            except urllib.error.HTTPError:
                return self._nan
        elif not os.path.exists(filename_or_url):
            return self._nan
        else:
            filename = filename_or_url

        with fits.open(filename) as hdulist:

            hdu = hdulist[0]
            data = hdu.data

            if data.shape != self.chunksize:

                # Need to add padding

                before = (hdu.header["TRIM3"], hdu.header["TRIM2"], hdu.header["TRIM1"])
                after = [
                    (c - s - b)
                    for (c, s, b) in zip(self.chunksize, data.shape, before, strict=False)
                ]

                data = np.pad(data, list(zip(before, after, strict=False)), constant_values=np.nan)

            data = data[:, ::-1, :]

            return data


def hips_as_dask(directory_or_url, level=None):
    array_wrapper = HiPSArray(directory_or_url, level=level)
    return da.from_array(
        array_wrapper,
        chunks=array_wrapper.chunksize,
        name=str(uuid.uuid4()),
    )


def hips3d_as_dask(directory_or_url, level=None):
    array_wrapper = HiPS3DArray(directory_or_url, level=level)
    return da.from_array(
        array_wrapper,
        chunks=array_wrapper.chunksize,
        name=str(uuid.uuid4()),
    )
