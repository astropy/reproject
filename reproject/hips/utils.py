import os
import urllib
from numbers import Number
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, SpectralCoord
from astropy.wcs.utils import celestial_frame_to_wcs
from astropy_healpix import (
    HEALPix,
    level_to_nside,
)

__all__ = [
    "map_header",
    "tile_header",
    "tile_filename",
    "make_tile_folders",
    "is_url",
    "load_properties",
    "save_properties",
    "spectral_coord_to_index",
    "spectral_index_to_coord",
    "skycoord_first",
]


FREQ_MIN = 1e-18  # Hz
FREQ_MAX = 1e38  # Hz
FREQ_MAX_ORDER = 51


def skycoord_first(worlds):
    """
    Convenience function which takes the output of pixel_to_world and puts
    the SkyCoord first
    """
    for w in worlds:
        if isinstance(w, SkyCoord):
            yield w
    for w in worlds:
        if not isinstance(w, SkyCoord):
            yield w


def spectral_index_to_coord(level, index):
    return SpectralCoord(
        10 ** (index / 2 ** (level + 1) * np.log10(FREQ_MAX / FREQ_MIN) + np.log10(FREQ_MIN)) * u.Hz
    )


def spectral_coord_to_index(level, coord):
    return (
        np.floor(
            2 ** (level + 1)
            * np.log10(coord.to_value(u.Hz) / FREQ_MIN)
            / np.log10(FREQ_MAX / FREQ_MIN)
        )
    ).astype(int)


def map_header(*, level, frame, tile_dims):
    if isinstance(level, Number):
        return map_header_2d(
            level=level,
            frame=frame,
            tile_size=tile_dims,
        )
    else:
        return map_header_3d(
            spatial_level=level[0],
            spectral_level=level[1],
            frame=frame,
            tile_size=tile_dims[0],
            tile_depth=tile_dims[1],
        )


def map_header_2d(*, level, frame, tile_size):
    """
    Return the WCS for a whole map stored as a 2D array in HPX projection
    """

    nside = level_to_nside(level)

    # Determine image size
    image_size = 5 * nside * tile_size

    map_wcs = celestial_frame_to_wcs(frame, projection="HPX")
    map_wcs.wcs.crval = 0.0, 0.0

    # Determine map resolution
    res = 45 / tile_size / 2**level
    map_wcs.wcs.cd = [[-res, -res], [res, -res]]

    # Set PV parameters to default values
    map_wcs.wcs.set_pv([(2, 1, 4), (2, 2, 3)])

    # Set origin to center of the image
    map_wcs.wcs.crpix = image_size / 2 + 0.5, image_size / 2 + 0.5

    # Construct header
    header = map_wcs.to_header()

    header["NAXIS"] = 2
    header["WCSAXES"] = 2
    header["NAXIS1"] = image_size
    header["NAXIS2"] = image_size

    return header


def map_header_3d(
    *,
    spatial_level,
    spectral_level,
    frame,
    tile_size,
    tile_depth,
):
    """
    Return the WCS for a whole map stored as a 3D array in HPX projection
    """

    # First get the 2D header
    header = map_header_2d(level=spatial_level, frame=frame, tile_size=tile_size)

    # Then modify it to be 3D
    header["NAXIS"] = 3
    header["WCSAXES"] = 3
    header["NAXIS3"] = tile_depth * 2 ** (spectral_level + 1)
    header["FORDER"] = spectral_level
    header["CTYPE3"] = "FREQ-LOG"
    header["CUNIT3"] = "Hz"
    header["CRPIX3"] = 1
    header["CRVAL3"] = FREQ_MIN
    header["CDELT3"] = (
        FREQ_MIN * np.log(FREQ_MAX / FREQ_MIN) / 2 ** (spectral_level + 1 + np.log2(tile_depth))
    )

    return header


def tile_header(*, level, index, frame, tile_dims):
    if isinstance(level, Number):
        return tile_header_2d(
            level=level,
            index=index,
            frame=frame,
            tile_size=tile_dims,
        )
    else:
        return tile_header_3d(
            spatial_level=level[0],
            spatial_index=index[0],
            spectral_level=level[1],
            spectral_index=index[1],
            frame=frame,
            tile_size=tile_dims[0],
            tile_depth=tile_dims[1],
        )


def tile_header_2d(*, level, index, frame, tile_size):
    """
    Return the WCS for a given HiPS tile
    """

    # PERF: we could optimize this by making it into a class, as very little
    # changes from tile to tile so we could first create the base tile header
    # and then just update values for each tile for performance.

    nside = level_to_nside(level)

    hp = HEALPix(nside=nside, order="nested", frame=frame)

    tile_wcs = celestial_frame_to_wcs(frame, projection="HPX")
    tile_wcs.wcs.crval = 0.0, 0.0

    # Determine tile resolution
    res = 45 / tile_size / 2**level  # degrees
    tile_wcs.wcs.cd = [[-res, -res], [res, -res]]

    # Set PV parameters to default values
    tile_wcs.wcs.set_pv([(2, 1, 4), (2, 2, 3)])

    # Determine CRPIX values by determining the position of the relevant corner
    # relative to the origin of the projection.
    offset_x, offset_y = tile_wcs.world_to_pixel(
        hp.healpix_to_skycoord(index, dx=[0.5, 0.9, 0.9, 0.1, 0.1], dy=[0.5, 0.1, 0.9, 0.9, 0.1])
    )
    border_tile = (
        np.max(np.hypot(offset_x[1:] - offset_x[0], offset_y[1:] - offset_y[0])) > tile_size
    )

    offset_x, offset_y = tile_wcs.world_to_pixel(hp.healpix_to_skycoord(index, dx=0.75, dy=0.25))

    tile_wcs.wcs.crpix[0] = -offset_x - 0.5 + tile_size / 4
    tile_wcs.wcs.crpix[1] = -offset_y - 0.5 + tile_size / 4

    # Construct header
    header = tile_wcs.to_header()

    header["NPIX"] = index
    header["ORDER"] = level
    header["NAXIS"] = 2
    header["WCSAXES"] = 2
    header["NAXIS1"] = tile_size
    header["NAXIS2"] = tile_size

    if border_tile:
        header2 = header.copy()
        header2["CRPIX1"] = -header["CRPIX2"] + tile_size + 1
        header2["CRPIX2"] = -header["CRPIX1"] + tile_size + 1
        return header, header2
    else:
        return header


def tile_header_3d(
    *,
    spatial_level,
    spatial_index,
    spectral_level,
    spectral_index,
    frame,
    tile_size,
    tile_depth,
):

    # First get the 2D header
    headers = tile_header_2d(
        level=spatial_level, index=spatial_index, frame=frame, tile_size=tile_size
    )

    if not isinstance(headers, tuple):
        headers = (headers,)

    for header in headers:

        # Then modify it to be 3D
        header["NAXIS"] = 3
        header["WCSAXES"] = 3
        header["NAXIS3"] = tile_depth
        header["FORDER"] = spectral_level
        header["FPIX"] = spectral_index
        header["CTYPE3"] = "FREQ-LOG"
        header["CUNIT3"] = "Hz"
        header["CRPIX3"] = 1 - spectral_index * tile_depth
        header["CRVAL3"] = FREQ_MIN
        header["CDELT3"] = (
            FREQ_MIN * np.log(FREQ_MAX / FREQ_MIN) / 2 ** (spectral_level + 1 + np.log2(tile_depth))
        )

    if len(headers) == 1:
        return headers[0]
    else:
        return headers


def _rounded_spatial_index(index):
    return 10000 * (index // 10000)


def _rounded_spectral_index(index):
    return 10 * (index // 10)


def tile_filename(*, level, index, output_directory, extension):
    if isinstance(level, Number):
        return tile_filename_2d(
            level=level, index=index, output_directory=output_directory, extension=extension
        )
    else:
        return tile_filename_3d(
            spatial_level=level[0],
            spatial_index=index[0],
            spectral_level=level[1],
            spectral_index=index[1],
            output_directory=output_directory,
            extension=extension,
        )


def tile_filename_2d(*, level, index, output_directory, extension):
    return os.path.join(
        output_directory,
        f"Norder{level}",
        f"Dir{_rounded_spatial_index(index)}",
        f"Npix{index}.{extension}",
    )


def tile_filename_3d(
    *, spatial_level, spectral_level, spatial_index, spectral_index, output_directory, extension
):
    return os.path.join(
        output_directory,
        f"Norder{spatial_level}_{spectral_level}",
        f"Dir{_rounded_spatial_index(spatial_index)}_{_rounded_spectral_index(spectral_index)}",
        f"Npix{spatial_index}_{spectral_index}.{extension}",
    )


def make_tile_folders(*, level, indices, output_directory):
    if isinstance(level, Number):
        make_tile_folders_2d(
            level=level,
            indices=indices,
            output_directory=output_directory,
        )
    else:
        indices = list(zip(*indices, strict=False))
        make_tile_folders_3d(
            spatial_level=level[0],
            spectral_level=level[1],
            spatial_indices=np.array(indices[0]),
            spectral_indices=np.array(indices[1]),
            output_directory=output_directory,
        )


def make_tile_folders_2d(*, level, indices, output_directory):

    rounded_indices = np.unique(_rounded_spatial_index(indices))
    for index in rounded_indices:
        dirname = os.path.dirname(
            tile_filename(level=level, index=index, output_directory=output_directory, extension="")
        )
        if not os.path.exists(dirname):
            os.makedirs(dirname)


def is_url(directory):
    if isinstance(directory, Path):
        return False
    else:
        return directory.startswith("http://") or directory.startswith("https://")


def make_tile_folders_3d(
    *, spatial_level, spectral_level, spatial_indices, spectral_indices, output_directory
):
    rounded_spatial_indices = np.unique(_rounded_spatial_index(spatial_indices))
    for spatial_index in rounded_spatial_indices:
        for spectral_index in np.unique(_rounded_spectral_index(spectral_indices)):
            dirname = os.path.dirname(
                tile_filename_3d(
                    spatial_level=spatial_level,
                    spectral_level=spectral_level,
                    spatial_index=spatial_index,
                    spectral_index=spectral_index,
                    output_directory=output_directory,
                    extension="",
                )
            )
            if not os.path.exists(dirname):
                os.makedirs(dirname)


def save_properties(directory, properties):
    with open(os.path.join(directory, "properties"), "w") as f:
        for key, value in properties.items():
            f.write(f"{key:20s} = {value}\n")


def load_properties(directory_or_url):

    if is_url(directory_or_url):
        properties_filename, _ = urllib.request.urlretrieve(f"{directory_or_url}/properties")
    else:
        properties_filename = os.path.join(directory_or_url, "properties")

    properties = {}
    with open(properties_filename) as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            key, value = line.split("=", 1)
            properties[key.strip()] = value.strip()

    return properties
