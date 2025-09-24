import os
import urllib

import numpy as np
from astropy.wcs.utils import celestial_frame_to_wcs
from astropy_healpix import (
    HEALPix,
    level_to_nside,
)

__all__ = [
    "tile_header",
    "tile_filename",
    "tile_filename_3d",
    "make_tile_folders",
    "is_url",
]


def tile_header(*, level, index, frame, tile_size):
    """
    Return the WCS for a given HiPS tile
    """

    # PERF: we could optimize this by making it into a class, as very little
    # changes from tile to tile so we could first create the base tile header
    # and then just update values for each tile for performance.

    nside = level_to_nside(level)
    hp = HEALPix(nside=nside, order="nested", frame=frame)

    tile_wcs = celestial_frame_to_wcs(frame, projection="HPX")

    # Determine tile resolution
    res = 45 / tile_size / 2**level  # degrees
    tile_wcs.wcs.cd = [[-res, -res], [res, -res]]

    # Set PV parameters to default values
    tile_wcs.wcs.set_pv([(2, 1, 4), (2, 2, 3)])

    # Determine CRPIX values by determining the position of the relevant corner
    # relative to the origin of the projection.
    offset_x, offset_y = tile_wcs.world_to_pixel(hp.healpix_to_skycoord(index, dx=1, dy=0))

    tile_wcs.wcs.crpix[0] = -offset_x - 0.5
    tile_wcs.wcs.crpix[1] = -offset_y - 0.5

    # Construct header
    header = tile_wcs.to_header()

    header["NPIX"] = index
    header["ORDER"] = level
    header["NAXIS"] = 2
    header["NAXIS1"] = tile_size
    header["NAXIS2"] = tile_size

    return header


def _rounded_spatial_index(index):
    return 10000 * (index // 10000)


def _rounded_spectral_index(index):
    return 10 * (index // 10)


def tile_filename(*, level, index, output_directory, extension):
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

    rounded_indices = np.unique(_rounded_spatial_index(indices))
    for index in rounded_indices:
        dirname = os.path.dirname(
            tile_filename(level=level, index=index, output_directory=output_directory, extension="")
        )
        if not os.path.exists(dirname):
            os.makedirs(dirname)


def make_tile_folders_3d(*, spatial_level, spectral_level, spatial_indices, spectral_indices, output_directory):

    rounded_spatial_indices = np.unique(_rounded_spatial_index(spatial_indices))
    for spatial_index in rounded_spatial_indices:
        dirname = os.path.dirname(
            tile_filename_3d(spatial_level=spatial_level, spectral_level=spectral_level,
            spatial_index=spatial_index, spectral_index=spectral_indices,
            output_directory=output_directory, extension="")
        )
        if not os.path.exists(dirname):
            os.makedirs(dirname)


def is_url(directory):
    return directory.startswith("http://") or directory.startswith("https://")


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
