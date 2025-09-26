import os
import urllib
from pathlib import Path

import numpy as np
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
]


def map_header(*, level, frame, tile_size):
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
    header["NAXIS1"] = image_size
    header["NAXIS2"] = image_size

    return header


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
    header["NAXIS1"] = tile_size
    header["NAXIS2"] = tile_size

    if border_tile:
        header2 = header.copy()
        header2["CRPIX1"] = -header["CRPIX2"] + tile_size + 1
        header2["CRPIX2"] = -header["CRPIX1"] + tile_size + 1
        return header, header2
    else:
        return header


def _rounded_index(index):
    return 10000 * (index // 10000)


def tile_filename(*, level, index, output_directory, extension):
    return os.path.join(
        output_directory,
        f"Norder{level}",
        f"Dir{_rounded_index(index)}",
        f"Npix{index}.{extension}",
    )


def make_tile_folders(*, level, indices, output_directory):

    rounded_indices = np.unique(_rounded_index(indices))
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
