import os

import numpy as np
from astropy.wcs.utils import celestial_frame_to_wcs
from astropy_healpix import (
    HEALPix,
    level_to_nside,
)

__all__ = ["tile_header", "tile_filename", "make_tile_folders"]


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
