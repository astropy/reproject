import math
import os
import warnings

import numpy as np
from astropy.io import fits
from astropy.nddata import block_reduce
from PIL import Image

from ._utils import tile_filename

__all__ = ["save_allsky"]


# Per the HiPS standard, Allsky preview files are only generated for the low
# orders (0 to 3), and each tile is downsampled to a small size that must stay a
# power of two (typically 64 pixels).
ALLSKY_MAX_ORDER = 3
ALLSKY_TILE_SIZE = 64


def _reduction_factor(tile_size):
    # Largest power-of-two factor that divides tile_size and keeps the
    # downsampled tile at least ALLSKY_TILE_SIZE pixels across.
    factor = 1
    while tile_size // (factor * 2) >= ALLSKY_TILE_SIZE and tile_size % (factor * 2) == 0:
        factor *= 2
    return factor


def save_allsky(*, output_directory, tile_format, extension, tile_size, spatial_level):
    """
    Write the ``Allsky`` preview files for a 2-d image HiPS.

    For each order between 0 and 3 (inclusive, and no deeper than the deepest
    order of the HiPS), all the tiles of that order are packed, downsampled, into
    a single ``Norder{order}/Allsky.{ext}`` mosaic. The tiles are laid out side
    by side in HEALPix index order, with a width of ``floor(sqrt(n_tiles))``, as
    described by the HiPS standard.
    """

    factor = _reduction_factor(tile_size)
    allsky_tile = tile_size // factor

    for order in range(min(ALLSKY_MAX_ORDER, spatial_level) + 1):

        n_tiles = 12 * 4**order
        width = math.isqrt(n_tiles)  # floor(sqrt(n_tiles))
        height = math.ceil(n_tiles / width)

        if tile_format == "fits":
            mosaic = np.full((height * allsky_tile, width * allsky_tile), np.nan, dtype=np.float32)
        else:
            channels = 4 if tile_format == "png" else 3
            mosaic = np.zeros((height * allsky_tile, width * allsky_tile, channels), dtype=np.uint8)

        found = False
        for index in range(n_tiles):

            filename = tile_filename(
                level=order, index=index, output_directory=output_directory, extension=extension
            )
            if not os.path.exists(filename):
                continue
            found = True

            row, col = divmod(index, width)
            ys = slice(row * allsky_tile, (row + 1) * allsky_tile)
            xs = slice(col * allsky_tile, (col + 1) * allsky_tile)

            if tile_format == "fits":
                data = fits.getdata(filename)
                if factor > 1:
                    # block_reduce with nanmean can warn on all-NaN tiles
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        data = block_reduce(data, factor, func=np.nanmean)
                mosaic[ys, xs] = data
            else:
                image = Image.open(filename).convert("RGBA" if tile_format == "png" else "RGB")
                if factor > 1:
                    image = image.reduce(factor)
                mosaic[ys, xs] = np.asarray(image)

        if not found:
            continue

        allsky_filename = os.path.join(output_directory, f"Norder{order}", f"Allsky.{extension}")
        if tile_format == "fits":
            fits.writeto(allsky_filename, mosaic, overwrite=True)
        else:
            image = Image.fromarray(mosaic)
            if tile_format == "jpeg":
                image = image.convert("RGB")
            image.save(allsky_filename)
