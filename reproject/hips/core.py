import os
import uuid
from datetime import datetime
from logging import getLogger

import numpy as np
from astropy.coordinates import ICRS, BarycentricTrueEcliptic, Galactic
from astropy.io import fits
from astropy.nddata import block_reduce
from astropy_healpix import HEALPix, level_to_nside
from PIL import Image

from ..utils import as_rgb_images
from .utils import make_tile_folders, tile_filename, tile_header

__all__ = ["image_to_hips"]

INDEX_HTML = """
<!DOCTYPE html>

<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, height=device-height, initial-scale=1.0, user-scalable=no">


    <script src="https://aladin.cds.unistra.fr/hips-templates/hips-landing-page.js" type="text/javascript"></script>
    <noscript>Please enable Javascript to view this page.</noscript>
</head>

<body></body>

<script type="text/javascript">
    buildLandingPage();
</script>

</html>
"""


VALID_COORD_SYSTEM = {
    "equatorial": ICRS(),
    "galactic": Galactic(),
    "ecliptic": BarycentricTrueEcliptic(),
}

VALID_TILE_FORMATS = {"fits", "png", "jpeg"}


def image_to_hips(
    array_in,
    wcs_in,
    coord_system_out,
    *,
    level,
    reproject_function,
    output_directory,
    tile_size,
    tile_format,
    progress_bar=None,
    **kwargs,
):
    """
    Convert image in a normal WCS projection to HiPS tiles.

    Parameters
    ----------
    data : `numpy.ndarray`
        Input data array to reproject
    wcs_in : `~astropy.wcs.WCS`
        The WCS of the input array
    coord_system_out : {'equatorial', 'galactic', 'ecliptic' }
        The target coordinate system for the HEALPIX projection
    level : int
        The number of levels of FITS tiles.
    reproject_function : callable
        The function to use for the reprojection.
    output_directory : str
        The name of the output directory.
    tile_size : int, optional
        The size of each individual tile (defaults to 512).
    tile_format : {'fits', 'png', 'jpeg'}
        The format of the output tiles
    progress_bar : callable, optional
        If specified, use this as a progress_bar to track loop iterations over
        data sets.
    """

    logger = getLogger(__name__)

    # Check tile size is even
    if tile_size % 2 != 0:
        raise ValueError("tile_size should be even")

    # Check coordinate system
    if coord_system_out in VALID_COORD_SYSTEM:
        frame = VALID_COORD_SYSTEM[coord_system_out]
    else:
        raise ValueError("coord_system_out should be one of " + "/".join(VALID_COORD_SYSTEM))

    # Check tile format
    if tile_format not in VALID_TILE_FORMATS:
        raise ValueError("tile_format should be one of " + "/".join(VALID_TILE_FORMATS))

    # Create output directory (and error if it already exists)
    os.makedirs(output_directory, exist_ok=False)

    # Determine center of image and radius to furthest corner, to determine
    # which HiPS tiles need to be generated

    ny, nx = array_in.shape[-2:]

    cen_x, cen_y = (nx - 1) / 2, (ny - 1) / 2

    cor_x = np.array([-0.5, -0.5, nx - 0.5, nx - 0.5])
    cor_y = np.array([-0.5, ny - 0.5, ny - 0.5, -0.5])

    cen_world = wcs_in.pixel_to_world(cen_x, cen_y)
    cor_world = wcs_in.pixel_to_world(cor_x, cor_y)

    radius = cor_world.separation(cen_world).max()

    # TODO: in future if astropy-healpix implements polygon searches, we could
    # use that instead

    # Determine all the indices at the highest level

    nside = level_to_nside(level)
    hp = HEALPix(nside=nside, order="nested", frame=frame)

    indices = hp.cone_search_skycoord(cen_world, radius=radius)

    logger.info(f"Found {len(indices)} tiles (at most) to generate at level {level}")

    # PERF: the code above may be prohibitive for large numbers of tiles,
    # so we may want to find a way to iterate over these in chunks.

    # Make all the folders required for the tiles
    make_tile_folders(level=level, indices=indices, output_directory=output_directory)

    # Iterate over the tiles and generate them
    generated_indices = []
    for index in progress_bar(indices):
        header = tile_header(level=level, index=index, frame=frame, tile_size=tile_size)
        array_out, footprint = reproject_function((array_in, wcs_in), header, **kwargs)
        array_out[np.isnan(array_out)] = 0.0
        if np.all(footprint == 0):
            continue
        if tile_format == "fits":
            fits.writeto(
                tile_filename(
                    level=level,
                    index=index,
                    output_directory=output_directory,
                    extension=tile_format,
                ),
                array_out,
            )
        else:
            image = as_rgb_images(array_out)[0]
            image.save(
                tile_filename(
                    level=level,
                    index=index,
                    output_directory=output_directory,
                    extension=tile_format,
                )
            )

        generated_indices.append(index)

    indices = np.array(generated_indices)

    # Iterate over higher levels and compute lower resolution tiles
    for ilevel in range(level - 1, -1, -1):

        # Find index of tiles to produce at lower-resolution levels
        indices = np.sort(np.unique(indices // 4))

        make_tile_folders(level=ilevel, indices=indices, output_directory=output_directory)

        for index in indices:

            header = tile_header(level=ilevel, index=index, frame=frame, tile_size=tile_size)

            if tile_format == "fits":
                array = np.zeros((tile_size, tile_size))
            else:
                array = np.zeros((tile_size, tile_size, 3))

            for subindex in range(4):

                current_index = 4 * index + subindex
                subtile_filename = tile_filename(
                    level=ilevel + 1,
                    index=current_index,
                    output_directory=output_directory,
                    extension=tile_format,
                )

                if os.path.exists(subtile_filename):

                    if tile_format == "fits":
                        data = block_reduce(fits.getdata(subtile_filename), 2, func=np.mean)
                    else:
                        data = block_reduce(
                            np.array(Image.open(subtile_filename))[::-1], (2, 2, 1), func=np.mean
                        )

                    if subindex == 0:
                        array[256:, :256] = data
                    elif subindex == 2:
                        array[256:, 256:] = data
                    elif subindex == 1:
                        array[:256, :256] = data
                    elif subindex == 3:
                        array[:256, 256:] = data

            if tile_format == "fits":
                fits.writeto(
                    tile_filename(
                        level=ilevel,
                        index=index,
                        output_directory=output_directory,
                        extension=tile_format,
                    ),
                    array,
                    header,
                )
            else:
                image = as_rgb_images(array.transpose(2, 0, 1))[0]
                image.save(
                    tile_filename(
                        level=ilevel,
                        index=index,
                        output_directory=output_directory,
                        extension=tile_format,
                    )
                )

    # Generate properties file

    cen_icrs = cen_world.icrs

    properties = {
        "creator_did": f"ivo://reproject/P/{str(uuid.uuid4())}",
        "obs_title": os.path.dirname(output_directory),
        "dataproduct_type": "image",
        "hips_version": "1.4",
        "hips_release_date": datetime.now().isoformat(),
        "hips_status": "public master clonableOnce",
        "hips_tile_format": tile_format,
        "hips_tile_width": tile_size,
        "hips_order": level,
        "hips_frame": coord_system_out,
        "hips_builder": "astropy/reproject",
        "hips_initial_ra": cen_icrs.ra.deg,
        "hips_initial_dec": cen_icrs.dec.deg,
        "hips_initial_fov": radius.deg,
    }

    if tile_format == "fits":
        properties["hips_pixel_bitpix"] = -64

    with open(os.path.join(output_directory, "properties"), "w") as f:
        for key, value in properties.items():
            f.write(f"{key:20s} = {value}\n")

    with open(os.path.join(output_directory, "index.html"), "w") as f:
        f.write(INDEX_HTML)
