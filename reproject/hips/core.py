import os
import uuid
from datetime import datetime
from logging import getLogger

import numpy as np
from astropy.coordinates import ICRS, BarycentricTrueEcliptic, Galactic
from astropy.io import fits
from astropy_healpix import HEALPix, level_to_nside

from .utils import make_tile_folders, tile_filename, tile_header

__all__ = ["image_to_hips"]


VALID_COORD_SYSTEM = {
    'equatorial': ICRS(),
    'galactic': Galactic(),
    'ecliptic': BarycentricTrueEcliptic(),
}

def image_to_hips(
    array_in,
    wcs_in,
    coord_system_out,
    *,
    level,
    reproject_function,
    output_directory,
    tile_size,
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

    # Create output directory (and error if it already exists)
    os.makedirs(output_directory, exist_ok=False)

    # Determine center of image and radius to furthest corner, to determine
    # which HiPS tiles need to be generated

    ny, nx = array_in.shape

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
        fits.writeto(tile_filename(level=level, index=index, output_directory=output_directory), array_out)
        generated_indices.append(index)

    indices = np.array(generated_indices)

    # Iterate over higher levels and compute lower resolution tiles
    for ilevel in range(level - 1, -1, -1):

        # Find index of tiles to produce at lower-resolution levels
        indices = np.sort(np.unique(indices // 4))

        make_tile_folders(level=ilevel, indices=indices, output_directory=output_directory)

        for index in indices:

            header = tile_header(level=ilevel, index=index, frame=frame, tile_size=tile_size)

            array = np.zeros((tile_size, tile_size))

            for subindex in range(4):

                current_index = 4 * index + subindex
                subtile_filename = tile_filename(level=ilevel+1, index=current_index, output_directory=output_directory)

                if os.path.exists(subtile_filename):

                    data = fits.getdata(subtile_filename)[::2,::2]

                    if subindex == 0:
                        array[256:, :256] = data
                    elif subindex == 2:
                        array[256:, 256:] = data
                    elif subindex == 1:
                        array[:256, :256] = data
                    elif subindex == 3:
                        array[:256, 256:] = data

            fits.writeto(tile_filename(level=ilevel, index=index, output_directory=output_directory), array, header)

    # Generate properties file

    cen_icrs = cen_world.icrs

    properties = {
        'creator_did': f'ivo://reproject/{str(uuid.uuid4())}',
        'obs_title': 'Placeholder title',
        'dataproduct_type': 'image',
        'hips_version': '1.4',
        'hips_release_date': datetime.now().isoformat(),
        'hips_status': 'public master clonableOnce',
        'hips_tile_format': 'fits',
        'hips_tile_width': tile_size,
        'hips_order': level,
        'hips_frame': coord_system_out,
        'hips_builder': 'astropy/reproject',
        'hips_initial_ra': cen_icrs.ra.deg,
        'hips_initial_dec': cen_icrs.dec.deg,
        'hips_initial_fov': radius.deg,
    }

    with open(os.path.join(output_directory, 'properties'), 'w') as f:
        for key, value in properties.items():
            f.write(f'{key:20s} = {value}\n')
