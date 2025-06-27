import os
import shutil
import uuid
from datetime import datetime
from logging import getLogger

import numpy as np
from astropy.coordinates import ICRS, BarycentricTrueEcliptic, Galactic
from astropy.io import fits
from astropy.nddata import block_reduce
from astropy_healpix import HEALPix, level_to_nside, nside_to_level
from PIL import Image
from astropy import units as u

from ..utils import as_rgb_images, as_transparent_rgb
from .utils import make_tile_folders, tile_filename, tile_header

__all__ = ["image_to_hips", "coadd_hips", "determine_healpix_level"]

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
    reproject_function,
    output_directory,
    tile_size,
    tile_format,
    output_id=None,
    level=None,
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
    level : int, optional
        The number of levels of FITS tiles. If not provided, will be determined
        automatically.
    reproject_function : callable
        The function to use for the reprojection.
    output_id : str, optional
        A unique identifier for the output. If not provided, will be generated
        from the output directory.  This string is the index name in HIPS
        aggregators and generally follows the form 'host/P/name', with host
        being the hosting data source (e.g., CDS) and name being a short descriptive
        name
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

    if progress_bar is None:
        progress_bar = lambda x: x

    if level is None:
        level = determine_healpix_level(wcs_in)

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
        if tile_format != "png":
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
            if tile_format == "png":
                image = as_transparent_rgb(array_out, footprint=footprint)
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
            elif tile_format == "png":
                array = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)
            else:
                array = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)

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
                if tile_format == "png":
                    image = Image.fromarray(array[::-1], mode="RGBA")
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

    if output_id is None:
        creator_did = f"ivo://reproject/P/{str(uuid.uuid4())}"
    else:
        creator_did = f"ivo://{output_id}"

    properties = {
        "creator_did": creator_did,
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

    save_properties(output_directory, properties)

    save_index(output_directory)


def save_index(directory):
    with open(os.path.join(directory, "index.html"), "w") as f:
        f.write(INDEX_HTML)


def save_properties(directory, properties):
    with open(os.path.join(directory, "properties"), "w") as f:
        for key, value in properties.items():
            f.write(f"{key:20s} = {value}\n")


def load_properties(directory):
    properties = {}
    with open(os.path.join(directory, "properties")) as f:
        for line in f:
            key, value = line.split("=")
            properties[key.strip()] = value.strip()
    return properties


def coadd_hips(input_directories, output_directory, overwrite=False):
    """
    Given multiple HiPS directories, combine these into a single HiPS

    The coordinate frame and tile format of the different input directories
    should match.

    In cases of overlap, the last image in the order of input_directories is used.

    Parameters
    ----------
    input_directories : iterable
        Iterable of HiPS directory names
    output_directory : str
        The path to the output directory
    overwrite : bool, optional
        If True, overwrite the output directory if it already exists. Default is
        False.  Overwriting follows the Aladin hipsgen behavior, such that if the
        output directory already exists and contains images, the images there will
        be used as the starting point for coadding (i.e., new images will be put
        on top of them).
    """

    all_properties = [load_properties(directory) for directory in input_directories]

    tile_formats = [p["hips_tile_format"] for p in all_properties]
    hips_frame = [p["hips_frame"] for p in all_properties]
    hips_order = [p["hips_order"] for p in all_properties]

    if len(set(tile_formats)) > 1:
        raise ValueError(f"tile_format values do not match: {tile_formats}")
    else:
        tile_format = tile_formats[0]

    if len(set(hips_frame)) > 1:
        raise ValueError("tile_format values do not match: {hips_frame}")

    reference_properties = all_properties[0]
    reference_properties["hips_order"] = max(hips_order)

    # Create output directory (and error if it already exists)
    os.makedirs(output_directory, exist_ok=overwrite)

    for directory in input_directories:

        for dirpath, _, filenames in os.walk(directory):
            for filename in filenames:
                if not filename.endswith("." + tile_format):
                    continue
                filepath = os.path.join(dirpath, filename)
                target_directory = os.path.join(
                    output_directory, os.path.relpath(dirpath, directory)
                )
                target_filepath = os.path.join(target_directory, filename)
                if not os.path.exists(target_directory):
                    os.makedirs(target_directory)
                if os.path.exists(target_filepath):
                    if tile_format == "png":
                        image1 = Image.open(filepath).convert("RGBA")
                        image2 = Image.open(target_filepath).convert("RGBA")
                        result = Image.alpha_composite(image1, image2)
                        result.save(target_filepath)
                    elif tile_format == "jpeg":
                        raise NotImplementedError("Convert jpg to png to allow for blending/coadding")
                    else:
                        raise NotImplementedError()
                else:
                    shutil.copyfile(filepath, target_filepath)

    save_properties(output_directory, reference_properties)

    save_index(output_directory)


def determine_healpix_level(wcs_in, max_level=25):
    """
    Determine the appropriate HEALPix level by matching the HEALPix pixel size
    to the input image pixel size.

    Parameters
    ----------
    wcs_in : `~astropy.wcs.WCS`
        The WCS of the input array
    max_level : int, optional
        The maximum level to consider. Default is 25, corresponding to 6 mas.
        Can be overridden, but included as as hint to users that images could
        get really huge at this level, as it contains 10^16 pixels over the sky

    Returns
    -------
    level : int
        The recommended HEALPix level
    """

    # Get the pixel scale from the input WCS in degrees
    # We use the geometric mean of the pixel scales in both dimensions
    pixel_scale = wcs_in.proj_plane_pixel_area()

    # HEALPix pixel area is 4π/(12*nside²) steradians
    # nside = 2^level
    # Approximate pixel "size" (side length) is sqrt(area) ≈ sqrt(4π/(12*nside²))
    # We want this to approximately match our input pixel size

    # Solve for nside: sqrt(4π/(12*nside²)) ≈ mean_pixel_scale_rad
    # nside² ≈ 4π/(12 * mean_pixel_scale_rad²)
    # nside ≈ sqrt(4π/(12 * mean_pixel_scale_rad²))

    target_nside = np.sqrt(4 * np.pi * u.sr / (12 * pixel_scale.to(u.sr)))

    # Convert nside to level
    # nside = 2^level, so level = log2(nside)
    target_level = nside_to_level(int(np.round(target_nside)))

    # Ensure level is within reasonable bounds
    target_level = max(0, min(target_level, max_level))

    return target_level
