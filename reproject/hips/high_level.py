import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from itertools import product
from logging import getLogger
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import (
    ICRS,
    BarycentricTrueEcliptic,
    Galactic,
)
from astropy.io import fits
from astropy.nddata import block_reduce
from astropy_healpix import (
    HEALPix,
    level_to_nside,
    nside_to_level,
    pixel_resolution_to_nside,
)
from PIL import Image

from ..array_utils import sample_array_edges
from ..utils import as_transparent_rgb, is_jpeg, is_png, parse_input_data
from ..wcs_utils import has_celestial, has_spectral, pixel_scale
from ._trim_utils import fits_getdata_untrimmed, fits_writeto_withtrim
from .utils import (
    load_properties,
    make_tile_folders,
    save_properties,
    skycoord_first,
    spectral_coord_to_index,
    tile_filename,
    tile_header,
)

__all__ = ["reproject_from_hips", "reproject_to_hips", "coadd_hips"]


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
    buildLandingPage({alScriptURL: 'https://aladin.cds.unistra.fr/AladinLite/api/v3/3.7.2-beta/aladin.js'});
</script>

</html>
"""

RESERVED_PROPERTIES = [
    "dataproduct_type",
    "hips_version",
    "hips_tile_format",
    "hips_tile_width",
    "hips_order",
    "hips_frame",
]

VALID_COORD_SYSTEM = {
    "equatorial": ICRS(),
    "galactic": Galactic(),
    "ecliptic": BarycentricTrueEcliptic(),
}

VALID_TILE_FORMATS = {"fits", "png", "jpeg"}


EXTENSION = {"fits": "fits", "png": "png", "jpeg": "jpg"}


def reproject_from_hips():
    raise NotImplementedError()


def reproject_to_hips(
    input_data,
    *,
    coord_system_out,
    reproject_function,
    output_directory,
    level=None,
    level_depth=None,
    hdu_in=0,
    tile_size=512,
    tile_depth=16,
    progress_bar=None,
    threads=False,
    properties=None,
    **kwargs,
):
    """
    Reproject data from a standard projection to a set of Hierarchical Progressive
    Surveys (HiPS) tiles.

    Parameters
    ----------
    input_data : object
        The input data to reproject. This can be:

            * The name of a FITS file as a `str` or a `pathlib.Path` object
            * An `~astropy.io.fits.HDUList` object
            * An image HDU object such as a `~astropy.io.fits.PrimaryHDU`,
              `~astropy.io.fits.ImageHDU`, or `~astropy.io.fits.CompImageHDU`
              instance
            * A tuple where the first element is a `~numpy.ndarray` and the
              second element is either a
              `~astropy.wcs.wcsapi.BaseLowLevelWCS`,
              `~astropy.wcs.wcsapi.BaseHighLevelWCS`, or a
              `~astropy.io.fits.Header` object
            * An `~astropy.nddata.NDData` object from which the ``.data`` and
              ``.wcs`` attributes will be used as the input data.
            * The name of a PNG or JPEG file with AVM metadata

    coord_system_out : {'equatorial', 'galactic', 'ecliptic' }
        The target coordinate system for the HEALPIX projection
    reproject_function : callable
        The function to use for the reprojection.
    output_directory : str
        The name of the output directory - if this already exists, an error
        will be raised.
    level : int, optional
        The number of levels of tiles for celestial coordinates.
    level_depth : int, optional
        The number of levels of tiles for the third (e.g. spectral) dimension.
    hdu_in : int or str, optional
        If ``input_data`` is a FITS file or an `~astropy.io.fits.HDUList`
        instance, specifies the HDU to use.
    tile_size : int or tuple, optional
        The size of each individual tile (defaults to 512) for celestial dimensions.
    tile_depth : int or tuple, optional
        The depth of each individual tile (defaults to 16) for the third (e.g. spectral) dimension when present.
    progress_bar : callable, optional
        If specified, use this as a progress_bar to track loop iterations over
        data sets.
    threads : bool or int
        If `False`, no multi-threading is used. If an integer, this number of
        threads will be used, and if `True`, the number of threads will be chosen
        automatically.
    properties : dict, optional
        Dictionary of properties that should be output to the ``properties``
        file inside the HiPS dataset. At list of properties and their meanings
        can be found in the `HiPS 1.0 <https://www.ivoa.net/documents/HiPS/20170406/PR-HIPS-1.0-20170406.pdf>`_
        description.
    **kwargs
        Keyword arguments to be passed to the reprojection function.

    Returns
    -------
    None
        This function does not return a value.
    """

    tile_format = "fits"

    if isinstance(input_data, str | Path):
        if is_png(input_data):
            tile_format = "png"
        elif is_jpeg(input_data):
            tile_format = "jpeg"

    array_in, wcs_in = parse_input_data(input_data, hdu_in=hdu_in)

    if not has_celestial(wcs_in):
        raise Exception("Only data with a celestial WCS can be reprojected to HiPS tiles")

    if wcs_in.low_level_wcs.pixel_n_dim != wcs_in.low_level_wcs.world_n_dim:
        raise Exception(
            f"Number of pixel ({wcs_in.low_level_wcs.pixel_n_dim}) "
            f"and world ({wcs_in.low_level_wcs.world_n_dim}) "
            f"dimensions do not match"
        )

    if wcs_in.low_level_wcs.pixel_n_dim == 2:
        ndim = 2
    elif wcs_in.low_level_wcs.pixel_n_dim == 3:
        if has_spectral(wcs_in):
            ndim = 3
        else:
            raise NotImplementedError(
                "Only 3-d data with a spectral axis are supported at this time"
            )
    else:
        raise Exception("Can only reproject data with 2-d or 3-d WCS")

    if ndim == 2:
        if array_in.ndim not in (2, 3):
            raise Exception("Input array should have 2 or 3 dimensions for 2-dimensional input WCS")
    else:
        if array_in.ndim != ndim:
            raise Exception(
                f"Input array dimensionality ({array_in.ndim}) should match WCS dimensionality ({ndim})"
            )

    if ndim == 3 and tile_format != "fits":
        raise ValueError("Only FITS tiles are supported in HiPS3D mode")

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

    if properties is None:
        properties = {}

    if progress_bar is None:
        progress_bar = lambda x: x

    # Determine celestial level if not specified

    if level is None:
        scale = pixel_scale(wcs_in, array_in.shape)
        nside = pixel_resolution_to_nside(scale * tile_size)
        level = int(nside_to_level(nside))
        logger.info(f"Automatically set the HEALPIX level to {level}")

    # Create output directory (and error if it already exists)
    os.makedirs(output_directory, exist_ok=False)

    # Determine center of image and radius to furthest corner, to determine
    # which HiPS tiles need to be generated

    # TODO: this will fail for e.g. allsky maps

    ny, nx = array_in.shape[-2:]

    centers = [(s - 1) / 2 for s in array_in.shape[-wcs_in.pixel_n_dim :]][::-1]
    edges = sample_array_edges(array_in.shape[-wcs_in.pixel_n_dim :], n_samples=2)[::-1]

    if ndim == 2:
        cen_skycoord = wcs_in.pixel_to_world(*centers)
        cor_skycoord = wcs_in.pixel_to_world(*edges)
    else:
        cen_skycoord, _ = skycoord_first(wcs_in.pixel_to_world(*centers))
        cor_skycoord, cor_spectralcoord = skycoord_first(wcs_in.pixel_to_world(*edges))

    separations = cor_skycoord.separation(cen_skycoord)

    if np.any(np.isnan(separations)):

        # At least one of the corners is outside of the region of validity of
        # the WCS, so we use a different approach where we randomly sample a
        # number of positions in the image and then check the maximum
        # separation between any pair of points.

        n_ran = 1000
        ran_x = np.random.uniform(-0.5, nx - 0.5, n_ran)
        ran_y = np.random.uniform(-0.5, nx - 0.5, n_ran)

        if ndim == 2:
            ran_world = wcs_in.pixel_to_world(ran_x, ran_y)
        elif ndim == 3:
            ran_world, _ = skycoord_first(wcs_in.pixel_to_world(ran_x, ran_y, np.zeros(n_ran)))

        separations = ran_world[:, None].separation(ran_world[None, :])

        max_separation = np.nanmax(separations)

    else:

        max_separation = separations.max()

    radius = 1.5 * max_separation

    # TODO: in future if astropy-healpix implements polygon searches, we could
    # use that instead

    # Determine all the celestial indices at the highest level

    nside = level_to_nside(level)
    hp = HEALPix(nside=nside, order="nested", frame=frame)

    if radius > 120 * u.deg:
        indices = np.arange(hp.npix)
    else:
        indices = hp.cone_search_skycoord(cen_skycoord, radius=radius)

    spatial_level = level

    if ndim == 3:

        # If depth level has not been specified, try and determine it
        if level_depth is None:

            for level_depth in range(52):  # FREQ_MAX_ORDER
                spectral_indices_edges = spectral_coord_to_index(level_depth, cor_spectralcoord)
                if np.ptp(spectral_indices_edges) > array_in.shape[0]:
                    break
            else:
                raise Exception(
                    "Could not determine depth level automatically, specify manually with level_depth="
                )

            level_depth = max(0, level_depth - int(np.log2(tile_depth)))

            logger.info(f"Automatically set the Spectral level to {level_depth}")

        # Determine all the spectral indices at the highest spectral level
        spectral_indices_edges = spectral_coord_to_index(level_depth, cor_spectralcoord)
        spectral_indices = np.arange(spectral_indices_edges.min(), spectral_indices_edges.max() + 1)
        indices = [
            (int(idx), int(spec_idx)) for (idx, spec_idx) in product(indices, spectral_indices)
        ]
        level = (level, level_depth)
        tile_dims = (tile_size, tile_depth)

    else:

        tile_dims = tile_size

    logger.info(f"Found {len(indices)} tiles (at most) to generate at level {level}")

    # PERF: the code above may be prohibitive for large numbers of tiles,
    # so we may want to find a way to iterate over these in chunks.

    # Make all the folders required for the tiles
    make_tile_folders(level=level, indices=indices, output_directory=output_directory)

    # Iterate over the tiles and generate them
    def process(index):
        if hasattr(wcs_in, "deepcopy"):
            wcs_in_copy = wcs_in.deepcopy()
        else:
            wcs_in_copy = deepcopy(wcs_in)

        header = tile_header(level=level, index=index, frame=frame, tile_dims=tile_dims)

        if isinstance(header, tuple):

            array_out1, footprint1 = reproject_function(
                (array_in, wcs_in_copy), header[0], **kwargs
            )
            array_out2, footprint2 = reproject_function(
                (array_in, wcs_in_copy), header[1], **kwargs
            )
            with np.errstate(invalid="ignore"):
                array_out = (
                    np.nan_to_num(array_out1) * footprint1 + np.nan_to_num(array_out2) * footprint2
                ) / (footprint1 + footprint2)
                footprint = (footprint1 + footprint2) / 2
            header = header[0]
        else:
            array_out, footprint = reproject_function((array_in, wcs_in_copy), header, **kwargs)

        if np.all(footprint == 0):
            return None

        if tile_format == "fits":
            array_out[footprint == 0] = np.nan
            pixel_min = np.nanmin(array_out)
            pixel_max = np.nanmax(array_out)
            fits_writeto_withtrim(
                tile_filename(
                    level=level,
                    index=index,
                    output_directory=output_directory,
                    extension=EXTENSION[tile_format],
                ),
                array_out,
                header,
            )

        else:
            if tile_format == "png":
                image = as_transparent_rgb(array_out, alpha=footprint[0])
            else:
                array_out[np.isnan(array_out)] = 0.0
                image = as_transparent_rgb(array_out).convert("RGB")
            image.save(
                tile_filename(
                    level=level,
                    index=index,
                    output_directory=output_directory,
                    extension=EXTENSION[tile_format],
                )
            )
            pixel_min, pixel_max = None, None

        return index, pixel_min, pixel_max

    if tile_format == "fits":
        pixel_min = np.inf
        pixel_max = -np.inf

    if threads:
        generated_indices = []
        with ThreadPoolExecutor(max_workers=None if threads is True else threads) as executor:
            futures = [executor.submit(process, index) for index in indices]
            for future in progress_bar(futures):
                result = future.result()
                if result is not None:
                    generated_indices.append(result[0])
                    if tile_format == "fits":
                        pixel_min = min(pixel_min, result[1])
                        pixel_max = max(pixel_max, result[2])
    else:
        generated_indices = []
        for index in progress_bar(indices):
            result = process(index)
            if result is not None:
                generated_indices.append(result[0])
                if tile_format == "fits":
                    pixel_min = min(pixel_min, result[1])
                    pixel_max = max(pixel_max, result[2])

    indices = generated_indices

    # Generate properties file

    cen_icrs = cen_skycoord.icrs

    for key in properties:
        if key in RESERVED_PROPERTIES:
            raise ValueError(f"Cannot override property {key}")

    generated_properties = {
        "creator_did": f"ivo://reproject/P/{str(uuid.uuid4())}",
        "obs_title": os.path.dirname(output_directory),
        "hips_version": "1.4",
        "hips_release_date": datetime.now().isoformat(),
        "hips_status": "public master clonableOnce",
        "hips_tile_format": tile_format,
        "hips_tile_width": tile_size,
        "hips_order": spatial_level,
        "hips_frame": coord_system_out,
        "hips_builder": "astropy/reproject",
        "hips_initial_ra": cen_icrs.ra.deg,
        "hips_initial_dec": cen_icrs.dec.deg,
        "hips_initial_fov": radius.deg,
    }

    if ndim == 2:
        generated_properties["dataproduct_type"] = "image"
    else:
        generated_properties["dataproduct_type"] = "spectral-cube"
        generated_properties["hips_order_freq"] = level_depth
        generated_properties["hips_order_min"] = 0
        generated_properties["hips_tile_depth"] = tile_depth
        wav = cor_spectralcoord.to_value(u.m)
        generated_properties["em_min"] = wav.min()
        generated_properties["em_max"] = wav.max()

    if tile_format == "fits":
        generated_properties["hips_pixel_bitpix"] = -64
        if not np.isinf(pixel_min) and not np.isinf(pixel_max):
            properties["hips_pixel_cut"] = (pixel_min, pixel_max)

    generated_properties.update(properties)

    save_properties(output_directory, generated_properties)

    save_index(output_directory)

    compute_lower_resolution_tiles(
        output_directory=output_directory,
        ndim=ndim,
        frame=frame,
        tile_dims=tile_dims,
        tile_format=tile_format,
        tile_size=tile_size,
        tile_depth=tile_depth,
        spatial_level=spatial_level,
        level_depth=level_depth,
    )


def find_indices(*, output_directory, ndim, spatial_level, level_depth):

    if ndim == 2:

        norder_directory = os.path.join(
            output_directory,
            f"Norder{spatial_level}",
        )

        for _, _, filenames in os.walk(norder_directory):
            for filename in filenames:
                yield int(filename.split(".")[0].replace("Npix", ""))

    else:

        norder_directory = os.path.join(
            output_directory,
            f"Norder{spatial_level}_{level_depth}",
        )

        for _, _, filenames in os.walk(norder_directory):
            for filename in filenames:
                indices = filename.split(".")[0].replace("Npix", "").split("_")
                yield int(indices[0]), int(indices[1])


def compute_lower_resolution_tiles(
    *,
    output_directory,
    ndim,
    frame,
    tile_dims,
    tile_format,
    tile_size,
    tile_depth,
    spatial_level,
    level_depth,
    indices=None,
):

    # Iterate over higher levels and compute lower resolution tiles

    half_tile_size = tile_size // 2
    if ndim == 3:
        half_tile_depth = tile_depth // 2

    for sub in range(1, spatial_level + 1):

        if ndim == 2:
            ilevel = spatial_level - sub
        else:
            ilevel = (spatial_level - sub, level_depth - sub)

        if indices is None:
            indices = list(
                find_indices(
                    ndim=ndim,
                    output_directory=output_directory,
                    spatial_level=spatial_level,
                    level_depth=level_depth,
                )
            )

        # Find index of tiles to produce at lower-resolution levels
        if ndim == 2:
            indices = np.sort(np.unique(np.asarray(indices) // 4))
        else:
            indices = sorted(
                set(
                    [
                        (spatial_index // 4, spectral_index // 2)
                        for (spatial_index, spectral_index) in indices
                    ]
                )
            )

        make_tile_folders(level=ilevel, indices=indices, output_directory=output_directory)

        for index in indices:

            header = tile_header(level=ilevel, index=index, frame=frame, tile_dims=tile_dims)

            if isinstance(header, tuple):
                header = header[0]

            if ndim == 2:

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
                        extension=EXTENSION[tile_format],
                    )

                    if os.path.exists(subtile_filename):

                        if tile_format == "fits":
                            tile_data = fits.getdata(subtile_filename)
                            data = block_reduce(tile_data, 2, func=np.mean)
                        else:
                            data = block_reduce(
                                np.array(Image.open(subtile_filename))[::-1],
                                (2, 2, 1),
                                func=np.mean,
                            )

                        if subindex == 0:
                            array[half_tile_size:, :half_tile_size] = data
                        elif subindex == 2:
                            array[half_tile_size:, half_tile_size:] = data
                        elif subindex == 1:
                            array[:half_tile_size, :half_tile_size] = data
                        elif subindex == 3:
                            array[:half_tile_size, half_tile_size:] = data

            elif ndim == 3:

                array = np.ones((tile_depth, tile_size, tile_size)) * np.nan

                for subindex in range(4):
                    for subindex_spec in range(2):

                        current_index = (4 * index[0] + subindex, 2 * index[1] + subindex_spec)
                        subtile_filename = tile_filename(
                            level=(ilevel[0] + 1, ilevel[1] + 1),
                            index=current_index,
                            output_directory=output_directory,
                            extension=EXTENSION[tile_format],
                        )

                        if os.path.exists(subtile_filename):

                            data = block_reduce(
                                fits_getdata_untrimmed(
                                    subtile_filename, tile_size=tile_size, tile_depth=tile_depth
                                ),
                                2,
                                func=np.mean,
                            )

                            if subindex_spec == 0:
                                subtile_slice = [slice(None, half_tile_depth)]
                            else:
                                subtile_slice = [slice(half_tile_depth, None)]

                            if subindex == 0:
                                subtile_slice.extend(
                                    [slice(half_tile_size, None), slice(None, half_tile_size)]
                                )
                            elif subindex == 2:
                                subtile_slice.extend(
                                    [slice(half_tile_size, None), slice(half_tile_size, None)]
                                )
                            elif subindex == 1:
                                subtile_slice.extend(
                                    [slice(None, half_tile_size), slice(None, half_tile_size)]
                                )
                            elif subindex == 3:
                                subtile_slice.extend(
                                    [slice(None, half_tile_size), slice(half_tile_size, None)]
                                )

                            array[tuple(subtile_slice)] = data

            if tile_format == "fits":
                fits_writeto_withtrim(
                    tile_filename(
                        level=ilevel,
                        index=index,
                        output_directory=output_directory,
                        extension=EXTENSION[tile_format],
                    ),
                    array,
                    header,
                )
            else:
                image = as_transparent_rgb(array.transpose(2, 0, 1))
                if tile_format == "jpeg":
                    image = image.convert("RGB")
                image.save(
                    tile_filename(
                        level=ilevel,
                        index=index,
                        output_directory=output_directory,
                        extension=EXTENSION[tile_format],
                    )
                )


def save_index(directory):
    with open(os.path.join(directory, "index.html"), "w") as f:
        f.write(INDEX_HTML)


def coadd_hips(input_directories, output_directory):
    """
    Given multiple HiPS directories, combine these into a single HiPS.

    The coordinate frame and tile format of the different input directories
    should match.

    In cases of overlap, the last image in the order of input_directories is used.

    Parameters
    ----------
    input_directories : iterable
        Iterable of HiPS directory names.
    output_directory : str
        The path to the output directory.
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
        raise ValueError(f"tile_format values do not match: {hips_frame}")

    reference_properties = all_properties[0]
    reference_properties["hips_order"] = max(hips_order)

    # Create output directory (and error if it already exists)
    os.makedirs(output_directory, exist_ok=False)

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
                        raise NotImplementedError(
                            "Convert jpg to png to allow for blending/coadding"
                        )
                    else:
                        raise NotImplementedError()
                else:
                    shutil.copyfile(filepath, target_filepath)

    save_properties(output_directory, reference_properties)

    save_index(output_directory)
