from pathlib import Path

from ..utils import is_jpeg, is_png, parse_input_data
from ..wcs_utils import has_celestial
from .core import image_to_hips, coadd_hips

__all__ = ["reproject_from_hips", "reproject_to_hips", "coadd_hips"]


def reproject_from_hips():
    raise NotImplementedError()


def reproject_to_hips(
    input_data,
    *,
    coord_system_out,
    level,
    reproject_function,
    output_directory,
    hdu_in=0,
    tile_size=512,
    progress_bar=None,
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
            * The name of a PNG or JPEG file

    coord_system_out : {'equatorial', 'galactic', 'ecliptic' }
        The target coordinate system for the HEALPIX projection
    level : int, optional
        The number of levels of FITS tiles.
    reproject_function : callable
        The function to use for the reprojection.
    output_directory : str
        The name of the output directory - if this already exists, an error
        will be raised.
    hdu_in : int or str, optional
        If ``input_data`` is a FITS file or an `~astropy.io.fits.HDUList`
        instance, specifies the HDU to use.
    tile_size : int, optional
        The size of each individual tile (defaults to 512).
    progress_bar : callable, optional
        If specified, use this as a progress_bar to track loop iterations over
        data sets.

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

    if (
        has_celestial(wcs_in)
        and wcs_in.low_level_wcs.pixel_n_dim == 2
        and wcs_in.low_level_wcs.world_n_dim == 2
    ):
        return image_to_hips(
            array_in,
            wcs_in,
            coord_system_out,
            level=level,
            reproject_function=reproject_function,
            output_directory=output_directory,
            tile_size=tile_size,
            tile_format=tile_format,
            progress_bar=progress_bar,
        )
    else:
        raise NotImplementedError(
            "Only data with a 2-d celestial WCS can be reprojected to HiPS tiles"
        )
