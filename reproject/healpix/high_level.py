from .core import healpix_to_image, image_to_healpix
from .utils import parse_input_healpix_data, parse_coord_system
from ..utils import parse_input_data, parse_output_projection

__all__ = ['reproject_from_healpix', 'reproject_to_healpix']


def reproject_from_healpix(input_data, output_projection, shape_out=None):
    """
    Reproject data from a HEALPIX projection to a standard projection.

    Parameters
    ----------
    input_data : str or `~astropy.io.fits.TableHDU` or `~astropy.io.fits.BinTableHDU` or tuple
        The input data to reproject. This can be the filename to a HEALPIX
        file, a `~astropy.io.fits.TableHDU` or `~astropy.io.fits.BinTableHDU`
        instance, or it can be a tuple where the first element is a
        `~numpy.ndarray` and the second element is a
        `~astropy.coordinates.BaseCoordinateFrame` instance.
    output_projection : `~astropy.wcs.WCS` or `~astropy.io.fits.Header`
        The output projection, which can be either a `~astropy.wcs.WCS`
        or a `~astropy.io.fits.Header` instance.
    shape_out : tuple, optional
        If ``output_projection`` is a `~astropy.wcs.WCS` instance, the
        shape of the output data should be specified separately.

    Returns
    -------
    array_new : `~numpy.ndarray`
        The reprojected array
    footprint : `~numpy.ndarray`
        Footprint of the input array in the output array. Values of 0 indicate
        no coverage or valid values in the input image, while values of 1
        indicate valid values. Intermediate values indicate partial coverage.
    """

    array_in, coord_system_in = parse_input_healpix_data(input_data)
    wcs_out, shape_out = parse_output_projection(output_projection, shape_out=shape_out)

    return healpix_to_image(array_in, coord_system_in, wcs_out, shape_out)

def reproject_to_healpix(input_data, coord_system_out):
    """
    Reproject data from a standard projection to a HEALPIX projection

    Parameters
    ----------
    input_data : `~astropy.io.fits.PrimaryHDU` or `~astropy.io.fits.ImageHDU` or tuple
        The input data to reproject. This can be an image HDU object from
        :mod:`astropy.io.fits`, such as a `~astropy.io.fits.PrimaryHDU`
        or `~astropy.io.fits.ImageHDU`, or it can be a tuple where the
        first element is a `~numpy.ndarray` and the second element is
        either a `~astropy.wcs.WCS` or a `~astropy.io.fits.Header` object
    coord_system_out : `~astropy.coordinates.BaseCoordinateFrame` or str
        The output coordinate system for the HEALPIX projection

    Returns
    -------
    array_new : `~numpy.ndarray`
        The reprojected array
    footprint : `~numpy.ndarray`
        Footprint of the input array in the output array. Values of 0 indicate
        no coverage or valid values in the input image, while values of 1
        indicate valid values. Intermediate values indicate partial coverage.
    """

    array_in, wcs_in = parse_input_data(input_data)
    wcs_out, shape_out = parse_coord_system(coord_system_out)

    if wcs_in.has_celestial and wcs_in.naxis == 2:
        return image_to_healpix(array_in, wcs_in, coord_system_out)
    else:
        raise NotImplementedError("Only data with a 2-d celestial WCS can be reprojected to a HEALPIX projection")

