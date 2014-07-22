import numpy as np
from astropy.io.fits import PrimaryHDU, ImageHDU, Header
from astropy.wcs import WCS

ORDER = {}
ORDER['nearest-neighbor'] = 0
ORDER['bilinear'] = 1
ORDER['biquadratic'] = 2
ORDER['bicubic']= 3


def reproject(input_data, output_projection, shape_out=None, projection_type='bilinear'):
    """
    Reproject data to a new projection

    Parameters
    ----------
    input_data : :class:`~astropy.io.fits.PrimaryHDU` or :class:`~astropy.io.fits.ImageHDU` or tuple
        The input data to reproject. This can be an image HDU object from
        :mod:`astropy.io.fits`, such as a :class:`~astropy.io.fits.PrimaryHDU`
        or :class:`~astropy.io.fits.ImageHDU`, or it can be a tuple where the
        first element is a :class:`~numpy.ndarray` and the second element is
        either a :class:`~astropy.wcs.WCS` or a :class:`~astropy.io.fits.Header` object
    output_projection : :class:`~astropy.wcs.WCS` or :class:`~astropy.io.fits.Header`
        The output projection, which can be either a :class:`~astropy.wcs.WCS`
        or a :class:`~astropy.io.fits.Header` instance.
    shape_out : tuple
        If ``output_projection`` is a :class:`~astropy.wcs.WCS` instance, the
        shape of the output data should be specified separately.
    projection_type : str
        The reprojection type, which can be one of:
            * 'nearest-neighbor'
            * 'bilinear'
            * 'biquadratic'
            * 'bicubic'
            * 'flux-conserving'

    Returns
    -------
    array_new : :class:`~numpy.ndarray`
        The reprojected data
    """

    if isinstance(input_data, (PrimaryHDU, ImageHDU)):
        array_in = input_data.data
        wcs_in = WCS(input_data.header)
    elif isinstance(input_data, tuple) and isinstance(input_data[0], np.ndarray):
        array_in = input_data[0]
        if isinstance(input_data[1], Header):
            wcs_in = WCS(input_data[1])
        else:
            wcs_in = input_data[1]
    else:
        raise TypeError("input_data should either be an HDU object or a tuple of (array, WCS) or (array, Header)")

    if isinstance(output_projection, Header):
        wcs_out = WCS(output_projection)
        shape_out = [output_projection['NAXIS{0}'.format(i+1)] for i in range(output_projection['NAXIS'])][::-1]
    elif isinstance(output_projection, WCS):
        wcs_out = output_projection
        if shape_out is None:
            raise ValueError("Need to specify shape when specifying output_projection as WCS object")


    if projection_type in ORDER:
        order = ORDER[projection_type]
        from .interpolation import reproject_2d
        return reproject_2d(array_in, wcs_in, wcs_out, shape_out=shape_out, order=order)
    elif projection_type == 'flux-conserving':
        from .spherical_intersect import reproject_2d
        return reproject_2d(array_in, wcs_in, wcs_out, shape_out=shape_out)
    else:
        raise ValueError("Unknown projection type: {0}".format(projection_type))
