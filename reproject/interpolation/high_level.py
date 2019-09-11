# Licensed under a 3-clause BSD style license - see LICENSE.rst

from ..utils import parse_input_data, parse_output_projection
from .core import _reproject_full

__all__ = ['reproject_interp']

ORDER = {}
ORDER['nearest-neighbor'] = 0
ORDER['bilinear'] = 1
ORDER['biquadratic'] = 2
ORDER['bicubic'] = 3


def reproject_interp(input_data, output_projection, shape_out=None, hdu_in=0,
                     order='bilinear', independent_celestial_slices=False,
                     output_array=None, return_footprint=True):
    """
    Reproject data to a new projection using interpolation (this is typically
    the fastest way to reproject an image).

    Parameters
    ----------
    input_data : str or `~astropy.io.fits.HDUList` or `~astropy.io.fits.PrimaryHDU` or `~astropy.io.fits.ImageHDU` or tuple
        The input data to reproject. This can be:

            * The name of a FITS file
            * An `~astropy.io.fits.HDUList` object
            * An image HDU object such as a `~astropy.io.fits.PrimaryHDU`,
              `~astropy.io.fits.ImageHDU`, or `~astropy.io.fits.CompImageHDU`
              instance
            * A tuple where the first element is a `~numpy.ndarray` and the
              second element is either a `~astropy.wcs.WCS` or a
              `~astropy.io.fits.Header` object

    output_projection : `~astropy.wcs.WCS` or `~astropy.io.fits.Header`
        The output projection, which can be either a `~astropy.wcs.WCS`
        or a `~astropy.io.fits.Header` instance.
    shape_out : tuple, optional
        If ``output_projection`` is a `~astropy.wcs.WCS` instance, the
        shape of the output data should be specified separately.
    hdu_in : int or str, optional
        If ``input_data`` is a FITS file or an `~astropy.io.fits.HDUList`
        instance, specifies the HDU to use.
    order : int or str, optional
        The order of the interpolation. This can be any of the
        following strings:

            * 'nearest-neighbor'
            * 'bilinear'
            * 'biquadratic'
            * 'bicubic'

        or an integer. A value of ``0`` indicates nearest neighbor
        interpolation.
    independent_celestial_slices : bool, optional
        This can be set to ``True`` for n-dimensional input in the following case
        (all conditions have to be fulfilled):

            * The number of pixels in each non-celestial dimension is the same
              between the input and target header.
            * The WCS coordinates along the non-celestial dimensions are the
              same between the input and target WCS.
            * The celestial WCS component is independent from other WCS
              coordinates.

        In this special case, we can make things a little faster by
        reprojecting each celestial slice independently using the same
        transformation.
    output_array : None or `~numpy.ndarray`
        An array in which to store the reprojected data.  This can be any numpy
        array including a memory map, which may be helpful when dealing with
        extremely large files.
    return_footprint : bool
        Return the footprint in addition to the output array?

    Returns
    -------
    array_new : `~numpy.ndarray`
        The reprojected array
    footprint : `~numpy.ndarray`
        Footprint of the input array in the output array. Values of 0 indicate
        no coverage or valid values in the input image, while values of 1
        indicate valid values.
    """

    array_in, wcs_in = parse_input_data(input_data, hdu_in=hdu_in)
    wcs_out, shape_out = parse_output_projection(output_projection, shape_out=shape_out,
                                                 output_array=output_array)

    if isinstance(order, str):
        order = ORDER[order]

    return _reproject_full(array_in, wcs_in, wcs_out, shape_out=shape_out, order=order,
                           array_out=output_array, return_footprint=return_footprint)
