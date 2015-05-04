# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import numpy as np
from astropy.wcs import WCSSUB_CELESTIAL
from astropy.extern import six

from ..wcs_utils import convert_world_coordinates
from ..array_utils import iterate_over_celestial_slices, pad_edge_1
from ..utils import parse_input_data, parse_output_projection

__all__ = ['reproject_interp']

ORDER = {}
ORDER['nearest-neighbor'] = 0
ORDER['bilinear'] = 1
ORDER['biquadratic'] = 2
ORDER['bicubic']= 3


def reproject_interp(input_data, output_projection, shape_out=None, hdu_in=None, order='bilinear'):
    """
    Reproject data to a new projection using interpolation (this is typically
    the fastest way to reproject an image).

    Parameters
    ----------
    input_data : str or `~astropy.io.fits.HDUList` or `~astropy.io.fits.PrimaryHDU` or `~astropy.io.fits.ImageHDU` or tuple
        The input data to reproject. This can be:
        
            * The name of a FITS file
            * An `~astropy.io.fits.HDUList` object
            * An image HDU object such as a `~astropy.io.fits.PrimaryHDU` or
              `~astropy.io.fits.ImageHDU`
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
        If ``input_data`` is a FITS file or an ~astropy.io.fits.HDUList`
        instance, specifies the HDU to use.
    order : int or str, optional
        The order of the interpolation (if ``mode`` is set to
        ``'interpolation'``). This can be either one of the following strings:

            * 'nearest-neighbor'
            * 'bilinear'
            * 'biquadratic'
            * 'bicubic'

        or an integer. A value of ``0`` indicates nearest neighbor
        interpolation. 

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
    wcs_out, shape_out = parse_output_projection(output_projection, shape_out=shape_out)
    
    if isinstance(order, six.string_types):
        order = ORDER[order]

    # For now only celestial reprojection is supported
    if wcs_in.has_celestial:
        return _reproject_celestial(array_in, wcs_in, wcs_out, shape_out=shape_out, order=order)
    else:
        raise NotImplementedError("Currently only data with a WCS that includes a celestial component can be reprojected")


def map_coordinates(image, coords, **kwargs):

    # In the built-in scipy map_coordinates, the values are defined at the
    # center of the pixels. This means that map_coordinates does not
    # correctly treat pixels that are in the outer half of the outer pixels.
    # We solve this by extending the array, updating the pixel coordinates,
    # then getting rid of values that were sampled in the range -1 to -0.5
    # and n to n - 0.5.

    from scipy.ndimage import map_coordinates as scipy_map_coordinates

    ny, nx = image.shape

    image = pad_edge_1(image)

    values = scipy_map_coordinates(image, coords + 1, **kwargs)

    reset = ((coords[0] < -0.5) | (coords[0] > nx - 0.5) |
             (coords[1] < -0.5) | (coords[1] > ny - 0.5))
    values[reset] = kwargs.get('cval', 0.)

    return values


def _get_input_pixels_celestial(wcs_in, wcs_out, shape_out):
    """
    Get the pixel coordinates of the pixels in an array of shape ``shape_out``
    in the input WCS.
    """

    # TODO: for now assuming that coordinates are spherical, not
    # necessarily the case. Also assuming something about the order of the
    # arguments.

    # Generate pixel coordinates of output image
    xp_out_ax = np.arange(shape_out[1])
    yp_out_ax = np.arange(shape_out[0])
    xp_out, yp_out = np.meshgrid(xp_out_ax, yp_out_ax)

    # Convert output pixel coordinates to pixel coordinates in original image
    # (using pixel centers).
    xw_out, yw_out = wcs_out.wcs_pix2world(xp_out, yp_out, 0)

    xw_in, yw_in = convert_world_coordinates(xw_out, yw_out, wcs_out, wcs_in)

    xp_in, yp_in = wcs_in.wcs_world2pix(xw_in, yw_in, 0)

    return xp_in, yp_in


def _reproject_celestial(array, wcs_in, wcs_out, shape_out, order=1):
    """
    Reproject data with celestial axes to a new projection using interpolation.
    """

    # Make sure image is floating point
    array = np.asarray(array, dtype=float)

    # For now, assume axes are independent in this routine

    # Check that WCSs are equivalent
    if wcs_in.naxis == wcs_out.naxis and np.any(wcs_in.wcs.axis_types != wcs_out.wcs.axis_types):
        raise ValueError("The input and output WCS are not equivalent")

    # Extract celestial part of WCS in lon/lat order
    wcs_in_celestial = wcs_in.sub([WCSSUB_CELESTIAL])
    wcs_out_celestial = wcs_out.sub([WCSSUB_CELESTIAL])

    # We create an output array with the required shape, then create an array
    # that is in order of [rest, lat, lon] where rest is the flattened
    # remainder of the array. We then operate on the view, but this will change
    # the original array with the correct shape.

    array_new = np.zeros(shape_out)

    coordinates = None

    # Loop over slices and interpolate
    for slice_in, slice_out in iterate_over_celestial_slices(array, array_new, wcs_in):

        if coordinates is None:  # Get position of output pixel centers in input image
            xp_in, yp_in = _get_input_pixels_celestial(wcs_in_celestial, wcs_out_celestial, slice_out.shape)
            coordinates = np.array([yp_in.ravel(), xp_in.ravel()])

        slice_out[:,:] = map_coordinates(slice_in,
                                         coordinates,
                                         order=order, cval=np.nan,
                                         mode='constant'
                                         ).reshape(slice_out.shape)

    return array_new, (~np.isnan(array_new)).astype(float)
