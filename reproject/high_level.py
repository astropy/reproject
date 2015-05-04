# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from astropy.io.fits import PrimaryHDU, ImageHDU, CompImageHDU, Header
from astropy.wcs import WCS


__all__ = ['reproject']



def reproject(input_data, output_projection, shape_out=None, projection_type='bilinear'):
    """
    Reproject data to a new projection.

    Parameters
    ----------
    input_data : `~astropy.io.fits.PrimaryHDU` or `~astropy.io.fits.ImageHDU` 
        or `~astropy.io.fits.CompImageHDU` or tuple.
        The input data to reproject. This can be an image HDU object from
        :mod:`astropy.io.fits`, such as a `~astropy.io.fits.PrimaryHDU`,
        `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.CompImageHDU`, 
        or it can be a tuple where the first element is a `~numpy.ndarray` 
        and the second element is either a `~astropy.wcs.WCS` or a 
        `~astropy.io.fits.Header` object
    output_projection : `~astropy.wcs.WCS` or `~astropy.io.fits.Header`
        The output projection, which can be either a `~astropy.wcs.WCS`
        or a `~astropy.io.fits.Header` instance.
    shape_out : tuple
        If ``output_projection`` is a `~astropy.wcs.WCS` instance, the
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
    array_new : `~numpy.ndarray`
        The reprojected data
    footprint : `~numpy.ndarray`
        Footprint of the input array in the output array. Values of 0 indicate
        no coverage or valid values in the input image, while values of 1
        indicate valid values. Intermediate values indicate partial coverage.
    """

    if projection_type == 'flux-conserving':
        from .spherical_intersect import reproject_flux_conserving
        return reproject_flux_conserving(input_data, output_projection, shape_out=shape_out)
    else:
        from .interpolation import reproject_interpolation
        return reproject_interpolation(input_data, output_projection, shape_out=shape_out, order=projection_type)

