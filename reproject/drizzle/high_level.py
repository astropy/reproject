# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.extern import six

from ..utils import parse_input_data, parse_output_projection
from .core import _reproject_drizzle

__all__ = ['reproject_drizzle']


def reproject_drizzle(input_data, output_projection, shape_out=None, hdu_in=0,
                      scale="exptime", pixel_fraction=1.0, kernel="square",
                      exposure_time=1.0, units="cps"):
    """
    Reproject data to a new projection using the drizzle algorithm.

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

    scale : str, optional
        How each input image should be scaled. The choices are `exptime`
        which scales each image by its exposure time, `expsq` which scales
        each image by the exposure time squared.

    pixel_fraction : float, optional
        The fraction of a pixel that the pixel flux is confined to. The
        default value of 1 has the pixel flux evenly spread across the image.
        A value of 0.5 confines it to half a pixel in the linear dimension,
        so the flux is confined to a quarter of the pixel area when the square
        kernel is used.

    kernel : str, optional
        The name of the kernel used to combine the inputs. The choice of
        kernel controls the distribution of flux over the kernel. The kernel
        names are: "square", "gaussian", "point", "tophat", "turbo", "lanczos2",
        and "lanczos3". The square kernel is the default.

    exposure_time : float, optional
        The exposure time of the input image, a positive number. The
        exposure time is used to scale the image if the units are counts.

    units : str, optional
        The units of the input image. The units can either be "counts"
        or "cps" (counts per second.) If the value is counts, before using
        the input image it is scaled by dividing it by the exposure time.

    Returns
    -------
    array_new : `~numpy.ndarray`
        The reprojected array
    footprint : `~numpy.ndarray`
        Footprint of the input array in the output array. Values of 0 indicate
        no coverage of valid values in the input image, while values of 1
        indicate valid values.
    """

    array_in, wcs_in = parse_input_data(input_data, hdu_in=hdu_in)
    wcs_out, shape_out = parse_output_projection(output_projection,
                                                 shape_out=shape_out)

    return _reproject_drizzle(array_in, wcs_in, wcs_out, shape_out=shape_out,
                              scale=scale, pixel_fraction=pixel_fraction,
                              kernel=kernel, exposure_time=exposure_time,
                              units=units)
