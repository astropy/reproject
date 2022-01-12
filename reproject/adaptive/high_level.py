# Licensed under a 3-clause BSD style license - see LICENSE.rst

from ..utils import parse_input_data, parse_output_projection
from .core import _reproject_adaptive_2d

__all__ = ['reproject_adaptive']


def reproject_adaptive(input_data, output_projection, shape_out=None, hdu_in=0,
                       return_footprint=True, center_jacobian=False,
                       roundtrip_coords=True, conserve_flux=False,
                       kernel='Hann', kernel_width=1.3, sample_region_width=4):
    """
    Reproject celestial slices from an 2d array from one WCS to another using
    the DeForest (2004) adaptive resampling algorithm.

    Parameters
    ----------
    input_data
        The input data to reproject. This can be:

            * The name of a FITS file
            * An `~astropy.io.fits.HDUList` object
            * An image HDU object such as a `~astropy.io.fits.PrimaryHDU`,
              `~astropy.io.fits.ImageHDU`, or `~astropy.io.fits.CompImageHDU`
              instance
            * A tuple where the first element is a `~numpy.ndarray` and the
              second element is either a `~astropy.wcs.WCS` or a
              `~astropy.io.fits.Header` object
            * An `~astropy.nddata.NDData` object from which the ``.data`` and
              ``.wcs`` attributes will be used as the input data.

    output_projection : `~astropy.wcs.WCS` or `~astropy.io.fits.Header`
        The output projection, which can be either a `~astropy.wcs.WCS`
        or a `~astropy.io.fits.Header` instance.
    shape_out : tuple, optional
        If ``output_projection`` is a `~astropy.wcs.WCS` instance, the
        shape of the output data should be specified separately.
    hdu_in : int or str, optional
        If ``input_data`` is a FITS file or an `~astropy.io.fits.HDUList`
        instance, specifies the HDU to use.
    return_footprint : bool
        Whether to return the footprint in addition to the output array.
    center_jacobian : bool
        A Jacobian matrix is calculated, representing
        d(input image coordinate) / d(output image coordinate),
        a local linearization of the coordinate transformation. When this flag
        is ``True``, the Jacobian is calculated at pixel grid points by
        calculating the transformation at locations offset by half a pixel.
        This is more accurate but carries the cost of tripling the number of
        coordinate transforms done by this routine. This is recommended if your
        coordinate transform varies significantly and non-smoothly between
        output pixels. When ``False``, the Jacobian is calculated using
        pixel-grid-point transforms, which produces Jacobian values at
        locations between pixel grid points, and nearby Jacobian values are
        averaged to produce values at the pixel grid points. This is more
        efficient, and the loss of accuracy is extremely small for
        transformations that vary smoothly between pixels. Defaults to
        ``False``.
    roundtrip_coords : bool
        Whether to verify that coordinate transformations are defined in both
        directions.
    conserve_flux : bool
        Whether to rescale output pixel values so flux is conserved.
    kernel : str
        The averaging kernel to use. Allowed values are 'Hann' and 'Gaussian'.
        Case-insensitive. The Gaussian kernel produces better photometric
        accuracy at the cost of some blurring (on the scale of a few pixels).
    kernel_width : double
        The width of the kernel in pixels, measuring to +/- 1 sigma for the
        Gaussian window. Does not apply to the Hann window. Reducing this width
        may introduce photometric errors or leave input pixels under-sampled,
        while increasing it may improve the degree of anti-aliasing but will
        increase blurring of the output image. If this width is changed from
        the default, a proportional change should be made to the value of
        sample_region_width to maintain an equivalent degree of photometric
        accuracy.
    sample_region_width : double
        The width in pixels of the output-image region which, when transformed
        to the input plane, defines the region to be sampled for each output
        pixel. Used only for the Gaussian kernel, which otherwise has infinite
        extent. This value sets a trade-off between accuracy and computation
        time, with better accuracy at higher values. The default value of 4,
        with the default kernel width, should limit the most extreme errors to
        less than one percent. Higher values will offer even more photometric
        accuracy.

    Returns
    -------
    array_new : `~numpy.ndarray`
        The reprojected array
    footprint : `~numpy.ndarray`
        Footprint of the input array in the output array. Values of 0 indicate
        no coverage or valid values in the input image, while values of 1
        indicate valid values.
    """

    # TODO: add support for output_array

    array_in, wcs_in = parse_input_data(input_data, hdu_in=hdu_in)
    wcs_out, shape_out = parse_output_projection(output_projection, shape_out=shape_out)

    return _reproject_adaptive_2d(array_in, wcs_in, wcs_out, shape_out,
                                  return_footprint=return_footprint,
                                  center_jacobian=center_jacobian,
                                  roundtrip_coords=roundtrip_coords,
                                  conserve_flux=conserve_flux,
                                  kernel=kernel, kernel_width=kernel_width,
                                  sample_region_width=sample_region_width)
