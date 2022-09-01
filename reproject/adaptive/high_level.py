# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.utils
import warnings

from ..utils import parse_input_data, parse_output_projection
from .core import _reproject_adaptive_2d

__all__ = ['reproject_adaptive']


@astropy.utils.deprecated_renamed_argument('order', None, since=0.9)
def reproject_adaptive(input_data, output_projection, shape_out=None, hdu_in=0,
                       order=None,
                       return_footprint=True, center_jacobian=False,
                       roundtrip_coords=True, conserve_flux=False,
                       kernel=None, kernel_width=1.3,
                       sample_region_width=4,
                       boundary_mode=None, boundary_fill_value=0,
                       boundary_ignore_threshold=0.5, x_cyclic=False,
                       y_cyclic=False):
    """
    Reproject a 2D array from one WCS to another using the DeForest (2004)
    adaptive, anti-aliased resampling algorithm, with optional flux
    conservation. This algorithm smoothly transitions between filtered
    interpolation and spatial averaging, depending on the scaling applied by
    the transformation at each output location.

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
    order : str
        Deprecated, and no longer has any effect. Will be removed in a future
        release.
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
        accuracy and stronger anti-aliasing at the cost of some blurring (on
        the scale of a few pixels). If not specified, the Hann kernel is used
        by default, but this will change to the Gaussian kernel in a future
        release.
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
    boundary_mode : str
        How to handle when the sampling region includes regions outside the
        bounds of the input image. The default is ``ignore``, but this will
        change to ``strict`` in a future release. Allowed values are:

            * ``strict`` --- Output pixels will be NaN if any input sample
              falls outside the input image.
            * ``constant`` --- Samples outside the input image are replaced by
              a constant value, set with the ``boundary_fill_value`` argument.
              Output values become NaN if there are no valid input samples.
            * ``grid-constant`` --- Samples outside the input image are
              replaced by a constant value, set with the
              ``boundary_fill_value`` argument. Output values will be
              ``boundary_fill_value`` if there are no valid input samples.
            * ``ignore`` --- Samples outside the input image are simply
              ignored, contributing neither to the output value nor the
              sum-of-weights normalization.
            * ``ignore_threshold`` --- Acts as ``ignore``, unless the total
              weight of the ignored samples exceeds a set fraction of the total
              weight across the entire sampling region, set by the
              ``boundary_ignore_threshold`` argument. In that case, acts as
              ``strict``.
            * ``nearest`` --- Samples outside the input image are replaced by
              the nearst in-bounds input pixel.

    boundary_fill_value : double
        The constant value used by the ``constant`` boundary mode.
    boundary_ignore_threshold : double
        The threshold used by the ``ignore_threshold`` boundary mode. Should be
        a value between 0 and 1, representing a fraction of the total weight
        across the sampling region.
    x_cyclic, y_cyclic : bool
        Indicates that the x or y axis of the input image should be treated as
        cyclic or periodic. Overrides the boundary mode for that axis, so that
        out-of-bounds samples wrap to the other side of the image.

    Returns
    -------
    array_new : `~numpy.ndarray`
        The reprojected array
    footprint : `~numpy.ndarray`
        Footprint of the input array in the output array. Values of 0 indicate
        no coverage or valid values in the input image, while values of 1
        indicate valid values.
    """

    if kernel is None:
        kernel = 'hann'
        warnings.warn(
                "The default kernel will change from 'Hann' to "
                " 'Gaussian' in a future release. To suppress this warning, "
                "explicitly select a kernel with the 'kernel' argument.",
                FutureWarning, stacklevel=3)

    if boundary_mode is None:
        boundary_mode = 'ignore'
        warnings.warn(
                "The default boundary mode will change from 'ignore' to "
                " 'strict' in a future release. To suppress this warning, "
                "explicitly select a mode with the 'boundary_mode' argument.",
                FutureWarning, stacklevel=3)

    # TODO: add support for output_array

    array_in, wcs_in = parse_input_data(input_data, hdu_in=hdu_in)
    wcs_out, shape_out = parse_output_projection(output_projection, shape_out=shape_out)

    return _reproject_adaptive_2d(array_in, wcs_in, wcs_out, shape_out,
                                  return_footprint=return_footprint,
                                  center_jacobian=center_jacobian,
                                  roundtrip_coords=roundtrip_coords,
                                  conserve_flux=conserve_flux,
                                  kernel=kernel, kernel_width=kernel_width,
                                  sample_region_width=sample_region_width,
                                  boundary_mode=boundary_mode,
                                  boundary_fill_value=boundary_fill_value,
                                  boundary_ignore_threshold=boundary_ignore_threshold,
                                  x_cyclic=x_cyclic, y_cyclic=y_cyclic)
