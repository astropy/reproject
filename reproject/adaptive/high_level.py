# Licensed under a 3-clause BSD style license - see LICENSE.rst

from ..common import _reproject_dispatcher
from ..utils import parse_input_data, parse_output_projection
from .core import _reproject_adaptive_2d

__all__ = ["reproject_adaptive"]


def reproject_adaptive(
    input_data,
    output_projection,
    shape_out=None,
    hdu_in=0,
    center_jacobian=False,
    despike_jacobian=False,
    roundtrip_coords=True,
    conserve_flux=False,
    kernel="gaussian",
    kernel_width=1.3,
    sample_region_width=4,
    boundary_mode="strict",
    boundary_fill_value=0,
    boundary_ignore_threshold=0.5,
    x_cyclic=False,
    y_cyclic=False,
    bad_value_mode="strict",
    bad_fill_value=0,
    output_array=None,
    output_footprint=None,
    return_footprint=True,
    block_size=None,
    parallel=False,
    return_type=None,
    dask_method=None,
):
    """
    Reproject a 2D array from one WCS to another using the DeForest (2004)
    adaptive, anti-aliased resampling algorithm, with optional flux
    conservation. This algorithm smoothly transitions between filtered
    interpolation and spatial averaging, depending on the scaling applied by
    the transformation at each output location.

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

        If the data array contains more dimensions than are described by the
        input header or WCS, the extra dimensions (assumed to be the first
        dimensions) are taken to represent multiple images with the same
        coordinate information. The coordinate transformation will be computed
        once and then each image will be reprojected, offering a speedup over
        reprojecting each image individually.
    output_projection : `~astropy.wcs.wcsapi.BaseLowLevelWCS` or `~astropy.wcs.wcsapi.BaseHighLevelWCS` or `~astropy.io.fits.Header`
        The output projection, which can be either a
        `~astropy.wcs.wcsapi.BaseLowLevelWCS`,
        `~astropy.wcs.wcsapi.BaseHighLevelWCS`, or a `~astropy.io.fits.Header`
        instance.
    shape_out : tuple, optional
        If ``output_projection`` is a WCS instance, the shape of the output
        data should be specified separately.
    hdu_in : int or str, optional
        If ``input_data`` is a FITS file or an `~astropy.io.fits.HDUList`
        instance, specifies the HDU to use.
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
    despike_jacobian : bool
        Whether to despike the computed Jacobian values. In some situations
        (e.g. an all-sky map, with a wrap point in the longitude), extremely
        large Jacobian values may be computed which are artifacts of the
        coordinate system definition, rather than reflecting the actual nature
        of the coordinate transformation. This may result in a band of ``nan``
        pixels in the output image. In these situations, if the actual
        transformation is approximately constant in the region of these
        artifacts, this option should be enabled. If enabled, the typical
        magnitude (distance from the determinant) of the Jacobian matrix,
        ``Jmag2 = sum_j sum_i (J_ij**2)``, is computed for each pixel and
        compared to the 25th percentile of that value in the local 3x3
        neighborhood (i.e. the third-lowest value). If it exceeds that
        percentile value by more than 10 times, the Jacobian matrix is deemed
        to be "spiking" and it is replaced by the average of the non-spiking
        values in the 3x3 neighborhood.
    roundtrip_coords : bool
        Whether to verify that coordinate transformations are defined in both
        directions.
    conserve_flux : bool
        Whether to rescale output pixel values so flux is conserved.
    kernel : str
        The averaging kernel to use. Allowed values are 'Hann' and 'Gaussian'.
        Case-insensitive. The Gaussian kernel produces better photometric
        accuracy and stronger anti-aliasing at the cost of some blurring (on
        the scale of a few pixels). If not specified, the Gaussain kernel is
        used by default.
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
        bounds of the input image. The default is ``strict``. Allowed values
        are:

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
              the nearest in-bounds input pixel.

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
    bad_value_mode : str
        How to handle values of ``nan`` and ``inf`` in the input data. The
        default is ``strct``. Allowed values are:

            * ``strict`` --- Values of ``nan`` or ``inf`` in the input data are
              propagated to every output value which samples them.
            * ``ignore`` --- When a sampled input value is ``nan`` or ``inf``,
              that input pixel is ignored (affected neither the accumulated sum
              of weighted samples nor the accumulated sum of weights).
            * ``constant`` --- Input values of ``nan`` and ``inf`` are replaced
              with a constant value, set via the ``bad_fill_value`` argument.

    bad_fill_value : double
        The constant value used by the ``constant`` bad-value mode.
    output_array : None or `~numpy.ndarray`
        An array in which to store the reprojected data.  This can be any numpy
        array including a memory map, which may be helpful when dealing with
        extremely large files.
    output_footprint : `~numpy.ndarray`, optional
        An array in which to store the footprint of reprojected data.  This can be
        any numpy array including a memory map, which may be helpful when dealing with
        extremely large files.
    return_footprint : bool
        Whether to return the footprint in addition to the output array.
    block_size : tuple or 'auto', optional
        The size of blocks in terms of output array pixels that each block will handle
        reprojecting. Extending out from (0,0) coords positively, block sizes
        are clamped to output space edges when a block would extend past edge.
        Specifying ``'auto'`` means that reprojection will be done in blocks with
        the block size automatically determined. If ``block_size`` is not
        specified or set to `None`, the reprojection will not be carried out in
        blocks.
    parallel : bool or int or str, optional
        If `True`, the reprojection is carried out in parallel, and if a
        positive integer, this specifies the number of threads to use.
        The reprojection will be parallelized over output array blocks specified
        by ``block_size`` (if the block size is not set, it will be determined
        automatically). To use the currently active dask scheduler (e.g.
        dask.distributed), set this to ``'current-scheduler'``.
    return_type : {'numpy', 'dask'}, optional
        Whether to return numpy or dask arrays.
    dask_method : {'memmap', 'none'}, optional
        Method to use when input array is a dask array. The methods are:
            * ``'memmap'``: write out the entire input dask array to a temporary
              memory-mapped array. This requires enough disk space to store
              the entire input array, but should avoid accidentally loading
              the entire array into memory.
            * ``'none'``: load the dask array into memory as needed. This may
              result in the entire array being loaded into memory. However,
              this can be efficient under two conditions: if the array easily
              fits into memory (as this will then be faster than ``'memmap'``),
              and when the data contains more dimensions than the input WCS and
              the block_size is chosen to iterate over the extra dimensions.

    Returns
    -------
    array_new : `~numpy.ndarray`
        The reprojected array.
    footprint : `~numpy.ndarray`
        Footprint of the input array in the output array. Values of 0 indicate
        no coverage or valid values in the input image, while values of 1
        indicate valid values.
    """

    # TODO: add support for output_array

    array_in, wcs_in = parse_input_data(input_data, hdu_in=hdu_in)
    wcs_out, shape_out = parse_output_projection(
        output_projection, shape_in=array_in.shape, shape_out=shape_out
    )

    return _reproject_dispatcher(
        _reproject_adaptive_2d,
        array_in=array_in,
        wcs_in=wcs_in,
        wcs_out=wcs_out,
        shape_out=shape_out,
        array_out=output_array,
        parallel=parallel,
        block_size=block_size,
        return_footprint=return_footprint,
        output_footprint=output_footprint,
        reproject_func_kwargs=dict(
            center_jacobian=center_jacobian,
            despike_jacobian=despike_jacobian,
            roundtrip_coords=roundtrip_coords,
            conserve_flux=conserve_flux,
            kernel=kernel,
            kernel_width=kernel_width,
            sample_region_width=sample_region_width,
            boundary_mode=boundary_mode,
            boundary_fill_value=boundary_fill_value,
            boundary_ignore_threshold=boundary_ignore_threshold,
            x_cyclic=x_cyclic,
            y_cyclic=y_cyclic,
            bad_value_mode=bad_value_mode,
            bad_fill_value=bad_fill_value,
        ),
        return_type=return_type,
    )
