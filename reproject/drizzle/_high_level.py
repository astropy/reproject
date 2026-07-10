# Licensed under a 3-clause BSD style license - see LICENSE.rst

from .._common import _reproject_dispatcher
from .._wcs_utils import has_celestial
from ..utils import parse_input_data, parse_output_projection
from ._core import _reproject_drizzle

__all__ = ["reproject_drizzle"]


def reproject_drizzle(
    input_data,
    output_projection,
    shape_out=None,
    hdu_in=0,
    kernel="square",
    pixfrac=1.0,
    output_array=None,
    output_footprint=None,
    return_footprint=True,
    block_size=None,
    parallel=False,
    return_type=None,
    dask_method=None,
):
    """
    Reproject data to a new projection using the drizzle algorithm from
    `Fruchter and Hook (2002) <https://doi.org/10.1086/338393>`_, as
    implemented in the `drizzle <https://pypi.org/project/drizzle/>`_ package
    (which needs to be installed to use this function).

    With the default ``kernel='square'`` and ``pixfrac=1``, this is a
    flux-conserving overlap-based algorithm equivalent to
    :func:`~reproject.reproject_exact` in the limit where individual pixels
    subtend small angles on the sky, with the difference that pixel edges are
    treated as straight lines in the output pixel plane rather than as arcs of
    great circles on the sky.

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
    kernel : str, optional
        The kernel with which the flux of each input pixel is distributed
        over the output pixels, matching the ``kernel`` argument of
        `drizzle.resample.Drizzle`. The default of ``'square'`` distributes
        the flux over the overlap between the (shrunken) input pixel and each
        output pixel and is flux-conserving; see the `drizzle documentation
        <https://spacetelescope-drizzle.readthedocs.io>`_ for the other
        kernels and their flux-conservation properties.
    pixfrac : float, optional
        The fraction of each input pixel's width by which the pixel is
        shrunk before its flux is distributed over the output pixels, between
        0 and 1 (the default). Values below 1 reduce the correlation between
        neighboring output pixels when combining multiple dithered images at
        the cost of a less uniform weight map, and are not in general useful
        when reprojecting a single image.
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
    block_size : None
        Included for compatibility with the other reprojection functions,
        but only `None` is accepted. The drizzle algorithm distributes the
        flux of each input pixel over the output pixels, so input pixels
        contribute across output block boundaries and the reprojection
        cannot currently be carried out in independent blocks.
    parallel : bool
        Included for compatibility with the other reprojection functions,
        but only `False` is accepted, since parallelization relies on
        blocked reprojection (see ``block_size``).
    return_type : {'numpy'}, optional
        Included for compatibility with the other reprojection functions,
        but only ``'numpy'`` is accepted, since the ``'dask'`` and ``'zarr'``
        return types rely on blocked reprojection (see ``block_size``).
    dask_method : {'memmap', 'none'}, optional
        Method to use when input array is a dask array. The methods are:
            * ``'memmap'``: write out the entire input dask array to a temporary
              memory-mapped array. This requires enough disk space to store
              the entire input array, but should avoid accidentally loading
              the entire array into memory.
            * ``'none'``: load the dask array into memory as needed. This may
              result in the entire array being loaded into memory.

    Returns
    -------
    array_new : `~numpy.ndarray`
        The reprojected array.
    footprint : `~numpy.ndarray`
        Footprint of the input array in the output array. Values of 0 indicate
        no coverage or valid values in the input image, while values of 1
        indicate valid values. Intermediate values indicate partial coverage.
        The normalization from the drizzle weight map to fractional coverage
        assumes the ratio of input to output pixel scales is constant across
        the image, so for images with strong distortion the values may deviate
        slightly from the fraction of each output pixel covered.
    """

    if block_size is not None:
        raise NotImplementedError(
            "The drizzle algorithm distributes the flux of each input pixel "
            "over the output pixels, so input pixels contribute across block "
            "boundaries and the reprojection cannot currently be carried out "
            "in blocks (block_size should be None)"
        )

    if parallel is not False:
        raise NotImplementedError(
            "Parallel reprojection relies on blocked reprojection, which is "
            "not currently supported by the drizzle algorithm (parallel "
            "should be False)"
        )

    if return_type not in (None, "numpy"):
        raise NotImplementedError(
            "The 'dask' and 'zarr' return types rely on blocked reprojection, "
            "which is not currently supported by the drizzle algorithm "
            "(return_type should be 'numpy')"
        )

    array_in, wcs_in = parse_input_data(input_data, hdu_in=hdu_in)
    wcs_out, shape_out = parse_output_projection(
        output_projection, shape_in=array_in.shape, shape_out=shape_out
    )

    if has_celestial(wcs_in) and wcs_in.pixel_n_dim == 2 and wcs_in.world_n_dim == 2:
        return _reproject_dispatcher(
            _reproject_drizzle,
            array_in=array_in,
            wcs_in=wcs_in,
            wcs_out=wcs_out,
            shape_out=shape_out,
            array_out=output_array,
            parallel=False,
            block_size=None,
            return_footprint=return_footprint,
            output_footprint=output_footprint,
            return_type=return_type,
            dask_method=dask_method,
            reproject_func_kwargs=dict(
                kernel=kernel,
                pixfrac=pixfrac,
            ),
        )
    else:
        raise NotImplementedError(
            "Currently only data with a 2-d celestial "
            "WCS can be reprojected using the drizzle algorithm"
        )
