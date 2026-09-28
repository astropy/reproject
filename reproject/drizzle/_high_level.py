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
    non_reprojected_dims=None,
    parallel=False,
    return_type=None,
    dask_method=None,
    zarr_path=None,
):
    """
    Reproject data to a new projection using the drizzle algorithm from
    `Fruchter and Hook (2002) <https://doi.org/10.1086/338393>`_, as
    implemented in the `drizzle <https://pypi.org/project/drizzle/>`_ package
    (which needs to be installed to use this function).

    Note that unlike the implementation in the `drizzle
    <https://pypi.org/project/drizzle/>`_ package, the function here does not
    require the input and output WCS coordinate frames to be the same, and will
    correctly convert between the two.

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
    block_size : tuple, optional
        The size of blocks in terms of output array pixels that each block
        will handle reprojecting. The drizzle algorithm distributes the flux
        of each input pixel over the output pixels, so input pixels
        contribute across output block boundaries and the reprojection
        cannot be carried out in blocks that split the celestial dimensions.
        The entries of ``block_size`` along the celestial dimensions must
        therefore match ``shape_out``, so that blocks only iterate over
        leading non-reprojected (broadcast) dimensions. When
        ``non_reprojected_dims`` is used, ``block_size`` can be left unset,
        in which case one block covering each non-reprojected slice in full
        is used automatically.
    non_reprojected_dims : tuple, optional
        Leading dimensions of the data that should not be reprojected but for
        which a one-to-one mapping between input and output pixels is assumed.
        This makes it possible to broadcast a reprojection over these dimensions
        even when the input and output WCS have the same number of dimensions as
        the data. The dimensions must be the leading ones, given as a tuple of
        sequential integers starting from zero (e.g. ``(0,)`` or ``(0, 1)``).
        The reprojection is done with one block per non-reprojected slice, so
        if ``block_size`` is specified, its entries along the reprojected
        dimensions have to match ``shape_out``; if not, this block size is
        used automatically.
    parallel : bool or int or str, optional
        If `True`, the reprojection is carried out in parallel, and if a
        positive integer, this specifies the number of threads to use.
        The reprojection will be parallelized over output array blocks that
        span the full extent of the celestial dimensions (see ``block_size``),
        so parallelization is only possible over leading non-reprojected
        dimensions. To use the currently active dask scheduler (e.g.
        dask.distributed), set this to ``'current-scheduler'``.
    return_type : {'numpy', 'dask', 'zarr'}, optional
        Whether to return numpy or dask arrays, or to write the output to a zarr
        array on disk. If ``'zarr'``, ``zarr_path`` must also be given. The
        ``'dask'`` and ``'zarr'`` return types compute the output in blocks, so
        they require a ``block_size`` that spans the full extent of the
        celestial dimensions (see ``block_size``).
    dask_method : {'memmap', 'none'}, optional
        Method to use when input array is a dask array. The methods are:
            * ``'memmap'``: write out the entire input dask array to a temporary
              memory-mapped array. This requires enough disk space to store
              the entire input array, but should avoid accidentally loading
              the entire array into memory.
            * ``'none'``: load the dask array into memory as needed. This may
              result in the entire array being loaded into memory.
    zarr_path : str, optional
        Path to use for the output zarr array when ``return_type='zarr'``. This
        must be a path that does not already exist.

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

    array_in, wcs_in = parse_input_data(input_data, hdu_in=hdu_in)
    wcs_out, shape_out = parse_output_projection(
        output_projection, shape_in=array_in.shape, shape_out=shape_out
    )

    n_non_reprojected = 0 if non_reprojected_dims is None else len(non_reprojected_dims)

    # Blocks are only acceptable if they span the full extent of the celestial
    # dimensions, so that they iterate over leading non-reprojected dimensions
    # and no flux is ever distributed across a block boundary. When
    # non_reprojected_dims is used with a WCS that has more dimensions than are
    # being reprojected, an unset (or 'auto') block size is also safe, since
    # the dispatcher then defaults to one block covering each non-reprojected
    # slice in full; without it, the automatic chunking may split the celestial
    # dimensions.
    if block_size is None or (isinstance(block_size, str) and block_size == "auto"):
        blocks_split_celestial = not (
            n_non_reprojected > 0 and wcs_in.pixel_n_dim == 2 + n_non_reprojected
        )
    else:
        blocks_split_celestial = tuple(block_size[-2:]) != tuple(shape_out)[-2:]

    if blocks_split_celestial and (
        block_size is not None or parallel is not False or return_type in ("dask", "zarr")
    ):
        raise NotImplementedError(
            "The drizzle algorithm distributes the flux of each input pixel "
            "over the output pixels, so input pixels contribute across block "
            "boundaries and the reprojection cannot be carried out in blocks "
            "that split the celestial dimensions. Blocked or parallel "
            "reprojection (including the 'dask' and 'zarr' return types) "
            "therefore requires a block_size whose entries along the celestial "
            "dimensions match shape_out, so that blocks only iterate over "
            "leading non-reprojected dimensions (e.g. non_reprojected_dims or "
            "extra leading dimensions of the data); when using "
            "non_reprojected_dims, block_size can also be left unset to use "
            "one block per non-reprojected slice automatically"
        )

    # When non_reprojected_dims is used with input and output WCS that have the
    # same number of dimensions as the data, the dispatcher slices the WCS down
    # to the celestial dimensions for each non-reprojected slice
    if (
        has_celestial(wcs_in)
        and wcs_in.pixel_n_dim in (2, 2 + n_non_reprojected)
        and wcs_in.world_n_dim == wcs_in.pixel_n_dim
    ):
        return _reproject_dispatcher(
            _reproject_drizzle,
            array_in=array_in,
            wcs_in=wcs_in,
            wcs_out=wcs_out,
            shape_out=shape_out,
            array_out=output_array,
            parallel=parallel,
            block_size=block_size,
            non_reprojected_dims=non_reprojected_dims,
            return_footprint=return_footprint,
            output_footprint=output_footprint,
            return_type=return_type,
            dask_method=dask_method,
            zarr_path=zarr_path,
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
