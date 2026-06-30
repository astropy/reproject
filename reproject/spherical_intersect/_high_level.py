# Licensed under a 3-clause BSD style license - see LICENSE.rst

from .._common import _reproject_dispatcher
from .._wcs_utils import has_celestial
from ..utils import parse_input_data, parse_output_projection
from ._core import _reproject_celestial

__all__ = ["reproject_exact"]


def reproject_exact(
    input_data,
    output_projection,
    shape_out=None,
    hdu_in=0,
    output_array=None,
    output_footprint=None,
    return_footprint=True,
    block_size=None,
    parallel=False,
    return_type=None,
    dask_method=None,
    zarr_path=None,
):
    """
    Reproject data to a new projection using flux-conserving spherical
    polygon intersection (this is the slowest algorithm).

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
    return_type : {'numpy', 'dask', 'zarr'}, optional
        Whether to return numpy or dask arrays, or to write the output to a zarr
        array on disk. If ``'zarr'``, ``zarr_path`` must also be given; the
        output is then computed in blocks (using dask, on the synchronous
        scheduler when ``parallel`` is `False`), ``block_size`` defaults to
        ``'auto'`` when not specified, and dask arrays backed by the zarr array
        are returned.
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
    """

    array_in, wcs_in = parse_input_data(input_data, hdu_in=hdu_in)
    wcs_out, shape_out = parse_output_projection(
        output_projection, shape_in=array_in.shape, shape_out=shape_out
    )

    if has_celestial(wcs_in) and wcs_in.pixel_n_dim == 2 and wcs_in.world_n_dim == 2:
        return _reproject_dispatcher(
            _reproject_celestial,
            array_in=array_in,
            wcs_in=wcs_in,
            wcs_out=wcs_out,
            shape_out=shape_out,
            array_out=output_array,
            parallel=parallel,
            block_size=block_size,
            return_footprint=return_footprint,
            output_footprint=output_footprint,
            return_type=return_type,
            zarr_path=zarr_path,
        )
    else:
        raise NotImplementedError(
            "Currently only data with a 2-d celestial "
            "WCS can be reprojected using flux-conserving algorithm"
        )
