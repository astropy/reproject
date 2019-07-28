# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import absolute_import, division, print_function


import numpy as np

from ..utils import parse_input_data, parse_output_projection
from .subset_array import ReprojectedArraySubset
from .background import determine_offset_matrix, solve_corrections_sgd

__all__ = ['reproject_and_coadd']


def reproject_and_coadd(input_data, output_projection, shape_out=None,
                        hdu_in=None, reproject_function=None,
                        combine_function='mean', match_background=False,
                        **kwargs):
    """
    Given a set of input images, reproject and co-add these to a single
    final image.

    This currently only works with 2-d images with celestial WCS.

    Parameters
    ----------
    input_data : iterable
        One or more input datasets to include in the calculation of the final
        WCS. This should be an iterable containing one entry for each dataset,
        where a single dataset is one of:

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
        If one or more items in ``input_data`` is a FITS file or an
        `~astropy.io.fits.HDUList` instance, specifies the HDU to use.
    reproject_function : callable
        The function to use for the reprojection
    combine_function : { 'mean', 'sum', 'median' }
        The type of function to use for combining the values into the final
        image.
    match_background : bool
        Whether to match the backgrounds of the images.
    kwargs
        Keyword arguments to be passed to the reprojection function.
    """

    # TODO: add support for saving intermediate files to disk to avoid blowing
    # up memory usage. We could probably still have references to array
    # objects, but we'd just make sure these were memory mapped

    # TODO: add support for specifying output array

    # Validate inputs

    if combine_function not in ('mean', 'sum', 'median'):
        raise ValueError("combine_function should be one of mean/sum/median")

    if reproject_function is None:
        raise ValueError("reprojection function should be specified with "
                         "the reprojection_function argument")

    # Parse the output projection to avoid having to do it for each

    wcs_out, shape_out = parse_output_projection(output_projection,
                                                 shape_out=shape_out)

    # Start off by reprojecting individual images to the final projection

    arrays = []

    for input_data_indiv in input_data:

        # We need to pre-parse the data here since we need to figure out how to
        # optimize/minimize the size of each output tile (see below).
        array_in, wcs_in = parse_input_data(input_data_indiv, hdu_in=hdu_in)

        # Since we might be reprojecting small images into a large mosaic we
        # want to make sure that for each image we reproject to an array with
        # minimal footprint. We therefore find the pixel coordinates of corners
        # in the initial image and transform this to pixel coordinates in the
        # final image to figure out the final WCS and shape to reproject to for
        # each tile. Note that in future if we are worried about significant
        # distortions of the edges in the reprojection process we could simply
        # add arbitrary numbers of midpoints to this list.
        ny, nx = array_in.shape
        xc = np.array([-0.5, nx - 0.5, nx - 0.5, -0.5])
        yc = np.array([-0.5, -0.5, ny - 0.5, ny - 0.5])
        xc_out, yc_out = wcs_out.world_to_pixel(wcs_in.pixel_to_world(xc, yc))

        # Determine the cutout parameters
        imin = max(0, int(np.floor(xc_out.min() + 0.5)))
        imax = min(shape_out[1], int(np.ceil(xc_out.max() + 0.5)))
        jmin = max(0, int(np.floor(yc_out.min() + 0.5)))
        jmax = min(shape_out[0], int(np.ceil(yc_out.max() + 0.5)))

        if imax < imin or jmax < jmin:
            continue

        # FIXME: for now, assume we are dealing with FITS-WCS, but once the
        # APE14 changes are merged in for reproject we can change to using a
        # sliced WCS
        wcs_out_indiv = wcs_out.deepcopy()
        wcs_out_indiv.wcs.crpix[0] -= imin
        wcs_out_indiv.wcs.crpix[1] -= jmin
        shape_out_indiv = (jmax - jmin, imax - imin)

        array, footprint = reproject_function(input_data_indiv,
                                              output_projection=wcs_out_indiv,
                                              shape_out=shape_out_indiv,
                                              hdu_in=hdu_in,
                                              **kwargs)

        array = ReprojectedArraySubset(array, footprint,
                                       imin, imax, jmin, jmax)

        # TODO: make sure we gracefully handle the case where the
        # output image is empty (due e.g. to no overlap).

        arrays.append(array)

    # If requested, try and match the backgrounds.
    if match_background:
        offset_matrix = determine_offset_matrix(arrays)
        corrections = solve_corrections_sgd(offset_matrix)
        for array, correction in zip(arrays, corrections):
            array.array -= correction

    # At this point, the images are now ready to be co-added.

    # TODO: provide control over final dtype

    final_array = np.zeros(shape_out)
    final_footprint = np.zeros(shape_out)

    if combine_function in ('mean', 'sum'):

        for array in arrays:

            # By default, values outside of the footprint are set to NaN
            # but we set these to 0 here to avoid getting NaNs in the
            # means/sums.
            array.array[array.footprint == 0] = 0

            final_array[array.view_in_original_array] += array.array
            final_footprint[array.view_in_original_array] += array.footprint

        if combine_function == 'mean':
            with np.errstate(invalid='ignore'):
                final_array /= final_footprint

    elif combine_function == 'median':

        # Here we need to operate in chunks since we could otherwise run
        # into memory issues

        raise NotImplementedError("combine_function='median' is "
                                  "not yet implemented")

    return final_array, final_footprint
