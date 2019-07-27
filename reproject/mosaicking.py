# Licensed under a 3-clause BSD style license - see LICENSE.rst

import operator

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import (pixel_to_skycoord, skycoord_to_pixel,
                               proj_plane_pixel_scales, wcs_to_celestial_frame)
from astropy.nddata import Cutout2D

from astropy.wcs.utils import celestial_frame_to_wcs
from .utils import parse_input_data, parse_output_projection


__all__ = ['find_optimal_celestial_wcs']


# Steps to mosaicking:
# - take input images and find optimal WCS and shape (optional)
# - reproject individual images, optionally with auto-cliping to avoid creating many large arrays
# - co-adding of individual images into a final image
#
# Possible add-ons:
# - background matching by looking at overlap of images
#
# Things to be aware of:
# - need a way of making sure that we can tell if individual tiles are destined.
#   for the same moasic. The easiest way is probably to make sure that the headers
#   are identical except for the CRPIX values.

class ReprojectedArraySubset:

    # NOTE: we can't use Cutout2D here because it's much more convenient
    # to work with position being the lower left corner of the cutout
    # rather than the center, which is not well defined for even-sized
    # cutouts.

    def __init__(self, array, footprint, position, shape):
        self.array = array
        self.footprint = footprint
        self.position = position
        self.shape = shape

    def __repr__(self):
        return '<ReprojectedArraySubset at {0} with shape {1}>'.format(self.position, self.shape)

    @property
    def view_in_original_array(self):
        return (slice(self.jmin, self.jmax), slice(self.imin, self.imax))

    @property
    def jmin(self):
        return self.position[0]

    @property
    def jmax(self):
        return self.position[0] + self.shape[0]

    @property
    def imin(self):
        return self.position[1]

    @property
    def imax(self):
        return self.position[1] + self.shape[1]

    def overlaps(self, other):
        # Note that the use of <= or >= instead of < and > is due to
        # the fact that the max values are exclusive (so +1 above the
        # last value).
        return not (self.imax <= other.imin or other.imax <= self.imin or
                    self.jmax <= other.jmin or other.jmax <= self.jmin)

    def __add__(self, other):
        return self._operation(other, operator.add)

    def __sub__(self, other):
        return self._operation(other, operator.sub)

    def __mul__(self, other):
        return self._operation(other, operator.mul)

    def __truediv__(self, other):
        return self._operation(other, operator.truediv)

    def _operation(self, other, op):

        # Determine cutout parameters for overlap region

        imin = max(self.imin, other.imin)
        imax = min(self.imax, other.imax)
        jmin = max(self.jmin, other.jmin)
        jmax = min(self.jmax, other.jmax)

        # Extract cutout from each

        self_array = self.array[jmin - self.jmin:jmax - self.jmin,
                                imin - self.imin:imax - self.imin]
        self_footprint = self.footprint[jmin - self.jmin:jmax - self.jmin,
                                        imin - self.imin:imax - self.imin]

        other_array = other.array[jmin - other.jmin:jmax - other.jmin,
                                  imin - other.imin:imax - other.imin]
        other_footprint = other.footprint[jmin - other.jmin:jmax - other.jmin,
                                          imin - other.imin:imax - other.imin]

        # Carry out operator and store result in ReprojectedArraySubset

        array = op(self_array, other_array)
        footprint = (self_footprint > 0) & (other_footprint > 0)

        return ReprojectedArraySubset(array, footprint, (jmin, imin),
                                      (jmax - jmin, imax - imin))


def _match_backgrounds(arrays):

    N = len(arrays)

    # Set up matrix to record differences
    offsets = np.ones((N, N)) * np.nan

    # Loop over all pairs of images and check for overlap
    for i1, array1 in enumerate(arrays):
        for i2, array2 in enumerate(arrays):
            if i2 <= i1:
                continue
            if array1.overlaps(array2):
                difference = array1 - array2
                if np.any(difference.footprint):
                    offsets[i1, i2] = np.median(difference.array[difference.footprint])
                    offsets[i2, i1] = -offsets[i1, i2]

    # We now need to iterate to find an optimal solution to the offsets

    inds = np.arange(N)
    b = np.zeros(N)
    eta0 = 1. / N  # Initial learning rate.

    red = 1.

    for main_iter in range(10):

        if main_iter > 0 and main_iter % 5000 == 0:
            red /= 2.

        np.random.shuffle(inds)

        # Update learning rate
        eta = eta0 * red

        for i in inds:

            if np.isnan(b[i]):
                continue

            keep = ~np.isnan(offsets[i, :])
            b[i] += eta * np.sum(offsets[i, keep] - b[i, np.newaxis] + b[keep][np.newaxis, :])

        mn = np.mean(b[~np.isnan(b)])
        b -= mn

    for array, offset in zip(arrays, b):
        array.array -= offset


def mosaic(input_data, output_projection, shape_out=None, hdu_in=None,
           reproject_function=None, combine_function='mean',
           match_background=False, **kwargs):
    """
    Given a set of input images, reproject and co-add these to a single
    final image.

    This currently only works with 2-d images with celestial WCS.

    Parameters
    ----------
    input_data : iterable
        One or more input datasets to include in the calculation of the final WCS.
        This should be an iterable containing one entry for each dataset, where
        a single dataset is one of:

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
    # up memory usage. We could probably still have references to array objects,
    # but we'd just make sure these were memory mapped

    # TODO: add support for specifying output array

    # Validate inputs

    if combine_function not in ('mean', 'sum', 'median'):
        raise ValueError("combine_function should be one of mean/sum/median")

    if reproject_function is None:
        raise ValueError("reprojection function should be specified with reprojection_function")

    # Parse the output projection to avoid having to do it for each

    wcs_out, shape_out = parse_output_projection(output_projection, shape_out=shape_out)

    # Start off by reprojecting individual images to the final projection

    arrays = []

    for input_data_indiv in input_data:
        array, footprint = reproject_function(input_data_indiv,
                                              output_projection=wcs_out,
                                              shape_out=shape_out,
                                              hdu_in=hdu_in,
                                              **kwargs)

        # TODO: in future, we should make the reprojection functions smart
        # and able to return cutouts rather than the full array. For now
        # we post-process the output above and convert it to a more
        # memory-efficient Cutout2D object which retains information

        # FIXME: make sure we gracefully handle the case where the
        # output image is empty (due e.g. to no overlap).

        iproj = np.nonzero(np.sum(footprint, axis=0))[0]
        jproj = np.nonzero(np.sum(footprint, axis=1))[0]
        imin, imax = iproj[0], iproj[-1] + 1
        jmin, jmax = jproj[0], jproj[-1] + 1

        array = ReprojectedArraySubset(array[jmin:jmax, imin:imax],
                                       footprint[jmin:jmax, imin:imax],
                                       (jmin, imin),
                                       (jmax - jmin, imax - imin))

        arrays.append(array)

    # If requested, try and match the backgrounds.
    if match_background:
        _match_backgrounds(arrays)

    # At this point, the images are now ready to be co-added.

    # TODO: provide control over final dtype?

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

        raise NotImplementedError("combine_function='median' is not yet implemented")

    return final_array, final_footprint


def find_optimal_celestial_wcs(input_data, frame=None, auto_rotate=False,
                               projection='TAN', resolution=None,
                               reference=None):
    """
    Given one or more images, return an optimal WCS projection object and shape.

    This currently only works with 2-d images with celestial WCS.

    Parameters
    ----------
    input_data : iterable
        One or more input datasets to include in the calculation of the final WCS.
        This should be an iterable containing one entry for each dataset, where
        a single dataset is one of:

            * The name of a FITS file
            * An `~astropy.io.fits.HDUList` object
            * An image HDU object such as a `~astropy.io.fits.PrimaryHDU`,
              `~astropy.io.fits.ImageHDU`, or `~astropy.io.fits.CompImageHDU`
              instance
            * A tuple where the first element is a `~numpy.ndarray` and the
              second element is either a `~astropy.wcs.WCS` or a
              `~astropy.io.fits.Header` object

    frame : `~astropy.coordinates.BaseCoordinateFrame`
        The coordinate system for the final image (defaults to the frame of
        the first image specified)
    auto_rotate : bool
        Whether to rotate the header to minimize the final image area (if
        `True`, requires shapely>=1.6 to be installed)
    projection : str
        Three-letter code for the WCS projection
    resolution : `~astropy.units.Quantity`
        The resolution of the final image. If not specified, this is the
        smallest resolution of the input images.
    reference : `~astropy.coordinates.SkyCoord`
        The reference coordinate for the final header. If not specified, this
        is determined automatically from the input images.

    Returns
    -------
    wcs : :class:`~astropy.wcs.WCS`
        The optimal WCS determined from the input images.
    shape : tuple
        The optimal shape required to cover all the output.
    """

    # TODO: support higher-dimensional datasets in future
    # TODO: take into account NaN values when determining the extent of the final WCS

    input_data = [parse_input_data(data) for data in input_data]

    # We start off by looping over images, checking that they are indeed
    # celestial images, and building up a list of all corners and all reference
    # coordinates in celestial (ICRS) coordinates.

    corners = []
    references = []
    resolutions = []

    for array, wcs in input_data:

        if array.ndim != 2:
            raise ValueError("Input data is not 2-dimensional")

        if wcs.naxis != 2:
            raise ValueError("Input WCS is not 2-dimensional")

        if not wcs.has_celestial:
            raise TypeError("WCS does not have celestial components")

        # Determine frame if it wasn't specified
        if frame is None:
            frame = wcs_to_celestial_frame(wcs)

        # Find pixel coordinates of corners. In future if we are worried about
        # significant distortions of the edges in the reprojection process we
        # could simply add arbitrary numbers of midpoints to this list.
        ny, nx = array.shape
        xc = np.array([-0.5, nx - 0.5, nx - 0.5, -0.5])
        yc = np.array([-0.5, -0.5, ny - 0.5, ny - 0.5])

        # We have to do .frame here to make sure that we get an ICRS object
        # without any 'hidden' attributes, otherwise the stacking below won't
        # work. TODO: check if we need to enable distortions here.
        corners.append(pixel_to_skycoord(xc, yc, wcs, origin=0).icrs.frame)

        # We now figure out the reference coordinate for the image in ICRS. The
        # easiest way to do this is actually to use pixel_to_skycoord with the
        # reference position in pixel coordinates. We have to set origin=1
        # because crpix values are 1-based.
        xp, yp = wcs.wcs.crpix
        references.append(pixel_to_skycoord(xp, yp, wcs, origin=1).icrs.frame)

        # Find the pixel scale at the reference position - we take the minimum
        # since we are going to set up a header with 'square' pixels with the
        # smallest resolution specified.
        scales = proj_plane_pixel_scales(wcs)
        resolutions.append(np.min(np.abs(scales)))

    # We now stack the coordinates - however the ICRS class can't do this
    # so we have to use the high-level SkyCoord class.
    corners = SkyCoord(corners)
    references = SkyCoord(references)

    # If no reference coordinate has been passed in for the final header, we
    # determine the reference coordinate as the mean of all the reference
    # positions. This choice is as good as any and if the user really cares,
    # they can set  it manually.
    if reference is None:
        reference = SkyCoord(references.data.mean(), frame=references.frame)

    # In any case, we need to convert the reference coordinate (either specified
    # or automatically determined) to the requested final frame.
    reference = reference.transform_to(frame)

    # Determine resolution if not specified
    if resolution is None:
        resolution = np.min(resolutions) * u.deg

    # Determine the resolution in degrees
    cdelt = resolution.to(u.deg).value

    # Construct WCS object centered on position
    wcs_final = celestial_frame_to_wcs(frame, projection=projection)

    rep = reference.represent_as('unitspherical')
    wcs_final.wcs.crval = rep.lon.degree, rep.lat.degree
    wcs_final.wcs.cdelt = -cdelt, cdelt

    # For now, set crpix to (1, 1) and we'll then figure out where all the images
    # fall in this projection, then we'll adjust crpix.
    wcs_final.wcs.crpix = (1, 1)

    # Find pixel coordinates of all corners in the final WCS projection. We use origin=1
    # since we are trying to determine crpix values.
    xp, yp = skycoord_to_pixel(corners, wcs_final, origin=1)

    if auto_rotate:

        # Use shapely to represent the points and find the minimum rotated rectangle
        from shapely.geometry import MultiPoint
        mp = MultiPoint(list(zip(xp, yp)))

        # The following returns a list of rectangle vertices - in fact there are
        # 5 coordinates because shapely represents it as a closed polygon with
        # the same first/last vertex.
        xr, yr = mp.minimum_rotated_rectangle.exterior.coords.xy
        xr, yr = xr[:4], yr[:4]

        # The order of the vertices is not guaranteed to be constant so we
        # take the vertices with the two smallest y values (which, for a
        # rectangle, guarantees that the vertices are neighboring)
        order = np.argsort(yr)
        x1, y1, x2, y2 = xr[order[0]], yr[order[0]], xr[order[1]], yr[order[1]]

        # Determine angle between two of the vertices. It doesn't matter which
        # ones they are, we just want to know how far from being straight the
        # rectangle is.
        angle = np.arctan2(y2 - y1, x2 - x1)

        # Determine the smallest angle that would cause the rectangle to be
        # lined up with the axes.
        angle = angle % (np.pi / 2)
        if angle > np.pi / 4:
            angle -= np.pi / 2

        # Set rotation matrix (use PC instead of CROTA2 since PC is the
        # recommended approach)
        pc = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
        wcs_final.wcs.pc = pc

        # Recompute pixel coordinates (more accurate than simply rotating xp, yp)
        xp, yp = skycoord_to_pixel(corners, wcs_final, origin=1)

    # Find the full range of values
    xmin = xp.min()
    xmax = xp.max()
    ymin = yp.min()
    ymax = yp.max()

    # Update crpix so that the lower range falls on the bottom and left. We add
    # 0.5 because in the final image the bottom left corner should be at (0.5,
    # 0.5) not (1, 1).
    wcs_final.wcs.crpix = (1 - xmin) + 0.5, (1 - ymin) + 0.5

    # Return the final image shape too
    naxis1 = int(round(xmax - xmin))
    naxis2 = int(round(ymax - ymin))

    return wcs_final, (naxis2, naxis1)
