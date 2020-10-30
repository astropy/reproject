# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, frame_transform_graph
from astropy.wcs.utils import (celestial_frame_to_wcs, pixel_to_skycoord, proj_plane_pixel_scales,
                               skycoord_to_pixel, wcs_to_celestial_frame)

from ..utils import parse_input_shape

__all__ = ['find_optimal_celestial_wcs']


def find_optimal_celestial_wcs(input_data, frame=None, auto_rotate=False,
                               projection='TAN', resolution=None,
                               reference=None):
    """
    Given one or more images, return an optimal WCS projection object and
    shape.

    This currently only works with 2-d images with celestial WCS.

    Parameters
    ----------
    input_data : iterable
        One or more input data specifications to include in the calculation of
        the final WCS. This should be an iterable containing one entry for each
        specification, where a single data specification is one of:

            * The name of a FITS file
            * An `~astropy.io.fits.HDUList` object
            * An image HDU object such as a `~astropy.io.fits.PrimaryHDU`,
              `~astropy.io.fits.ImageHDU`, or `~astropy.io.fits.CompImageHDU`
              instance
            * A tuple where the first element is an Numpy array shape tuple
              the second element is either a `~astropy.wcs.WCS` or a
              `~astropy.io.fits.Header` object
            * A tuple where the first element is a `~numpy.ndarray` and the
              second element is either a `~astropy.wcs.WCS` or a
              `~astropy.io.fits.Header` object

    frame : str or `~astropy.coordinates.BaseCoordinateFrame`
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
    # TODO: take into account NaN values when determining the extent of the
    #       final WCS

    if isinstance(frame, str):
        frame = frame_transform_graph.lookup_name(frame)()

    input_shapes = [parse_input_shape(shape) for shape in input_data]

    # We start off by looping over images, checking that they are indeed
    # celestial images, and building up a list of all corners and all reference
    # coordinates in celestial (ICRS) coordinates.

    corners = []
    references = []
    resolutions = []

    for shape, wcs in input_shapes:

        if len(shape) != 2:
            raise ValueError("Input data is not 2-dimensional (got shape {!r})".format(shape))

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
        ny, nx = shape
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

    # In any case, we need to convert the reference coordinate (either
    # specified or automatically determined) to the requested final frame.
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

    # For now, set crpix to (1, 1) and we'll then figure out where all the
    # images fall in this projection, then we'll adjust crpix.
    wcs_final.wcs.crpix = (1, 1)

    # Find pixel coordinates of all corners in the final WCS projection. We use
    # origin=1 since we are trying to determine crpix values.
    xp, yp = skycoord_to_pixel(corners, wcs_final, origin=1)

    if auto_rotate:

        # Use shapely to represent the points and find the minimum rotated
        # rectangle
        from shapely.geometry import MultiPoint
        mp = MultiPoint(list(zip(xp, yp)))

        # The following returns a list of rectangle vertices - in fact there
        # are 5 coordinates because shapely represents it as a closed polygon
        # with the same first/last vertex.
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
