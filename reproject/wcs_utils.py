# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
WCS-related utilities
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, ICRS, UnitSphericalRepresentation
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import (wcs_to_celestial_frame, pixel_to_skycoord, skycoord_to_pixel,
                               celestial_frame_to_wcs, proj_plane_pixel_scales)

from .utils import parse_input_data

__all__ = ['convert_world_coordinates', 'find_optimal_celestial_wcs']


def convert_world_coordinates(lon_in, lat_in, wcs_in, wcs_out):
    """
    Convert longitude/latitude coordinates from an input frame to an output
    frame.

    Parameters
    ----------
    lon_in, lat_in : `~numpy.ndarray`
        The longitude and latitude to convert
    wcs_in, wcs_out : tuple or `~astropy.wcs.WCS`
        The input and output frames, which can be passed either as a tuple of
        ``(frame, lon_unit, lat_unit)`` or as a `~astropy.wcs.WCS` instance.

    Returns
    -------
    lon_out, lat_out : `~numpy.ndarray`
        The output longitude and latitude
    """

    if isinstance(wcs_in, WCS):
        # Extract the celestial component of the WCS in (lon, lat) order
        wcs_in = wcs_in.celestial
        frame_in = wcs_to_celestial_frame(wcs_in)
        lon_in_unit = u.Unit(wcs_in.wcs.cunit[0])
        lat_in_unit = u.Unit(wcs_in.wcs.cunit[1])
    else:
        frame_in, lon_in_unit, lat_in_unit = wcs_in

    if isinstance(wcs_out, WCS):
        # Extract the celestial component of the WCS in (lon, lat) order
        wcs_out = wcs_out.celestial
        frame_out = wcs_to_celestial_frame(wcs_out)
        lon_out_unit = u.Unit(wcs_out.wcs.cunit[0])
        lat_out_unit = u.Unit(wcs_out.wcs.cunit[1])
    else:
        frame_out, lon_out_unit, lat_out_unit = wcs_out

    data = UnitSphericalRepresentation(lon_in * lon_in_unit,
                                       lat_in * lat_in_unit)

    coords_in = frame_in.realize_frame(data)
    coords_out = coords_in.transform_to(frame_out)

    lon_out = coords_out.represent_as('unitspherical').lon.to(lon_out_unit).value
    lat_out = coords_out.represent_as('unitspherical').lat.to(lat_out_unit).value

    return lon_out, lat_out


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
        The coordinate system for the final image (defaults to ICRS)
    auto_rotate : bool
        Whether to rotate the header to minimize the final image area
    projection : str
        Three-letter code for the WCS projection
    resolution : `~astropy.units.Quantity`
        The resolution of the final image. If not specified, this is the
        smallest resolution of the input images.
    reference : `~astropy.coordinates.SkyCoord`
        The reference coordinate for the final header. If not specified, this
        is determined automatically from the input images.
    """

    # TODO: support higher-dimensional datasets in future
    # TODO: take into account NaN values when determining the extent of the final WCS

    if frame is None:
        frame = ICRS()

    input_data = [parse_input_data(data) for data in input_data]

    # We start off by looping over images, checking that they are indeed
    # celestial images, and building up a list of all corners and all reference
    # coordinates in celestial (ICRS) coordinates.

    corners = []
    references = []
    resolutions = []

    for array, wcs in input_data:

        if array.ndim != 2:
            raise ValueError("input data is not 2-dimensional")

        if wcs.naxis != 2:
            raise ValueError("input WCS is not 2-dimensional")

        if not wcs.has_celestial:
            raise TypeError("WCS does not have celestial component")

        # Find pixel coordinates of corners. In future if we are worried about
        # significant distortions of the edges in the reprojection process we
        # could simply add arbitrary numbers of midpoints to this list.
        ny, nx = array.shape
        xc = np.array([-0.5, nx - 0.5, nx - 0.5, -0.5])
        yc = np.array([-0.5, -0.5, ny - 0.5, ny - 0.5])

        # We have to do .frame here to make sure that we get an ICRS object
        # without any 'hidden' attributes, otherwise the stacking below won't
        # work. TODO: check if we need to enable distortions here.
        corners.append(pixel_to_skycoord(xc, yc, wcs).icrs.frame)

        # We now figure out the reference coordinate for the image in ICRS. The
        # easiest way to do this is actually to use pixel_to_skycoord with the
        # reference position in pixel coordinates.
        xp, yp = wcs.wcs.crpix
        references.append(pixel_to_skycoord(xp, yp, wcs).icrs.frame)

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
    wcs_final.wcs.crval = reference.spherical.lon.degree, reference.spherical.lat.degree
    wcs_final.wcs.cdelt = -cdelt, cdelt

    # For now, set crpix to 0 and we'll then figure out where all the images
    # fall in this projection, then we'll adjust crpix.
    wcs_final.wcs.crpix = 0, 0

    # Find pixel coordinates of all corners in the final WCS projection
    xp, yp = skycoord_to_pixel(corners, wcs_final)

    if auto_rotate:

        # Use shapely to represent the points and find the minimum rotated rectangle
        from shapely.geometry import MultiPoint
        mp = MultiPoint(list(zip(xp, yp)))

        # The following returns a list of rectangle vertices - in fact there are
        # 5 coordinates because shapely represents it as a closed polygon with
        # the same first/last vertex.
        xr, yr = mp.minimum_rotated_rectangle.exterior.coords.xy

        # Determine angle between two of the vertices. It doesn't matter which
        # ones they are, we just want to know how far from being straight the
        # rectangle is.
        angle = np.arctan2(yr[1] - yr[0], xr[1] - xr[0])

        # Determine the smallest angle that would cause the rectangle to be
        # lined up with the axes.
        angle = angle % 90
        if angle > 45:
            angle -= 90

        # Rotate the original corner coordinates by this angle and then find
        # the range of coordinates. We do the following in one go so we can
        # overwrite xp and yp in one go.
        xp, yp = (xp * np.cos(-angle) - yp * np.sin(-angle),
                  xp * np.sin(-angle) + yp * np.cos(-angle))

        # Set rotation matrix (use PC instead of CROTA2 since PC is the
        # recommended approach)
        pc = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
        wcs_final.wcs.pc = pc

    # Find the full range of values
    xmin = xp.min()
    xmax = xp.max()
    ymin = yp.min()
    ymax = yp.max()

    # Update crpix so that the lower range falls on the bottom and left
    wcs_final.wcs.crpix = -xmin, -ymin

    # Return the final image shape too
    naxis1 = int(round(xmax - xmin)) + 1
    naxis2 = int(round(ymax - ymin)) + 1

    return wcs_final, (naxis2, naxis1)
