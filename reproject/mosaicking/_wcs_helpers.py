# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, frame_transform_graph
from astropy.io.fits import Header
from astropy.wcs import WCS
from astropy.wcs.utils import (
    celestial_frame_to_wcs,
    pixel_to_skycoord,
    skycoord_to_pixel,
    wcs_to_celestial_frame,
)
from astropy.wcs.wcsapi import BaseHighLevelWCS, BaseLowLevelWCS

from ..utils import parse_input_shape
from ..wcs_utils import pixel_scale

__all__ = ["find_optimal_celestial_wcs"]


# Note that if this is modified, the docstring should be updated
NEGATIVE_CDELT_CTYPES = ["RA--", "GLON", "ELON", "HLON", "SLON"]


def find_optimal_celestial_wcs(
    input_data,
    hdu_in=None,
    frame=None,
    auto_rotate=False,
    projection="TAN",
    resolution=None,
    reference=None,
    negative_lon_cdelt=None,
):
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

            * The name of a FITS file as a `str` or a `pathlib.Path` object
            * An `~astropy.io.fits.HDUList` object
            * An image HDU object such as a `~astropy.io.fits.PrimaryHDU`,
              `~astropy.io.fits.ImageHDU`, or `~astropy.io.fits.CompImageHDU`
              instance
            * A tuple where the first element is an Numpy array shape tuple and
              the second element is either a
              `~astropy.wcs.wcsapi.BaseLowLevelWCS`,
              `~astropy.wcs.wcsapi.BaseHighLevelWCS`, or a
              `~astropy.io.fits.Header` object
            * A tuple where the first element is a `~numpy.ndarray` and the
              second element is either a
              `~astropy.wcs.wcsapi.BaseLowLevelWCS`,
              `~astropy.wcs.wcsapi.BaseHighLevelWCS`, or a
              `~astropy.io.fits.Header` object
            * An `~astropy.nddata.NDData` object from which the ``.data`` and
              ``.wcs`` attributes will be used as the input data.
            * A `~astropy.wcs.wcsapi.BaseLowLevelWCS` object with ``array_shape`` set
              or a `~astropy.wcs.wcsapi.BaseHighLevelWCS` object whose
              underlying low level WCS object has ``array_shape`` set.

        If only one input data needs to be provided, it is also possible to
        pass it in without including it in an iterable.

    hdu_in : int or str, optional
        If ``input_data`` is a FITS file or an `~astropy.io.fits.HDUList`
        instance, specifies the HDU to use.
    frame : str or `~astropy.coordinates.BaseCoordinateFrame`
        The coordinate system for the final image (defaults to the frame of
        the first image specified).
    auto_rotate : bool
        Whether to rotate the header to minimize the final image area (if
        `True`, requires shapely>=1.6 to be installed).
    projection : str
        Three-letter code for the WCS projection.
    resolution : `~astropy.units.Quantity`
        The resolution of the final image. If not specified, this is the
        smallest resolution of the input images.
    reference : `~astropy.coordinates.SkyCoord`
        The reference coordinate for the final header. If not specified, this
        is determined automatically from the input images.
    negative_lon_cdelt : bool or str, optional
        Whether the CDELT value for the longitude coordinate should be negative
        (`True`) or positive (`False`), or determined automatically (``'auto'``).
        For astronomical observations of the sky CDELT is usually negative,
        while for coordinate systems used in solar physics this is usually
        positive. If this is ``'auto'``, the value will be `True` if the
        first four characters for CTYPE for the longitude is ``RA--``,
        ``GLON``, ``ELON``, ``HLON``, or ``SLON``, and `False` otherwise.
        The default is currently ``True``, and will become ``'auto'`` in
        future.

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

    # Determine whether an iterable of input values was given or a single
    # input data.

    if isinstance(input_data, str):
        # Handle this explicitly as str is iterable too
        iterable = False
    elif np.iterable(input_data):
        if len(input_data) == 2 and isinstance(
            input_data[1], BaseLowLevelWCS | BaseHighLevelWCS | Header
        ):
            # Since 2-element tuples are valid single inputs we need to check for this
            iterable = False
        else:
            iterable = True
    else:
        iterable = False

    if iterable:
        input_shapes = [parse_input_shape(shape, hdu_in=hdu_in) for shape in input_data]
    else:
        input_shapes = [parse_input_shape(input_data, hdu_in=hdu_in)]

    # We start off by looping over images, checking that they are indeed
    # celestial images, and building up a list of all corners and all reference
    # coordinates in the frame of reference of the first image.

    corners = []
    references = []
    resolutions = []

    for shape, wcs in input_shapes:

        if len(shape) > wcs.pixel_n_dim:
            shape = shape[-wcs.pixel_n_dim :]

        if len(shape) != 2:
            raise ValueError(f"Input data is not 2-dimensional (got shape {shape!r})")

        if wcs.pixel_n_dim != 2 or wcs.world_n_dim != 2:
            raise ValueError("Input WCS is not 2-dimensional")

        if isinstance(wcs, WCS):
            if not wcs.has_celestial:
                raise TypeError("WCS does not have celestial components")

            # Determine frame if it wasn't specified
            if frame is None:
                frame = wcs_to_celestial_frame(wcs)

        else:
            # Convert a single position to determine type of output and make
            # sure there is only a single SkyCoord returned.
            coord = wcs.pixel_to_world(0, 0)

            if not isinstance(coord, SkyCoord):
                raise TypeError("WCS does not have celestial components")

            if frame is None:
                frame = coord.frame.replicate_without_data()

        # Find pixel coordinates of corners. In future if we are worried about
        # significant distortions of the edges in the reprojection process we
        # could simply add arbitrary numbers of midpoints to this list.
        ny, nx = shape
        xc = np.array([-0.5, nx - 0.5, nx - 0.5, -0.5])
        yc = np.array([-0.5, -0.5, ny - 0.5, ny - 0.5])

        # We have to do .frame here to make sure that we get a frame object
        # without any 'hidden' attributes, otherwise the stacking below won't
        # work.
        corners.append(wcs.pixel_to_world(xc, yc).transform_to(frame).frame)

        if isinstance(wcs, WCS):
            # We now figure out the reference coordinate for the image in the
            # frame of the first image. The easiest way to do this is actually
            # to use pixel_to_skycoord with the reference position in pixel
            # coordinates. We have to set origin=1 because crpix values are
            # 1-based.
            xp, yp = wcs.wcs.crpix
            references.append(pixel_to_skycoord(xp, yp, wcs, origin=1).transform_to(frame).frame)
        else:
            xp, yp = (nx - 1) / 2, (ny - 1) / 2
            references.append(wcs.pixel_to_world(xp, yp).transform_to(frame).frame)

        resolutions.append(pixel_scale(wcs, shape))

    # We now stack the coordinates - however the frame classes can't do this
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
        resolution = np.min(u.Quantity(resolutions))

    # Construct WCS object centered on position
    wcs_final = celestial_frame_to_wcs(frame, projection=projection)

    negative_lon_cdelt_auto = wcs_final.wcs.ctype[0][:4] in NEGATIVE_CDELT_CTYPES

    if negative_lon_cdelt == "auto":
        negative_lon_cdelt = negative_lon_cdelt_auto
    elif negative_lon_cdelt is None:
        if not negative_lon_cdelt_auto:
            warnings.warn(
                "negative_lon_cdelt is not set, and currently defaults to True, "
                "but in future will change to 'auto', and for this WCS this will "
                "evaluate to False in future. It is recommended that you set "
                "negative_lon_cdelt explicitly, either to 'auto', or to True/False.",
                DeprecationWarning,
                stacklevel=2,
            )
        negative_lon_cdelt = True

    if wcs_final.wcs.cunit[0] == "":
        wcs_final.wcs.cunit[0] = "deg"

    if wcs_final.wcs.cunit[1] == "":
        wcs_final.wcs.cunit[1] = "deg"

    rep = reference.represent_as("unitspherical")
    wcs_final.wcs.crval = (
        rep.lon.to_value(wcs_final.wcs.cunit[0]),
        rep.lat.to_value(wcs_final.wcs.cunit[1]),
    )

    lon_factor = -1 if negative_lon_cdelt else 1

    wcs_final.wcs.cdelt = (
        lon_factor * resolution.to_value(wcs_final.wcs.cunit[0]),
        resolution.to_value(wcs_final.wcs.cunit[1]),
    )

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

        mp = MultiPoint(list(zip(xp, yp, strict=True)))

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
        pc = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
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
