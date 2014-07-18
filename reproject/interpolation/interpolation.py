# Reprojection by interpolation

import numpy as np

from ..wcs_utils import wcs_to_celestial_frame
from astropy.coordinates import UnitSphericalRepresentation
from astropy import units as u
from astropy.wcs import WCSSUB_CELESTIAL

__all__ = ['interpolate_2d', 'interpolate_2d_slices']


def get_input_pixels_celestial(wcs_in, wcs_out, shape_out):
    """
    Get the pixel coordinates of the pixels in an array of shape ``shape_out``
    in the input WCS.
    """

    # Extract celestial component of WCS
    wcs_in = wcs_in.sub([WCSSUB_CELESTIAL])
    wcs_out = wcs_out.sub([WCSSUB_CELESTIAL])

    # TODO: for now assuming that coordinates are spherical, not
    # necessarily the case. Also assuming something about the order of the
    # arguments.

    # Find input/output frames
    frame_in = wcs_to_celestial_frame(wcs_in)
    frame_out = wcs_to_celestial_frame(wcs_out)

    # Generate pixel coordinates of output image
    xp_out_ax = np.arange(shape_out[1])
    yp_out_ax = np.arange(shape_out[0])
    xp_out, yp_out = np.meshgrid(xp_out_ax, yp_out_ax)

    # Convert output pixel coordinates to pixel coordinates in original image
    # (using pixel centers).
    xw_out, yw_out = wcs_out.wcs_pix2world(xp_out, yp_out, 0)

    xw_out_unit = u.Unit(wcs_out.wcs.cunit[0])
    yw_out_unit = u.Unit(wcs_out.wcs.cunit[1])

    data = UnitSphericalRepresentation(xw_out * xw_out_unit,
                                       yw_out * yw_out_unit)

    coords_out = frame_out.realize_frame(data)
    coords_in = coords_out.transform_to(frame_in)

    xw_unit_in = u.Unit(wcs_in.wcs.cunit[0])
    yw_unit_in = u.Unit(wcs_in.wcs.cunit[1])

    xw_in = coords_in.spherical.lon.to(xw_unit_in).value
    yw_in = coords_in.spherical.lat.to(yw_unit_in).value

    xp_in, yp_in = wcs_in.wcs_world2pix(xw_in, yw_in, 0)

    return xp_in, yp_in


def interpolate_2d(array, wcs_in, wcs_out, shape_out, order=1):
    """
    Reproject a 2D array from one WCS to another using interpolation.

    Parameters
    ----------
    array : :class:`~numpy.ndarray`
        The array to reproject
    wcs_in : :class:`~astropy.wcs.WCS`
        The input WCS
    wcs_out : :class:`~astropy.wcs.WCS`
        The output WCS
    shape_out : tuple
        The shape of the output array
    order : int
        The order of the interpolation (if ``mode`` is set to
        ``'interpolation'``). A value of ``0`` indicates nearest neighbor
        interpolation (the default).
    """

    # Get position of output pixel centers in input image
    xp_in, yp_in = get_input_pixels_celestial(wcs_in, wcs_out, shape_out)
    coordinates = [yp_in.ravel(), xp_in.ravel()]

    # Interpolate values to new grid
    from scipy.ndimage import map_coordinates
    array_new = map_coordinates(array, coordinates,
                                order=order, cval=np.nan,
                                mode='constant').reshape(shape_out)

    return array_new


def interpolate_2d_slices(array, wcs_in, wcs_out, shape_out, order=1):
    """
    Reproject 2D slices from a 3D array from one WCS to another using
    interpolation.

    The spatial dimensions should be the last two dimensions, and the third
    axis should be independent of the spatial axes.

    Parameters
    ----------
    array : :class:`~numpy.ndarray`
        The array to reproject
    wcs_in : :class:`~astropy.wcs.WCS`
        The input WCS
    wcs_out : :class:`~astropy.wcs.WCS`
        The output WCS
    shape_out : tuple
        The shape of the output array
    order : int
        The order of the interpolation (if ``mode`` is set to
        ``'interpolation'``). A value of ``0`` indicates nearest neighbor
        interpolation (the default).
    """

    # TODO: check that third axis is independent

    # Get position of output pixel centers in input image
    xp_in, yp_in = get_input_pixels_celestial(wcs_in, wcs_out, shape_out[1:])
    coordinates = [yp_in.ravel(), xp_in.ravel()]

    # Interpolate values to new grid
    from scipy.ndimage import map_coordinates
    array_new = np.zeros(shape_out)
    for slice_index in range(array.shape[0]):
        array_new[slice_index] = map_coordinates(array[slice_index],
                                                 coordinates,
                                                 order=order, cval=np.nan,
                                                 mode='constant'
                                                 ).reshape(shape_out[1:])

    return array_new
