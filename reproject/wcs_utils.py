# Licensed under a 2-clause BSD style license - see LICENSE.rst

"""
WCS-related utilities
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy import units as u
from astropy.coordinates import UnitSphericalRepresentation
from astropy.wcs.utils import wcs_to_celestial_frame
from astropy.wcs import WCS

__all__ = ['convert_world_coordinates']


def convert_world_coordinates(xw_in, yw_in, wcs_in, wcs_out):
    """
    Convert world coordinates from an input frame to an output frame.

    Parameters
    ----------
    xw_in, yw_in : `~numpy.ndarray`
        The input coordinates to convert
    wcs_in, wcs_out : tuple or `~astropy.wcs.WCS`
        The input and output frames, which can be passed either as a tuple of
        ``(frame, x_unit, y_unit)`` or as a `~astropy.wcs.WCS` instance.
    """

    if isinstance(wcs_in, WCS):
        frame_in = wcs_to_celestial_frame(wcs_in)
        xw_in_unit = u.Unit(wcs_in.wcs.cunit[0])
        yw_in_unit = u.Unit(wcs_in.wcs.cunit[1])
    else:
        frame_in, xw_in_unit, yw_in_unit = wcs_in

    if isinstance(wcs_out, WCS):
        frame_out = wcs_to_celestial_frame(wcs_out)
        xw_out_unit = u.Unit(wcs_out.wcs.cunit[0])
        yw_out_unit = u.Unit(wcs_out.wcs.cunit[1])
    else:
        frame_out, xw_out_unit, yw_out_unit = wcs_out

    data = UnitSphericalRepresentation(xw_in * xw_in_unit,
                                       yw_in * yw_in_unit)

    coords_in = frame_in.realize_frame(data)
    coords_out = coords_in.transform_to(frame_out)

    xw_out = coords_out.spherical.lon.to(xw_out_unit).value
    yw_out = coords_out.spherical.lat.to(yw_out_unit).value

    return xw_out, yw_out
