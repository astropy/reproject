# Licensed under a 3-clause BSD style license - see LICENSE.rst

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
