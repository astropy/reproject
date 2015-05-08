"""
HEALPIX (Hierarchical Equal-Area and Isolatitude Pixelization) utility functions.

These are convenience functions that are thin wrappers around `healpy`
(http://code.google.com/p/healpy/) functionality.

See https://github.com/healpy/healpy/issues/129 and https://github.com/gammapy/gammapy/blob/master/gammapy/image/healpix.py
"""

from __future__ import print_function, division

import numpy as np

from astropy import units as u
from astropy.extern import six

from ..wcs_utils import convert_world_coordinates
from .utils import parse_coord_system

__all__ = ['healpix_to_image', 'image_to_healpix']

ORDER = {}
ORDER['nearest-neighbor'] = 0
ORDER['bilinear'] = 1
ORDER['biquadratic'] = 2
ORDER['bicubic'] = 3


def healpix_to_image(healpix_data, coord_system_in, wcs_out, shape_out,
                     order='bilinear', nested=False):
    """
    Convert image in HEALPIX format to a normal FITS projection image (e.g.
    CAR or AIT).

    .. note:: This function uses healpy, which is licensed
              under the GPLv2, so any package using this funtions has to (for
              now) abide with the GPLv2 rather than the BSD license.

    Parameters
    ----------
    healpix_data : `numpy.ndarray`
        HEALPIX data array
    coord_system_in : str or `~astropy.coordinates.BaseCoordinateFrame`
        The coordinate system for the input HEALPIX data, as an Astropy
        coordinate frame or corresponding string alias (e.g. ``'icrs'`` or
        ``'galactic'``)
    wcs_out : `~astropy.wcs.WCS`
        The WCS of the output array
    shape_out : tuple
        The shape of the output array
    order : int or str, optional
        The order of the interpolation (if ``mode`` is set to
        ``'interpolation'``). This can be either one of the following strings:

            * 'nearest-neighbor'
            * 'bilinear'

        or an integer. A value of ``0`` indicates nearest neighbor
        interpolation.
    nested : bool
        The order of the healpix_data, either nested or ring.  Stored in
        FITS headers in the ORDERING keyword.

    Returns
    -------
    reprojected_data : `numpy.ndarray`
        HEALPIX image resampled onto the reference image
    footprint : `~numpy.ndarray`
        Footprint of the input array in the output array. Values of 0 indicate
        no coverage or valid values in the input image, while values of 1
        indicate valid values.
    """
    import healpy as hp

    healpix_data = np.asarray(healpix_data, dtype=float)

    # Look up lon, lat of pixels in reference system
    yinds, xinds = np.indices(shape_out)
    lon_out, lat_out = wcs_out.wcs_pix2world(xinds, yinds, 0)

    # Convert between celestial coordinates
    coord_system_in = parse_coord_system(coord_system_in)
    with np.errstate(invalid='ignore'):
        lon_in, lat_in = convert_world_coordinates(lon_out, lat_out, wcs_out, (coord_system_in, u.deg, u.deg))

    # Convert from lon, lat in degrees to colatitude theta, longitude phi,
    # in radians
    theta = np.radians(90. - lat_in)
    phi = np.radians(lon_in)

    # hp.ang2pix() raises an exception for invalid values of theta, so only
    # process values for which WCS projection gives non-nan value
    good = np.isfinite(theta)
    data = np.empty(theta.shape, healpix_data.dtype)
    data[~good] = np.nan

    if isinstance(order, six.string_types):
        order = ORDER[order]

    if order == 1:
        data[good] = hp.get_interp_val(healpix_data, theta[good], phi[good], nested)
    elif order == 0:
        npix = len(healpix_data)
        nside = hp.npix2nside(npix)
        ipix = hp.ang2pix(nside, theta[good], phi[good], nested)
        data[good] = healpix_data[ipix]
    else:
        raise ValueError("Only nearest-neighbor and bilinear interpolation are supported")

    footprint = good.astype(int)

    return data, footprint


def image_to_healpix(data, wcs_in, coord_system_out,
                     nside, order='bilinear', nested=False):
    """
    Convert image in a normal WCS projection to HEALPIX format.

    .. note:: This function uses healpy, which is licensed
              under the GPLv2, so any package using this funtions has to (for
              now) abide with the GPLv2 rather than the BSD license.

    Parameters
    ----------
    data : `numpy.ndarray`
        Input data array to reproject
    wcs_in : `~astropy.wcs.WCS`
        The WCS of the input array
    coord_system_out : str or `~astropy.coordinates.BaseCoordinateFrame`
        The target coordinate system for the HEALPIX projection, as an Astropy
        coordinate frame or corresponding string alias (e.g. ``'icrs'`` or
        ``'galactic'``)
    order : int or str, optional
        The order of the interpolation (if ``mode`` is set to
        ``'interpolation'``). This can be either one of the following strings:

            * 'nearest-neighbor'
            * 'bilinear'
            * 'biquadratic'
            * 'bicubic'

        or an integer. A value of ``0`` indicates nearest neighbor
        interpolation.
    nested : bool
        The order of the healpix_data, either nested or ring.  Stored in
        FITS headers in the ORDERING keyword.

    Returns
    -------
    reprojected_data : `numpy.ndarray`
        A HEALPIX array of values
    footprint : `~numpy.ndarray`
        Footprint of the input array in the output array. Values of 0 indicate
        no coverage or valid values in the input image, while values of 1
        indicate valid values.
    """
    import healpy as hp
    from scipy.ndimage import map_coordinates

    npix = hp.nside2npix(nside)

    # Look up lon, lat of pixels in output system and convert colatitude theta
    # and longitude phi to longitude and latitude.
    theta, phi = hp.pix2ang(nside, np.arange(npix), nested)
    lon_out = np.degrees(phi)
    lat_out = 90. - np.degrees(theta)

    # Convert between celestial coordinates
    coord_system_out = parse_coord_system(coord_system_out)
    with np.errstate(invalid='ignore'):
        lon_in, lat_in = convert_world_coordinates(lon_out, lat_out, (coord_system_out, u.deg, u.deg), wcs_in)

    # Look up pixels in input system
    yinds, xinds = wcs_in.wcs_world2pix(lon_in, lat_in, 0)

    # Interpolate

    if isinstance(order, six.string_types):
        order = ORDER[order]

    healpix_data = map_coordinates(data, [xinds, yinds],
                                   order=order,
                                   mode='constant', cval=np.nan)

    return healpix_data, (~np.isnan(healpix_data)).astype(float)
