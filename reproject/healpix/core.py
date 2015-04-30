"""HEALPIX (Hierarchical Equal-Area and Isolatitude Pixelization) utility functions.

This is a thin wrapper convenience functions around
`healpy` (http://code.google.com/p/healpy/) functionality.

Refer to https://github.com/healpy/healpy/issues/129 and https://github.com/gammapy/gammapy/blob/master/gammapy/image/healpix.py
"""
from __future__ import print_function, division

import numpy as np

from astropy.io import fits
from astropy import units as u
from astropy.extern import six
from astropy.wcs import WCS
from astropy.coordinates import BaseCoordinateFrame, frame_transform_graph, Galactic, ICRS

from ..wcs_utils import convert_world_coordinates

__all__ = ['healpix_reproject_file', 'healpix_to_image', 'image_to_healpix']


FRAMES = {
    'g': Galactic(),
    'c': ICRS()
}

def parse_coord_system(system):
    if isinstance(system, BaseCoordinateFrame):
        return system
    elif isinstance(system, six.string_types):
        system = system.lower()
        if system == 'e':
            raise ValueError("Ecliptic coordinate frame not yet supported")
        elif system in FRAMES:
            return FRAMES[system]
        else:
            system_new = frame_transform_graph.lookup_name(system)
            if system_new is None:
                raise ValueError("Could not determine frame for system={0}".format(system))
            else:
                return system_new

def healpix_reproject_file(hp_filename, reference, outfilename=None, clobber=False, field=0, **kwargs):
    """
    Reproject a HEALPIX file

    Parameters
    ----------
    hp_filename : str
        A HEALPIX FITS file name
    reference : fits.Header, fits.PrimaryHDU, fits.HDUList, or str
        A fits.Header or HDU or FITS filename containing the target for projection
    outfilename : str or None
        The filename to write to
    clobber : bool
        Overwrite the outfilename if it exists?
    field : int
        The field number containing the data to be reprojected.  If not
        specifies, defaults to the first field in the BinTable
    kwargs : dict
        passed to healpix_to_image

    Returns
    -------
    fits.PrimaryHDU containing the reprojected image
    """
    import healpy as hp
    hp_data, hp_header = hp.read_map(
        hp_filename, verbose=False, h=True, field=field)
    hp_header = dict(hp_header)
    hp_coordsys = hp_header['COORDSYS']

    if isinstance(reference, str):
        reference_header = fits.getheader(reference)
    elif isinstance(reference, fits.Header):
        reference_header = reference
    elif isinstance(reference, fits.PrimaryHDU):
        reference_header = reference.header
    elif isinstance(reference, fits.HDUList):
        reference_header = reference[0].header
    else:
        raise TypeError("Reference was not a valid type; must be some sort of FITS header representation")

    wcs_out = WCS(reference_header)
    shape_out = reference_header['NAXIS2'], reference_header['NAXIS1']

    image_data = healpix_to_image(hp_data, hp_coordsys, wcs_out, shape_out, **kwargs)

    new_hdu = fits.PrimaryHDU(data=image_data, header=reference_header)

    if outfilename is not None:
        new_hdu.writeto(outfilename, clobber=clobber)

    return new_hdu


def healpix_to_image(healpix_data, coord_system_in, wcs_out, shape_out,
                     interp=True, nest=False):
    """
    Convert image in HEALPIX format to a normal FITS projection image (e.g.
    CAR or AIT).

    Parameters
    ----------
    healpix_data : `numpy.ndarray`
        HEALPIX data array
    coord_system_in : str or `~astropy.coordinate.BaseCoordinateFrame`
        The coordinate system for the input HEALPIX data, as an Astropy
        coordinate frame or corresponding string alias (e.g. ``'icrs'`` or
        ``'galactic'``)
    wcs_out : `~astropy.wcs.WCS`
        The WCS of the output array
    shape_out : tuple
        The shape of the output array
    nest : bool
        The order of the healpix_data, either nested or ring.  Stored in
        FITS headers in the ORDERING keyword.
    interp : bool
        Get the bilinear interpolated data?  If not, returns a set of neighbors

    Returns
    -------
    reprojected_data : `numpy.ndarray`
        HEALPIX image resampled onto the reference image

    Examples
    --------
    >>> import os
    >>> import healpy as hp
    >>> from astropy.io import fits
    >>> from reproject.healpix import healpix_to_image
    >>> reference_header = fits.Header()
    >>> os.system('curl -O http://www.ligo.org/scientists/first2years/2015/compare/12157/bayestar.fits.gz')
    0
    >>> reference_header.update({
    ...     'COORDSYS': 'icrs',
    ...     'CDELT1': -0.4,
    ...     'CDELT2': 0.4,
    ...     'CRPIX1': 500,
    ...     'CRPIX2': 400,
    ...     'CRVAL1': 180.0,
    ...     'CRVAL2': 0.0,
    ...     'CTYPE1': 'RA---MOL',
    ...     'CTYPE2': 'DEC--MOL',
    ...     'CUNIT1': 'deg',
    ...     'CUNIT2': 'deg',
    ...     'NAXIS': 2,
    ...     'NAXIS1': 1000,
    ...     'NAXIS2': 800})
    >>> healpix_data, healpix_header = hp.read_map('bayestar.fits.gz', h=True, verbose=False)
    >>> healpix_system = dict(healpix_header)['COORDSYS']
    >>> reprojected_data = healpix_to_image(healpix_data, reference_header, healpix_system)
    >>> fits.writeto('new_image.fits', reprojected_data, reference_header)
    """
    import healpy as hp

    print(wcs_out.to_header())

    # Look up lon, lat of pixels in reference system
    yinds, xinds = np.indices(shape_out)
    lon_out, lat_out = wcs_out.wcs_pix2world(xinds, yinds, 0)

    # Convert between celestial coordinates
    coord_system_in = parse_coord_system(coord_system_in)
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

    if interp:
        data[good] = hp.get_interp_val(healpix_data, theta[good], phi[good], nest)
    else:
        npix = len(healpix_data)
        nside = hp.npix2nside(npix)
        ipix = hp.ang2pix(nside, theta[good], phi[good], nest)
        data[good] = healpix_data[ipix]

    return data


def image_to_healpix(data, wcs_in, coord_system_out,
                     nside, interp=True, nest=False):
    """
    Convert image in a normal WCS projection to HEALPIX format.

    Parameters
    ----------
    data : `numpy.ndarray`
        Input data array to reproject
    wcs_in : `~astropy.wcs.WCS`
        The WCS of the input array
    coord_system_out : str or `~astropy.coordinate.BaseCoordinateFrame`
        The target coordinate system for the HEALPIX projection, as an Astropy
        coordinate frame or corresponding string alias (e.g. ``'icrs'`` or
        ``'galactic'``)
    nest : bool
        The order of the healpix_data, either nested or ring.  Stored in
        FITS headers in the ORDERING keyword.
    interp : bool
        Get the bilinear interpolated data?  If not, returns a set of neighbors

    Returns
    -------
    reprojected_data : `numpy.ndarray`
        A HEALPIX array of values
    """
    import healpy as hp
    from scipy.ndimage import map_coordinates

    npix = hp.nside2npix(nside)

    if interp:
        raise NotImplementedError

    # Look up lon, lat of pixels in output system and convert colatitude theta
    # and longitude phi to longitude and latitude.
    theta, phi = hp.pix2ang(nside, np.arange(npix), nest)
    lon_out = np.degrees(phi)
    lat_out = 90. - np.degrees(theta)

    # Convert between celestial coordinates
    coord_system_out = parse_coord_system(coord_system_out)
    lon_in, lat_in = convert_world_coordinates(lon_out, lat_out, (coord_system_out, u.deg, u.deg), wcs_in)

    # Look up pixels in input system
    yinds, xinds = wcs_in.wcs_world2pix(lon_in, lat_in, 0)

    # Interpolate
    healpix_data = map_coordinates(data, [xinds, yinds],
                                   order=(3 if interp else 0),
                                   mode='constant', cval=np.nan)

    return healpix_data
