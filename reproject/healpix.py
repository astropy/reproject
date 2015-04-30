"""HEALPIX (Hierarchical Equal-Area and Isolatitude Pixelization) utility functions.

This is a thin wrapper convenience functions around
`healpy` (http://code.google.com/p/healpy/) functionality.

Refer to https://github.com/healpy/healpy/issues/129 and https://github.com/gammapy/gammapy/blob/master/gammapy/image/healpix.py
"""
from __future__ import print_function, division
from astropy import wcs
from astropy.io import fits
import numpy as np

__all__ = ['healpix_reproject_file', 'healpix_to_image', 'image_to_healpix']

# Mapping between HEALPix and WCS coordinate frames
healpix_to_wcs_coordsys = {
    'E': 'ecliptic',
    'G': 'galactic',
    'C': 'icrs'}

# Reverse mapping
wcs_to_healpix_coordsys = dict(zip(*list(zip(*healpix_to_wcs_coordsys.items()))[::-1]))

def normalize_healpix_coordsys(coordsys):
    if coordsys.upper() in healpix_to_wcs_coordsys:
        return coordsys.upper()
    try:
        return wcs_to_healpix_coordsys[coordsys.lower()]
    except KeyError:
        raise ValueError('Unrecognized coordinate system: "{0}"'.format(coordsys))

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

    if isinstance(reference,str):
        reference_header = fits.getheader(reference)
    elif isinstance(reference,fits.Header):
        reference_header = reference
    elif isinstance(reference,fits.PrimaryHDU):
        reference_header = reference.header
    elif isinstance(reference,fits.HDUList):
        reference_header = reference[0].header
    else:
        raise TypeError("Reference was not a valid type; must be some sort of FITS header representation")

    image_data = healpix_to_image(hp_data, reference_header, hp_coordsys, **kwargs)
    new_hdu = fits.PrimaryHDU(data=image_data, header=reference_header)

    if outfilename is not None:
        new_hdu.writeto(outfilename, clobber=clobber)

    return new_hdu

def healpix_to_image(healpix_data, reference_header, hpx_coord_system,
                     interp=True, nest=False):
    """
    Convert image in HEALPIX format to a normal FITS projection image (e.g.
    CAR or AIT).

    Parameters
    ----------
    healpix_data : `numpy.ndarray`
        HEALPIX data array
    reference_header : `astropy.io.fits.ImageHDU` or `astropy.io.fits.Header`
        A reference image or header to project to.  Must have a 'COORDSYS'
        keyword of either 'galactic' or 'icrs'
    hpx_coord_system : 'galactic' or 'icrs'
        The target coordinate system.  Should be derived from the HEALPIX
        COORDSYS keyword if it is a FITS file
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

    # Look up lon, lat of pixels in reference system
    refwcs = wcs.WCS(reference_header)
    yinds,xinds = np.indices([reference_header['NAXIS2'],reference_header['NAXIS1']])
    lon_deg, lat_deg = refwcs.wcs_pix2world(xinds,yinds,0)

    # Convert from lon, lat in degrees to colatitude theta, longitude phi,
    # in radians
    theta = 0.5 * np.pi - np.deg2rad(lat_deg)
    phi = np.deg2rad(lon_deg)

    # If the reference image uses a different celestial coordinate system from
    # the HEALPIX image we need to transform the coordinates
    ref_coord_system = reference_header['COORDSYS']
    theta, phi = sky_to_sky(theta, phi, ref_coord_system, hpx_coord_system)

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


def sky_to_sky(theta, phi, in_system, out_system):
    """Convert between sky coordinates.

    Parameters
    ----------
    theta, phi : array_like
        Coordinate arrays
    in_system, out_system : {'galactic', 'icrs'}
        Input / output coordinate system

    Returns
    -------
    """
    import healpy as hp

    in_system = normalize_healpix_coordsys(in_system)
    out_system = normalize_healpix_coordsys(out_system)

    if in_system != out_system:
        r = hp.Rotator(coord=[in_system, out_system])
        new_theta, new_phi = r(theta.flatten(), phi.flatten())
        theta = new_theta.reshape(theta.shape)
        phi = new_phi.reshape(phi.shape)

    return theta, phi


def image_to_healpix(data, reference_header, hpx_coord_system,
                     nside, interp=True, nest=False):
    """
    Convert image in a normal FITS projection image (e.g. CAR or AIT)
    to HEALPIX format.

    Parameters
    ----------
    data : `numpy.ndarray`
        FITS data array
    reference_header : `astropy.io.fits.ImageHDU` or `astropy.io.fits.Header`
        A reference image or header to project from.  Must have a 'COORDSYS'
        keyword of either 'galactic' or 'icrs'
    hpx_coord_system : 'galactic' or 'icrs'
        The target coordinate system.  Should be derived from the HEALPIX
        COORDSYS keyword if it is a FITS file
    nest : bool
        The order of the healpix_data, either nested or ring.  Stored in 
        FITS headers in the ORDERING keyword.
    interp : bool
        Get the bilinear interpolated data?  If not, returns a set of neighbors

    Returns
    -------
    reprojected_data : `numpy.ndarray`
        FITS image resampled into HEALPIX array
    """
    import healpy as hp
    from scipy.ndimage import map_coordinates

    # If the reference image uses a different celestial coordinate system from
    # the HEALPIX image we need to transform the coordinates
    ref_coord_system = reference_header['COORDSYS']

    npix = hp.nside2npix(nside)

    if interp:
        raise NotImplementedError

    # Look up lon, lat of pixels in reference system
    theta, phi = hp.pix2ang(nside, np.arange(npix), nest)
    theta, phi = sky_to_sky(theta, phi, hpx_coord_system, ref_coord_system)
    lon_deg = np.rad2deg(phi)
    lat_deg = np.rad2deg(0.5 * np.pi - theta)

    # Look up pixels in reference system
    refwcs = wcs.WCS(reference_header)
    yinds, xinds = refwcs.wcs_world2pix(lon_deg, lat_deg, 0)

    healpix_data = map_coordinates(
        data, [xinds, yinds],
        order=(3 if interp else 0),
        mode='constant', cval=np.nan)

    return healpix_data
