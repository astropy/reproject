import numpy as np
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix, npix_to_nside

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

    healpix_data = np.asarray(healpix_data, dtype=float)

    # Look up lon, lat of pixels in reference system and convert celestial coordinates
    yinds, xinds = np.indices(shape_out)
    world_in = wcs_out.pixel_to_world(xinds, yinds).transform_to(coord_system_in)
    world_in_unitsph = world_in.represent_as('unitspherical')
    lon_in, lat_in = world_in_unitsph.lon, world_in_unitsph.lat

    if isinstance(order, str):
        order = ORDER[order]

    nside = npix_to_nside(len(healpix_data))

    hp = HEALPix(nside=nside, order='nested' if nested else 'ring')

    if order == 1:
        data = hp.interpolate_bilinear_lonlat(lon_in, lat_in, healpix_data)
    elif order == 0:
        ipix = hp.lonlat_to_healpix(lon_in, lat_in)
        data = healpix_data[ipix]
    else:
        raise ValueError("Only nearest-neighbor and bilinear interpolation are supported")

    footprint = np.ones(data.shape, bool)

    return data, footprint


def image_to_healpix(data, wcs_in, coord_system_out,
                     nside, order='bilinear', nested=False):
    """
    Convert image in a normal WCS projection to HEALPIX format.

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
    from scipy.ndimage import map_coordinates

    hp = HEALPix(nside=nside, order='nested' if nested else 'ring')

    npix = hp.npix

    # Look up lon, lat of pixels in output system and convert colatitude theta
    # and longitude phi to longitude and latitude.
    lon_out, lat_out = hp.healpix_to_lonlat(np.arange(npix))

    world_out = SkyCoord(lon_out, lat_out, frame=coord_system_out)

    # Look up pixels in input WCS
    yinds, xinds = wcs_in.world_to_pixel(world_out)

    # Interpolate

    if isinstance(order, str):
        order = ORDER[order]

    healpix_data = map_coordinates(data, [xinds, yinds],
                                   order=order,
                                   mode='constant', cval=np.nan)

    return healpix_data, (~np.isnan(healpix_data)).astype(float)
