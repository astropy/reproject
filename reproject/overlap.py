from ._overlap_wrapper import _computeOverlap

__all__ = ['compute_overlap', 'solid_angle']


def compute_overlap(ilon, ilat, olon, olat):
    """Compute the overlap between two 'pixels' in spherical coordinates.
    
    Parameters
    ----------
    ilon : np.ndarray
        The longitudes (in deg) defining the four corners of the input pixel
    ilat : np.ndarray
        The latitudes (in deg) defining the four corners of the input pixel
    olon : np.ndarray
        The longitudes (in deg) defining the four corners of the output pixel
    olat : np.ndarray
        The latitudes (in deg) defining the four corners of the output pixel
    
    Returns
    -------
    overlap : np.ndarray
        Pixel overlap solid angle in steradians
    area_ratio : np.ndarray
        TODO
    """
    return _computeOverlap(ilon, ilat, olon, olat, 0, 1.)


def solid_angle(lon, lat):
    """Compute the area of 'pixels' in spherical coordinates.     

    Parameters
    ----------
    lon : np.ndarray
        The longitudes defining the four corners of the pixel
    lat : np.ndarray
        The latitudes defining the four corners of the pixel

    Returns
    -------
    solid_angle : np.ndarray
        Pixel solid angle in steradians
    """
    return _computeOverlap(lon, lat, lon, lat, 0, 1.)[0]


def test_solid_angle():
    import numpy as np
    lon, lat = [0, 1, 1, 0], [0, 0, 1, 1]
    omega = solid_angle(lon, lat)
    np.testing.assert_allclose(omega, np.radians(1) ** 2, rtol=1e-3)
