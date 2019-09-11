import numpy as np

from ._overlap import _compute_overlap

__all__ = ['compute_overlap']


def compute_overlap(ilon, ilat, olon, olat):
    """Compute the overlap between two 'pixels' in spherical coordinates.

    Parameters
    ----------
    ilon : np.ndarray with shape (N, 4)
        The longitudes (in radians) defining the four corners of the input pixel
    ilat : np.ndarray with shape (N, 4)
        The latitudes (in radians) defining the four corners of the input pixel
    olon : np.ndarray with shape (N, 4)
        The longitudes (in radians) defining the four corners of the output pixel
    olat : np.ndarray with shape (N, 4)
        The latitudes (in radians) defining the four corners of the output pixel

    Returns
    -------
    overlap : np.ndarray of length N
        Pixel overlap solid angle in steradians
    area_ratio : np.ndarray of length N
        TODO
    """
    ilon = np.ascontiguousarray(ilon, dtype=np.float64)
    ilat = np.ascontiguousarray(ilat, dtype=np.float64)
    olon = np.ascontiguousarray(olon, dtype=np.float64)
    olat = np.ascontiguousarray(olat, dtype=np.float64)

    return _compute_overlap(ilon, ilat, olon, olat)
