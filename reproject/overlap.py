from ._overlap_wrapper import _computeOverlap

__all__ = ['compute_overlap']


def compute_overlap(ilon, ilat, olon, olat):
    """
    Compute the overlap between two 'pixels' in spherical coordinates
    
    Parameters
    ----------
    ilon : np.ndarray
        The longitudes defining the four corners of the input pixel
    ilat : np.ndarray
        The latitudes defining the four corners of the input pixel
    olon : np.ndarray
        The longitudes defining the four corners of the output pixel
    olat : np.ndarray
        The latitudes defining the four corners of the output pixel
    """
    return _computeOverlap(ilon, ilat, olon, olat, 0, 1.)
