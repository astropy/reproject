from ._overlap_wrapper import _computeOverlap

def compute_overlap(ilon, ilat, olon, olat, energy_mode=True, reference_area=1.):
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
    energy_mode : bool
        Whether to work in energy-conserving or surface-brightness-conserving mode
    reference_area : float
        To be determined
    """
    return _computeOverlap(ilon, ilat, olon, olat, int(energy_mode), reference_area)
    
    
