import os

import numpy as np
from astropy import units as u

from ._utils import spectral_index_to_coord

__all__ = ["save_moc"]


# Mapping from HiPS coordinate frame to the MOC FITS COORDSYS keyword
COORDSYS = {"equatorial": "C", "galactic": "G", "ecliptic": "E"}


def save_moc(*, output_directory, indices, coord_system, spatial_level, level_depth=None):
    """
    Write a ``Moc.fits`` coverage file at the root of a HiPS dataset.

    For 2-d image HiPS this is a spatial MOC (S-MOC), and for 3-d spectral-cube
    HiPS this is a space-frequency MOC (SF-MOC), as recommended by the HiPS and
    HiPS3D standards. The frequency axis uses the same logarithmic discretization
    as the rest of the HiPS3D code (see `~reproject.hips._utils`).

    Parameters
    ----------
    output_directory : str
        The root directory of the HiPS dataset.
    indices : iterable
        The indices of the tiles generated at the deepest level. For 2-d data
        these are integer HEALPix indices, and for 3-d data these are
        ``(spatial_index, spectral_index)`` tuples.
    coord_system : {'equatorial', 'galactic', 'ecliptic'}
        The coordinate frame of the HiPS.
    spatial_level : int
        The deepest spatial HEALPix order of the HiPS.
    level_depth : int, optional
        The deepest spectral (frequency) order of the HiPS. If `None`, a 2-d
        spatial MOC is written, otherwise a space-frequency MOC is written.
    """

    # mocpy is imported lazily so that it is only required when a MOC is
    # actually being generated.
    from mocpy import MOC, SFMOC

    indices = list(indices)

    # If no tiles were generated there is no coverage to describe.
    if len(indices) == 0:
        return

    filename = os.path.join(output_directory, "Moc.fits")
    fits_keywords = {"COORDSYS": COORDSYS[coord_system]}

    if level_depth is None:

        ipix = np.array(sorted(indices), dtype=np.int64)
        moc = MOC.from_healpix_cells(
            ipix=ipix, depth=np.full(ipix.size, spatial_level), max_depth=spatial_level
        )
        moc.save(filename, format="fits", overwrite=True, fits_keywords=fits_keywords)

    else:

        # Group the spatial cells by the frequency cell they belong to, so that
        # each frequency range maps to the spatial coverage observed within it.
        spatial_by_spectral = {}
        for spatial_index, spectral_index in indices:
            spatial_by_spectral.setdefault(spectral_index, []).append(spatial_index)

        frequencies_min = []
        frequencies_max = []
        spatial_coverages = []
        for spectral_index in sorted(spatial_by_spectral):
            ipix = np.array(sorted(spatial_by_spectral[spectral_index]), dtype=np.int64)
            spatial_coverages.append(
                MOC.from_healpix_cells(
                    ipix=ipix, depth=np.full(ipix.size, spatial_level), max_depth=spatial_level
                )
            )
            # The frequency tile covers the FMOC cell at the deepest spectral
            # order, i.e. the range between this index and the next.
            frequencies_min.append(spectral_index_to_coord(level_depth, spectral_index).to_value(u.Hz))
            frequencies_max.append(
                spectral_index_to_coord(level_depth, spectral_index + 1).to_value(u.Hz)
            )

        sfmoc = SFMOC.from_spatial_coverages(
            np.array(frequencies_min) * u.Hz,
            np.array(frequencies_max) * u.Hz,
            spatial_coverages,
            max_order_frequency=level_depth,
        )
        sfmoc.save(filename, format="fits", overwrite=True, fits_keywords=fits_keywords)
