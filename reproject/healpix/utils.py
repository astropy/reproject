import numpy as np
from astropy.coordinates import (
    ICRS,
    BaseCoordinateFrame,
    Galactic,
    frame_transform_graph,
)
from astropy.io import fits
from astropy.io.fits import BinTableHDU, TableHDU

FRAMES = {"g": Galactic(), "c": ICRS()}


def parse_coord_system(system):
    if isinstance(system, BaseCoordinateFrame):
        return system
    elif isinstance(system, str):
        system = system.lower()
        if system == "e":
            raise ValueError("Ecliptic coordinate frame not yet supported")
        elif system in FRAMES:
            return FRAMES[system]
        else:
            system_new = frame_transform_graph.lookup_name(system)
            if system_new is None:
                raise ValueError(f"Could not determine frame for system={system}")
            else:
                return system_new()


def parse_input_healpix_data(input_data, field=0, hdu_in=None, nested=None):
    """
    Parse input HEALPIX data to return a Numpy array and coordinate frame object.
    """

    if isinstance(input_data, TableHDU | BinTableHDU):
        data = input_data.data
        header = input_data.header
        coordinate_system_in = parse_coord_system(header["COORDSYS"])
        array_in = data[data.columns[field].name].ravel()
        if "ORDERING" in header:
            nested = header["ORDERING"].lower() == "nested"
    elif isinstance(input_data, str):
        # NOTE: hdu is not closed here.
        hdu = fits.open(input_data)[hdu_in or 1]
        return parse_input_healpix_data(hdu, field=field)
    elif isinstance(input_data, tuple) and isinstance(input_data[0], np.ndarray):
        array_in = input_data[0]
        coordinate_system_in = parse_coord_system(input_data[1])
    else:
        raise TypeError("input_data should either be an HDU object or a tuple of (array, frame)")

    return array_in, coordinate_system_in, nested
