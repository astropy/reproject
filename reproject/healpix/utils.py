import tempfile

import numpy as np

from astropy.io.fits import TableHDU, BinTableHDU
from astropy.extern import six
from astropy.coordinates import BaseCoordinateFrame, frame_transform_graph, Galactic, ICRS

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
                return system_new()


def parse_input_healpix_data(input_data, field=0, hdu_in=None):
    """
    Parse input HEALPIX data to return a Numpy array and coordinate frame object.
    """

    if isinstance(input_data, (TableHDU, BinTableHDU)):

        # TODO: for now we have to write out to a temporary file. A pull
        # request to healpy has been merged to allow ``read_map`` to take
        # HDUList objects and HDUs, but we have to wait for a stable release
        # before we can use that:
        #
        # https://github.com/healpy/healpy/pull/249

        filename = tempfile.mktemp()
        input_data.writeto(filename)
        input_data = filename

    if isinstance(input_data, six.string_types):
        from healpy import read_map
        array_in, header = read_map(input_data, verbose=False, h=True, field=field, hdu=1 if hdu_in is None else hdu_in)
        coordinate_system_in = parse_coord_system(dict(header)['COORDSYS'])
    elif isinstance(input_data, tuple) and isinstance(input_data[0], np.ndarray):
        array_in = input_data[0]
        coordinate_system_in = parse_coord_system(input_data[1])
    else:
        raise TypeError("input_data should either be an HDU object or a tuple of (array, frame)")

    return array_in, coordinate_system_in
