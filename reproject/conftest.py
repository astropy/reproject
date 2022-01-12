# This file is used to configure the behavior of pytest when using the Astropy
# test infrastructure. It needs to live inside the package in order for it to
# get picked up when running the tests inside an interpreter using
# packagename.test

import os

try:
    from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS
    ASTROPY_HEADER = True
except ImportError:
    ASTROPY_HEADER = False

os.environ['MPLBACKEND'] = 'Agg'


def pytest_configure(config):

    if ASTROPY_HEADER:

        config.option.astropy_header = True

        PYTEST_HEADER_MODULES.pop('Pandas', None)
        PYTEST_HEADER_MODULES.pop('h5py', None)
        PYTEST_HEADER_MODULES.pop('Matplotlib', None)
        PYTEST_HEADER_MODULES['Astropy'] = 'astropy'
        PYTEST_HEADER_MODULES['astropy-healpix'] = 'astropy_healpix'
        PYTEST_HEADER_MODULES['Cython'] = 'cython'

        from reproject import __version__
        TESTED_VERSIONS['reproject'] = __version__
