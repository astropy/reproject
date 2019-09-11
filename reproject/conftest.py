# This file is used to configure the behavior of pytest when using the Astropy
# test infrastructure.
import os

from astropy.tests.plugins.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS
from astropy.tests.helper import enable_deprecations_as_exceptions

enable_deprecations_as_exceptions()

PYTEST_HEADER_MODULES['Astropy'] = 'astropy'
PYTEST_HEADER_MODULES['astropy-healpix'] = 'astropy_healpix'
PYTEST_HEADER_MODULES['Cython'] = 'cython'
del PYTEST_HEADER_MODULES['h5py']
del PYTEST_HEADER_MODULES['Matplotlib']

from .version import version, astropy_helpers_version  # noqa

packagename = os.path.basename(os.path.dirname(__file__))
TESTED_VERSIONS[packagename] = version
TESTED_VERSIONS['astropy_helpers'] = astropy_helpers_version
