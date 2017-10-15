# this contains imports plugins that configure py.test for astropy tests.
# by importing them here in conftest.py they are discoverable by py.test
# no matter how it is invoked within the source tree.

from astropy.tests.pytest_plugins import *
from distutils.version import LooseVersion

# Uncomment the following line to treat all DeprecationWarnings as
# exceptions
# TODO: remove warnings_to_ignore_entire_module once
# https://github.com/astrofrog/pytest-arraydiff/pull/10 is sorted out and a
# new release is available
import astropy
if LooseVersion(astropy.__version__) < LooseVersion('2.0'):
    enable_deprecations_as_exceptions()
else:
    enable_deprecations_as_exceptions(warnings_to_ignore_entire_module=['astropy.io.fits'])

# Uncomment and customize the following lines to add/remove entries from
# the list of packages for which version numbers are displayed when running
# the tests. Making it pass for KeyError is essential in some cases when
# the package uses other astropy affiliated packages.
try:
    PYTEST_HEADER_MODULES['Astropy'] = 'astropy'
    PYTEST_HEADER_MODULES['Healpy'] = 'healpy'
    PYTEST_HEADER_MODULES['Cython'] = 'cython'
    del PYTEST_HEADER_MODULES['h5py']
    del PYTEST_HEADER_MODULES['Matplotlib']
except (NameError, KeyError):  # NameError is needed to support Astropy < 1.0
    pass

# Uncomment the following lines to display the version number of the
# package rather than the version number of Astropy in the top line when
# running the tests.
import os

# This is to figure out the affiliated package version, rather than
# using Astropy's
try:
    from .version import version
except ImportError:
    version = 'dev'

try:
    packagename = os.path.basename(os.path.dirname(__file__))
    TESTED_VERSIONS[packagename] = version
except NameError:   # Needed to support Astropy <= 1.0.0
    pass
