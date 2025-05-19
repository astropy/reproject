import os

import numpy as np
from setuptools import Extension

REPROJECT_ROOT = os.path.relpath(os.path.dirname(__file__))


def get_extensions():
    libraries = []

    sources = []
    sources.append(os.path.join(REPROJECT_ROOT, "_overlap.pyx"))
    sources.append(os.path.join(REPROJECT_ROOT, "overlapArea.c"))
    sources.append(os.path.join(REPROJECT_ROOT, "reproject_slice_c.c"))

    include_dirs = [np.get_include()]
    include_dirs.append(REPROJECT_ROOT)

    # Note that to set the DEBUG variable in the overlapArea.c code, which
    # results in debugging information being printed out, you can set
    # DEBUG_OVERLAP_AREA=1 at build-time.
    if int(os.environ.get("DEBUG_OVERLAP_AREA", 0)):
        define_macros = [("DEBUG_OVERLAP_AREA", 1)]
    else:
        define_macros = []

    define_macros.append(('CYTHON_LIMITED_API', '0x030800f0'))
    define_macros.append(('Py_LIMITED_API', '0x030800f0'))

    extension = Extension(
        name="reproject.spherical_intersect._overlap",
        sources=sources,
        include_dirs=include_dirs,
        libraries=libraries,
        language="c",
        extra_compile_args=["-O2"],
        define_macros=define_macros,
        py_limited_api=True,
    )

    return [extension]
