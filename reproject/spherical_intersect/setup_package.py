import os
from distutils.core import Extension

REPROJECT_ROOT = os.path.relpath(os.path.dirname(__file__))


def get_extensions():

    libraries = []

    sources = []
    sources.append(os.path.join(REPROJECT_ROOT, "_overlap.c"))
    sources.append(os.path.join(REPROJECT_ROOT, "overlapArea.c"))
    sources.append(os.path.join(REPROJECT_ROOT, "reproject_slice_c.c"))

    include_dirs = ['numpy']
    include_dirs.append(REPROJECT_ROOT)

    extension = Extension(
        name="reproject.spherical_intersect._overlap",
        sources=sources,
        include_dirs=include_dirs,
        libraries=libraries,
        language="c",
        extra_compile_args=['-O2'])

    return [extension]


def get_package_data():

    header_files = ['overlapArea.h', 'reproject_slice_c.h', 'mNaN.h']

    return {'reproject.spherical_intersect': header_files}
