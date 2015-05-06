import os
from distutils.core import Extension

REPROJECT_ROOT = os.path.relpath(os.path.dirname(__file__))


def get_extensions():
    sources = [os.path.join(REPROJECT_ROOT, "_overlap.pyx")]
    include_dirs = ['numpy']
    libraries = []

    sources.append("reproject/spherical_intersect/overlapArea.c")
    sources.append("reproject/spherical_intersect/reproject_slice_c.c")
    include_dirs.append('reproject/spherical_intersect')

    extension = Extension(
        name="reproject.spherical_intersect._overlap",
        sources=sources,
        include_dirs=include_dirs,
        libraries=libraries,
        language="c",
        extra_compile_args=['-O2'])

    return [extension]
