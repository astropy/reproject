import os

import numpy as np
from setuptools import Extension


def get_extensions():
    libraries = []

    sources = []
    sources.append(os.path.join(REPROJECT_ROOT, "deforest.pyx"))

    include_dirs = [np.get_include()]

    define_macros = []

    define_macros.append(("CYTHON_LIMITED_API", "0x030800f0"))
    define_macros.append(("Py_LIMITED_API", "0x030800f0"))

    extension = Extension(
        name="reproject.adaptive.deforest",
        sources=sources,
        include_dirs=include_dirs,
        libraries=libraries,
        language="c",
        extra_compile_args=["-O2"],
        define_macros=define_macros,
        py_limited_api=True,
    )

    return [extension]
