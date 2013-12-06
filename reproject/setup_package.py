from distutils.core import Extension


def get_extensions():

    from numpy import get_include as get_numpy_include

    numpy_includes = get_numpy_include()

    ext_modules = [Extension("reproject._overlap_wrapper",
                             ['reproject/_overlap_wrapper.c', 'reproject/overlapArea.c'],
                             include_dirs=[numpy_includes])]

    return ext_modules


def requires_2to3():
    return False
