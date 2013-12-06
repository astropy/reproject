#!/usr/bin/env python

import os

from distutils.core import setup, Extension
from distutils.command.sdist import sdist

try:  # Python 3.x
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:  # Python 2.x
    from distutils.command.build_py import build_py

from numpy import get_include as get_numpy_include

numpy_includes = get_numpy_include()

ext_modules = [Extension("reproject._overlap_wrapper",
                         ['reproject/_overlap_wrapper.c', 'reproject/overlapArea.c'],
                         include_dirs=[numpy_includes])]

setup(name='reproject',
      version="0.1.0",
      author='Thomas Robitaille',
      author_email='thomas.robitaille@gmail.com',
      packages=['reproject'],
      ext_modules = ext_modules
     )
