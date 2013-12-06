#!/usr/bin/env python

import os

from distutils.core import setup, Extension, Command
from distutils.command.sdist import sdist

from distutils.command.build_py import build_py

from numpy import get_include as get_numpy_include

numpy_includes = get_numpy_include()

ext_modules = [Extension("reproject._overlap_wrapper",
                         ['reproject/_overlap_wrapper.c', 'reproject/overlapArea.c'],
                         include_dirs=[numpy_includes])]

class PyTest(Command):
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        import sys,subprocess
        errno = subprocess.call([sys.executable, 'runtests.py'])
        raise SystemExit(errno)


setup(name='reproject',
      version="0.1.0",
      author='Thomas Robitaille',
      author_email='thomas.robitaille@gmail.com',
      packages=['reproject'],
      cmdclass = {'test': PyTest},
      ext_modules = ext_modules
     )
