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

        import os
        import shutil
        import tempfile

        # First ensure that we build the package so that 2to3 gets executed
        self.reinitialize_command('build')
        self.run_command('build')
        build_cmd = self.get_finalized_command('build')
        new_path = os.path.abspath(build_cmd.build_lib)

        # Copy the build to a temporary directory for the purposes of testing
        # - this avoids creating pyc and __pycache__ directories inside the
        # build directory
        tmp_dir = tempfile.mkdtemp(prefix='reprojection-test-')
        testing_path = os.path.join(tmp_dir, os.path.basename(new_path))
        shutil.copytree(new_path, testing_path)

        import sys
        import subprocess

        errno = subprocess.call([sys.executable, os.path.abspath('runtests.py')], cwd=testing_path)
        raise SystemExit(errno)


setup(name='reproject',
      version="0.1.0",
      author='Thomas Robitaille',
      author_email='thomas.robitaille@gmail.com',
      packages=['reproject', 'reproject.tests'],
      cmdclass = {'test': PyTest},
      ext_modules = ext_modules
     )
