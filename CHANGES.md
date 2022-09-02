0.9 (2022-09-02)
----------------

- Drop support for Python 3.7.
- Infrastructure and packaging updates.
- Made many improvements, bug fixes, and significant speed-ups for the adaptive
  resampling algorithm, ``reproject_adaptive``. These bug fixes may cause
  changes to the reprojected images, which are typically negligible.
  Improvements include the addition of a flux-conserving mode, support for a
  Gaussian filter kernel, a menu of boundary-handling modes, and a
  ``center_jacobian`` flag to trade speed for accuracy with rapidly-varying
  transformations.
- Added a ``roundtrip_coords`` argument to ``reproject_adaptive`` and
  ``reproject_interp``. By default, all coordinate transformations are run in
  both directions to handle some situations where they are ambiguous. This can
  be disabled by setting ``roundtrip_coords=False`` which may offer a
  significant speed increase.

0.8 (2021-08-11)
----------------

- Improve ``find_optimal_celestial_wcs`` to accept input data descriptions as
  just array shapes, not necessarily fully populated arrays. This makes it
  possible to solve for the optimal WCS for a set of images that couldn't fit
  into memory all at once, since the actual data aren't needed for optimal WCS
  determination. [#242]

- Fix implementation of ``hdu_weights`` in ``reproject_and_coadd``. [#249]

0.7.1 (2020-05-29)
------------------

- Fixed compatibility with Astropy 4.1. [#234]

- Updated minimum requirement for SciPy. [#236]

0.7 (2020-04-02)
----------------

- Made C extension in overlapArea.c thread-safe by removing global
  variables. [#211]

- Made it possible to control whether to output debugging information
  from overlapArea.c by setting DEBUG_OVERLAP_AREA=1 at build-time. [#211]

- Fix compatibility with astropy v4.0.1. [#227]

- Disable parallelization by default in ``reproject_exact`` - this can be
  enabled with ``parallel=True``. [#227]

- Fixed a bug with ``reproject_exact`` with ``parallel=False`` and
  ``return_footprint=False``, which caused the footprint to be returned
  anyway. [#227]

- The infrastructure of the package has been updated in line with the
  APE 17 roadmap (https://github.com/astropy/astropy-APEs/blob/main/APE17.rst).
  The main changes are that the ``python setup.py test`` and
  ``python setup.py build_docs`` commands will no longer work. The
  easiest way to replicate these commands is to install the tox
  (https://tox.readthedocs.io) package and run ``tox -e test`` and
  ``tox -e build_docs``. It is also possible to run pytest and sphinx
  directly. [#228]

0.6 (2019-11-01)
----------------

- Added support for using any WCS that conforms to the WCS API described
  in the Astropy Proposal for Enhancements 14 (APE 14). The
  ``independent_celestial_slices=`` argument to ``reproject_interp`` has
  been deprecated since it is no longer needed, as transformations are
  automatically done in the most efficient way possible. [#166]

- Include a warning for high resolution images with ``reproject_exact``,
  since if the pixels are <0.05", precision issues can occur. [#200]

- Added a new ``reproject_and_coadd`` function for doing mosaicking of
  individual images, and added section in documentation about mosaicking.
  [#186]

- Added a new reproject.adaptive sub-package that implements the DeForest
  (2004) algorithm for reprojection. [#52]

- Fixed a bug that caused 'exact' reprojection results to have numerical
  issues when doing identity transformations. [#190]

0.5.1 (2019-09-01)
------------------

- Fixed a bug that caused 'exact' reprojection to fail if one or more of
  the WCSes was oriented such that E and W were flipped. [#188]

0.5 (2019-06-13)
----------------

- Improve parse_output_projection to make it so that the output projection
  can be specified as a filename. [#150]

- Fixed a bug that caused HEALPix maps in RING order to not be correctly
  interpreted. [#163]

- Make it possible to specify the output array for reprojection using the
  ``output_array=`` keyword argument. [#115]

0.4 (2018-01-29)
----------------

- Refactored HEALPix reprojection code to use the astropy-healpix package
  instead of healpy. [#139]

- Added the ability to specify an output array in ``reproject_interp``, which
  permits the use of memory-mapped arrays and therefore provides the capability
  to handle data cubes much larger than memory [#115]

- Fix test 32-bit test failures. [#146]

- Fix an issue with reprojecting images where there are two solutions along
  the line of sight by forcing round-tripping of coordinate conversions [#129]

- Explicitly define default HDU as 0 for normal reprojection and 1 for
  HEALPix reprojection. [#119]

- Added a function to find the optimal WCS for a set of images. [#136, #137]

0.3.2 (2017-10-22)
------------------

- Fix a regression that caused certain all-sky images (e.g. the Mellinger Milky
  Way Panorama, http://www.milkywaysky.com) to be reprojected to all NaNs when
  the output WCS was in Mollweide coordinates. [#124]

0.3.1 (2016-07-07)
------------------

- Include missing license file in tarball.

- Updated documentation to remove warnings about early versions.

0.3 (2016-07-06)
----------------

- Allow users to pass a ``field=`` option to ``reproject_from_healpix``
  to access different fields in a HEALPIX file. [#86]

- Significant improvements to performance when the input data is a large
  memory-mapped array. [#105]

- Significant refactoring of interpolating reprojection to improve support for
  n-dimensional arrays, optionally including two celestial axes (in which
  case the coordinate transformation is taken into account). [#96, #102]

0.2 (2015-10-29)
----------------

- Fixed a bug that caused reprojection by interpolation to be truncated for
  rectangular output images.

0.1 (2015-05-08)
----------------

- Initial Release.
