0.6 (unreleased)
----------------

- No changes yet.

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
