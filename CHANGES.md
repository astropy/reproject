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
