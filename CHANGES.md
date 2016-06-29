0.3 (unreleased)
----------------

- Allow users to pass a ``field=`` option to ``reproject_from_healpix``
  to access different fields in a HEALPIX file. [#86]

- Significant improvements to performance when the input data is a large
  memory-mapped array. [#105]

0.2 (2015-10-29)
----------------

- Fixed a bug that caused reprojection by interpolation to be truncated for
  rectangular output images.

0.1 (2015-05-08)
----------------

- Initial Release.
