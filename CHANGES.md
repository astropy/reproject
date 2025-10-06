## v0.16.0 - 2025-09-30

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

* Added 2-d HiPS dask array class and fix a number of bugs with HiPS generation by @astrofrog in https://github.com/astropy/reproject/pull/554

**Full Changelog**: https://github.com/astropy/reproject/compare/v0.15.0...v0.16.0

## v0.15.0 - 2025-08-01

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

#### New Features

* Flag out NaN weights in coadd & allow for weights to have different WCS than data by @keflavich in https://github.com/astropy/reproject/pull/474
* Add support for generating HiPS and reprojecting RGB images with AVM metadata by @astrofrog in https://github.com/astropy/reproject/pull/509
* Implement parallelization across broadcasted dimensions by @astrofrog in https://github.com/astropy/reproject/pull/376
* Allow dask arrays to not be written to memmap by @astrofrog in https://github.com/astropy/reproject/pull/543

#### Bug Fixes

* Fix a bug that caused big-endian dask arrays to not be reprojected correctly by @astrofrog in https://github.com/astropy/reproject/pull/487
* Force little endian input for zarr by @olebole in https://github.com/astropy/reproject/pull/492
* Support FITS files with distortion maps when given as file name by @svank in https://github.com/astropy/reproject/pull/477
* Don't mosaic empty empty slices by @lpsinger in https://github.com/astropy/reproject/pull/502
* FIx issue with co-adding with broadcasting with images that have inhomogeneous footprints by @astrofrog in https://github.com/astropy/reproject/pull/535
* Fix reprojection of compressed FITS files by @astrofrog in https://github.com/astropy/reproject/pull/540
* Fix threads=True option in reproject.hips by @astrofrog in https://github.com/astropy/reproject/pull/541
* Add a new keyword argument `negative_lon_cdelt` to `find_optimal_celestial_wcs` by @astrofrog in https://github.com/astropy/reproject/pull/528

#### Other Changes

* MNT: Replace ubuntu-20.04 with ubuntu-22.04 by @pllim in https://github.com/astropy/reproject/pull/490
* Add py313 and aarch64 to Actions by @Cadair in https://github.com/astropy/reproject/pull/494
* Avoid using `astropy.utils.isiterable()` by @eerovaher in https://github.com/astropy/reproject/pull/498
* Bump minimum version of Python to 3.11 and enable limited API builds/wheels by @astrofrog in https://github.com/astropy/reproject/pull/506
* Clean up RGB code and API by @astrofrog in https://github.com/astropy/reproject/pull/542
* Use APE-14 compliant pixel scale calculation in reproject.hips by @astrofrog in https://github.com/astropy/reproject/pull/545
* Pin more oldest dependencies by @astrofrog in https://github.com/astropy/reproject/pull/546
* Added a test to make sure that transparent PNGs generate HiPS tiles with transparency preserved by @astrofrog in https://github.com/astropy/reproject/pull/548

### New Contributors

* @eerovaher made their first contribution in https://github.com/astropy/reproject/pull/498

**Full Changelog**: https://github.com/astropy/reproject/compare/v0.14.1...v0.15.0

## v0.14.1 - 2024-11-19

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

#### Bug Fixes

* Enforce readonly mode for memmap when reading input array by @olebole in https://github.com/astropy/reproject/pull/461
* Fix Continuous Integration following changes in Sunpy 6.0.0 and 6.0.1 by @astrofrog in https://github.com/astropy/reproject/pull/472

#### Other Changes

* MNT: Use hash for Action workflow versions and update if needed by @pllim in https://github.com/astropy/reproject/pull/469
* Bump actions/checkout from 4.2.0 to 4.2.2 in the actions group by @dependabot in https://github.com/astropy/reproject/pull/476

**Full Changelog**: https://github.com/astropy/reproject/compare/v0.14.0...v0.14.1

## v0.14.0 - 2024-07-26

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

#### Bug Fixes

* Support readonly arrays in adaptive by @svank in https://github.com/astropy/reproject/pull/452
* Fix bug with artifacts in reproject_to_healpix by @astrofrog in https://github.com/astropy/reproject/pull/459

#### New Features

* Fix multi-threaded reprojection when using Astropy WCS by @astrofrog in https://github.com/astropy/reproject/pull/434
* Generalize reproject_and_coadd for N-dimensional data, and add option to specify blank pixel value and progress bar by @keflavich in https://github.com/astropy/reproject/pull/351
* Improve performance for large datasets and switch to multi-threading by default by @astrofrog in https://github.com/astropy/reproject/pull/443

#### Documentation

* Reorganized performance docs/tips by @astrofrog in https://github.com/astropy/reproject/pull/444

#### Other Changes

* Don't use --pre on Python 3.12 by @astrofrog in https://github.com/astropy/reproject/pull/445
* Bump minimum required version of astropy-healpix by @astrofrog in https://github.com/astropy/reproject/pull/446
* Improvements to performance when using dask.distributed by @astrofrog in https://github.com/astropy/reproject/pull/447
* Add logging calls and fix a couple of dask-related issues by @astrofrog in https://github.com/astropy/reproject/pull/450
* Fix CI following Sunpy 6.0.0 release by @astrofrog in https://github.com/astropy/reproject/pull/457
* Add a script to regenerate the aia asdf file and do so by @Cadair in https://github.com/astropy/reproject/pull/439
* Fix devdeps CI by @astrofrog in https://github.com/astropy/reproject/pull/458
* Performance improvements for interpolation with map_coordinates by @astrofrog in https://github.com/astropy/reproject/pull/448

**Full Changelog**: https://github.com/astropy/reproject/compare/v0.13.1...v0.14.0

## v0.13.1 - 2024-04-05

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

#### Bug Fixes

* Ensure reproject_and_coadd handles bg-matching with one input image by @svank in https://github.com/astropy/reproject/pull/412
* Fixes for mosaic output pixels not covered by inputs by @svank in https://github.com/astropy/reproject/pull/413

#### Documentation

* Updated docstrings for output_projection and shape_out to indicate that any APE-14 WCS is acceptable by @astrofrog in https://github.com/astropy/reproject/pull/407

#### Other Changes

* Add tests for full range of inputs/outputs in healpix functions by @astrofrog in https://github.com/astropy/reproject/pull/408
* Fix Cython warnings by @svank in https://github.com/astropy/reproject/pull/418
* Fix devdeps job by @astrofrog in https://github.com/astropy/reproject/pull/415
* BLD: pin extension-helpers to 1.* following upstream recommendation by @neutrinoceros in https://github.com/astropy/reproject/pull/420
* Added sp-repo-review to pre-commit by @astrofrog in https://github.com/astropy/reproject/pull/411
* Bump actions/checkout from 2 to 4 by @dependabot in https://github.com/astropy/reproject/pull/422
* Bump stefanzweifel/git-auto-commit-action from 4 to 5 by @dependabot in https://github.com/astropy/reproject/pull/423
* Add configuration for nightly wheels by @astrofrog in https://github.com/astropy/reproject/pull/417
* Fix pre-commit errors by @astrofrog in https://github.com/astropy/reproject/pull/429
* Enable testing of wheels on silicon mac by @astrofrog in https://github.com/astropy/reproject/pull/428
* Build against Numpy 2.0.0rc1 or later by @astrofrog in https://github.com/astropy/reproject/pull/436

### New Contributors

* @neutrinoceros made their first contribution in https://github.com/astropy/reproject/pull/420
* @dependabot made their first contribution in https://github.com/astropy/reproject/pull/422

**Full Changelog**: https://github.com/astropy/reproject/compare/v0.13.0...v0.13.1

## v0.13.0 - 2023-10-24

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

#### Bug Fixes

- Fix TestReprojectAndCoAdd failure on i386 by @olebole in https://github.com/astropy/reproject/pull/386
- Fixed a bug that caused reprojected dask arrays to not be computable due to a temporary directory being removed by @astrofrog in https://github.com/astropy/reproject/pull/390

#### New Features

- Add ability to specify output array and footprint in reproject_and_coadd by @astrofrog in https://github.com/astropy/reproject/pull/387
- Added ability to set `block_size='auto'` and fix missing parameters in docstrings by @astrofrog in https://github.com/astropy/reproject/pull/392
- Compute footprint in healpix_to_image by @lpsinger in https://github.com/astropy/reproject/pull/400

#### Other Changes

- Fix compatibility with Cython 3.0.2 and update version in pyproject.toml by @astrofrog in https://github.com/astropy/reproject/pull/391
- Add tests on Python 3.11 by @dstansby in https://github.com/astropy/reproject/pull/401
- Add testing on Python 3.12 by @dstansby in https://github.com/astropy/reproject/pull/399
- Python 3.12 testing by @dstansby in https://github.com/astropy/reproject/pull/403
- Add testing on Python 3.12 by @astrofrog in https://github.com/astropy/reproject/pull/402
- Enable Python 3.12 wheel building by @astrofrog in https://github.com/astropy/reproject/pull/405

### New Contributors

- @olebole made their first contribution in https://github.com/astropy/reproject/pull/386

**Full Changelog**: https://github.com/astropy/reproject/compare/v0.12.0...v0.13.0

## v0.12.0 - 2023-09-07

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

#### Bug Fixes

- Fix support for NDData objects with dask .data attributes by @astrofrog in https://github.com/astropy/reproject/pull/365
- Fix docs mosaic page rendering by @jdavies-st in https://github.com/astropy/reproject/pull/381

#### New Features

- Add despike_jacobian option for adaptive resampling by @svank in https://github.com/astropy/reproject/pull/366
- Refactor blocked/parallel reprojection by @astrofrog in https://github.com/astropy/reproject/pull/374
- Add 'first' and 'last' moasicking modes to reproject_and_coadd by @svank in https://github.com/astropy/reproject/pull/383
- Add modes for nan and inf handling to adaptive algo by @svank in https://github.com/astropy/reproject/pull/380
- Added new combine function to compute the minimum and maximum by @fjankowsk in https://github.com/astropy/reproject/pull/369

#### Other Changes

- TST: Update URL for Scientific Python nightlies by @pllim in https://github.com/astropy/reproject/pull/368
- Dask support improvements by @astrofrog in https://github.com/astropy/reproject/pull/367
- Fix --remote-data tests by @astrofrog in https://github.com/astropy/reproject/pull/375
- Update docstring for adaptive defaults by @svank in https://github.com/astropy/reproject/pull/378
- Use more points to find image bounds in moasics by @svank in https://github.com/astropy/reproject/pull/382
- Skip Python 3.12 wheels by @astrofrog in https://github.com/astropy/reproject/pull/385

### New Contributors

- @jdavies-st made their first contribution in https://github.com/astropy/reproject/pull/381
- @fjankowsk made their first contribution in https://github.com/astropy/reproject/pull/369

**Full Changelog**: https://github.com/astropy/reproject/compare/v0.11.0...v0.12.0

## v0.11.0 - 2023-05-19

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

#### Bug Fixes

- Fix for HighLevelWCS attribute error by @keflavich in https://github.com/astropy/reproject/pull/349
- Fixes for solar frames and non-degree units by @astrofrog in https://github.com/astropy/reproject/pull/360
- If shape_out is specified, use this over the array_shape attribute of a WCS object by @astrofrog in https://github.com/astropy/reproject/pull/361

#### New Features

- Allow single inputs to find_optimal_celestial_wcs and add ability to specify HDU by @astrofrog in https://github.com/astropy/reproject/pull/344
- Add support for specifying output projection as APE 14 WCS with array_shape defined by @astrofrog in https://github.com/astropy/reproject/pull/345
- Started adding support for allowing dask arrays as input by @astrofrog in https://github.com/astropy/reproject/pull/352

#### Other Changes

- Mark tests that use remote data by @smaret in https://github.com/astropy/reproject/pull/339
- Fix code style by @astrofrog in https://github.com/astropy/reproject/pull/340
- Simplify blocked reprojection implementation by using dask and improve efficiency of parallel reprojection by @astrofrog in https://github.com/astropy/reproject/pull/314
- Remove code that was required for astropy<4 by @astrofrog in https://github.com/astropy/reproject/pull/346
- Add a new 'all' extras for shapely by @astrofrog in https://github.com/astropy/reproject/pull/363

### New Contributors

- @smaret made their first contribution in https://github.com/astropy/reproject/pull/339

**Full Changelog**: https://github.com/astropy/reproject/compare/v0.10.0...v0.11.0

## v0.10.0 - 2023-01-30

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

#### Bug Fixes

- Close FITS files after loading by @svank in https://github.com/astropy/reproject/pull/330

#### New Features

- Add support for blocked and parallel reprojection in `reproject_interp` by @AlistairSymonds in https://github.com/astropy/reproject/pull/214
- Add support for efficiently reprojecting multiple images with the same wcs by @svank in https://github.com/astropy/reproject/pull/332
- Add support for APE 14 WCSes in find_optimal_celestial_wcs by @astrofrog in https://github.com/astropy/reproject/pull/334

#### Other Changes

- Update package infrastructure by @Cadair in https://github.com/astropy/reproject/pull/304
- Changed default filter kernel and boundary mode in `reproject_adaptive`, and removed `order` argument. by @svank in https://github.com/astropy/reproject/pull/291
- Skip wheel tests on manylinux_aarch64 by @astrofrog in https://github.com/astropy/reproject/pull/307
- Reformat all Python code using Black by @Cadair in https://github.com/astropy/reproject/pull/308
- Use pixel_to_pixel from astropy.wcs.utils by @astrofrog in https://github.com/astropy/reproject/pull/315
- Test CI on Python 3.11 beta by @dstansby in https://github.com/astropy/reproject/pull/298
- Update pinned version of Cython by @astrofrog in https://github.com/astropy/reproject/pull/316
- Speed up test_blocked_against_single by increasing smallest block size by @astrofrog in https://github.com/astropy/reproject/pull/319
- Fix weird quotation marks from Black auto-formatting by @svank in https://github.com/astropy/reproject/pull/331
- Fix CI by @astrofrog in https://github.com/astropy/reproject/pull/333

### New Contributors

- @AlistairSymonds made their first contribution in https://github.com/astropy/reproject/pull/214

**Full Changelog**: https://github.com/astropy/reproject/compare/v0.9...v0.10.0

## 0.9 - 2022-09-02

- Drop support for Python 3.7.
-
- Infrastructure and packaging updates.
-
- Made many improvements, bug fixes, and significant speed-ups for the adaptive
- resampling algorithm, `reproject_adaptive`. These bug fixes may cause
- changes to the reprojected images, which are typically negligible.
- Improvements include the addition of a flux-conserving mode, support for a
- Gaussian filter kernel, a menu of boundary-handling modes, and a
- `center_jacobian` flag to trade speed for accuracy with rapidly-varying
- transformations.
-
- Added a `roundtrip_coords` argument to `reproject_adaptive` and
- `reproject_interp`. By default, all coordinate transformations are run in
- both directions to handle some situations where they are ambiguous. This can
- be disabled by setting `roundtrip_coords=False` which may offer a
- significant speed increase.
-

## 0.8 - 2021-08-11

- Improve `find_optimal_celestial_wcs` to accept input data descriptions as
- just array shapes, not necessarily fully populated arrays. This makes it
- possible to solve for the optimal WCS for a set of images that couldn't fit
- into memory all at once, since the actual data aren't needed for optimal WCS
- determination. [#242]
-
- Fix implementation of `hdu_weights` in `reproject_and_coadd`. [#249]
-

## 0.7.1 - 2020-05-29

- Fixed compatibility with Astropy 4.1. [#234]
-
- Updated minimum requirement for SciPy. [#236]
-

## 0.7 - 2020-04-02

- Made C extension in overlapArea.c thread-safe by removing global
- variables. [#211]
-
- Made it possible to control whether to output debugging information
- from overlapArea.c by setting DEBUG_OVERLAP_AREA=1 at build-time. [#211]
-
- Fix compatibility with astropy v4.0.1. [#227]
-
- Disable parallelization by default in `reproject_exact` - this can be
- enabled with `parallel=True`. [#227]
-
- Fixed a bug with `reproject_exact` with `parallel=False` and
- `return_footprint=False`, which caused the footprint to be returned
- anyway. [#227]
-
- The infrastructure of the package has been updated in line with the
- APE 17 roadmap (https://github.com/astropy/astropy-APEs/blob/main/APE17.rst).
- The main changes are that the `python setup.py test` and
- `python setup.py build_docs` commands will no longer work. The
- easiest way to replicate these commands is to install the tox
- (https://tox.readthedocs.io) package and run `tox -e test` and
- `tox -e build_docs`. It is also possible to run pytest and sphinx
- directly. [#228]
-

## 0.6 - 2019-11-01

- Added support for using any WCS that conforms to the WCS API described
- in the Astropy Proposal for Enhancements 14 (APE 14). The
- `independent_celestial_slices=` argument to `reproject_interp` has
- been deprecated since it is no longer needed, as transformations are
- automatically done in the most efficient way possible. [#166]
-
- Include a warning for high resolution images with `reproject_exact`,
- since if the pixels are <0.05", precision issues can occur. [#200]
-
- Added a new `reproject_and_coadd` function for doing mosaicking of
- individual images, and added section in documentation about mosaicking.
- [#186]
-
- Added a new reproject.adaptive sub-package that implements the DeForest
- (2004) algorithm for reprojection. [#52]
-
- Fixed a bug that caused 'exact' reprojection results to have numerical
- issues when doing identity transformations. [#190]
-

## 0.5.1 - 2019-09-01

- Fixed a bug that caused 'exact' reprojection to fail if one or more of
- the WCSes was oriented such that E and W were flipped. [#188]

## 0.5 - 2019-06-13

- Improve parse_output_projection to make it so that the output projection
- can be specified as a filename. [#150]
-
- Fixed a bug that caused HEALPix maps in RING order to not be correctly
- interpreted. [#163]
-
- Make it possible to specify the output array for reprojection using the
- `output_array=` keyword argument. [#115]
-

## 0.4 - 2018-01-29

- Refactored HEALPix reprojection code to use the astropy-healpix package
- instead of healpy. [#139]
-
- Added the ability to specify an output array in `reproject_interp`, which
- permits the use of memory-mapped arrays and therefore provides the capability
- to handle data cubes much larger than memory [#115]
-
- Fix test 32-bit test failures. [#146]
-
- Fix an issue with reprojecting images where there are two solutions along
- the line of sight by forcing round-tripping of coordinate conversions [#129]
-
- Explicitly define default HDU as 0 for normal reprojection and 1 for
- HEALPix reprojection. [#119]
-
- Added a function to find the optimal WCS for a set of images. [#136, #137]
-

## 0.3.2 - 2017-10-22

- Fix a regression that caused certain all-sky images (e.g. the Mellinger Milky
- Way Panorama, http://www.milkywaysky.com) to be reprojected to all NaNs when
- the output WCS was in Mollweide coordinates. [#124]

## 0.3.1 - 2016-07-07

- Include missing license file in tarball.
-
- Updated documentation to remove warnings about early versions.
-

## 0.3 - 2016-07-06

- Allow users to pass a `field=` option to `reproject_from_healpix`
- to access different fields in a HEALPIX file. [#86]
-
- Significant improvements to performance when the input data is a large
- memory-mapped array. [#105]
-
- Significant refactoring of interpolating reprojection to improve support for
- n-dimensional arrays, optionally including two celestial axes (in which
- case the coordinate transformation is taken into account). [#96, #102]
-

## 0.2 - 2015-10-29

- Fixed a bug that caused reprojection by interpolation to be truncated for
- rectangular output images.

## 0.1 - 2015-05-08

- Initial Release.
