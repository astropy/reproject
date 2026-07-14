.. _broadcasting:

**********************************************
Reprojecting only some dimensions of a dataset
**********************************************

The reprojection functions can operate on data with more dimensions than you
actually want to reproject - for example a spectral cube where only the
celestial dimensions should be reprojected, or a stack of images that share
the same coordinates. For the dimensions that are not being reprojected, a
one-to-one mapping between input and output pixels is assumed, and the
coordinate mapping for the reprojected dimensions is computed only once and
reused for each slice, which is significantly faster than reprojecting the
slices one at a time. There are two ways of doing this, described in the
sections below.

Data with more dimensions than the WCS
======================================

When the input array contains more dimensions than the input WCS describes,
the extra leading dimensions are automatically treated as non-reprojected
dimensions, and the reprojection loops over those dimensions after computing
the pixel mapping. For example, consider a spectral cube with 500 spectral
channels, where ``wcs_in`` and ``wcs_out`` describe only the two celestial
dimensions:

.. doctest-skip::

    >>> cube.shape
    (500, 2048, 2048)
    >>> reprojected, footprint = reproject_interp(
    ...         (cube, wcs_in), wcs_out, shape_out=(2048, 2048))
    >>> reprojected.shape
    (500, 2048, 2048)

When the output coordinates are provided as a WCS and a ``shape_out`` tuple,
``shape_out`` may describe the output shape of a single slice, as in the
example above, in which case the extra leading dimensions are prepended
automatically, or it may include the extra dimensions, in which case the size
of the extra dimensions must match those of the input data exactly.

.. _non-reprojected-dims:

When the WCS also describes the extra dimensions
================================================

In some cases, the WCS describes all of the dimensions of the data - for
example a spectral cube with a 3-dimensional WCS - and you may still want to
reproject only the celestial dimensions, keeping the remaining dimensions
untouched. Rather than manually slicing the WCS down to its celestial
dimensions, you can use the ``non_reprojected_dims`` option to specify the
leading dimensions that should not be reprojected. Taking the same cube as
above, but with ``wcs_in`` and ``wcs_out`` that are now both 3-dimensional:

.. doctest-skip::

    >>> cube.shape
    (500, 2048, 2048)
    >>> reprojected, footprint = reproject_interp(
    ...         (cube, wcs_in), wcs_out, shape_out=(500, 2048, 2048),
    ...         non_reprojected_dims=(0,), parallel=True)
    >>> reprojected.shape
    (500, 2048, 2048)

In this case, each spectral channel in the output is computed from the same
channel in the input. The remaining part of the WCS is sliced at the position
of each channel, so it does not need to be identical from slice to slice - for
example, this can be used to reproject a cube whose celestial coordinates
drift along a non-reprojected time axis. The dimensions must be the leading
ones, given as a tuple of sequential integers starting from zero (e.g. ``(0,)``
or ``(0, 1)``).

In this mode, each slice along the non-reprojected dimensions is processed as
a separate block, and setting ``parallel=True`` (or an integer number of
threads) as in the example above processes the slices concurrently, which can
result in significant speedups. If ``block_size`` is specified explicitly, its
entries along the reprojected dimensions have to match ``shape_out``.

Note that ``non_reprojected_dims`` is currently supported by
:func:`~reproject.reproject_interp` and :func:`~reproject.reproject_adaptive`,
but not yet by :func:`~reproject.reproject_exact`.
