.. _drizzle-options:

****************
Drizzle options
****************

This page describes the options that are specific to
:func:`~reproject.reproject_drizzle`, which carries out reprojection using
the drizzle algorithm described in `Fruchter and Hook (2002)
<https://doi.org/10.1086/338393>`_, as implemented in the `drizzle
<https://pypi.org/project/drizzle/>`_ package.

Kernel
======

The ``kernel`` argument can be used to select the kernel used to distribute
the flux of each input pixel onto the output pixels. The default
(``kernel='square'``) distributes the flux based on the overlap area of the
input and output pixels, and is therefore a flux-conserving algorithm
equivalent to :func:`~reproject.reproject_exact` (see
:ref:`choosing-algorithm` for a discussion of when to use each). See the
`drizzle documentation <https://spacetelescope-drizzle.readthedocs.io>`_ for
full details of the other kernels and their flux-conservation properties.

Pixel fraction
==============

The ``pixfrac`` argument can be used to shrink each input pixel before its
flux is distributed onto the output grid, which reduces the correlation
between neighboring output pixels when combining multiple dithered images.
Note that ``pixfrac`` values below 1 are not in general useful when
reprojecting a single image, since they leave gaps in the output.

Blocked and parallel reprojection
=================================

Since the drizzle algorithm distributes the flux of each input pixel over the
output pixels, input pixels contribute across output block boundaries, and
blocked (and therefore parallel) reprojection (see :doc:`../howto/chunked`) is
only supported when the blocks span the full extent of the celestial
dimensions, iterating only over leading non-reprojected dimensions (either
extra leading dimensions of the data, or dimensions designated with
``non_reprojected_dims``). When using ``non_reprojected_dims``, ``block_size``
can be left unset, in which case one block covering each non-reprojected
slice in full is used automatically.
