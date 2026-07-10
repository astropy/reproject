************************
Working with dask arrays
************************

This page describes how to use dask arrays as input to and output from the
reprojection functions, as well as considerations specific to mosaicking.

.. testsetup::

    >>> import numpy as np
    >>> import dask.array as da
    >>> input_array = np.random.random((1024, 1024))
    >>> dask_array = da.from_array(input_array, chunks=(128, 128))
    >>> from astropy.wcs import WCS
    >>> wcs_in = WCS(naxis=2)
    >>> wcs_out = WCS(naxis=2)
    >>> from reproject import reproject_interp

Input dask arrays
=================

The three main reprojection functions can accept dask arrays as inputs, e.g.
assuming you have already constructed a dask array named ``dask_array``::

    >>> dask_array
    dask.array<array, shape=(1024, 1024), dtype=float64, chunksize=(128, 128), chunktype=numpy.ndarray>

you can pass this in as part of the first argument to one of the reprojection
functions::

    >>> array, footprint = reproject_interp((dask_array, wcs_in), wcs_out,
    ...                                     shape_out=(2048, 2048))

In general however, we cannot benefit much from the chunking of the input arrays
because any input pixel might in principle contribute to any output pixel.
Therefore, for now, when a dask array is passed as input, it is computed using
the current default scheduler and converted to a Numpy memory-mapped array. This
is done efficiently in terms of memory and never results in the whole dataset
being loaded into memory at any given time. However, this does require
sufficient space on disk to store the array. If your default system temporary
directory does not have sufficient space, you can set the ``TMPDIR`` environment
variable to point at another directory:

    >>> import os
    >>> os.environ['TMPDIR'] = '/home/lancelot/tmp'


Output dask arrays
==================

By default, the reprojection functions will do the computation immediately and
return Numpy arrays for the reprojected array and optionally the footprint (this
is regardless of whether dask or Numpy arrays were passed in, or of the
parallelization options described in :ref:`multithreading`). However, by setting
``return_type='dask'``, you can make the functions delay any computation and
return dask arrays::

    >>> array, footprint = reproject_interp((input_array, wcs_in), wcs_out,
    ...                                     shape_out=(2048, 2048), block_size=(256, 256),
    ...                                     return_type='dask')
    >>> array
    dask.array<getitem, shape=(2048, 2048), dtype=float64, chunksize=(256, 256), ...>

You can then compute the array or a section of the array yourself whenever you need, or use the
result in further dask expressions.

Mosaicking
==========

The :func:`~reproject.mosaicking.reproject_and_coadd` function also supports
dask arrays as input and output. Input datasets can be passed in as ``(array,
wcs)`` tuples where the array is a dask array. In this case, make sure that
each input dask array has a unique name (as in ``array.name``), since dask
treats arrays with the same name as being the same array - a warning is
emitted if two input arrays share a name.

Setting ``return_type='dask'`` returns uncomputed dask arrays for the mosaic
and footprint, in which each chunk of the output is assembled from the images
that overlap it, so that the whole co-addition is computed lazily in one go.
Note that ``match_background``, ``output_array``, ``output_footprint`` and
``intermediate_memmap`` are not supported in this mode, since the result is an
uncomputed graph rather than arrays filled in place.

For large mosaics, however, computing the ``return_type='dask'`` result in one
go may not be optimal, since depending on the order in which the scheduler
chooses to compute the output chunks, a large number of intermediate arrays
may need to be held in memory at any given time. In this case, it may be
better to use ``return_type='zarr'``, which computes the same result but does
so here, slab by slab along the first dimension of the output, into a zarr
store on disk at ``zarr_path``, and returns dask arrays that read from that
store:

.. doctest-skip::

    >>> array, footprint = reproject_and_coadd(input_data,
    ...                                        wcs_out, shape_out=shape_out,
    ...                                        reproject_function=reproject_interp,
    ...                                        return_type='zarr',
    ...                                        zarr_path='mosaic.zarr')

This bounds the memory used by the computation to a single slab regardless of
the size of the mosaic, at the cost of recomputing the reprojected chunks of
any image that spans a slab boundary. The size of the slabs can be controlled
with the ``zarr_batch_size`` option, which gives the number of output chunks
to compute per slab - by default this is chosen such that one slab of the
output is around 2 GB. The slabs are computed with the scheduler implied by
the ``parallel`` keyword, with the same semantics as for the individual
reprojection functions.

Using dask.distributed
======================

The `dask.distributed <https://distributed.dask.org/en/stable/>`_ package makes it
possible to use distributed schedulers for dask. In order to compute
reprojections or mosaics with dask.distributed, set up the client and then call the
reprojection functions or :func:`~reproject.mosaicking.reproject_and_coadd`
with ``parallel='current-scheduler'``. Alternatively, you can make use of the
``return_type='dask'`` option mentioned above so that you can compute the dask
array once the distributed scheduler has been set up.
