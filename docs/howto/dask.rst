************************
Working with dask arrays
************************

This page describes how to use dask arrays as input to and output from the
reprojection functions. Note that this does not yet apply to the mosaicking
functions.

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

Using dask.distributed
======================

The `dask.distributed <https://distributed.dask.org/en/stable/>`_ package makes it
possible to use distributed schedulers for dask. In order to compute
reprojections with dask.distributed, set up the client and then call the reprojection
functions with ``parallel='current-scheduler'``. Alternatively, you can make use of the
``return_type='dask'`` option mentioned above so that you can compute the dask
array once the distributed scheduler has been set up.
