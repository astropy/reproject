Integration with dask and parallel processing
=============================================

The following functions all integrate well with the `dask <https://www.dask.org/>`_ library.

* :func:`~reproject.reproject_interp`
* :func:`~reproject.reproject_adaptive`
* :func:`~reproject.reproject_exact`

This integration has several aspects that we will discuss in the following sections.

.. testsetup::

    >>> import numpy as np
    >>> import dask.array as da
    >>> input_array = np.random.random((1024, 1024))
    >>> dask_array = da.from_array(input_array, chunks=(128, 128))
    >>> from astropy.wcs import WCS
    >>> wcs_in = WCS(naxis=2)
    >>> wcs_out = WCS(naxis=2)

Input dask arrays
-----------------

The three reprojection functions mentioned above can accept dask arrays as
inputs, e.g. assuming you have already constructed a dask array named
``dask_array``::

    >>> dask_array
    dask.array<array, shape=(1024, 1024), dtype=float64, chunksize=(128, 128), chunktype=numpy.ndarray>

you can pass this in as part of the first argument to one of the reprojection
functions::

    >>> from reproject import reproject_interp
    >>> array, footprint = reproject_interp((dask_array, wcs_in), wcs_out,
    ...                                     shape_out=(2048, 2048))

In general however, we cannot benefit much from the chunking of the input arrays
because any input pixel might in principle contribute to any output pixel.
Therefore, for now, when a dask array is passed as input, it is computed using
the current default scheduler and converted to a Numpy memory-mapped array. This
is done efficiently in terms of memory and never results in the whole dataset
being loaded into memory at any given time. However, this does require
sufficient space on disk to store the array.

Chunk by chunk reprojection and parallel processing
---------------------------------------------------

Regardless of whether a dask or Numpy array is passed in as input to the
reprojection functions, you can specify a block size to use for the
reprojection, and this is used to iterate over chunks in the output array in
chunks. For instance, if you pass in a (1024, 1024) array and specify that the
shape of the output should be a (2048, 2048) array (e.g., via ``shape_out``),
then if you set ``block_size=(256, 256)``::

    >>> input_array.shape
    (1024, 1024)
    >>> array, footprint = reproject_interp((input_array, wcs_in), wcs_out,
    ...                                     shape_out=(2048, 2048), block_size=(256, 256))

the reprojection will be done in 64 separate output chunks. Note however that
this does not break up the input array into chunks since in the general case any
input pixel may contribute to any output pixel.

By default, the iteration over the output chunks is done in a single
process/thread, but you may specify ``parallel=True`` to process these in
parallel. If you do this, reproject will use multiple processes (rather than
threads) to parallelize the computation (this is because the core reprojection
algorithms we use are not currently thread-safe). If you specify
``parallel=True``, then ``block_size`` will be automatically set to a sensible
default, but you can also set ``block_size`` manually for more control. Note
that you can also set ``parallel=`` to an integer to indicate the number of
processes to use.

Output dask arrays
------------------

By default, the reprojection functions will do the computation immediately and
return Numpy arrays for the reprojected array and optionally the footprint (this
is regardless of whether dask or Numpy arrays were passed in, or any of the
parallelization options above). However, by setting ``return_type='dask'``, you
can make the functions delay any computation and return dask arrays::

    >>> array, footprint = reproject_interp((input_array, wcs_in), wcs_out,
    ...                                     shape_out=(2048, 2048), block_size=(256, 256),
    ...                                     return_type='dask')
    >>> array
    dask.array<getitem, shape=(2048, 2048), dtype=float64, chunksize=(256, 256), ...>

You can then compute the array or a section of the array yourself whenever you need, or use the
result in further dask expressions.

.. warning:: The reprojection does not currently work reliably when using multiple threads, so
             it is important to make sure you use a dask scheduler that is not multi-threaded.
             At the time of writing, the default dask scheduler is ``threads``, so the scheduler
             needs to be explicitly set to a different one.

Using dask.distributed
----------------------

The `dask.distributed <https://distributed.dask.org/en/stable/>`_ package makes it
possible to use distributed schedulers for dask. In order to compute
reprojections with dask.distributed, you should make use of the
``return_type='dask'`` option mentioned above so that you can compute the dask
array once the distributed scheduler has been set up. As mentioned in `Output
dask arrays`_, you should make sure that you limit any cluster to have one
thread per process or the results may be unreliable.
