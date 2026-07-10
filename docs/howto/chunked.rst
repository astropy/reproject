**************************************
Reprojecting in chunks and in parallel
**************************************

Chunk by chunk reprojection
===========================

.. testsetup::

    >>> import numpy as np
    >>> input_array = np.random.random((1024, 1024))
    >>> from astropy.wcs import WCS
    >>> wcs_in = WCS(naxis=2)
    >>> wcs_out = WCS(naxis=2)

When calling one of the reprojection functions, you can specify a block size to use for the
reprojection, and this is used to iterate over chunks in the output array in
chunks. For instance, if you pass in a (1024, 1024) array and specify that the
shape of the output should be a (2048, 2048) array (e.g., via ``shape_out``),
then if you set ``block_size=(256, 256)``::

    >>> from reproject import reproject_interp
    >>> input_array.shape
    (1024, 1024)
    >>> array, footprint = reproject_interp((input_array, wcs_in), wcs_out,
    ...                                     shape_out=(2048, 2048), block_size=(256, 256))

the reprojection will be done in 64 separate output chunks. Note however that
this does not break up the input array into chunks since in the general case any
input pixel may contribute to any output pixel.

Even without parallelization, reprojecting chunk by chunk can be useful to
avoid using too much memory, since the coordinate transformations and the
intermediate arrays needed for the reprojection only ever cover a single chunk
of the output at a time. For very large output arrays, this can be combined
with a memory-mapped output array (see :doc:`performance`) to keep the total
memory usage low.

.. _multithreading:

Multi-threaded reprojection
===========================

By default, the iteration over the output chunks is done in a single
process/thread, but you may specify ``parallel=True`` to process these in
parallel. If you do this, reproject will use multiple threads to parallelize the
computation. If you specify ``parallel=True``, then ``block_size`` will be
automatically set to a sensible default, but you can also set ``block_size``
manually for more control. Note that you can also set ``parallel=`` to an
integer to indicate the number of threads to use.

By default, in parallel mode, the entire input array will be written to a
temporary file that is then memory-mapped - this is to avoid loading the whole
input array into memory in each process. If you are specifying a WCS with fewer
dimensions than the data to be reprojected, as described in :ref:`broadcasting`,
you can set the block size to be such that the block size along the dimensions
being reprojected cover the whole image, while the other dimensions can be
smaller. For example, if you are reprojecting a spectral cube with dimensions
(500, 2048, 2048) where 500 is the number of spectral channels and (2048, 2048)
is the celestial plane, then if you are reprojecting just the celestial part of
the WCS you can specify a block size of (N, 2048, 2048) and this will enable a
separate reprojection mode where the input array is not written to disk but
where the reprojection is done in truly independent chunks with size (N, 2048, 2048).

Multi-threading is not limited to single-image reprojection: the
:func:`~reproject.mosaicking.reproject_and_coadd` function also accepts the
``parallel=`` option, with the same semantics as above.
