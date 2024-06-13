*************************************
Optimizing performance for mosaicking
*************************************

Using memory-mapped output arrays
=================================

If you are producing a large mosaic, you may be want to write the mosaic and
footprint to an array of your choice, such as for example a memory-mapped array.
For example:

.. doctest-skip::

    >>> output_array = np.memmap(filename='array.np', mode='w+',
    ...                          shape=shape_out, dtype='float32')
    >>> output_footprint = np.memmap(filename='footprint.np', mode='w+',
    ...                              shape=shape_out, dtype='float32')
    >>> reproject_and_coadd(...,
                            output_array=output_array,
                            output_footprint=output_footprint)

Using memory-mapped intermediate arrays
=======================================

During the mosaicking process, each cube is reprojected to the minimal subset of
the final header that it covers. In some cases, this can result in arrays that
may not fit in memory. In this case, you can use the ``intermediate_memmap``
option to indicate that all intermediate arrays in the mosaicking process should
use memory-mapped arrays rather than in-memory arrays:

    >>> reproject_and_coadd(...,
                            intermediate_memmap=True)

Combined with the above option to specify the output array and footprint for the
final mosaic, it is possible to make sure that no large arrays are ever loaded
into memory. Note however that you will need to make sure you have sufficient disk
space in your temporary directory. If your default system temporary directory does
not have sufficient space, you can set the ``TMPDIR`` environment variable to point
at another directory:

    >>> import os
    >>> os.environ['TMPDIR'] = '/home/lancelot/tmp'

Multi-threading
===============

Similarly to single-image reprojection (see :ref:`multithreading`), it is possible
to make use of multi-threading during the mosaicking process by setting the
``parallel=`` option to True or to an integer value to indicate the number of
threads to use.
