.. _background-matching:

*****************************
How background matching works
*****************************

When combining images into a mosaic, each input image may have an arbitrary
additive offset in its background level, due for example to different
observing conditions for different tiles. The
:func:`~reproject.mosaicking.reproject_and_coadd` function can correct for
this if the ``match_background`` option is set.

The background matching works by finding all overlapping pairs of images and
determining the median difference for each pair, then using a `stochastic
gradient descent <https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`_
method to find the optimal additive corrections (a positive or negative constant
for each image) to minimize differences. We additionally place the constraint
that the average correction should be zero, but since there's no reason that
the average correction should be exactly zero, you should be aware that the
final mosaic may be offset from the absolute surface brightness/flux by a
constant additive factor. The algorithm should be robust for many situations
and does not currently have any exposed options for fine tuning.
