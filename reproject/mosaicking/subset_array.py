# Licensed under a 3-clause BSD style license - see LICENSE.rst

import operator

__all__ = ['ReprojectedArraySubset']


class ReprojectedArraySubset:
    """
    The aim of this class is to represent a subset of an array and
    footprint extracted (or meant to represent extracted) versions
    from larger arrays and footprints.
    """

    # NOTE: we can't use Cutout2D here because it's much more convenient
    # to work with position being the lower left corner of the cutout
    # rather than the center, which is not well defined for even-sized
    # cutouts.

    def __init__(self, array, footprint, imin, imax, jmin, jmax):
        self.array = array
        self.footprint = footprint
        self.imin = imin
        self.imax = imax
        self.jmin = jmin
        self.jmax = jmax

    def __repr__(self):
        return ('<ReprojectedArraySubset at [{}:{},{}:{}]>'
                .format(self.jmin, self.jmax, self.imin, self.imax))

    @property
    def view_in_original_array(self):
        return (slice(self.jmin, self.jmax), slice(self.imin, self.imax))

    @property
    def shape(self):
        return (self.jmax - self.jmin, self.imax - self.imin)

    def overlaps(self, other):
        # Note that the use of <= or >= instead of < and > is due to
        # the fact that the max values are exclusive (so +1 above the
        # last value).
        return not (self.imax <= other.imin or other.imax <= self.imin or
                    self.jmax <= other.jmin or other.jmax <= self.jmin)

    def __add__(self, other):
        return self._operation(other, operator.add)

    def __sub__(self, other):
        return self._operation(other, operator.sub)

    def __mul__(self, other):
        return self._operation(other, operator.mul)

    def __truediv__(self, other):
        return self._operation(other, operator.truediv)

    def _operation(self, other, op):

        # Determine cutout parameters for overlap region

        imin = max(self.imin, other.imin)
        imax = min(self.imax, other.imax)
        jmin = max(self.jmin, other.jmin)
        jmax = min(self.jmax, other.jmax)

        if imax < imin:
            imax = imin

        if jmax < jmin:
            jmax = jmin

        # Extract cutout from each

        self_array = self.array[jmin - self.jmin:jmax - self.jmin,
                                imin - self.imin:imax - self.imin]
        self_footprint = self.footprint[jmin - self.jmin:jmax - self.jmin,
                                        imin - self.imin:imax - self.imin]

        other_array = other.array[jmin - other.jmin:jmax - other.jmin,
                                  imin - other.imin:imax - other.imin]
        other_footprint = other.footprint[jmin - other.jmin:jmax - other.jmin,
                                          imin - other.imin:imax - other.imin]

        # Carry out operator and store result in ReprojectedArraySubset

        array = op(self_array, other_array)
        footprint = (self_footprint > 0) & (other_footprint > 0)

        return ReprojectedArraySubset(array, footprint, imin, imax, jmin, jmax)
