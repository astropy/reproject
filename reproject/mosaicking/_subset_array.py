# Licensed under a 3-clause BSD style license - see LICENSE.rst

import operator

__all__ = ["ReprojectedArraySubset"]


class ReprojectedArraySubset:
    # The aim of this class is to represent a subset of an array and
    # footprint extracted (or meant to represent extracted) versions
    # from larger arrays and footprints.

    # NOTE: we can't use Cutout2D here because it's much more convenient
    # to work with position being the lower left corner of the cutout
    # rather than the center, which is not well defined for even-sized
    # cutouts.

    def __init__(self, array, footprint, bounds):
        self.array = array
        self.footprint = footprint
        self.bounds = bounds

    def __repr__(self):
        bounds_str = "[" + ",".join(f"{imin}:{imax}" for (imin, imax) in self.bounds) + "]"
        return f"<ReprojectedArraySubset at {bounds_str}>"

    @property
    def view_in_original_array(self):
        return tuple([slice(imin, imax) for (imin, imax) in self.bounds])

    @property
    def shape(self):
        return tuple((imax - imin) for (imin, imax) in self.bounds)

    def overlaps(self, other):
        # Note that the use of <= or >= instead of < and > is due to
        # the fact that the max values are exclusive (so +1 above the
        # last value).
        if len(self.bounds) != len(other.bounds):
            raise ValueError(
                f"Mismatch in number of dimensions, expected "
                f"{len(self.bounds)} dimensions and got {len(other.bounds)}"
            )
        for (imin, imax), (imin_other, imax_other) in zip(self.bounds, other.bounds, strict=False):
            if imax <= imin_other or imax_other <= imin:
                return False
        return True

    def __add__(self, other):
        return self._operation(other, operator.add)

    def __sub__(self, other):
        return self._operation(other, operator.sub)

    def __mul__(self, other):
        return self._operation(other, operator.mul)

    def __truediv__(self, other):
        return self._operation(other, operator.truediv)

    def _operation(self, other, op):
        if len(self.bounds) != len(other.bounds):
            raise ValueError(
                f"Mismatch in number of dimensions, expected "
                f"{len(self.bounds)} dimensions and got {len(other.bounds)}"
            )

        # Determine cutout parameters for overlap region

        overlap_bounds = []
        self_slices = []
        other_slices = []
        for (imin, imax), (imin_other, imax_other) in zip(self.bounds, other.bounds, strict=False):
            imin_overlap = max(imin, imin_other)
            imax_overlap = min(imax, imax_other)
            if imax_overlap < imin_overlap:
                imax_overlap = imin_overlap
            overlap_bounds.append((imin_overlap, imax_overlap))
            self_slices.append(slice(imin_overlap - imin, imax_overlap - imin))
            other_slices.append(slice(imin_overlap - imin_other, imax_overlap - imin_other))

        self_slices = tuple(self_slices)

        self_array = self.array[self_slices]
        self_footprint = self.footprint[self_slices]

        other_slices = tuple(other_slices)

        other_array = other.array[other_slices]
        other_footprint = other.footprint[other_slices]

        # Carry out operator and store result in ReprojectedArraySubset

        array = op(self_array, other_array)
        footprint = (self_footprint > 0) & (other_footprint > 0)

        return ReprojectedArraySubset(array, footprint, overlap_bounds)
