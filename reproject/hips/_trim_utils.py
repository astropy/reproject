import numpy as np
from astropy.io import fits

__all__ = ["fits_writeto_withtrim", "fits_getdata_untrimmed"]


def fits_writeto_withtrim(filename, array, header, **kwargs):

    if array.ndim == 3:

        mask = ~np.isnan(array)

        slices = []
        for axis in range(3):
            mask_1d = np.any(mask, axis=tuple(a for a in range(3) if a != axis))
            indices = np.nonzero(mask_1d)[0]
            if len(indices) == 0:
                return
            imin, imax = indices[0], indices[-1]
            slices.append(slice(imin, imax + 1))
            fits_axis = 3 - axis
            header[f"TRIM{fits_axis}"] = imin
            header[f"CRPIX{fits_axis}"] -= imin
            header[f"NAXIS{fits_axis}"] = imax + 1 - imin
        array = array[tuple(slices)]

    array = array.astype(np.float32)
    header["BITPIX"] = -32

    fits.writeto(filename, array, header, **kwargs)


def fits_getdata_untrimmed(filename, *, tile_size, tile_depth):

    with fits.open(filename) as hdulist:
        data = hdulist[0].data
        header = hdulist[0].header

    pad_before = tuple(header.get(f"TRIM{3 - axis}", 0) for axis in range(3))
    shape = data.shape
    pad_after = (
        tile_depth - shape[0] - pad_before[0],
        tile_size - shape[1] - pad_before[1],
        tile_size - shape[2] - pad_before[2],
    )

    data = np.pad(data, list(zip(pad_before, pad_after, strict=False)))

    assert data.shape == (tile_depth, tile_size, tile_size)

    return data
