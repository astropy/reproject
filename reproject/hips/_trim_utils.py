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
            # Store the original (untrimmed) size so that a reader can restore
            # the full tile, and record the size of the leading margin removed.
            header[f"ONAXIS{fits_axis}"] = array.shape[axis]
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

    # Prefer the original tile size recorded at write time (ONAXISn), falling
    # back to the requested tile dimensions for tiles written without it.
    onaxis = (
        int(header.get("ONAXIS3", tile_depth)),
        int(header.get("ONAXIS2", tile_size)),
        int(header.get("ONAXIS1", tile_size)),
    )

    pad_before = tuple(header.get(f"TRIM{3 - axis}", 0) for axis in range(3))
    shape = data.shape
    pad_after = (
        onaxis[0] - shape[0] - pad_before[0],
        onaxis[1] - shape[1] - pad_before[1],
        onaxis[2] - shape[2] - pad_before[2],
    )

    data = np.pad(
        data,
        list(zip(pad_before, pad_after, strict=False)),
        mode="constant",
        constant_values=np.nan,
    )

    assert data.shape == onaxis

    return data
