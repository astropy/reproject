import os
import re
import warnings
from math import prod

import numpy as np
import pytest
from astropy.coordinates import ICRS
from astropy.io import fits
from astropy.io.fits import Header
from astropy.wcs import WCS
from PIL import Image
from pyavm import AVM

from ... import reproject_interp
from .._high_level import compute_lower_resolution_tiles, reproject_to_hips
from .._trim_utils import fits_getdata_untrimmed, fits_writeto_withtrim
from .._utils import load_properties, tile_header_3d

EXPECTED_FILES = [
    "Moc.fits",
    "Norder0/Allsky.fits",
    "Norder0/Dir0/Npix0.fits",
    "Norder1/Allsky.fits",
    "Norder1/Dir0/Npix2.fits",
    "Norder2/Allsky.fits",
    "Norder2/Dir0/Npix9.fits",
    "Norder3/Allsky.fits",
    "Norder3/Dir0/Npix38.fits",
    "Norder4/Dir0/Npix152.fits",
    "Norder4/Dir0/Npix153.fits",
    "Norder4/Dir0/Npix154.fits",
    "Norder5/Dir0/Npix609.fits",
    "Norder5/Dir0/Npix611.fits",
    "Norder5/Dir0/Npix612.fits",
    "Norder5/Dir0/Npix614.fits",
    "Norder5/Dir0/Npix617.fits",
    "Norder6/Dir0/Npix2439.fits",
    "Norder6/Dir0/Npix2444.fits",
    "Norder6/Dir0/Npix2445.fits",
    "Norder6/Dir0/Npix2446.fits",
    "Norder6/Dir0/Npix2447.fits",
    "Norder6/Dir0/Npix2450.fits",
    "Norder6/Dir0/Npix2456.fits",
    "Norder6/Dir0/Npix2458.fits",
    "Norder6/Dir0/Npix2468.fits",
    "Norder6/Dir0/Npix2469.fits",
    "index.html",
    "properties",
]


def assert_files_expected(directory, expected):
    actual = sorted(
        [f.relative_to(directory).as_posix() for f in directory.rglob("*") if f.is_file()]
    )
    assert actual == expected


def test_reproject_to_hips(tmp_path, valid_celestial_input_data):

    _, _, input_value, kwargs_in = valid_celestial_input_data

    output_directory = tmp_path / "output"

    reproject_to_hips(
        input_value,
        coord_system_out="equatorial",
        level=6,
        reproject_function=reproject_interp,
        output_directory=output_directory,
        **kwargs_in,
    )

    if str(input_value).endswith("png"):
        expected = [
            f.replace(".fits", ".png") if f.startswith("Norder") else f for f in EXPECTED_FILES
        ]
    elif str(input_value).endswith("jpg"):
        expected = [
            f.replace(".fits", ".jpg") if f.startswith("Norder") else f for f in EXPECTED_FILES
        ]
    else:
        expected = EXPECTED_FILES

    assert_files_expected(output_directory, expected)


EXPECTED_FILES_GALACTIC = [
    "Moc.fits",
    "Norder0/Allsky.fits",
    "Norder0/Dir0/Npix9.fits",
    "Norder1/Allsky.fits",
    "Norder1/Dir0/Npix39.fits",
    "Norder2/Allsky.fits",
    "Norder2/Dir0/Npix156.fits",
    "Norder2/Dir0/Npix157.fits",
    "Norder3/Allsky.fits",
    "Norder3/Dir0/Npix627.fits",
    "Norder3/Dir0/Npix630.fits",
    "Norder4/Dir0/Npix2511.fits",
    "Norder4/Dir0/Npix2522.fits",
    "Norder5/Dir10000/Npix10045.fits",
    "Norder5/Dir10000/Npix10047.fits",
    "Norder5/Dir10000/Npix10088.fits",
    "Norder5/Dir10000/Npix10090.fits",
    "Norder6/Dir40000/Npix40182.fits",
    "Norder6/Dir40000/Npix40183.fits",
    "Norder6/Dir40000/Npix40188.fits",
    "Norder6/Dir40000/Npix40189.fits",
    "Norder6/Dir40000/Npix40190.fits",
    "Norder6/Dir40000/Npix40191.fits",
    "Norder6/Dir40000/Npix40354.fits",
    "Norder6/Dir40000/Npix40360.fits",
    "index.html",
    "properties",
]


def test_reproject_to_hips_galactic(tmp_path, simple_celestial_fits_wcs):

    array_in = np.ones((30, 40))
    wcs_in = simple_celestial_fits_wcs

    output_directory = tmp_path / "output"

    reproject_to_hips(
        (array_in, wcs_in),
        coord_system_out="galactic",
        level=6,
        reproject_function=reproject_interp,
        output_directory=output_directory,
    )

    assert_files_expected(output_directory, EXPECTED_FILES_GALACTIC)


def test_reproject_to_hips_invalid_parameters(tmp_path, simple_celestial_fits_wcs):

    array_in = np.ones((30, 40))
    wcs_in = simple_celestial_fits_wcs

    output_directory = tmp_path / "output"

    with pytest.raises(
        ValueError,
        match=re.escape("coord_system_out should be one of equatorial/galactic/ecliptic"),
    ):
        reproject_to_hips(
            (array_in, wcs_in),
            coord_system_out="intragalactic",
            reproject_function=reproject_interp,
            output_directory=output_directory,
        )

    with pytest.raises(ValueError, match=re.escape("tile_size should be even")):
        reproject_to_hips(
            (array_in, wcs_in),
            tile_size=311,
            coord_system_out="galactic",
            reproject_function=reproject_interp,
            output_directory=output_directory,
        )


EXPECTED_FILES_AUTO_1 = [
    "Moc.fits",
    "Norder0/Allsky.fits",
    "Norder0/Dir0/Npix0.fits",
    "Norder1/Allsky.fits",
    "Norder1/Dir0/Npix2.fits",
    "Norder2/Allsky.fits",
    "Norder2/Dir0/Npix9.fits",
    "index.html",
    "properties",
]


EXPECTED_FILES_AUTO_2 = [
    "Moc.fits",
    "Norder0/Allsky.fits",
    "Norder0/Dir0/Npix0.fits",
    "Norder1/Allsky.fits",
    "Norder1/Dir0/Npix2.fits",
    "Norder2/Allsky.fits",
    "Norder2/Dir0/Npix9.fits",
    "Norder3/Allsky.fits",
    "Norder3/Dir0/Npix38.fits",
    "Norder4/Dir0/Npix153.fits",
    "Norder5/Dir0/Npix612.fits",
    "Norder6/Dir0/Npix2450.fits",
    "Norder7/Dir0/Npix9802.fits",
    "index.html",
    "properties",
]


def test_reproject_to_hips_automatic(tmp_path, simple_celestial_wcs):

    array_in = np.ones((30, 40))
    wcs_in = simple_celestial_wcs

    output_directory = tmp_path / "output_1"

    reproject_to_hips(
        (array_in, wcs_in),
        coord_system_out="equatorial",
        reproject_function=reproject_interp,
        output_directory=output_directory,
    )

    assert_files_expected(output_directory, EXPECTED_FILES_AUTO_1)

    if isinstance(wcs_in, WCS):

        output_directory = tmp_path / "output_2"
        wcs_in.wcs.cdelt = -0.001, 0.001

        reproject_to_hips(
            (array_in, wcs_in),
            coord_system_out="equatorial",
            reproject_function=reproject_interp,
            output_directory=output_directory,
        )

        assert_files_expected(output_directory, EXPECTED_FILES_AUTO_2)


def test_reproject_to_hips_alpha(tmp_path, simple_celestial_fits_wcs):

    # Make sure that any input alpha channel is preserved

    data = np.arange(900).reshape((30, 30)) / 1200
    layer = (data * 255).astype(np.uint8)
    alpha = np.zeros((30, 30)).astype(np.uint8)
    alpha[10:15, 10:15] = 255
    array_rgba = np.dstack([layer, np.rot90(layer, k=1), np.rot90(layer, k=2), alpha])
    image = Image.fromarray(array_rgba)
    image.save(tmp_path / "rgb.png")

    avm = AVM.from_wcs(simple_celestial_fits_wcs, shape=(30, 30))
    avm.embed(tmp_path / "rgb.png", tmp_path / "rgb_tagged.png")

    output_directory = tmp_path / "output_1"

    reproject_to_hips(
        tmp_path / "rgb_tagged.png",
        coord_system_out="equatorial",
        reproject_function=reproject_interp,
        output_directory=output_directory,
    )

    result = np.array(Image.open(output_directory / "Norder1" / "Dir0" / "Npix2.png"))

    count = result.sum(axis=(0, 1))

    # There should be many pixels that are valid but transparent
    assert result[:, :, 0].size == 262144
    assert np.all(count[:3] > 50000)
    assert count[3] < 4000


HEADER_SPECTRAL = """
WCSAXES = 3
CRPIX1  = 10.
CRPIX2  = 20.
CRPIX3  = 30.
CDELT1  = -0.02
CDELT2  = 0.02
CDELT3  = 0.1
CUNIT1  = 'deg'
CUNIT2  = 'deg'
CUNIT3  = 'GHz'
CTYPE1  = 'RA---SIN'
CTYPE2  = 'DEC--SIN'
CTYPE3  = 'FREQ'
CRVAL1  = 50.
CRVAL2  = 70.
CRVAL3  = 230.
SPECSYS = 'LSRK'
"""


EXPECTED_FILES_CUBE = [
    "Moc.fits",
    "Norder0_7/Dir0_130/Npix0_134.fits",
    "Norder1_8/Dir0_260/Npix3_268.fits",
    "Norder2_9/Dir0_530/Npix15_536.fits",
    "Norder3_10/Dir0_1070/Npix60_1073.fits",
    "Norder4_11/Dir0_2140/Npix240_2147.fits",
    "Norder4_11/Dir0_2140/Npix241_2147.fits",
    "Norder5_12/Dir0_4290/Npix961_4294.fits",
    "Norder5_12/Dir0_4290/Npix964_4294.fits",
    "Norder6_13/Dir0_8580/Npix3845_8588.fits",
    "Norder6_13/Dir0_8580/Npix3845_8589.fits",
    "Norder6_13/Dir0_8580/Npix3856_8588.fits",
    "Norder6_13/Dir0_8580/Npix3856_8589.fits",
    "Norder7_14/Dir10000_17170/Npix15381_17177.fits",
    "Norder7_14/Dir10000_17170/Npix15381_17178.fits",
    "Norder7_14/Dir10000_17170/Npix15381_17179.fits",
    "Norder7_14/Dir10000_17170/Npix15383_17177.fits",
    "Norder7_14/Dir10000_17170/Npix15383_17178.fits",
    "Norder7_14/Dir10000_17170/Npix15383_17179.fits",
    "Norder7_14/Dir10000_17170/Npix15426_17177.fits",
    "Norder7_14/Dir10000_17170/Npix15426_17178.fits",
    "Norder7_14/Dir10000_17170/Npix15426_17179.fits",
    "Norder8_15/Dir60000_34350/Npix61527_34355.fits",
    "Norder8_15/Dir60000_34350/Npix61527_34356.fits",
    "Norder8_15/Dir60000_34350/Npix61527_34357.fits",
    "Norder8_15/Dir60000_34350/Npix61527_34358.fits",
    "Norder8_15/Dir60000_34350/Npix61532_34355.fits",
    "Norder8_15/Dir60000_34350/Npix61532_34356.fits",
    "Norder8_15/Dir60000_34350/Npix61532_34357.fits",
    "Norder8_15/Dir60000_34350/Npix61532_34358.fits",
    "Norder8_15/Dir60000_34350/Npix61533_34355.fits",
    "Norder8_15/Dir60000_34350/Npix61533_34356.fits",
    "Norder8_15/Dir60000_34350/Npix61533_34357.fits",
    "Norder8_15/Dir60000_34350/Npix61533_34358.fits",
    "Norder8_15/Dir60000_34350/Npix61534_34355.fits",
    "Norder8_15/Dir60000_34350/Npix61534_34356.fits",
    "Norder8_15/Dir60000_34350/Npix61534_34357.fits",
    "Norder8_15/Dir60000_34350/Npix61534_34358.fits",
    "Norder8_15/Dir60000_34350/Npix61535_34355.fits",
    "Norder8_15/Dir60000_34350/Npix61535_34356.fits",
    "Norder8_15/Dir60000_34350/Npix61535_34357.fits",
    "Norder8_15/Dir60000_34350/Npix61535_34358.fits",
    "Norder8_15/Dir60000_34350/Npix61704_34355.fits",
    "Norder8_15/Dir60000_34350/Npix61704_34356.fits",
    "Norder8_15/Dir60000_34350/Npix61704_34357.fits",
    "Norder8_15/Dir60000_34350/Npix61704_34358.fits",
    "index.html",
    "properties",
]


def test_reproject_to_hips3d_spectral(tmp_path):

    shape = (15, 16, 17)

    cube_data = np.arange(prod(shape)).reshape(shape)
    cube_wcs = WCS(Header.fromstring(HEADER_SPECTRAL, sep="\n"))

    output_directory = tmp_path / "output"

    reproject_to_hips(
        (cube_data, cube_wcs),
        coord_system_out="equatorial",
        reproject_function=reproject_interp,
        output_directory=output_directory,
        threads=True,
        tile_size=16,
        tile_depth=8,
    )

    assert_files_expected(output_directory, EXPECTED_FILES_CUBE)


def test_properties_regressions(tmp_path, simple_celestial_fits_wcs):
    # Regression tests for the HiPS properties file:
    #  - obs_title should be the dataset directory name, not its parent
    #  - hips_pixel_bitpix should match the float32 tiles that are written
    #  - hips_release_date should follow the ISO 8601 YYYY-mm-ddTHH:MMZ form
    output_directory = tmp_path / "some" / "nested" / "my_survey"

    reproject_to_hips(
        (np.ones((30, 40)), simple_celestial_fits_wcs),
        coord_system_out="equatorial",
        level=3,
        reproject_function=reproject_interp,
        output_directory=output_directory,
    )

    properties = load_properties(output_directory)

    assert properties["obs_title"] == "my_survey"
    assert properties["hips_pixel_bitpix"] == "-32"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z", properties["hips_release_date"])

    tile = next(output_directory.glob("Norder3/*/*.fits"))
    assert fits.getheader(tile)["BITPIX"] == -32


def test_trimmed_tiles_store_original_size(tmp_path):
    # Regression test: trimmed 3D tiles must record their original (untrimmed)
    # size via ONAXIS1/2/3 so that a reader can reconstruct the full tile.
    tile_size, tile_depth = 16, 8
    header = tile_header_3d(
        spatial_level=2,
        spatial_index=0,
        spectral_level=2,
        spectral_index=0,
        frame=ICRS(),
        tile_size=tile_size,
        tile_depth=tile_depth,
    )
    if isinstance(header, tuple):
        header = header[0]
    crpix_before = (header["CRPIX1"], header["CRPIX2"], header["CRPIX3"])

    array = np.full((tile_depth, tile_size, tile_size), np.nan)
    array[2:5, 3:10, 4:12] = 1.0  # off-centre block so every axis is trimmed

    filename = str(tmp_path / "tile.fits")
    fits_writeto_withtrim(filename, array, header)

    written = fits.getheader(filename)
    # The original full size is preserved in ONAXISn ...
    assert (written["ONAXIS1"], written["ONAXIS2"], written["ONAXIS3"]) == (
        tile_size,
        tile_size,
        tile_depth,
    )
    # ... while NAXISn and the reference pixel reflect the trimmed, stored data
    assert (written["NAXIS1"], written["NAXIS2"], written["NAXIS3"]) == (8, 7, 3)
    assert (written["TRIM1"], written["TRIM2"], written["TRIM3"]) == (4, 3, 2)
    # The reference pixel on each axis is shifted by the leading trim
    assert written["CRPIX1"] == crpix_before[0] - 4
    assert written["CRPIX2"] == crpix_before[1] - 3
    assert written["CRPIX3"] == crpix_before[2] - 2

    # The reader restores the full tile, relying on ONAXISn even without hints
    restored = fits_getdata_untrimmed(filename, tile_size=1, tile_depth=1)
    assert restored.shape == (tile_depth, tile_size, tile_size)
    np.testing.assert_array_equal(np.nan_to_num(restored), np.nan_to_num(array.astype(np.float32)))


CUBE_CLAMP_HEADER = """
WCSAXES = 3
CRPIX1  = 16.
CRPIX2  = 16.
CRPIX3  = 1.
CDELT1  = -0.15
CDELT2  = 0.15
CDELT3  = {cdelt3}
CUNIT1  = 'deg'
CUNIT2  = 'deg'
CUNIT3  = 'Hz'
CTYPE1  = 'RA---SIN'
CTYPE2  = 'DEC--SIN'
CTYPE3  = 'FREQ-LOG'
CRVAL1  = 50.
CRVAL2  = 70.
CRVAL3  = 1.0E9
SPECSYS = 'LSRK'
"""


def test_reproject_to_hips3d_clamps_spectral_order(tmp_path):
    # Regression test: when the spectral (Lmax) order is smaller than the
    # spatial (Kmax) order, lower-resolution tiles must clamp the spectral
    # order at 0 (never negative) and keep degrading spatially.
    #
    # This needs Lmax < Kmax, which means a low spectral order. Because the
    # frequency grid spans FREQ_MIN=1e-18 to FREQ_MAX=1e38 Hz, a low order only
    # arises for a band covering many decades, so we use a coarse spatial grid
    # (small Kmax) together with a wide, log-sampled spectral axis (small Lmax).
    level, level_depth = 4, 1  # Kmax > Lmax forces the clamp

    nz = 8
    # CDELT3 for the -LOG algorithm so the nz channels span ~4 decades
    cdelt3 = 1e9 * np.log(10**4) / (nz - 1)
    header = Header.fromstring(CUBE_CLAMP_HEADER.format(cdelt3=cdelt3), sep="\n")
    cube = np.arange(nz * 32 * 32).reshape(nz, 32, 32).astype(float)

    output_directory = tmp_path / "cube"

    reproject_to_hips(
        (cube, WCS(header)),
        coord_system_out="equatorial",
        reproject_function=reproject_interp,
        output_directory=output_directory,
        tile_size=16,
        tile_depth=8,
        level=level,
        level_depth=level_depth,
    )

    pairs = sorted(
        tuple(int(value) for value in path.name.replace("Norder", "").split("_"))
        for path in output_directory.glob("Norder*")
    )

    # Orders follow L = max(0, Lmax - (Kmax - K)), clamped at 0, down to spatial 0
    expected = sorted({(level - sub, max(0, level_depth - sub)) for sub in range(level + 1)})
    assert pairs == expected
    assert all(spectral_order >= 0 for _, spectral_order in pairs)
    assert min(spatial_order for spatial_order, _ in pairs) == 0
    assert min(spectral_order for _, spectral_order in pairs) == 0

    # Clamped lower-resolution tiles still carry the propagated data
    clamped = list(output_directory.glob("Norder0_0/*/*.fits"))
    assert clamped
    assert np.isfinite(fits.getdata(clamped[0])).any()


def test_compute_lower_resolution_no_tiles_warns(tmp_path):
    # Regression test: if no tiles were generated at the deepest level, the
    # lower-resolution step should warn and return rather than raise.
    output_directory = tmp_path / "empty"
    os.makedirs(output_directory / "Norder4_2")

    with pytest.warns(UserWarning, match="No tiles were generated"):
        compute_lower_resolution_tiles(
            output_directory=output_directory,
            ndim=3,
            frame=ICRS(),
            tile_format="fits",
            tile_size=8,
            tile_depth=4,
            spatial_level=4,
            level_depth=2,
        )

    assert list(output_directory.glob("Norder*")) == [output_directory / "Norder4_2"]


def test_moc_2d(tmp_path, simple_celestial_fits_wcs):
    # A spatial MOC (Moc.fits) is written for a 2D HiPS and its coverage matches
    # the generated tiles.
    from mocpy import MOC

    from .._high_level import find_indices

    output_directory = tmp_path / "output"
    reproject_to_hips(
        (np.ones((30, 40)), simple_celestial_fits_wcs),
        coord_system_out="equatorial",
        level=4,
        reproject_function=reproject_interp,
        output_directory=output_directory,
    )

    moc_path = output_directory / "Moc.fits"
    assert moc_path.exists()
    assert fits.getheader(moc_path, 1)["COORDSYS"] == "C"

    tiles = sorted(
        find_indices(output_directory=output_directory, ndim=2, spatial_level=4, level_depth=None)
    )
    expected = MOC.from_healpix_cells(
        ipix=np.array(tiles), depth=np.full(len(tiles), 4), max_depth=4
    )
    assert MOC.from_fits(moc_path) == expected


def test_moc_2d_galactic_coordsys(tmp_path, simple_celestial_fits_wcs):
    # The MOC COORDSYS keyword reflects the HiPS frame.
    output_directory = tmp_path / "output"
    reproject_to_hips(
        (np.ones((30, 40)), simple_celestial_fits_wcs),
        coord_system_out="galactic",
        level=4,
        reproject_function=reproject_interp,
        output_directory=output_directory,
    )
    assert fits.getheader(output_directory / "Moc.fits", 1)["COORDSYS"] == "G"


def test_moc_3d(tmp_path):
    # A space-frequency MOC (SF-MOC) is written for a 3D HiPS3D dataset.
    from mocpy import SFMOC

    level, level_depth = 4, 1
    nz = 8
    cdelt3 = 1e9 * np.log(10**4) / (nz - 1)
    header = Header.fromstring(CUBE_CLAMP_HEADER.format(cdelt3=cdelt3), sep="\n")
    cube = np.arange(nz * 32 * 32).reshape(nz, 32, 32).astype(float)

    output_directory = tmp_path / "cube"
    reproject_to_hips(
        (cube, WCS(header)),
        coord_system_out="equatorial",
        reproject_function=reproject_interp,
        output_directory=output_directory,
        tile_size=16,
        tile_depth=8,
        level=level,
        level_depth=level_depth,
    )

    moc_path = output_directory / "Moc.fits"
    assert moc_path.exists()
    moc_header = fits.getheader(moc_path, 1)
    assert moc_header["MOCDIM"] == "FREQUENCY.SPACE"
    assert moc_header["MOCORD_S"] == level
    assert moc_header["MOCORD_F"] == level_depth
    # The range column is named so that generic FITS table readers can open it -
    # mocpy/CDS otherwise omit TTYPE1, which makes the table unreadable.
    assert moc_header["TTYPE1"] == "RANGE"
    with fits.open(moc_path) as hdulist:
        assert len(hdulist[1].data) > 0
    # The file is a valid SF-MOC that mocpy can read back
    SFMOC.from_fits(moc_path)


def test_moc_3d_extreme_frequency_index(tmp_path):
    # The SF-MOC is built directly from integer FMOC indices, so even the
    # highest frequency cell (whose upper edge is FREQ_MAX) is handled without
    # error - converting it to a frequency in Hz would overflow the allowed range.
    from mocpy import SFMOC

    from .._moc import save_moc

    level_depth = 4
    topmost = 2 ** (level_depth + 1) - 1  # last valid FMOC cell at this order
    save_moc(
        output_directory=tmp_path,
        indices=[(10, 0), (10, topmost)],
        coord_system="equatorial",
        spatial_level=3,
        level_depth=level_depth,
    )

    moc_path = tmp_path / "Moc.fits"
    assert moc_path.exists()
    SFMOC.from_fits(moc_path)


def test_moc_disabled(tmp_path, simple_celestial_fits_wcs):
    # No Moc.fits is written when generate_moc=False.
    output_directory = tmp_path / "output"
    reproject_to_hips(
        (np.ones((30, 40)), simple_celestial_fits_wcs),
        coord_system_out="equatorial",
        level=4,
        reproject_function=reproject_interp,
        output_directory=output_directory,
        generate_moc=False,
    )
    assert not (output_directory / "Moc.fits").exists()


def test_allsky_2d(tmp_path, simple_celestial_fits_wcs):
    # An Allsky preview is written for each low order (0-3), packing all the
    # tiles of that order into a single downsampled mosaic.
    from astropy.nddata import block_reduce

    output_directory = tmp_path / "output"
    reproject_to_hips(
        (np.arange(30 * 40).reshape(30, 40).astype(float), simple_celestial_fits_wcs),
        coord_system_out="equatorial",
        level=5,
        tile_size=128,
        reproject_function=reproject_interp,
        output_directory=output_directory,
    )

    # Allsky files exist for orders 0-3 only (not for the deeper orders 4-5)
    for order in range(4):
        assert (output_directory / f"Norder{order}" / "Allsky.fits").exists()
    for order in (4, 5):
        assert not (output_directory / f"Norder{order}" / "Allsky.fits").exists()

    # The order-3 mosaic has the standard layout: width = floor(sqrt(768)) = 27
    # columns of 64-pixel tiles (128 downsampled by 2)
    allsky = fits.getdata(output_directory / "Norder3" / "Allsky.fits")
    assert allsky.shape == (29 * 64, 27 * 64)

    # Each generated tile appears, downsampled, in its mosaic cell
    tile_path = next((output_directory / "Norder3").glob("Dir*/Npix*.fits"))
    index = int(tile_path.name[:-5].replace("Npix", ""))
    row, col = divmod(index, 27)
    cell = allsky[row * 64 : (row + 1) * 64, col * 64 : (col + 1) * 64]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks in nanmean
        expected = block_reduce(fits.getdata(tile_path), 2, func=np.nanmean)
    mask = np.isfinite(expected) & np.isfinite(cell)
    assert mask.any()
    np.testing.assert_allclose(cell[mask], expected[mask])


def test_allsky_disabled(tmp_path, simple_celestial_fits_wcs):
    # No Allsky files are written when allsky=False.
    output_directory = tmp_path / "output"
    reproject_to_hips(
        (np.ones((30, 40)), simple_celestial_fits_wcs),
        coord_system_out="equatorial",
        level=4,
        reproject_function=reproject_interp,
        output_directory=output_directory,
        allsky=False,
    )
    assert not list(output_directory.glob("Norder*/Allsky.*"))


# TODO: Add tests of different spectral frames
# TODO: Add test of auto level determination for all sky maps in 2D and 3D
