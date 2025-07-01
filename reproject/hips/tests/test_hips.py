import os

import numpy as np

from ... import reproject_interp
from ..high_level import reproject_to_hips

EXPECTED_FILES = [
    "Norder0/Dir0/Npix0.fits",
    "Norder1/Dir0/Npix2.fits",
    "Norder2/Dir0/Npix9.fits",
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
    actual = sorted([str(f.relative_to(directory)) for f in directory.rglob("*") if f.is_file()])
    print(actual)
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
        expected = [filename.replace(".fits", ".png") for filename in EXPECTED_FILES]
    elif str(input_value).endswith("jpg"):
        expected = [filename.replace(".fits", ".jpg") for filename in EXPECTED_FILES]
    else:
        expected = EXPECTED_FILES

    assert_files_expected(output_directory, expected)


EXPECTED_FILES_GALACTIC = [
    "Norder0/Dir0/Npix9.fits",
    "Norder1/Dir0/Npix39.fits",
    "Norder2/Dir0/Npix156.fits",
    "Norder2/Dir0/Npix157.fits",
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


def test_reproject_to_hips_galactic(tmp_path, simple_celestial_wcs):

    array_in = np.ones((30, 40))
    wcs_in = simple_celestial_wcs

    output_directory = tmp_path / "output"

    reproject_to_hips(
        (array_in, wcs_in),
        coord_system_out="galactic",
        level=6,
        reproject_function=reproject_interp,
        output_directory=output_directory,
    )

    assert_files_expected(output_directory, EXPECTED_FILES_GALACTIC)
