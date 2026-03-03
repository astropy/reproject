py.install_sources(
  '__init__.py',
  'test_core.py',
  subdir: 'reproject/interpolation/tests',
)

install_data(
  'reference/test_reproject_celestial_2d_gal2equ.fits',
  'reference/test_reproject_celestial_3d_equ2gal.fits',
  'reference/test_reproject_roundtrip.fits',
  'reference/test_small_cutout.fits',
  install_dir: py.get_install_dir() / 'reproject/interpolation/tests/reference',
)
