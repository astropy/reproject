py.install_sources(
  '__init__.py',
  'test_core.py',
  subdir: 'reproject/adaptive/tests',
)

install_data(
  'reference/test_reproject_adaptive_2d.fits',
  'reference/test_reproject_adaptive_2d_rotated.fits',
  'reference/test_reproject_adaptive_roundtrip.fits',
  'reference/test_reproject_adaptive_uncentered_jacobian.fits',
  install_dir: py.get_install_dir() / 'reproject/adaptive/tests/reference',
)
