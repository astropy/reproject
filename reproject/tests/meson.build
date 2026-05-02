py.install_sources(
  '__init__.py',
  'helpers.py',
  'test_array_utils.py',
  'test_high_level.py',
  'test_utils.py',
  subdir: 'reproject/tests',
)

install_data(
  'data/aia_171_level1.asdf',
  'data/aia_171_level1.fits',
  'data/cube.hdr',
  'data/equatorial_3d.fits',
  'data/galactic_2d.fits',
  'data/gc_eq.hdr',
  'data/gc_ga.hdr',
  'data/generate_asdf.py',
  'data/image_with_distortion_map.fits',
  'data/mwpan2_RGB_3600.hdr',
  'data/secchi_l0_a.fits',
  'data/secchi_l0_b.fits',
  install_dir: py.get_install_dir() / 'reproject/tests/data',
)
