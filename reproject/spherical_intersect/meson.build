py.install_sources(
  '__init__.py',
  'core.py',
  'high_level.py',
  'overlap.py',
  subdir: 'reproject/spherical_intersect',
)

# Install header files needed at runtime for potential rebuilds
py.install_sources(
  'mNaN.h',
  'overlapArea.h',
  'reproject_slice_c.h',
  subdir: 'reproject/spherical_intersect',
)

py.extension_module(
  '_overlap',
  '_overlap.pyx',
  'overlapArea.c',
  'reproject_slice_c.c',
  dependencies: dependency('numpy'),
  include_directories: include_directories('.'),
  c_args: ['-O2'],
  install: true,
  subdir: 'reproject/spherical_intersect',
)

subdir('tests')
