py.install_sources(
  '__init__.py',
  'core.py',
  'high_level.py',
  subdir: 'reproject/adaptive',
)

py.extension_module(
  'deforest',
  'deforest.pyx',
  dependencies: dependency('numpy'),
  install: true,
  subdir: 'reproject/adaptive',
)

subdir('tests')
