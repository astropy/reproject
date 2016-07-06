def get_package_data():
    return {
        _ASTROPY_PACKAGE_NAME_ + '.tests': ['coveragerc', 'data/*'],
        _ASTROPY_PACKAGE_NAME_ + '.interpolation.tests': ['baseline/*']
    }
