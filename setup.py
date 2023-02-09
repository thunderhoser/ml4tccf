"""Setup file for ml4tccf."""

from setuptools import setup

PACKAGE_NAMES = [
    'ml4tccf', 'ml4tccf.io', 'ml4tccf.utils', 'ml4tccf.machine_learning',
    'ml4tccf.plotting', 'ml4tccf.scripts'
]
KEYWORDS = [
    'machine learning', 'deep learning', 'artificial intelligence',
    'data mining', 'weather', 'meteorology', 'thunderstorm', 'wind', 'tornado'
]
SHORT_DESCRIPTION = 'Machine learning for center-fixing of tropical cyclones.'
LONG_DESCRIPTION = (
    'ml4tccf is an end-to-end library for using machine learning to estimate '
    'the center location (e.g., the middle of the eye) of tropical cyclones.'
)
CLASSIFIERS = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3'
]

# You also need to install the following packages, which are not available in
# pip.  They can both be installed by "git clone" and "python setup.py install",
# the normal way one installs a GitHub package.
#
# https://github.com/matplotlib/basemap

PACKAGE_REQUIREMENTS = [
    'numpy',
    'scipy',
    'tensorflow',
    'keras',
    'scikit-learn',
    'scikit-image',
    'netCDF4',
    'pyproj',
    'opencv-python',
    'matplotlib',
    'pandas',
    'shapely',
    'descartes',
    'geopy',
    'metpy'
]

if __name__ == '__main__':
    setup(
        name='ml4tccf',
        version='0.1',
        description=SHORT_DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author='Ryan Lagerquist',
        author_email='ryan.lagerquist@noaa.gov',
        url='https://github.com/thunderhoser/ml4tccf',
        packages=PACKAGE_NAMES,
        scripts=[],
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        include_package_data=True,
        zip_safe=False,
        install_requires=PACKAGE_REQUIREMENTS
    )
