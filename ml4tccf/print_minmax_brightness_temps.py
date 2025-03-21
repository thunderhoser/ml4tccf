"""Prints min/max brightness temperature from a set of files."""

import os
import sys
import glob
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import satellite_io
import satellite_utils

TOLERANCE = 1e-6

INPUT_FILE_PATTERN_ARG_NAME = 'input_satellite_file_pattern'
INPUT_FILE_PATTERN_HELP_STRING = (
    'Glob pattern for input files, which will be read by '
    '`satellite_io.read_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_PATTERN_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_PATTERN_HELP_STRING
)


def _run(satellite_file_pattern):
    """Prints min/max brightness temperature from a set of files.

    This is effectively the main method.

    :param satellite_file_pattern: See documentation at top of this script.
    """

    satellite_file_names = glob.glob(satellite_file_pattern)
    satellite_file_names.sort()

    min_brightness_temp_kelvins = numpy.inf
    max_brightness_temp_kelvins = -numpy.inf
    num_brightness_temp_le170 = 0

    for this_file_name in satellite_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        satellite_table_xarray = satellite_io.read_file(this_file_name)
        stx = satellite_table_xarray

        min_brightness_temp_kelvins = min([
            min_brightness_temp_kelvins,
            numpy.min(stx[satellite_utils.BRIGHTNESS_TEMPERATURE_KEY].values)
        ])
        max_brightness_temp_kelvins = max([
            max_brightness_temp_kelvins,
            numpy.max(stx[satellite_utils.BRIGHTNESS_TEMPERATURE_KEY].values)
        ])
        num_brightness_temp_le170 += numpy.sum(
            stx[satellite_utils.BRIGHTNESS_TEMPERATURE_KEY].values <
            170 + TOLERANCE
        )

    print((
        'Min brightness temperature = {0:.4f} K\n'
        'Max brightness temperature = {1:.4f} K\n'
        'Number of values <= 170 K: {2:d}'
    ).format(
        min_brightness_temp_kelvins,
        max_brightness_temp_kelvins,
        num_brightness_temp_le170
    ))


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        satellite_file_pattern=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_PATTERN_ARG_NAME
        )
    )
