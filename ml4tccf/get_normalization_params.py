"""Computes normalization parameters."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import satellite_io
import misc_utils
import normalization

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'input_satellite_dir_name'
YEARS_ARG_NAME = 'years'
NUM_FILES_ARG_NAME = 'num_satellite_files'
NUM_VALUES_PER_HIGH_RES_ARG_NAME = 'num_values_per_high_res_channel'
NUM_VALUES_PER_LOW_RES_ARG_NAME = 'num_values_per_low_res_channel'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`satellite_io.find_file` and read by `satellite_io.read_file`.'
)
YEARS_HELP_STRING = (
    'Will use data from these years (list) to compute normalization params.'
)
NUM_FILES_HELP_STRING = (
    'Will randomly select this many satellite files from the years in `{0:s}`.'
).format(YEARS_ARG_NAME)

NUM_VALUES_PER_HIGH_RES_HELP_STRING = (
    'Number of reference values to keep for each high-resolution channel.'
)
NUM_VALUES_PER_LOW_RES_HELP_STRING = (
    'Number of reference values to keep for each low-resolution channel.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Normalization parameters (i.e., reference values for'
    ' uniformization) will be saved here by `normalization.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + YEARS_ARG_NAME, type=int, nargs='+', required=True,
    help=YEARS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_FILES_ARG_NAME, type=int, required=True,
    help=NUM_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_VALUES_PER_HIGH_RES_ARG_NAME, type=int, required=True,
    help=NUM_VALUES_PER_HIGH_RES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_VALUES_PER_LOW_RES_ARG_NAME, type=int, required=True,
    help=NUM_VALUES_PER_LOW_RES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(satellite_dir_name, years, num_satellite_files,
         num_values_per_high_res_channel, num_values_per_low_res_channel,
         output_file_name):
    """Computes normalization parameters.

    This is effectively the main method.

    :param satellite_dir_name: See documentation at top of file.
    :param years: Same.
    :param num_satellite_files: Same.
    :param num_values_per_high_res_channel: Same.
    :param num_values_per_low_res_channel: Same.
    :param output_file_name: Same.
    """

    error_checking.assert_is_geq(num_satellite_files, 10)

    cyclone_id_strings = satellite_io.find_cyclones(
        directory_name=satellite_dir_name, raise_error_if_all_missing=True
    )
    cyclone_years = numpy.array(
        [misc_utils.parse_cyclone_id(c)[0] for c in cyclone_id_strings],
        dtype=int
    )

    good_flags = numpy.array([c in years for c in cyclone_years], dtype=float)
    good_indices = numpy.where(good_flags)[0]
    cyclone_id_strings = [cyclone_id_strings[k] for k in good_indices]
    cyclone_id_strings.sort()

    satellite_file_names_list_list = [
        satellite_io.find_files_one_cyclone(
            directory_name=satellite_dir_name, cyclone_id_string=c,
            raise_error_if_all_missing=True
        )
        for c in cyclone_id_strings
    ]
    satellite_file_names = sum(satellite_file_names_list_list, [])

    if len(satellite_file_names) > num_satellite_files:
        file_indices = numpy.linspace(
            0, len(satellite_file_names) - 1, num=len(satellite_file_names),
            dtype=int
        )
        file_indices = numpy.random.choice(
            file_indices, size=num_satellite_files, replace=False
        )

        satellite_file_names = [satellite_file_names[k] for k in file_indices]

    normalization_param_table_xarray = normalization.get_normalization_params(
        satellite_file_names=satellite_file_names,
        num_values_per_low_res_channel=num_values_per_low_res_channel,
        num_values_per_high_res_channel=num_values_per_high_res_channel
    )
    print(SEPARATOR_STRING)

    print('Writing normalization params to file: "{0:s}"...'.format(
        output_file_name
    ))
    normalization.write_file(
        normalization_param_table_xarray=normalization_param_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        satellite_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        years=numpy.array(getattr(INPUT_ARG_OBJECT, YEARS_ARG_NAME), dtype=int),
        num_satellite_files=getattr(INPUT_ARG_OBJECT, NUM_FILES_ARG_NAME),
        num_values_per_high_res_channel=getattr(
            INPUT_ARG_OBJECT, NUM_VALUES_PER_HIGH_RES_ARG_NAME
        ),
        num_values_per_low_res_channel=getattr(
            INPUT_ARG_OBJECT, NUM_VALUES_PER_LOW_RES_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
