"""Normalizes satellite data."""

import os
import sys
import argparse

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import satellite_io
import normalization

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'input_satellite_dir_name'
CYCLONE_ID_ARG_NAME = 'cyclone_id_string'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
# COMPRESS_ARG_NAME = 'compress_output_files'
OUTPUT_DIR_ARG_NAME = 'output_satellite_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing unnormalized data.  Files therein '
    'will be found by `satellite_io.find_file` and read by '
    '`satellite_io.read_file`.'
)
CYCLONE_ID_HELP_STRING = 'Will normalize satellite data for this cyclone.'
NORMALIZATION_FILE_HELP_STRING = (
    'Path to file with normalization params (will be read by '
    '`normalization.read_file`).'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Normalized data will be written here by '
    '`satellite_io.write_file`, to exact locations determined by '
    '`satellite_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CYCLONE_ID_ARG_NAME, type=str, required=True,
    help=CYCLONE_ID_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=True,
    help=NORMALIZATION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_satellite_dir_name, cyclone_id_string, normalization_file_name,
         output_satellite_dir_name):
    """Normalizes satellite data.

    This is effectively the main method.

    :param input_satellite_dir_name: See documentation at top of file.
    :param cyclone_id_string: Same.
    :param normalization_file_name: Same.
    :param output_satellite_dir_name: Same.
    """

    print('Reading data from: "{0:s}"...'.format(normalization_file_name))
    normalization_param_table_xarray = normalization.read_file(
        normalization_file_name
    )

    input_satellite_file_names = satellite_io.find_files_one_cyclone(
        directory_name=input_satellite_dir_name,
        cyclone_id_string=cyclone_id_string, raise_error_if_all_missing=True
    )
    input_satellite_file_names.sort()

    output_satellite_file_names = [
        satellite_io.find_file(
            directory_name=output_satellite_dir_name,
            cyclone_id_string=cyclone_id_string,
            valid_date_string=satellite_io.file_name_to_date(f),
            raise_error_if_missing=False
        )
        for f in input_satellite_file_names
    ]

    for i in range(len(input_satellite_file_names)):
        print('Reading unnormalized data from: "{0:s}"...'.format(
            input_satellite_file_names[i]
        ))
        satellite_table_xarray = satellite_io.read_file(
            input_satellite_file_names[i]
        )

        satellite_table_xarray = normalization.normalize_data(
            satellite_table_xarray=satellite_table_xarray,
            normalization_param_table_xarray=normalization_param_table_xarray
        )

        print('Writing normalized data to: "{0:s}"...'.format(
            output_satellite_file_names[i]
        ))
        satellite_io.write_file(
            satellite_table_xarray=satellite_table_xarray,
            zarr_file_name=output_satellite_file_names[i]
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_satellite_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        cyclone_id_string=getattr(INPUT_ARG_OBJECT, CYCLONE_ID_ARG_NAME),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        output_satellite_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
