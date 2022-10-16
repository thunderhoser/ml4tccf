"""Processes XBT (extended best-track) data, all into one NetCDF file."""

import glob
import argparse
import xarray
from ml4tccf.io import raw_extended_best_track_io as raw_xbt_io
from ml4tccf.io import extended_best_track_io as xbt_io
from ml4tccf.utils import extended_best_track_utils as xbt_utils

INPUT_DIR_ARG_NAME = 'input_dir_name'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing all raw XBT files.  This script will '
    'look for files therein with the ".txt" extension.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Processed XBT data will be stored here in NetCDF '
    'format.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_dir_name, output_file_name):
    """Processes XBT (extended best-track) data, all into one NetCDF file.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param output_file_name: Same.
    """

    input_file_names = glob.glob('{0:s}/*.txt'.format(input_dir_name))

    num_input_files = len(input_file_names)
    xbt_tables_xarray = [None] * num_input_files

    for i in range(num_input_files):
        print('Reading data from: "{0:s}"...'.format(input_file_names[i]))
        xbt_tables_xarray[i] = raw_xbt_io.read_file(input_file_names[i])

    xbt_table_xarray = xarray.concat(
        objs=xbt_tables_xarray, dim=xbt_utils.STORM_OBJECT_DIM
    )
    print(xbt_table_xarray)

    # for this_key in xbt_table_xarray:
    #     print(xbt_table_xarray[this_key])

    print('Writing processed data to: "{0:s}"...'.format(output_file_name))
    xbt_io.write_file(
        xbt_table_xarray=xbt_table_xarray, netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
