"""Processes A-deck data, from many cyclones, into one NetCDF file."""

import argparse
import numpy
from gewittergefahr.gg_utils import error_checking
from ml4tccf.io import a_deck_io
from ml4tccf.io import raw_a_deck_io

# TODO(thunderhoser): Not putting this file in the stand-alone repository,
# since it depends on Chris Slocum's ATCF library.

INPUT_DIR_ARG_NAME = 'input_dir_name'
FIRST_YEAR_ARG_NAME = 'first_year'
LAST_YEAR_ARG_NAME = 'last_year'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing all raw A-deck files.  Files therein '
    'will be found by `raw_a_deck_io.find_file`.'
)
YEAR_HELP_STRING = (
    'Will process all cyclones in the period `{0:s}`...`{1:s}`.'
).format(FIRST_YEAR_ARG_NAME, LAST_YEAR_ARG_NAME)

OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Processed data will be stored here in NetCDF format.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_YEAR_ARG_NAME, type=int, required=True, help=YEAR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_YEAR_ARG_NAME, type=int, required=True, help=YEAR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_dir_name, first_year, last_year, output_file_name):
    """Processes A-deck data, from many cyclones, into one NetCDF file.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param first_year: Same.
    :param last_year: Same.
    :param output_file_name: Same.
    """

    error_checking.assert_is_geq(last_year, first_year)
    years = numpy.linspace(
        first_year, last_year, num=last_year - first_year + 1, dtype=int
    )

    cyclone_id_strings = []
    for this_year in years:
        cyclone_id_strings += raw_a_deck_io.find_cyclones_one_year(
            directory_name=input_dir_name, year=this_year,
            raise_error_if_all_missing=True
        )

    num_cyclones = len(cyclone_id_strings)
    a_deck_tables_xarray = [None] * num_cyclones

    for i in range(num_cyclones):
        this_file_name = raw_a_deck_io.find_file(
            directory_name=input_dir_name,
            cyclone_id_string=cyclone_id_strings[i],
            raise_error_if_missing=True
        )

        print('Reading data from: "{0:s}"...'.format(this_file_name))
        a_deck_tables_xarray[i] = raw_a_deck_io.read_file(this_file_name)

    a_deck_table_xarray = a_deck_io.concat_tables_over_storm_object(
        a_deck_tables_xarray
    )

    print('Writing data to: "{0:s}"...'.format(output_file_name))
    a_deck_io.write_file(
        netcdf_file_name=output_file_name,
        a_deck_table_xarray=a_deck_table_xarray
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_year=getattr(INPUT_ARG_OBJECT, FIRST_YEAR_ARG_NAME),
        last_year=getattr(INPUT_ARG_OBJECT, LAST_YEAR_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
