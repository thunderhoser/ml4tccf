"""Converts short-track data from raw Pickle format to NetCDF format."""

import argparse
from ml4tccf.io import short_track_io
from ml4tccf.io import raw_short_track_io

TIME_FORMAT = '%Y-%m-%d-%H%M'
TEN_MINUTES_TO_SECONDS = 600

INPUT_DIR_ARG_NAME = 'input_raw_dir_name'
CYCLONE_ID_ARG_NAME = 'cyclone_id_string'
NUM_MINUTES_BACK_ARG_NAME = 'num_minutes_back'
NUM_MINUTES_AHEAD_ARG_NAME = 'num_minutes_ahead'
OUTPUT_DIR_ARG_NAME = 'output_processed_dir_name'

INPUT_DIR_HELP_STRING = (
    'Path to input directory.  Raw files therein will be found by '
    '`raw_short_track_io.find_file`.'
)
CYCLONE_ID_HELP_STRING = (
    'Cyclone ID.  All raw files for this cyclone will be processed into one '
    'NetCDF file.'
)
NUM_MINUTES_BACK_HELP_STRING = (
    'For each init time, will process short-track forecasts valid this many '
    'minutes into the past.'
)
NUM_MINUTES_AHEAD_HELP_STRING = (
    'For each init time, will process short-track forecasts valid this many '
    'minutes into the future.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  The processed NetCDF file will be written here '
    'by `short_track_io.write_file`, to an exact location determined by '
    '`short_track_io.find_file`.'
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
    '--' + NUM_MINUTES_BACK_ARG_NAME, type=int, required=True,
    help=NUM_MINUTES_BACK_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_MINUTES_AHEAD_ARG_NAME, type=int, required=True,
    help=NUM_MINUTES_AHEAD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_dir_name, cyclone_id_string, num_minutes_back, num_minutes_ahead,
         output_dir_name):
    """Converts short-track data from raw Pickle format to NetCDF format.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of this script.
    :param cyclone_id_string: Same.
    :param num_minutes_back: Same.
    :param num_minutes_ahead: Same.
    :param output_dir_name: Same.
    """

    raw_file_names = raw_short_track_io.find_files_one_cyclone(
        directory_name=input_dir_name,
        cyclone_id_string=cyclone_id_string,
        raise_error_if_all_missing=True
    )
    short_track_tables_xarray = [None] * len(raw_file_names)

    for i in range(len(raw_file_names)):
        print('Reading data from: "{0:s}"...'.format(raw_file_names[i]))
        short_track_tables_xarray[i] = raw_short_track_io.read_file(
            pickle_file_name=raw_file_names[i],
            num_minutes_back=num_minutes_back,
            num_minutes_ahead=num_minutes_ahead
        )

    short_track_table_xarray = short_track_io.concat_over_init_time(
        short_track_tables_xarray
    )

    output_file_name = short_track_io.find_file(
        directory_name=output_dir_name,
        cyclone_id_string=cyclone_id_string,
        raise_error_if_missing=False
    )

    print('Writing processed data to: "{0:s}"...'.format(output_file_name))
    short_track_io.write_file(
        short_track_table_xarray=short_track_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        cyclone_id_string=getattr(INPUT_ARG_OBJECT, CYCLONE_ID_ARG_NAME),
        num_minutes_back=getattr(INPUT_ARG_OBJECT, NUM_MINUTES_BACK_ARG_NAME),
        num_minutes_ahead=getattr(INPUT_ARG_OBJECT, NUM_MINUTES_AHEAD_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
