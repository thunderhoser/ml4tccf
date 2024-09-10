"""Splits predictions by day vs. night."""

import os
import sys
import glob
import shutil
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import prediction_io
import extended_best_track_io as ebtrk_io
import misc_utils
import scalar_prediction_utils
import split_predictions_by_basin as split_by_basin

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H%M'

INPUT_PREDICTION_FILES_ARG_NAME = 'input_prediction_file_pattern'
EBTRK_FILE_ARG_NAME = 'input_extended_best_track_file_name'
ALTITUDE_ANGLE_EXE_ARG_NAME = 'altitude_angle_exe_name'
TEMPORARY_DIR_ARG_NAME = 'temporary_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_prediction_dir_name'

INPUT_PREDICTION_FILES_HELP_STRING = (
    'Glob pattern for input files.  Each will be read by '
    '`prediction_io.read_file`.'
)
EBTRK_FILE_HELP_STRING = (
    'Path to file with extended best-track data (contains TC-center locations).'
    '  Will be read by `extended_best_track_io.read_file`.'
)
ALTITUDE_ANGLE_EXE_HELP_STRING = (
    'Path to Fortran executable (pathless file name should probably be '
    '"solarpos") that computes solar altitude angles.'
)
TEMPORARY_DIR_HELP_STRING = (
    'Path to directory for temporary files with solar altitude angles.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Within this directory, a "day" '
    'subdirectory and a "night" subdirectory will be created.  Then subset '
    'predictions will be written to these subdirectories by '
    '`scalar_prediction_io.write_file` or `gridded_prediction_io.write_file`, '
    'to exact locations determined by `prediction_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_PREDICTION_FILES_ARG_NAME, type=str, required=True,
    help=INPUT_PREDICTION_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EBTRK_FILE_ARG_NAME, type=str, required=True,
    help=EBTRK_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ALTITUDE_ANGLE_EXE_ARG_NAME, type=str, required=False,
    default=misc_utils.DEFAULT_EXE_NAME_FOR_ALTITUDE_ANGLE,
    help=ALTITUDE_ANGLE_EXE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TEMPORARY_DIR_ARG_NAME, type=str, required=True,
    help=TEMPORARY_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_prediction_file_pattern, ebtrk_file_name,
         altitude_angle_exe_name, temporary_dir_name,
         top_output_prediction_dir_name):
    """Splits predictions by day vs. night.

    This is effectively the main method.

    :param input_prediction_file_pattern: See documentation at top of file.
    :param ebtrk_file_name: Same.
    :param altitude_angle_exe_name: Same.
    :param temporary_dir_name: Same.
    :param top_output_prediction_dir_name: Same.
    :raises: ValueError: if no prediction files could be found.
    """

    # Read prediction files.
    input_prediction_file_names = glob.glob(input_prediction_file_pattern)
    if len(input_prediction_file_names) == 0:
        error_string = (
            'Could not find any prediction file with the following pattern: '
            '"{0:s}"'
        ).format(input_prediction_file_pattern)

        raise ValueError(error_string)

    input_prediction_file_names.sort()

    num_files = len(input_prediction_file_names)
    prediction_tables_xarray = [None] * num_files
    are_predictions_gridded = False

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(
            input_prediction_file_names[i]
        ))
        prediction_tables_xarray[i] = prediction_io.read_file(
            input_prediction_file_names[i]
        )

        are_predictions_gridded = (
            scalar_prediction_utils.PREDICTED_ROW_OFFSET_KEY
            not in prediction_tables_xarray[i]
        )

        if are_predictions_gridded:
            raise ValueError(
                'This script does not yet work for gridded predictions.'
            )

    prediction_table_xarray = scalar_prediction_utils.concat_over_examples(
        prediction_tables_xarray
    )

    # Read EBTRK (extended best-track) file.
    print('Reading data from: "{0:s}"...'.format(ebtrk_file_name))
    ebtrk_table_xarray = ebtrk_io.read_file(ebtrk_file_name)
    longitudes_deg_e, latitudes_deg_n = (
        misc_utils.match_predictions_to_tc_centers(
            prediction_table_xarray=prediction_table_xarray,
            ebtrk_table_xarray=ebtrk_table_xarray,
            return_xy=False
        )
    )

    # Compute solar altitude angle for every TC sample.
    target_times_unix_sec = (
        prediction_table_xarray[scalar_prediction_utils.TARGET_TIME_KEY].values
    )

    solar_altitude_angles_deg = [
        misc_utils.get_solar_altitude_angles(
            valid_time_unix_sec=t,
            latitudes_deg_n=numpy.array([y], dtype=float),
            longitudes_deg_e=numpy.array([x], dtype=float),
            temporary_dir_name=temporary_dir_name,
            fortran_exe_name=altitude_angle_exe_name
        )[0]
        for t, y, x in
        zip(target_times_unix_sec, latitudes_deg_n, longitudes_deg_e)
    ]

    solar_altitude_angles_deg = numpy.array(
        solar_altitude_angles_deg, dtype=int
    )

    day_indices = numpy.where(solar_altitude_angles_deg > 0)[0]
    this_output_dir_name = '{0:s}/day'.format(top_output_prediction_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=this_output_dir_name
    )
    split_by_basin._write_scalar_predictions_1basin(
        prediction_table_1basin_xarray=prediction_table_xarray.isel(
            indexers=
            {scalar_prediction_utils.EXAMPLE_DIM_KEY: day_indices}
        ),
        output_dir_name_1basin=this_output_dir_name
    )

    night_indices = numpy.where(solar_altitude_angles_deg < 0)[0]
    this_output_dir_name = '{0:s}/night'.format(top_output_prediction_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=this_output_dir_name
    )
    split_by_basin._write_scalar_predictions_1basin(
        prediction_table_1basin_xarray=prediction_table_xarray.isel(
            indexers=
            {scalar_prediction_utils.EXAMPLE_DIM_KEY: night_indices}
        ),
        output_dir_name_1basin=this_output_dir_name
    )

    shutil.rmtree(temporary_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_file_pattern=getattr(
            INPUT_ARG_OBJECT, INPUT_PREDICTION_FILES_ARG_NAME
        ),
        ebtrk_file_name=getattr(INPUT_ARG_OBJECT, EBTRK_FILE_ARG_NAME),
        altitude_angle_exe_name=getattr(
            INPUT_ARG_OBJECT, ALTITUDE_ANGLE_EXE_ARG_NAME
        ),
        temporary_dir_name=getattr(INPUT_ARG_OBJECT, TEMPORARY_DIR_ARG_NAME),
        top_output_prediction_dir_name=getattr(
            INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME
        )
    )
