"""Splits predictions by x-y (nadir-relative) coordinates."""

import os
import sys
import glob
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import number_rounding
import file_system_utils
import error_checking
import prediction_io
import extended_best_track_io as ebtrk_io
import misc_utils
import scalar_prediction_utils
import neural_net_training_cira_ir as nn_training
import split_predictions_by_intensity as split_by_intensity

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H%M'

METRES_TO_KM = 0.001
KM_TO_METRES = 1000.
HOURS_TO_SECONDS = 3600

MAX_XY_COORD_METRES = misc_utils.MAX_XY_COORD_METRES
SYNOPTIC_TIME_TOLERANCE_SEC = nn_training.SYNOPTIC_TIME_TOLERANCE_SEC

INPUT_PREDICTION_FILES_ARG_NAME = 'input_prediction_file_pattern'
EBTRK_FILE_ARG_NAME = 'input_extended_best_track_file_name'
X_COORD_CUTOFFS_ARG_NAME = 'x_coord_cutoffs_metres'
Y_COORD_CUTOFFS_ARG_NAME = 'y_coord_cutoffs_metres'
OUTPUT_DIR_ARG_NAME = 'output_prediction_dir_name'

INPUT_PREDICTION_FILES_HELP_STRING = (
    'Glob pattern for input files.  Each will be read by '
    '`prediction_io.read_file`.'
)
EBTRK_FILE_HELP_STRING = (
    'Path to file with extended best-track data (contains TC-center locations).'
    '  Will be read by `extended_best_track_io.read_file`.'
)
X_COORD_CUTOFFS_HELP_STRING = (
    'Category cutoffs for nadir-relative x-coordinate.  Please leave -inf and '
    '+inf out of this list, as they will be added automatically.'
)
Y_COORD_CUTOFFS_HELP_STRING = (
    'Category cutoffs for nadir-relative y-coordinate.  Please leave -inf and '
    '+inf out of this list, as they will be added automatically.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Within this directory, one category '
    'will be created for every 2-D bin (based on both x- and y-coords).  Then '
    'subset predictions will be written to these subdirectories by '
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
    '--' + X_COORD_CUTOFFS_ARG_NAME, type=float, nargs='+', required=True,
    help=X_COORD_CUTOFFS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + Y_COORD_CUTOFFS_ARG_NAME, type=float, nargs='+', required=True,
    help=Y_COORD_CUTOFFS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def check_input_args(x_coord_cutoffs_metres, y_coord_cutoffs_metres):
    """Checks input arguments.

    :param x_coord_cutoffs_metres: See documentation at top of file.
    :param y_coord_cutoffs_metres: Same.
    :return: x_coord_cutoffs_metres: Same as input but with end values
        (-inf and inf).
    :return: y_coord_cutoffs_metres: Same as input but with end values
        (-inf and inf).
    """

    x_coord_cutoffs_metres = number_rounding.round_to_nearest(
        x_coord_cutoffs_metres, KM_TO_METRES
    )
    error_checking.assert_is_greater_numpy_array(
        x_coord_cutoffs_metres, -MAX_XY_COORD_METRES
    )
    error_checking.assert_is_less_than_numpy_array(
        x_coord_cutoffs_metres, MAX_XY_COORD_METRES
    )
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(x_coord_cutoffs_metres), 0.
    )
    x_coord_cutoffs_metres = numpy.concatenate((
        numpy.array([-numpy.inf]),
        x_coord_cutoffs_metres,
        numpy.array([numpy.inf])
    ))

    y_coord_cutoffs_metres = number_rounding.round_to_nearest(
        y_coord_cutoffs_metres, KM_TO_METRES
    )
    error_checking.assert_is_greater_numpy_array(
        y_coord_cutoffs_metres, -MAX_XY_COORD_METRES
    )
    error_checking.assert_is_less_than_numpy_array(
        y_coord_cutoffs_metres, MAX_XY_COORD_METRES
    )
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(y_coord_cutoffs_metres), 0.
    )
    y_coord_cutoffs_metres = numpy.concatenate((
        numpy.array([-numpy.inf]),
        y_coord_cutoffs_metres,
        numpy.array([numpy.inf])
    ))

    return x_coord_cutoffs_metres, y_coord_cutoffs_metres


def _run(input_prediction_file_pattern, ebtrk_file_name,
         x_coord_cutoffs_metres, y_coord_cutoffs_metres,
         top_output_prediction_dir_name):
    """Splits predictions by x-y (nadir-relative) coordinates.

    This is effectively the main method.

    :param input_prediction_file_pattern: See documentation at top of file.
    :param ebtrk_file_name: Same.
    :param x_coord_cutoffs_metres: Same.
    :param y_coord_cutoffs_metres: Same.
    :param top_output_prediction_dir_name: Same.
    :raises: ValueError: if no prediction files could be found.
    """

    x_coord_cutoffs_metres, y_coord_cutoffs_metres = check_input_args(
        x_coord_cutoffs_metres=x_coord_cutoffs_metres,
        y_coord_cutoffs_metres=y_coord_cutoffs_metres
    )

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

    print('Reading data from: "{0:s}"...'.format(ebtrk_file_name))
    ebtrk_table_xarray = ebtrk_io.read_file(ebtrk_file_name)
    prediction_x_coords_metres, prediction_y_coords_metres = (
        misc_utils.match_predictions_to_tc_centers(
            prediction_table_xarray=prediction_table_xarray,
            ebtrk_table_xarray=ebtrk_table_xarray,
            return_xy=True
        )
    )

    num_examples = len(prediction_x_coords_metres)

    for i in range(len(x_coord_cutoffs_metres)):
        for j in range(len(y_coord_cutoffs_metres)):
            if i == len(x_coord_cutoffs_metres) - 1:
                x_flags = numpy.full(num_examples, True, dtype=bool)
            else:
                x_flags = numpy.logical_and(
                    prediction_x_coords_metres >= x_coord_cutoffs_metres[i],
                    prediction_x_coords_metres < x_coord_cutoffs_metres[i + 1]
                )

            if j == len(y_coord_cutoffs_metres) - 1:
                y_flags = numpy.full(num_examples, True, dtype=bool)
            else:
                y_flags = numpy.logical_and(
                    prediction_y_coords_metres >= y_coord_cutoffs_metres[j],
                    prediction_y_coords_metres < y_coord_cutoffs_metres[j + 1]
                )

            these_indices = numpy.where(numpy.logical_and(
                x_flags, y_flags
            ))[0]

            if i == len(x_coord_cutoffs_metres) - 1:
                this_output_dir_name = '{0:s}/x=all'.format(
                    top_output_prediction_dir_name
                )
            else:
                this_output_dir_name = (
                    '{0:s}/x={1:+04.0f}_{2:+04.0f}km'
                ).format(
                    top_output_prediction_dir_name,
                    METRES_TO_KM * x_coord_cutoffs_metres[i],
                    METRES_TO_KM * x_coord_cutoffs_metres[i + 1]
                )

            if j == len(y_coord_cutoffs_metres) - 1:
                this_output_dir_name += '_y=all'
            else:
                this_output_dir_name += (
                    '_y={0:+04.0f}_{1:+04.0f}km'
                ).format(
                    METRES_TO_KM * y_coord_cutoffs_metres[j],
                    METRES_TO_KM * y_coord_cutoffs_metres[j + 1]
                )

            file_system_utils.mkdir_recursive_if_necessary(
                directory_name=this_output_dir_name
            )

            if len(these_indices) == 0:
                continue

            split_by_intensity._write_scalar_predictions_1category(
                prediction_table_1cat_xarray=prediction_table_xarray.isel(
                    indexers=
                    {scalar_prediction_utils.EXAMPLE_DIM_KEY: these_indices}
                ),
                output_dir_name_1cat=this_output_dir_name
            )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_file_pattern=getattr(
            INPUT_ARG_OBJECT, INPUT_PREDICTION_FILES_ARG_NAME
        ),
        ebtrk_file_name=getattr(INPUT_ARG_OBJECT, EBTRK_FILE_ARG_NAME),
        x_coord_cutoffs_metres=numpy.array(
            getattr(INPUT_ARG_OBJECT, X_COORD_CUTOFFS_ARG_NAME),
            dtype=float
        ),
        y_coord_cutoffs_metres=numpy.array(
            getattr(INPUT_ARG_OBJECT, Y_COORD_CUTOFFS_ARG_NAME),
            dtype=float
        ),
        top_output_prediction_dir_name=getattr(
            INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME
        )
    )
