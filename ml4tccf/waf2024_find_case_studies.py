"""Finds interesting case studies for 2024 WAF paper.

WAF = Weather and Forecasting
"""

import os
import sys
import glob
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import error_checking
import prediction_io
import scalar_prediction_utils as prediction_utils

# TODO(thunderhoser): Maybe let storm type also be an input arg.

TIME_FORMAT_FOR_LOG = '%Y-%m-%d-%H%M'

INPUT_FILE_PATTERN_ARG_NAME = 'input_prediction_file_pattern'
LATITUDE_LIMITS_ARG_NAME = 'latitude_limits_deg_n'
EUCLIDEAN_ERROR_LIMITS_ARG_NAME = 'euclidean_error_limits_km'
X_ERROR_LIMITS_ARG_NAME = 'x_error_limits_km'
Y_ERROR_LIMITS_ARG_NAME = 'y_error_limits_km'

INPUT_FILE_PATTERN_HELP_STRING = (
    'Glob pattern for input files, containing predictions and targets.  Each '
    'will be read by `prediction_io.read_file`, and TC samples in all these '
    'files will be searched together.'
)
LATITUDE_LIMITS_HELP_STRING = (
    'List with [min, max] latitude in deg north.  Will look for TC samples in '
    'this latitude range.  If you do not care about latitude range, leave this '
    'argument alone.'
)
EUCLIDEAN_ERROR_LIMITS_HELP_STRING = (
    'List with [min, max] Euclidean errors.  Will look for TC samples with '
    'this range of Euclidean errors.  If you do not care about Euclidean '
    'error, leave this argument alone.'
)
X_ERROR_LIMITS_HELP_STRING = (
    'List with [min, max] x-coord errors.  Will look for TC samples with this '
    'range of x-errors.  If you do not care about x-error, leave this argument '
    'alone.'
)
Y_ERROR_LIMITS_HELP_STRING = (
    'List with [min, max] y-coord errors.  Will look for TC samples with this '
    'range of y-errors.  If you do not care about y-error, leave this argument '
    'alone.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_PATTERN_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_PATTERN_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LATITUDE_LIMITS_ARG_NAME, type=float, nargs=2, required=False,
    default=[1, -1], help=LATITUDE_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EUCLIDEAN_ERROR_LIMITS_ARG_NAME, type=float, nargs=2, required=False,
    default=[1, -1], help=EUCLIDEAN_ERROR_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + X_ERROR_LIMITS_ARG_NAME, type=float, nargs=2, required=False,
    default=[1, -1], help=X_ERROR_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + Y_ERROR_LIMITS_ARG_NAME, type=float, nargs=2, required=False,
    default=[1, -1], help=Y_ERROR_LIMITS_HELP_STRING
)


def _create_pred_and_target_matrices(prediction_table_xarray):
    """Creates simple prediction and target matrices from xarray table.

    E = number of examples

    :param prediction_table_xarray: xarray table in format returned by
        `prediction_io.read_file`.
    :return: prediction_matrix_km: E-by-2 numpy array with predicted y-offsets
        (first column) and x-offsets (second column) in km.
    :return: target_matrix_km: E-by-2 numpy array with actual y-offsets (first
        column) and x-offsets (second column) in km.
    """

    ptx = prediction_table_xarray

    grid_spacings_km = ptx[prediction_utils.GRID_SPACING_KEY].values
    prediction_matrix_km = numpy.transpose(numpy.vstack([
        grid_spacings_km *
        ptx[prediction_utils.PREDICTED_ROW_OFFSET_KEY].values[:, 0],
        grid_spacings_km *
        ptx[prediction_utils.PREDICTED_COLUMN_OFFSET_KEY].values[:, 0]
    ]))
    target_matrix_km = numpy.transpose(numpy.vstack([
        grid_spacings_km * ptx[prediction_utils.ACTUAL_ROW_OFFSET_KEY].values,
        grid_spacings_km * ptx[prediction_utils.ACTUAL_COLUMN_OFFSET_KEY].values
    ]))

    return prediction_matrix_km, target_matrix_km


def _run(prediction_file_pattern, latitude_limits_deg_n,
         euclidean_error_limits_km, x_error_limits_km, y_error_limits_km):
    """Finds interesting case studies for 2024 WAF paper.

    This is effectively the main method.

    :param prediction_file_pattern: See documentation at top of this script.
    :param latitude_limits_deg_n: Same.
    :param euclidean_error_limits_km: Same.
    :param x_error_limits_km: Same.
    :param y_error_limits_km: Same.
    """

    # Check input args.
    prediction_file_names = glob.glob(prediction_file_pattern)
    if len(prediction_file_names) == 0:
        error_string = (
            'Could not find any prediction file with the following pattern: '
            '"{0:s}"'
        ).format(prediction_file_pattern)

        raise ValueError(error_string)

    prediction_file_names.sort()

    if numpy.diff(latitude_limits_deg_n)[0] <= 0:
        latitude_limits_deg_n = None
    if latitude_limits_deg_n is not None:
        error_checking.assert_is_valid_lat_numpy_array(latitude_limits_deg_n)

    if numpy.diff(euclidean_error_limits_km)[0] <= 0:
        euclidean_error_limits_km = None
    if euclidean_error_limits_km is not None:
        error_checking.assert_is_geq_numpy_array(euclidean_error_limits_km, 0.)

    if numpy.diff(x_error_limits_km)[0] <= 0:
        x_error_limits_km = None
    if x_error_limits_km is not None:
        error_checking.assert_is_numpy_array_without_nan(x_error_limits_km)

    if numpy.diff(y_error_limits_km)[0] <= 0:
        y_error_limits_km = None
    if y_error_limits_km is not None:
        error_checking.assert_is_numpy_array_without_nan(y_error_limits_km)

    print('Reading data from: "{0:s}"...'.format(prediction_file_names[0]))
    first_prediction_table_xarray = prediction_io.read_file(
        prediction_file_names[0]
    )

    are_predictions_gridded = (
        prediction_utils.PREDICTED_ROW_OFFSET_KEY
        not in first_prediction_table_xarray
    )
    if are_predictions_gridded:
        raise ValueError(
            'This script does not yet work for gridded predictions.'
        )

    # Do actual stuff.
    num_files = len(prediction_file_names)
    prediction_tables_xarray = [None] * num_files

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(prediction_file_names[i]))
        prediction_tables_xarray[i] = prediction_io.read_file(
            prediction_file_names[i]
        )

    prediction_table_xarray = prediction_utils.concat_over_examples(
        prediction_tables_xarray
    )
    prediction_table_xarray = prediction_utils.get_ensemble_mean(
        prediction_table_xarray
    )
    ptx = prediction_table_xarray

    if latitude_limits_deg_n is not None:
        orig_num_examples = len(
            ptx[prediction_utils.ACTUAL_CENTER_LATITUDE_KEY].values
        )

        keep_indices = numpy.where(numpy.logical_and(
            ptx[prediction_utils.ACTUAL_CENTER_LATITUDE_KEY].values >=
            latitude_limits_deg_n[0],
            ptx[prediction_utils.ACTUAL_CENTER_LATITUDE_KEY].values <=
            latitude_limits_deg_n[1]
        ))[0]
        ptx = ptx.isel({prediction_utils.EXAMPLE_DIM_KEY: keep_indices})

        num_examples = len(
            ptx[prediction_utils.ACTUAL_CENTER_LATITUDE_KEY].values
        )

        print((
            'Found {0:d} of {1:d} examples in the latitude range '
            '[{2:.2f}, {3:.2f}] deg N!'
        ).format(
            num_examples,
            orig_num_examples,
            latitude_limits_deg_n[0],
            latitude_limits_deg_n[1]
        ))

    if euclidean_error_limits_km is not None:
        prediction_matrix_km, target_matrix_km = (
            _create_pred_and_target_matrices(prediction_table_xarray)
        )
        all_euclidean_errors_km = numpy.sqrt(
            (prediction_matrix_km[:, 0] - target_matrix_km[:, 0]) ** 2 +
            (prediction_matrix_km[:, 1] - target_matrix_km[:, 1]) ** 2
        )
        orig_num_examples = len(all_euclidean_errors_km)

        keep_indices = numpy.where(numpy.logical_and(
            all_euclidean_errors_km >= euclidean_error_limits_km[0],
            all_euclidean_errors_km <= euclidean_error_limits_km[1]
        ))[0]
        ptx = ptx.isel({prediction_utils.EXAMPLE_DIM_KEY: keep_indices})

        num_examples = len(
            ptx[prediction_utils.ACTUAL_CENTER_LATITUDE_KEY].values
        )

        print((
            'Found {0:d} of {1:d} examples in the Euclidean-error range '
            '[{2:.2f}, {3:.2f}] km!'
        ).format(
            num_examples,
            orig_num_examples,
            euclidean_error_limits_km[0],
            euclidean_error_limits_km[1]
        ))

    if x_error_limits_km is not None:
        prediction_matrix_km, target_matrix_km = (
            _create_pred_and_target_matrices(prediction_table_xarray)
        )
        all_x_errors_km = prediction_matrix_km[:, 1] - target_matrix_km[:, 1]
        orig_num_examples = len(all_x_errors_km)

        keep_indices = numpy.where(numpy.logical_and(
            all_x_errors_km >= x_error_limits_km[0],
            all_x_errors_km <= x_error_limits_km[1]
        ))[0]
        ptx = ptx.isel({prediction_utils.EXAMPLE_DIM_KEY: keep_indices})

        num_examples = len(
            ptx[prediction_utils.ACTUAL_CENTER_LATITUDE_KEY].values
        )

        print((
            'Found {0:d} of {1:d} examples in the x-error range '
            '[{2:.2f}, {3:.2f}] km!'
        ).format(
            num_examples,
            orig_num_examples,
            x_error_limits_km[0],
            x_error_limits_km[1]
        ))

    if y_error_limits_km is not None:
        prediction_matrix_km, target_matrix_km = (
            _create_pred_and_target_matrices(prediction_table_xarray)
        )
        all_y_errors_km = prediction_matrix_km[:, 0] - target_matrix_km[:, 0]
        orig_num_examples = len(all_y_errors_km)

        keep_indices = numpy.where(numpy.logical_and(
            all_y_errors_km >= y_error_limits_km[0],
            all_y_errors_km <= y_error_limits_km[1]
        ))[0]
        ptx = ptx.isel({prediction_utils.EXAMPLE_DIM_KEY: keep_indices})

        num_examples = len(
            ptx[prediction_utils.ACTUAL_CENTER_LATITUDE_KEY].values
        )

        print((
            'Found {0:d} of {1:d} examples in the y-error range '
            '[{2:.2f}, {3:.2f}] km!'
        ).format(
            num_examples,
            orig_num_examples,
            y_error_limits_km[0],
            y_error_limits_km[1]
        ))

    sort_indices = numpy.argsort(ptx[prediction_utils.TARGET_TIME_KEY].values)
    ptx = ptx.isel({prediction_utils.EXAMPLE_DIM_KEY: sort_indices})

    prediction_matrix_km, target_matrix_km = (
        _create_pred_and_target_matrices(prediction_table_xarray)
    )
    y_errors_km = prediction_matrix_km[:, 0] - target_matrix_km[:, 0]
    x_errors_km = prediction_matrix_km[:, 1] - target_matrix_km[:, 1]
    euclidean_errors_km = numpy.sqrt(
        (prediction_matrix_km[:, 0] - target_matrix_km[:, 0]) ** 2 +
        (prediction_matrix_km[:, 1] - target_matrix_km[:, 1]) ** 2
    )
    cyclone_id_strings = ptx[prediction_utils.CYCLONE_ID_KEY].values
    try:
        cyclone_id_strings = [c.decode('utf-8') for c in cyclone_id_strings]
    except:
        pass

    for i in range(len(cyclone_id_strings)):
        print((
            '{0:d}th example remaining ... '
            '{1:s} at {2:s} and {3:.2f} deg N ... '
            'Euclidean/x/y errors = {4:.2f}, {5:.2f}, {6:.2f} km'
        ).format(
            i + 1,
            cyclone_id_strings[i],
            time_conversion.unix_sec_to_string(
                ptx[prediction_utils.TARGET_TIME_KEY].values[i],
                TIME_FORMAT_FOR_LOG
            ),
            ptx[prediction_utils.ACTUAL_CENTER_LATITUDE_KEY].values[i],
            euclidean_errors_km[i],
            x_errors_km[i],
            y_errors_km[i]
        ))


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_pattern=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_PATTERN_ARG_NAME
        ),
        latitude_limits_deg_n=numpy.array(
            getattr(INPUT_ARG_OBJECT, LATITUDE_LIMITS_ARG_NAME), dtype=float
        ),
        euclidean_error_limits_km=numpy.array(
            getattr(INPUT_ARG_OBJECT, EUCLIDEAN_ERROR_LIMITS_ARG_NAME),
            dtype=float
        ),
        x_error_limits_km=numpy.array(
            getattr(INPUT_ARG_OBJECT, X_ERROR_LIMITS_ARG_NAME), dtype=float
        ),
        y_error_limits_km=numpy.array(
            getattr(INPUT_ARG_OBJECT, Y_ERROR_LIMITS_ARG_NAME), dtype=float
        )
    )
