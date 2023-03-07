"""Evaluates trained neural net."""

import os
import sys
import glob
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import prediction_io
import scalar_prediction_utils
import scalar_evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILE_PATTERN_ARG_NAME = 'input_prediction_file_pattern'
NUM_BOOTSTRAP_REPS_ARG_NAME = 'num_bootstrap_reps'
NUM_XY_OFFSET_BINS_ARG_NAME = 'num_xy_offset_bins'
XY_OFFSET_LIMITS_ARG_NAME = 'xy_offset_limits_metres'
XY_OFFSET_LIMITS_PRCTILE_ARG_NAME = 'xy_offset_limits_percentile'
NUM_OFFSET_DIST_BINS_ARG_NAME = 'num_offset_distance_bins'
OFFSET_DIST_LIMITS_ARG_NAME = 'offset_distance_limits_metres'
OFFSET_DIST_LIMITS_PRCTILE_ARG_NAME = 'offset_distance_limits_percentile'
NUM_OFFSET_DIR_BINS_ARG_NAME = 'num_offset_direction_bins'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_PATTERN_HELP_STRING = (
    'Glob pattern for input files.  Each will be read by '
    '`prediction_io.read_file`, and predictions in all these files will be '
    'evaluated together.'
)
NUM_BOOTSTRAP_REPS_HELP_STRING = 'Number of bootstrap replicates.'
NUM_XY_OFFSET_BINS_HELP_STRING = (
    'Number of bins in reliability curve for two target variables: x-offset '
    'distance and y-offset distance.'
)
XY_OFFSET_LIMITS_HELP_STRING = (
    'Min and max offsets (list of two values, in metres), used to create bins '
    'for reliability curve for two target variables: x-offset distance and '
    'y-offset distance.  If you want to specify min/max by percentiles instead'
    ' -- chosen independently for both x-offset and y-offset -- leave this '
    'argument alone.'
)
XY_OFFSET_LIMITS_PRCTILE_HELP_STRING = (
    'Min and max percentiles (list of two values, in range 0...100), used to '
    'create bins for reliability curve for two target variables: x-offset '
    'distance and y-offset distance.  If you want to specify min/max by '
    'physical values instead, leave this argument alone.'
)
NUM_OFFSET_DIST_BINS_HELP_STRING = (
    'Number of bins in reliability curve for total (Euclidean) offset distance.'
)
OFFSET_DIST_LIMITS_HELP_STRING = (
    'Min and max distances (list of two values, in metres), used to create '
    'bins for reliability curve for total (Euclidean) offset distance.  If you '
    'want to specify min/max by percentiles instead, leave this argument alone.'
)
OFFSET_DIST_LIMITS_PRCTILE_HELP_STRING = (
    'Min and max percentiles (list of two values, in range 0...100), used to '
    'create bins for reliability curve for total (Euclidean) offset distance.  '
    'If you want to specify min/max by physical values instead, leave this '
    'argument alone.'
)
NUM_OFFSET_DIR_BINS_HELP_STRING = (
    'Number of bins in reliability curve for offset direction.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output (NetCDF) file.  Evaluation results will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_PATTERN_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_PATTERN_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BOOTSTRAP_REPS_ARG_NAME, type=int, required=True,
    help=NUM_BOOTSTRAP_REPS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_XY_OFFSET_BINS_ARG_NAME, type=int, required=False, default=20,
    help=NUM_XY_OFFSET_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + XY_OFFSET_LIMITS_ARG_NAME, type=float, nargs=2, required=False,
    default=[1, -1], help=XY_OFFSET_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + XY_OFFSET_LIMITS_PRCTILE_ARG_NAME, type=float, nargs=2,
    required=False, default=[0.5, 99.5],
    help=XY_OFFSET_LIMITS_PRCTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_OFFSET_DIST_BINS_ARG_NAME, type=int, required=False, default=20,
    help=NUM_OFFSET_DIST_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OFFSET_DIST_LIMITS_ARG_NAME, type=float, nargs=2, required=False,
    default=[1, -1], help=OFFSET_DIST_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OFFSET_DIST_LIMITS_PRCTILE_ARG_NAME, type=float, nargs=2,
    required=False, default=[0.5, 99.5],
    help=OFFSET_DIST_LIMITS_PRCTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_OFFSET_DIR_BINS_ARG_NAME, type=int, required=False, default=36,
    help=NUM_OFFSET_DIR_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(prediction_file_pattern, num_bootstrap_reps, num_xy_offset_bins,
         xy_offset_limits_metres, xy_offset_limits_percentile,
         num_offset_distance_bins, offset_distance_limits_metres,
         offset_distance_limits_percentile,
         num_offset_direction_bins, output_file_name):
    """Evaluates trained neural net.

    This is effectively the main method.

    :param prediction_file_pattern: See documentation at top of file.
    :param num_bootstrap_reps: Same.
    :param num_xy_offset_bins: Same.
    :param xy_offset_limits_metres: Same.
    :param xy_offset_limits_percentile: Same.
    :param num_offset_distance_bins: Same.
    :param offset_distance_limits_metres: Same.
    :param offset_distance_limits_percentile: Same.
    :param num_offset_direction_bins: Same.
    :param output_file_name: Same.
    :raises: ValueError: if no prediction files could be found.
    """

    min_xy_offset_metres = xy_offset_limits_metres[0]
    max_xy_offset_metres = xy_offset_limits_metres[1]
    min_xy_offset_percentile = xy_offset_limits_percentile[0]
    max_xy_offset_percentile = xy_offset_limits_percentile[1]

    if min_xy_offset_metres >= max_xy_offset_metres:
        min_xy_offset_metres = None
        max_xy_offset_metres = None

        error_checking.assert_is_leq(min_xy_offset_percentile, 10.)
        error_checking.assert_is_geq(max_xy_offset_percentile, 90.)
    else:
        min_xy_offset_percentile = None
        max_xy_offset_percentile = None

    min_offset_distance_metres = offset_distance_limits_metres[0]
    max_offset_distance_metres = offset_distance_limits_metres[1]
    min_offset_distance_percentile = offset_distance_limits_percentile[0]
    max_offset_distance_percentile = offset_distance_limits_percentile[1]

    if min_offset_distance_metres >= max_offset_distance_metres:
        min_offset_distance_metres = None
        max_offset_distance_metres = None

        error_checking.assert_is_leq(min_offset_distance_percentile, 10.)
        error_checking.assert_is_geq(max_offset_distance_percentile, 90.)
    else:
        min_offset_distance_percentile = None
        max_offset_distance_percentile = None

        error_checking.assert_is_geq(min_offset_distance_metres, 0.)
        error_checking.assert_is_greater(max_offset_distance_metres, 0.)

    prediction_file_names = glob.glob(prediction_file_pattern)
    if len(prediction_file_names) == 0:
        error_string = (
            'Could not find any prediction file with the following pattern: '
            '"{0:s}"'
        ).format(prediction_file_pattern)

        raise ValueError(error_string)

    prediction_file_names.sort()

    print('Reading data from: "{0:s}"...'.format(prediction_file_names[0]))
    first_prediction_table_xarray = prediction_io.read_file(
        prediction_file_names[0]
    )

    are_predictions_gridded = (
        scalar_prediction_utils.PREDICTED_ROW_OFFSET_KEY
        not in first_prediction_table_xarray
    )
    if are_predictions_gridded:
        raise ValueError(
            'This script does not yet work for gridded predictions.'
        )

    result_table_xarray = scalar_evaluation.get_scores_all_variables(
        prediction_file_names=prediction_file_names,
        num_bootstrap_reps=num_bootstrap_reps,
        num_xy_offset_bins=num_xy_offset_bins,
        min_xy_offset_metres=min_xy_offset_metres,
        max_xy_offset_metres=max_xy_offset_metres,
        min_xy_offset_percentile=min_xy_offset_percentile,
        max_xy_offset_percentile=max_xy_offset_percentile,
        num_offset_distance_bins=num_offset_distance_bins,
        min_offset_distance_metres=min_offset_distance_metres,
        max_offset_distance_metres=max_offset_distance_metres,
        min_offset_distance_percentile=min_offset_distance_percentile,
        max_offset_distance_percentile=max_offset_distance_percentile,
        num_offset_direction_bins=num_offset_direction_bins
    )
    print(SEPARATOR_STRING)

    t = result_table_xarray
    target_field_names = t.coords[scalar_evaluation.TARGET_FIELD_DIM].values

    for j in range(len(target_field_names)):
        print((
            'Variable = {0:s} ... '
            'stdev of target and predicted values = {1:f}, {2:f} ... '
            'MSE and skill score = {3:f}, {4:f} ... '
            'MAE and skill score = {5:f}, {6:f} ... '
            'bias = {7:f} ... correlation = {8:f} ... KGE = {9:f} ... '
            'reliability = {10:f} ... resolution = {11:f}'
        ).format(
            target_field_names[j],
            numpy.nanmean(t[scalar_evaluation.TARGET_STDEV_KEY].values[j, :]),
            numpy.nanmean(
                t[scalar_evaluation.PREDICTION_STDEV_KEY].values[j, :]
            ),
            numpy.nanmean(
                t[scalar_evaluation.MEAN_SQUARED_ERROR_KEY].values[j, :]
            ),
            numpy.nanmean(
                t[scalar_evaluation.MSE_SKILL_SCORE_KEY].values[j, :]
            ),
            numpy.nanmean(
                t[scalar_evaluation.MEAN_ABSOLUTE_ERROR_KEY].values[j, :]
            ),
            numpy.nanmean(
                t[scalar_evaluation.MAE_SKILL_SCORE_KEY].values[j, :]
            ),
            numpy.nanmean(t[scalar_evaluation.BIAS_KEY].values[j, :]),
            numpy.nanmean(t[scalar_evaluation.CORRELATION_KEY].values[j, :]),
            numpy.nanmean(t[scalar_evaluation.KGE_KEY].values[j, :]),
            numpy.nanmean(t[scalar_evaluation.RELIABILITY_KEY].values[j, :]),
            numpy.nanmean(t[scalar_evaluation.RESOLUTION_KEY].values[j, :])
        ))

    print((
        'Mean distance and skill score = {0:f}, {1:f} ... '
        'mean squared distance and skill score = {2:f}, {3:f}'
    ).format(
        numpy.nanmean(t[scalar_evaluation.MEAN_DISTANCE_KEY].values),
        numpy.nanmean(t[scalar_evaluation.MEAN_DIST_SKILL_SCORE_KEY].values),
        numpy.nanmean(t[scalar_evaluation.MEAN_SQUARED_DISTANCE_KEY].values),
        numpy.nanmean(t[scalar_evaluation.MEAN_SQ_DIST_SKILL_SCORE_KEY].values)
    ))

    print(SEPARATOR_STRING)

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    scalar_evaluation.write_file(
        result_table_xarray=result_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_pattern=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_PATTERN_ARG_NAME
        ),
        num_bootstrap_reps=getattr(
            INPUT_ARG_OBJECT, NUM_BOOTSTRAP_REPS_ARG_NAME
        ),
        num_xy_offset_bins=getattr(
            INPUT_ARG_OBJECT, NUM_XY_OFFSET_BINS_ARG_NAME
        ),
        xy_offset_limits_metres=numpy.array(
            getattr(INPUT_ARG_OBJECT, XY_OFFSET_LIMITS_ARG_NAME),
            dtype=float
        ),
        xy_offset_limits_percentile=numpy.array(
            getattr(INPUT_ARG_OBJECT, XY_OFFSET_LIMITS_PRCTILE_ARG_NAME),
            dtype=float
        ),
        num_offset_distance_bins=getattr(
            INPUT_ARG_OBJECT, NUM_OFFSET_DIST_BINS_ARG_NAME
        ),
        offset_distance_limits_metres=numpy.array(
            getattr(INPUT_ARG_OBJECT, OFFSET_DIST_LIMITS_ARG_NAME),
            dtype=float
        ),
        offset_distance_limits_percentile=numpy.array(
            getattr(INPUT_ARG_OBJECT, OFFSET_DIST_LIMITS_PRCTILE_ARG_NAME),
            dtype=float
        ),
        num_offset_direction_bins=getattr(
            INPUT_ARG_OBJECT, NUM_OFFSET_DIR_BINS_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
