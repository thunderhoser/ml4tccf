"""Computes spread-skill relationship for trained model."""

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
import spread_skill_utils as ss_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILE_PATTERN_ARG_NAME = 'input_prediction_file_pattern'
NUM_XY_OFFSET_BINS_ARG_NAME = 'num_xy_offset_bins'
XY_OFFSET_LIMITS_ARG_NAME = 'xy_offset_limits_metres'
XY_OFFSET_LIMITS_PRCTILE_ARG_NAME = 'xy_offset_limits_percentile'
NUM_OFFSET_DIST_BINS_ARG_NAME = 'num_offset_distance_bins'
OFFSET_DIST_LIMITS_ARG_NAME = 'offset_distance_limits_metres'
OFFSET_DIST_LIMITS_PRCTILE_ARG_NAME = 'offset_distance_limits_percentile'
NUM_OFFSET_DIR_BINS_ARG_NAME = 'num_offset_direction_bins'
OFFSET_DIR_LIMITS_ARG_NAME = 'offset_direction_limits_deg'
OFFSET_DIR_LIMITS_PRCTILE_ARG_NAME = 'offset_direction_limits_percentile'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_PATTERN_HELP_STRING = (
    'Glob pattern for input files.  Each will be read by '
    '`prediction_io.read_file`, and predictions in all these files will be '
    'evaluated together.'
)
NUM_XY_OFFSET_BINS_HELP_STRING = (
    'Number of spread bins for both x- and y-offset.'
)
XY_OFFSET_LIMITS_HELP_STRING = (
    'Min and max bin edges [lower edge of lowest bin, upper edge of highest '
    'bin] for both x- and y-offset.  If you want to choose bin edges by '
    'specifying percentiles over the data, rather than direct values, leave '
    'this argument alone and use {0:s}.'
).format(XY_OFFSET_LIMITS_PRCTILE_ARG_NAME)

XY_OFFSET_LIMITS_PRCTILE_HELP_STRING = (
    'List of two percentiles [p, q].  Determines min and max bin edges for '
    'both x- and y-offset.  Specifically, the lower edge of the lowest bin '
    'will be the [p]th percentile over all spread values in the data, and the '
    'upper edge of the highest bin will be the [q]th percentile.  Percentiles '
    'are taken independently for the two variables, x and y.  If you want to '
    'choose bin edges directly, leave this argument alone and use {0:s}.'
).format(XY_OFFSET_LIMITS_ARG_NAME)

NUM_OFFSET_DIST_BINS_HELP_STRING = (
    'Number of spread bins for total (Euclidean) offset.'
)
OFFSET_DIST_LIMITS_HELP_STRING = (
    'Same as {0:s} but for total (Euclidean) offset.'
).format(XY_OFFSET_LIMITS_ARG_NAME)

OFFSET_DIST_LIMITS_PRCTILE_HELP_STRING = (
    'Same as {0:s} but for total (Euclidean) offset.'
).format(XY_OFFSET_LIMITS_PRCTILE_ARG_NAME)

NUM_OFFSET_DIR_BINS_HELP_STRING = (
    'Number of spread bins for offset direction (angle).'
)
OFFSET_DIR_LIMITS_HELP_STRING = (
    'Same as {0:s} but for offset direction (angle).'
).format(XY_OFFSET_LIMITS_ARG_NAME)

OFFSET_DIR_LIMITS_PRCTILE_HELP_STRING = (
    'Same as {0:s} but for offset direction (angle).'
).format(XY_OFFSET_LIMITS_PRCTILE_ARG_NAME)

OUTPUT_FILE_HELP_STRING = (
    'Path to output (NetCDF) file.  Results will be written here by '
    '`spread_skill_utils.write_results`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_PATTERN_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_PATTERN_HELP_STRING
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
    '--' + NUM_OFFSET_DIR_BINS_ARG_NAME, type=int, required=False, default=20,
    help=NUM_OFFSET_DIR_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OFFSET_DIR_LIMITS_ARG_NAME, type=float, nargs=2, required=False,
    default=[1, -1], help=OFFSET_DIR_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OFFSET_DIR_LIMITS_PRCTILE_ARG_NAME, type=float, nargs=2,
    required=False, default=[0.5, 99.5],
    help=OFFSET_DIR_LIMITS_PRCTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(prediction_file_pattern, num_xy_offset_bins,
         xy_offset_limits_metres, xy_offset_limits_percentile,
         num_offset_distance_bins, offset_distance_limits_metres,
         offset_distance_limits_percentile, num_offset_direction_bins,
         offset_direction_limits_deg, offset_direction_limits_percentile,
         output_file_name):
    """Computes spread-skill relationship for trained model.

    This is effectively the main method.

    :param prediction_file_pattern: See documentation at top of file.
    :param num_xy_offset_bins: Same.
    :param xy_offset_limits_metres: Same.
    :param xy_offset_limits_percentile: Same.
    :param num_offset_distance_bins: Same.
    :param offset_distance_limits_metres: Same.
    :param offset_distance_limits_percentile: Same.
    :param num_offset_direction_bins: Same.
    :param offset_direction_limits_deg: Same.
    :param offset_direction_limits_percentile: Same.
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
        error_checking.assert_is_greater(
            max_offset_distance_metres, min_offset_distance_metres
        )

    min_offset_direction_deg = offset_direction_limits_deg[0]
    max_offset_direction_deg = offset_direction_limits_deg[1]
    min_offset_direction_percentile = offset_direction_limits_percentile[0]
    max_offset_direction_percentile = offset_direction_limits_percentile[1]

    if min_offset_direction_deg >= max_offset_direction_deg:
        min_offset_direction_deg = None
        max_offset_direction_deg = None

        error_checking.assert_is_leq(min_offset_direction_percentile, 10.)
        error_checking.assert_is_geq(max_offset_direction_percentile, 90.)
    else:
        min_offset_direction_percentile = None
        max_offset_direction_percentile = None

        error_checking.assert_is_geq(min_offset_direction_deg, 0.)
        error_checking.assert_is_less_than(max_offset_direction_deg, 360.)
        error_checking.assert_is_greater(
            max_offset_direction_deg, min_offset_direction_deg
        )

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

    result_table_xarray = ss_utils.get_results_all_vars(
        prediction_file_names=prediction_file_names,
        num_xy_offset_bins=num_xy_offset_bins,
        bin_min_xy_offset_metres=min_xy_offset_metres,
        bin_max_xy_offset_metres=max_xy_offset_metres,
        bin_min_xy_offset_percentile=min_xy_offset_percentile,
        bin_max_xy_offset_percentile=max_xy_offset_percentile,
        num_offset_distance_bins=num_offset_distance_bins,
        bin_min_offset_distance_metres=min_offset_distance_metres,
        bin_max_offset_distance_metres=max_offset_distance_metres,
        bin_min_offset_distance_percentile=min_offset_distance_percentile,
        bin_max_offset_distance_percentile=max_offset_distance_percentile,
        num_offset_direction_bins=num_offset_direction_bins,
        bin_min_offset_direction_deg=min_offset_direction_deg,
        bin_max_offset_direction_deg=max_offset_direction_deg,
        bin_min_offset_dir_percentile=min_offset_direction_percentile,
        bin_max_offset_dir_percentile=max_offset_direction_percentile
    )
    print(SEPARATOR_STRING)

    t = result_table_xarray
    target_names = t.coords[ss_utils.TARGET_FIELD_DIM].values.tolist()

    xy_indices = numpy.array([
        target_names.index(ss_utils.X_OFFSET_NAME),
        target_names.index(ss_utils.Y_OFFSET_NAME)
    ], dtype=int)

    for j in xy_indices:
        print('Variable = {0:s} ... SSREL = {1:f} ... SSRAT = {2:f}'.format(
            target_names[j],
            t[ss_utils.XY_SSREL_KEY].values[j],
            t[ss_utils.XY_SSRAT_KEY].values[j]
        ))

    direction_indices = numpy.array([
        target_names.index(ss_utils.OFFSET_DIRECTION_NAME)
    ], dtype=int)

    for j in direction_indices:
        print('Variable = {0:s} ... SSREL = {1:f} ... SSRAT = {2:f}'.format(
            target_names[j],
            t[ss_utils.OFFSET_DIR_SSREL_KEY].values[j],
            t[ss_utils.OFFSET_DIR_SSRAT_KEY].values[j]
        ))

    distance_indices = numpy.array([
        target_names.index(ss_utils.OFFSET_DISTANCE_NAME)
    ], dtype=int)

    for j in distance_indices:
        print('Variable = {0:s} ... SSREL = {1:f} ... SSRAT = {2:f}'.format(
            target_names[j],
            t[ss_utils.OFFSET_DIST_SSREL_KEY].values[j],
            t[ss_utils.OFFSET_DIST_SSRAT_KEY].values[j]
        ))

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    ss_utils.write_results(
        result_table_xarray=result_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_pattern=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_PATTERN_ARG_NAME
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
        offset_direction_limits_deg=numpy.array(
            getattr(INPUT_ARG_OBJECT, OFFSET_DIR_LIMITS_ARG_NAME),
            dtype=float
        ),
        offset_direction_limits_percentile=numpy.array(
            getattr(INPUT_ARG_OBJECT, OFFSET_DIR_LIMITS_PRCTILE_ARG_NAME),
            dtype=float
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
