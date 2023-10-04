"""Plots evaluation metrics as a function of x-y (nadir-relative) coords."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import scalar_evaluation
import scalar_evaluation_plotting as scalar_eval_plotting
import split_predictions_by_xy_coords as split_predictions

METRES_TO_KM = 0.001

BIAS_COLOUR_MAP_NAME = 'seismic'
MAIN_COLOUR_MAP_NAME = 'viridis'
MIN_COLOUR_PERCENTILE = 1.
MAX_COLOUR_PERCENTILE = 99.

FIGURE_RESOLUTION_DPI = 300

X_COORD_CUTOFFS_ARG_NAME = 'x_coord_cutoffs_metres'
Y_COORD_CUTOFFS_ARG_NAME = 'y_coord_cutoffs_metres'
INPUT_FILES_ARG_NAME = 'input_evaluation_file_names'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

X_COORD_CUTOFFS_HELP_STRING = (
    'Category cutoffs for nadir-relative x-coordinate.  Please leave -inf and '
    '+inf out of this list, as they will be added automatically.'
)
Y_COORD_CUTOFFS_HELP_STRING = (
    'Category cutoffs for nadir-relative y-coordinate.  Please leave -inf and '
    '+inf out of this list, as they will be added automatically.'
)
INPUT_FILES_HELP_STRING = (
    'List of paths to coord-specific evaluation files (each will be read by '
    '`scalar_evaluation.read_file` or `gridded_evaluation.read_file`).  This '
    'should have length (M + 1) * (N + 1), where M = length of {0:s} and '
    'N = length of {1:s}.  Within this list, the x-coord should vary fastest; '
    'y-coord bin should vary slowest.'
).format(Y_COORD_CUTOFFS_ARG_NAME, X_COORD_CUTOFFS_ARG_NAME)

CONFIDENCE_LEVEL_HELP_STRING = (
    'Confidence level for uncertainty intervals.  Must be in range 0...1.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + X_COORD_CUTOFFS_ARG_NAME, type=float, nargs='+', required=True,
    help=X_COORD_CUTOFFS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + Y_COORD_CUTOFFS_ARG_NAME, type=float, nargs='+', required=True,
    help=Y_COORD_CUTOFFS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(x_coord_cutoffs_metres, y_coord_cutoffs_metres, evaluation_file_names,
         confidence_level, output_dir_name):
    """Plots evaluation metrics as a function of x-y (nadir-relative) coords.

    This is effectively the main method.

    :param x_coord_cutoffs_metres: See documentation at top of file.
    :param y_coord_cutoffs_metres: Same.
    :param evaluation_file_names: Same.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    error_checking.assert_is_geq(confidence_level, 0.8)
    error_checking.assert_is_less_than(confidence_level, 1.)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    x_coord_cutoffs_metres, y_coord_cutoffs_metres = (
        split_predictions.check_input_args(
            x_coord_cutoffs_metres=x_coord_cutoffs_metres,
            y_coord_cutoffs_metres=y_coord_cutoffs_metres
        )
    )

    num_x_coord_bins = len(x_coord_cutoffs_metres) - 1
    num_y_coord_bins = len(y_coord_cutoffs_metres) - 1

    expected_dim = numpy.array(
        [(num_x_coord_bins + 1) * (num_y_coord_bins + 1)], dtype=int
    )
    error_checking.assert_is_numpy_array(
        numpy.array(evaluation_file_names), exact_dimensions=expected_dim
    )

    # Do actual stuff.
    eval_table_by_xy_bin = (
        [None] * len(evaluation_file_names)
    )
    num_bootstrap_reps = -1

    for i in range(len(eval_table_by_xy_bin)):
        print('Reading data from: "{0:s}"...'.format(evaluation_file_names[i]))
        if not os.path.isfile(evaluation_file_names[i]):
            continue

        try:
            eval_table_by_xy_bin[i] = scalar_evaluation.read_file(
                evaluation_file_names[i]
            )
            assert (
                scalar_evaluation.TARGET_FIELD_DIM
                in eval_table_by_xy_bin[i].coords
            )
        except:
            raise ValueError(
                'This script does not yet work for gridded predictions.'
            )

        etbxy = eval_table_by_xy_bin

        if num_bootstrap_reps == -1:
            num_bootstrap_reps = len(
                etbxy[i].coords[scalar_evaluation.BOOTSTRAP_REP_DIM].values
            )

        this_num_bootstrap_reps = len(
            etbxy[i].coords[scalar_evaluation.BOOTSTRAP_REP_DIM].values
        )
        assert num_bootstrap_reps == this_num_bootstrap_reps

    eval_table_by_xy_listlist = [
        [None] * (num_x_coord_bins + 1) for _ in range(num_y_coord_bins + 1)
    ]
    k = -1

    for i in range(num_y_coord_bins + 1):
        for j in range(num_x_coord_bins + 1):
            k += 1

            print('{0:d}th x-coord bin; {1:d}th y-coord bin; {2:s}'.format(
                j + 1, i + 1, evaluation_file_names[k]
            ))
            eval_table_by_xy_listlist[i][j] = eval_table_by_xy_bin[k]

    del eval_table_by_xy_bin

    x_description_strings = [
        '[{0:.0f}, {1:.0f})'.format(x1, x2)
        for x1, x2 in zip(
            METRES_TO_KM * x_coord_cutoffs_metres[:-1],
            METRES_TO_KM * x_coord_cutoffs_metres[1:]
        )
    ]
    x_description_strings[0] = '< {0:.0f}'.format(
        METRES_TO_KM * x_coord_cutoffs_metres[1]
    )
    x_description_strings[-1] = '>= {0:.0f}'.format(
        METRES_TO_KM * x_coord_cutoffs_metres[-2]
    )
    x_description_strings.append('All')

    y_description_strings = [
        '[{0:.0f}, {1:.0f})'.format(y1, y2)
        for y1, y2 in zip(
            METRES_TO_KM * y_coord_cutoffs_metres[:-1],
            METRES_TO_KM * y_coord_cutoffs_metres[1:]
        )
    ]
    y_description_strings[0] = '< {0:.0f}'.format(
        METRES_TO_KM * y_coord_cutoffs_metres[1]
    )
    y_description_strings[-1] = '>= {0:.0f}'.format(
        METRES_TO_KM * y_coord_cutoffs_metres[-2]
    )
    y_description_strings.append('All')

    etbxy = eval_table_by_xy_listlist

    for metric_name in scalar_eval_plotting.BASIC_METRIC_NAMES:
        for target_field_name in scalar_eval_plotting.BASIC_TARGET_FIELD_NAMES:
            these_dim = (
                num_y_coord_bins + 1, num_x_coord_bins + 1, num_bootstrap_reps
            )
            metric_matrix = numpy.full(these_dim, numpy.nan)

            for j in range(num_x_coord_bins + 1):
                for i in range(num_y_coord_bins + 1):
                    if etbxy[i][j] is None:
                        continue

                    k = numpy.where(
                        etbxy[i][j].coords[
                            scalar_evaluation.TARGET_FIELD_DIM
                        ].values
                        == target_field_name
                    )[0][0]

                    metric_matrix[i, j, :] = (
                        etbxy[i][j][metric_name].values[k, :]
                    )

            label_format_string = (
                '{0:.1f}'
                if numpy.nanmax(numpy.absolute(metric_matrix)) > 1
                else '{0:.2f}'
            )

            figure_object = scalar_eval_plotting.plot_metric_by_2categories(
                metric_matrix=numpy.nanmean(metric_matrix, axis=-1),
                metric_name=metric_name,
                target_field_name=target_field_name,
                y_category_description_strings=y_description_strings,
                y_label_string='Meridional distance from nadir (km)',
                x_category_description_strings=x_description_strings,
                x_label_string='Zonal distance from nadir (km)',
                colour_map_name=(
                    BIAS_COLOUR_MAP_NAME
                    if metric_name == scalar_evaluation.BIAS_KEY
                    else MAIN_COLOUR_MAP_NAME
                ),
                min_colour_percentile=MIN_COLOUR_PERCENTILE,
                max_colour_percentile=MAX_COLOUR_PERCENTILE,
                label_format_string=label_format_string
            )[0]

            output_file_name = '{0:s}/{1:s}_{2:s}_by-xy.jpg'.format(
                output_dir_name,
                target_field_name.replace('_', '-'),
                metric_name.replace('_', '-')
            )

            print('Saving figure to: "{0:s}"...'.format(output_file_name))
            figure_object.savefig(
                output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                bbox_inches='tight'
            )
            pyplot.close(figure_object)

    for metric_name in scalar_eval_plotting.ADVANCED_METRIC_NAMES:
        these_dim = (
            num_y_coord_bins + 1, num_x_coord_bins + 1, num_bootstrap_reps
        )
        metric_matrix = numpy.full(these_dim, numpy.nan)

        for j in range(num_x_coord_bins + 1):
            for i in range(num_y_coord_bins + 1):
                if etbxy[i][j] is None:
                    continue

                metric_matrix[i, j, :] = etbxy[i][j][metric_name].values[:]

        label_format_string = (
            '{0:.1f}' if numpy.nanmax(numpy.absolute(metric_matrix)) > 1
            else '{0:.2f}'
        )

        figure_object = scalar_eval_plotting.plot_metric_by_2categories(
            metric_matrix=numpy.nanmean(metric_matrix, axis=-1),
            metric_name=metric_name,
            target_field_name=scalar_evaluation.OFFSET_DISTANCE_NAME,
            y_category_description_strings=y_description_strings,
            y_label_string='Meridional distance from nadir (km)',
            x_category_description_strings=x_description_strings,
            x_label_string='Zonal distance from nadir (km)',
            colour_map_name=(
                BIAS_COLOUR_MAP_NAME
                if metric_name == scalar_evaluation.BIAS_KEY
                else MAIN_COLOUR_MAP_NAME
            ),
            min_colour_percentile=MIN_COLOUR_PERCENTILE,
            max_colour_percentile=MAX_COLOUR_PERCENTILE,
            label_format_string=label_format_string
        )[0]

        output_file_name = '{0:s}/{1:s}_by-xy.jpg'.format(
            output_dir_name,
            metric_name.replace('_', '-')
        )

        print('Saving figure to: "{0:s}"...'.format(output_file_name))
        figure_object.savefig(
            output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
            bbox_inches='tight'
        )
        pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        x_coord_cutoffs_metres=numpy.array(
            getattr(INPUT_ARG_OBJECT, X_COORD_CUTOFFS_ARG_NAME),
            dtype=float
        ),
        y_coord_cutoffs_metres=numpy.array(
            getattr(INPUT_ARG_OBJECT, Y_COORD_CUTOFFS_ARG_NAME),
            dtype=float
        ),
        evaluation_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
