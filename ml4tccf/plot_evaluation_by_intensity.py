"""Plots evaluation metrics as a function of TC intensity."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import number_rounding
import file_system_utils
import error_checking
import scalar_evaluation
import scalar_evaluation_plotting as scalar_eval_plotting

MAX_WIND_KT = 250.
MAX_PRESSURE_MB = 1030.

FIGURE_RESOLUTION_DPI = 300

MAX_WIND_CUTOFFS_ARG_NAME = 'max_wind_cutoffs_kt'
MIN_PRESSURE_CUTOFFS_ARG_NAME = 'min_pressure_cutoffs_mb'
MAX_WIND_BASED_FILES_ARG_NAME = 'input_max_wind_based_eval_file_names'
MIN_PRESSURE_BASED_FILES_ARG_NAME = 'input_min_pressure_based_eval_file_names'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MAX_WIND_CUTOFFS_HELP_STRING = (
    'Category cutoffs for max sustained wind (one measure of intensity).  '
    'Please leave 0 and infinity out of this list, as they will be added '
    'automatically.'
)
MIN_PRESSURE_CUTOFFS_HELP_STRING = (
    'Category cutoffs for min surface pressure (one measure of intensity).  '
    'Please leave 0 and infinity out of this list, as they will be added '
    'automatically.'
)
MAX_WIND_BASED_FILES_HELP_STRING = (
    'List of paths to max-wind-specific evaluation files (each will be read by '
    '`scalar_evaluation.read_file` or `gridded_evaluation.read_file`).  This '
    'list should have length N + 1, where N = length of {0:s}.'
).format(MAX_WIND_CUTOFFS_ARG_NAME)

MIN_PRESSURE_BASED_FILES_HELP_STRING = (
    'List of paths to min-pressure-specific evaluation files (each will be '
    'read by `scalar_evaluation.read_file` or `gridded_evaluation.read_file`).'
    '  This list should have length N + 1, where N = length of {0:s}.'
).format(MIN_PRESSURE_CUTOFFS_ARG_NAME)

CONFIDENCE_LEVEL_HELP_STRING = (
    'Confidence level for uncertainty intervals.  Must be in range 0...1.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_WIND_CUTOFFS_ARG_NAME, type=float, nargs='+', required=False,
    default=[50, 85], help=MAX_WIND_CUTOFFS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_PRESSURE_CUTOFFS_ARG_NAME, type=float, nargs='+', required=False,
    default=[], help=MIN_PRESSURE_CUTOFFS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_WIND_BASED_FILES_ARG_NAME, type=str, nargs='+', required=False,
    default=[], help=MAX_WIND_BASED_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_PRESSURE_BASED_FILES_ARG_NAME, type=str, nargs='+',
    required=False, default=[], help=MIN_PRESSURE_BASED_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(max_wind_cutoffs_kt, min_pressure_cutoffs_mb,
         max_wind_based_eval_file_names, min_pressure_based_eval_file_names,
         confidence_level, output_dir_name):
    """Plots evaluation metrics as a function of TC intensity.

    This is effectively the main method.

    :param max_wind_cutoffs_kt: See documentation at top of file.
    :param min_pressure_cutoffs_mb: Same.
    :param max_wind_based_eval_file_names: Same.
    :param min_pressure_based_eval_file_names: Same.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    """

    error_checking.assert_is_geq(confidence_level, 0.8)
    error_checking.assert_is_less_than(confidence_level, 1.)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # TODO(thunderhoser): Modularize this!
    if len(max_wind_cutoffs_kt) > 0:
        max_wind_cutoffs_kt = number_rounding.round_to_nearest(
            max_wind_cutoffs_kt, 0.1
        )

        error_checking.assert_is_greater_numpy_array(max_wind_cutoffs_kt, 0.)
        error_checking.assert_is_leq_numpy_array(
            max_wind_cutoffs_kt, MAX_WIND_KT
        )
        error_checking.assert_is_greater_numpy_array(
            numpy.diff(max_wind_cutoffs_kt), 0.
        )

        max_wind_cutoffs_kt = numpy.concatenate((
            numpy.array([0.]),
            max_wind_cutoffs_kt,
            numpy.array([numpy.inf])
        ))

        expected_dim = numpy.array([len(max_wind_cutoffs_kt) - 1], dtype=int)
        error_checking.assert_is_numpy_array(
            numpy.array(max_wind_based_eval_file_names),
            exact_dimensions=expected_dim
        )

    if len(min_pressure_cutoffs_mb) > 0:
        min_pressure_cutoffs_mb = number_rounding.round_to_nearest(
            min_pressure_cutoffs_mb, 0.1
        )

        error_checking.assert_is_greater_numpy_array(
            min_pressure_cutoffs_mb, 0.
        )
        error_checking.assert_is_leq_numpy_array(
            min_pressure_cutoffs_mb, MAX_PRESSURE_MB
        )
        error_checking.assert_is_greater_numpy_array(
            numpy.diff(min_pressure_cutoffs_mb), 0.
        )

        min_pressure_cutoffs_mb = numpy.concatenate((
            numpy.array([0.]),
            min_pressure_cutoffs_mb,
            numpy.array([numpy.inf])
        ))

        expected_dim = numpy.array(
            [len(min_pressure_cutoffs_mb) - 1], dtype=int
        )
        error_checking.assert_is_numpy_array(
            numpy.array(min_pressure_based_eval_file_names),
            exact_dimensions=expected_dim
        )

    error_checking.assert_is_greater(
        len(max_wind_cutoffs_kt) + len(min_pressure_cutoffs_mb), 0
    )

    num_wind_categories = len(max_wind_cutoffs_kt) - 1
    eval_table_by_wind_category = [xarray.Dataset()] * num_wind_categories
    num_bootstrap_reps = -1

    for i in range(num_wind_categories):
        print('Reading data from: "{0:s}"...'.format(
            max_wind_based_eval_file_names[i]
        ))

        try:
            eval_table_by_wind_category[i] = scalar_evaluation.read_file(
                max_wind_based_eval_file_names[i]
            )
            assert (
                scalar_evaluation.TARGET_FIELD_DIM
                in eval_table_by_wind_category[i].coords
            )
        except:
            raise ValueError(
                'This script does not yet work for gridded predictions.'
            )

        etbwc = eval_table_by_wind_category

        num_bootstrap_reps = len(
            etbwc[0].coords[scalar_evaluation.BOOTSTRAP_REP_DIM].values
        )
        this_num_bootstrap_reps = len(
            etbwc[i].coords[scalar_evaluation.BOOTSTRAP_REP_DIM].values
        )
        assert num_bootstrap_reps == this_num_bootstrap_reps

    if num_wind_categories > 0:
        category_description_strings = [
            '[{0:.1f}, {1:.1f})'.format(v1, v2)
            for v1, v2 in zip(max_wind_cutoffs_kt[:-1], max_wind_cutoffs_kt[1:])
        ]
        category_description_strings[0] = '< {0:.1f}'.format(
            max_wind_cutoffs_kt[1]
        )
        category_description_strings[-1] = '>= {0:.1f}'.format(
            max_wind_cutoffs_kt[-2]
        )

        etbwc = eval_table_by_wind_category

        for metric_name in scalar_eval_plotting.BASIC_METRIC_NAMES:
            for target_field_name in scalar_eval_plotting.BASIC_TARGET_FIELD_NAMES:
                metric_matrix = numpy.full(
                    (num_wind_categories, num_bootstrap_reps), numpy.nan
                )

                for i in range(num_wind_categories):
                    j = numpy.where(
                        etbwc[i].coords[
                            scalar_evaluation.TARGET_FIELD_DIM
                        ].values
                        == target_field_name
                    )[0][0]

                    metric_matrix[i, :] = etbwc[i][metric_name].values[j, :]

                figure_object = scalar_eval_plotting.plot_metric_by_category(
                    metric_matrix=metric_matrix,
                    metric_name=metric_name,
                    target_field_name=target_field_name,
                    category_description_strings=category_description_strings,
                    x_label_string='Max sustained wind (kt)',
                    confidence_level=confidence_level
                )[0]

                output_file_name = '{0:s}/{1:s}_{2:s}_by-max-wind.jpg'.format(
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
            metric_matrix = numpy.full(
                (num_wind_categories, num_bootstrap_reps), numpy.nan
            )

            for i in range(num_wind_categories):
                metric_matrix[i, :] = etbwc[i][metric_name].values[:]

            figure_object = scalar_eval_plotting.plot_metric_by_category(
                metric_matrix=metric_matrix,
                metric_name=metric_name,
                target_field_name=scalar_evaluation.OFFSET_DISTANCE_NAME,
                category_description_strings=category_description_strings,
                x_label_string='Max sustained wind (kt)',
                confidence_level=confidence_level
            )[0]

            output_file_name = '{0:s}/{1:s}_by-max-wind.jpg'.format(
                output_dir_name,
                metric_name.replace('_', '-')
            )

            print('Saving figure to: "{0:s}"...'.format(output_file_name))
            figure_object.savefig(
                output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                bbox_inches='tight'
            )
            pyplot.close(figure_object)

    num_pressure_categories = len(min_pressure_cutoffs_mb) - 1
    eval_table_by_pressure_category = (
        [xarray.Dataset()] * num_pressure_categories
    )
    num_bootstrap_reps = -1

    for i in range(num_pressure_categories):
        print('Reading data from: "{0:s}"...'.format(
            min_pressure_based_eval_file_names[i]
        ))
        eval_table_by_pressure_category[i] = scalar_evaluation.read_file(
            min_pressure_based_eval_file_names[i]
        )

        etbpc = eval_table_by_pressure_category

        num_bootstrap_reps = len(
            etbpc[0].coords[scalar_evaluation.BOOTSTRAP_REP_DIM].values
        )
        this_num_bootstrap_reps = len(
            etbpc[i].coords[scalar_evaluation.BOOTSTRAP_REP_DIM].values
        )
        assert num_bootstrap_reps == this_num_bootstrap_reps

    if num_pressure_categories < 1:
        return

    category_description_strings = [
        '[{0:.1f}, {1:.1f})'.format(v1, v2)
        for v1, v2 in zip(
            min_pressure_cutoffs_mb[:-1], min_pressure_cutoffs_mb[1:]
        )
    ]
    category_description_strings[0] = '< {0:.1f}'.format(
        min_pressure_cutoffs_mb[1]
    )
    category_description_strings[-1] = '>= {0:.1f}'.format(
        min_pressure_cutoffs_mb[-2]
    )

    etbpc = eval_table_by_pressure_category

    for metric_name in scalar_eval_plotting.BASIC_METRIC_NAMES:
        for target_field_name in scalar_eval_plotting.BASIC_TARGET_FIELD_NAMES:
            metric_matrix = numpy.full(
                (num_pressure_categories, num_bootstrap_reps), numpy.nan
            )

            for i in range(num_pressure_categories):
                j = numpy.where(
                    etbpc[i].coords[scalar_evaluation.TARGET_FIELD_DIM].values
                    == target_field_name
                )[0][0]

                metric_matrix[i, :] = etbpc[i][metric_name].values[j, :]

            figure_object = scalar_eval_plotting.plot_metric_by_category(
                metric_matrix=metric_matrix,
                metric_name=metric_name,
                target_field_name=target_field_name,
                category_description_strings=category_description_strings,
                x_label_string='Minimum central pressure (mb)',
                confidence_level=confidence_level
            )[0]

            output_file_name = '{0:s}/{1:s}_{2:s}_by-min-pressure.jpg'.format(
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
        metric_matrix = numpy.full(
            (num_wind_categories, num_bootstrap_reps), numpy.nan
        )

        for i in range(num_wind_categories):
            metric_matrix[i, :] = etbpc[i][metric_name].values[:]

        figure_object = scalar_eval_plotting.plot_metric_by_category(
            metric_matrix=metric_matrix,
            metric_name=metric_name,
            target_field_name=scalar_evaluation.OFFSET_DISTANCE_NAME,
            category_description_strings=category_description_strings,
            x_label_string='Minimum central pressure (mb)',
            confidence_level=confidence_level
        )[0]

        output_file_name = '{0:s}/{1:s}_by-min-pressure.jpg'.format(
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
        max_wind_cutoffs_kt=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_WIND_CUTOFFS_ARG_NAME),
            dtype=float
        ),
        min_pressure_cutoffs_mb=numpy.array(
            getattr(INPUT_ARG_OBJECT, MIN_PRESSURE_CUTOFFS_ARG_NAME),
            dtype=float
        ),
        max_wind_based_eval_file_names=getattr(
            INPUT_ARG_OBJECT, MAX_WIND_BASED_FILES_ARG_NAME
        ),
        min_pressure_based_eval_file_names=getattr(
            INPUT_ARG_OBJECT, MIN_PRESSURE_BASED_FILES_ARG_NAME
        ),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
