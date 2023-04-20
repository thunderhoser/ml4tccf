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

import file_system_utils
import error_checking
import scalar_evaluation
import scalar_evaluation_plotting as scalar_eval_plotting
import split_predictions_by_intensity as split_predictions

MAX_WIND_KT = 250.
MAX_PRESSURE_MB = 1030.

BIAS_COLOUR_MAP_NAME = 'seismic'
MAIN_COLOUR_MAP_NAME = 'viridis'
MIN_COLOUR_PERCENTILE = 1.
MAX_COLOUR_PERCENTILE = 99.

FIGURE_RESOLUTION_DPI = 300

MAX_WIND_CUTOFFS_ARG_NAME = 'max_wind_cutoffs_kt'
MIN_PRESSURE_CUTOFFS_ARG_NAME = 'min_pressure_cutoffs_mb'
LATITUDE_CUTOFFS_ARG_NAME = 'tc_center_latitude_cutoffs_deg_n'
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
LATITUDE_CUTOFFS_HELP_STRING = (
    '[use ONLY if you have split predictions into 2-D bins, accounting for '
    'both intensity/latitude] Category cutoffs for actual TC-center latitude '
    '(deg north).  Please leave -inf and +inf out of this list, as they will '
    'be added automatically.'
)
MAX_WIND_BASED_FILES_HELP_STRING = (
    'List of paths to max-wind-specific evaluation files (each will be read by '
    '`scalar_evaluation.read_file` or `gridded_evaluation.read_file`).  If '
    '{0:s} is empty, this list should have length N + 1, where N = length of '
    '{1:s}.  If {0:s} is used, this list should have length (N + 1) * (L + 1), '
    'where L = length of {0:s}.  Within this list, latitude bin should vary '
    'fastest; max-wind bin should vary slowest.'
).format(LATITUDE_CUTOFFS_ARG_NAME, MAX_WIND_CUTOFFS_ARG_NAME)

MIN_PRESSURE_BASED_FILES_HELP_STRING = (
    'List of paths to min-pressure-specific evaluation files (each will be '
    'read by `scalar_evaluation.read_file` or `gridded_evaluation.read_file`).'
    '  If {0:s} is empty, this list should have length N + 1, where N = length '
    'of {1:s}.  If {0:s} is used, this list should have length '
    '(N + 1) * (L + 1), where L = length of {0:s}.  Within this list, latitude '
    'bin should vary fastest; min-pressure bin should vary slowest.'
).format(LATITUDE_CUTOFFS_ARG_NAME, MIN_PRESSURE_CUTOFFS_ARG_NAME)

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
    '--' + LATITUDE_CUTOFFS_ARG_NAME, type=float, nargs='+', required=False,
    default=[], help=LATITUDE_CUTOFFS_HELP_STRING
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
         tc_center_latitude_cutoffs_deg_n, max_wind_based_eval_file_names,
         min_pressure_based_eval_file_names, confidence_level, output_dir_name):
    """Plots evaluation metrics as a function of TC intensity.

    This is effectively the main method.

    :param max_wind_cutoffs_kt: See documentation at top of file.
    :param min_pressure_cutoffs_mb: Same.
    :param tc_center_latitude_cutoffs_deg_n: Same.
    :param max_wind_based_eval_file_names: Same.
    :param min_pressure_based_eval_file_names: Same.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    error_checking.assert_is_geq(confidence_level, 0.8)
    error_checking.assert_is_less_than(confidence_level, 1.)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    (
        max_wind_cutoffs_kt,
        min_pressure_cutoffs_mb,
        tc_center_latitude_cutoffs_deg_n
    ) = split_predictions.check_input_args(
        max_wind_cutoffs_kt=max_wind_cutoffs_kt,
        min_pressure_cutoffs_mb=min_pressure_cutoffs_mb,
        tc_center_latitude_cutoffs_deg_n=tc_center_latitude_cutoffs_deg_n
    )

    split_into_2d_bins = len(tc_center_latitude_cutoffs_deg_n) > 2
    num_latitude_bins = len(tc_center_latitude_cutoffs_deg_n) - 1
    num_wind_bins = len(max_wind_cutoffs_kt) - 1
    num_pressure_bins = len(min_pressure_cutoffs_mb) - 1

    if num_wind_bins == 1:
        num_wind_bins = 0
    if num_pressure_bins == 1:
        num_pressure_bins = 0

    error_checking.assert_is_greater(num_wind_bins + num_pressure_bins, 0)

    if num_wind_bins > 0:
        expected_dim = numpy.array(
            [num_latitude_bins * num_wind_bins], dtype=int
        )
        error_checking.assert_is_numpy_array(
            numpy.array(max_wind_based_eval_file_names),
            exact_dimensions=expected_dim
        )

    if num_pressure_bins > 0:
        expected_dim = numpy.array(
            [num_latitude_bins * num_pressure_bins], dtype=int
        )
        error_checking.assert_is_numpy_array(
            numpy.array(min_pressure_based_eval_file_names),
            exact_dimensions=expected_dim
        )

    # Do actual stuff.
    eval_table_by_wind = (
        [xarray.Dataset()] * len(max_wind_based_eval_file_names)
    )
    num_bootstrap_reps = -1

    for i in range(len(eval_table_by_wind)):
        print('Reading data from: "{0:s}"...'.format(
            max_wind_based_eval_file_names[i]
        ))

        try:
            eval_table_by_wind[i] = scalar_evaluation.read_file(
                max_wind_based_eval_file_names[i]
            )
            assert (
                scalar_evaluation.TARGET_FIELD_DIM
                in eval_table_by_wind[i].coords
            )
        except:
            raise ValueError(
                'This script does not yet work for gridded predictions.'
            )

        etbw = eval_table_by_wind

        num_bootstrap_reps = len(
            etbw[0].coords[scalar_evaluation.BOOTSTRAP_REP_DIM].values
        )
        this_num_bootstrap_reps = len(
            etbw[i].coords[scalar_evaluation.BOOTSTRAP_REP_DIM].values
        )
        assert num_bootstrap_reps == this_num_bootstrap_reps

    eval_table_by_wind_listlist = [[None] * num_latitude_bins] * num_wind_bins
    k = -1

    for i in range(num_wind_bins):
        for j in range(num_latitude_bins):
            k += 1
            eval_table_by_wind_listlist[i][j] = eval_table_by_wind[k]

    # del eval_table_by_wind

    for i in range(num_wind_bins):
        for j in range(num_latitude_bins):
            print(eval_table_by_wind_listlist[i][j][scalar_evaluation.MEAN_DISTANCE_KEY])

    latitude_description_strings = [
        '[{0:.1f}, {1:.1f})'.format(l1, l2)
        for l1, l2 in zip(
            tc_center_latitude_cutoffs_deg_n[:-1],
            tc_center_latitude_cutoffs_deg_n[1:]
        )
    ]
    latitude_description_strings[0] = '< {0:.1f}'.format(
        tc_center_latitude_cutoffs_deg_n[1]
    )
    latitude_description_strings[-1] = '>= {0:.1f}'.format(
        tc_center_latitude_cutoffs_deg_n[-2]
    )

    if num_wind_bins > 0:
        wind_description_strings = [
            '[{0:.1f}, {1:.1f})'.format(v1, v2)
            for v1, v2 in zip(max_wind_cutoffs_kt[:-1], max_wind_cutoffs_kt[1:])
        ]
        wind_description_strings[0] = '< {0:.1f}'.format(
            max_wind_cutoffs_kt[1]
        )
        wind_description_strings[-1] = '>= {0:.1f}'.format(
            max_wind_cutoffs_kt[-2]
        )

        etbwll = eval_table_by_wind_listlist

        for metric_name in scalar_eval_plotting.BASIC_METRIC_NAMES:
            for target_field_name in scalar_eval_plotting.BASIC_TARGET_FIELD_NAMES:
                metric_matrix = numpy.full(
                    (num_latitude_bins, num_wind_bins, num_bootstrap_reps),
                    numpy.nan
                )

                for j in range(num_latitude_bins):
                    for i in range(num_wind_bins):
                        k = numpy.where(
                            etbwll[i][j].coords[
                                scalar_evaluation.TARGET_FIELD_DIM
                            ].values
                            == target_field_name
                        )[0][0]

                        metric_matrix[j, i, :] = (
                            etbwll[i][j][metric_name].values[k, :]
                        )

                        # print(j)
                        # print(i)
                        # print(etbwll[i][j][metric_name].values[k, :])
                        # print('\n\n\n\n\n\n\n\n\n')

                if split_into_2d_bins:
                    figure_object = (
                        scalar_eval_plotting.plot_metric_by_2categories(
                            metric_matrix=numpy.nanmean(metric_matrix, axis=-1),
                            metric_name=metric_name,
                            target_field_name=target_field_name,
                            y_category_description_strings=
                            latitude_description_strings,
                            y_label_string=r'TC-center latitude ($^{\circ}$N)',
                            x_category_description_strings=
                            wind_description_strings,
                            x_label_string='Max sustained wind (kt)',
                            colour_map_name=(
                                BIAS_COLOUR_MAP_NAME
                                if metric_name == scalar_evaluation.BIAS_KEY
                                else MAIN_COLOUR_MAP_NAME
                            ),
                            min_colour_percentile=MIN_COLOUR_PERCENTILE,
                            max_colour_percentile=MAX_COLOUR_PERCENTILE
                        )[0]
                    )
                else:
                    figure_object = (
                        scalar_eval_plotting.plot_metric_by_category(
                            metric_matrix=metric_matrix,
                            metric_name=metric_name,
                            target_field_name=target_field_name,
                            category_description_strings=
                            wind_description_strings,
                            x_label_string='Max sustained wind (kt)',
                            confidence_level=confidence_level
                        )[0]
                    )

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
                (num_latitude_bins, num_wind_bins, num_bootstrap_reps),
                numpy.nan
            )

            for j in range(num_latitude_bins):
                for i in range(num_wind_bins):
                    metric_matrix[j, i, :] = etbwll[i][j][metric_name].values[:]

            if split_into_2d_bins:
                figure_object = (
                    scalar_eval_plotting.plot_metric_by_2categories(
                        metric_matrix=numpy.nanmean(metric_matrix, axis=-1),
                        metric_name=metric_name,
                        target_field_name=
                        scalar_evaluation.OFFSET_DISTANCE_NAME,
                        y_category_description_strings=
                        latitude_description_strings,
                        y_label_string=r'TC-center latitude ($^{\circ}$N)',
                        x_category_description_strings=
                        wind_description_strings,
                        x_label_string='Max sustained wind (kt)',
                        colour_map_name=(
                            BIAS_COLOUR_MAP_NAME
                            if metric_name == scalar_evaluation.BIAS_KEY
                            else MAIN_COLOUR_MAP_NAME
                        ),
                        min_colour_percentile=MIN_COLOUR_PERCENTILE,
                        max_colour_percentile=MAX_COLOUR_PERCENTILE
                    )[0]
                )
            else:
                figure_object = scalar_eval_plotting.plot_metric_by_category(
                    metric_matrix=metric_matrix,
                    metric_name=metric_name,
                    target_field_name=scalar_evaluation.OFFSET_DISTANCE_NAME,
                    category_description_strings=wind_description_strings,
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

    if num_pressure_bins < 1:
        return

    eval_table_by_pressure = (
        [xarray.Dataset()] * len(min_pressure_based_eval_file_names)
    )
    num_bootstrap_reps = -1

    for i in range(num_pressure_bins):
        print('Reading data from: "{0:s}"...'.format(
            min_pressure_based_eval_file_names[i]
        ))
        eval_table_by_pressure[i] = scalar_evaluation.read_file(
            min_pressure_based_eval_file_names[i]
        )

        etbp = eval_table_by_pressure

        num_bootstrap_reps = len(
            etbp[0].coords[scalar_evaluation.BOOTSTRAP_REP_DIM].values
        )
        this_num_bootstrap_reps = len(
            etbp[i].coords[scalar_evaluation.BOOTSTRAP_REP_DIM].values
        )
        assert num_bootstrap_reps == this_num_bootstrap_reps

    eval_table_by_pressure_listlist = (
        [[None] * num_latitude_bins] * num_pressure_bins
    )
    k = -1

    for i in range(num_pressure_bins):
        for j in range(num_latitude_bins):
            k += 1
            eval_table_by_pressure_listlist[i][j] = eval_table_by_pressure[k]

    del eval_table_by_pressure

    pressure_description_strings = [
        '[{0:.1f}, {1:.1f})'.format(v1, v2)
        for v1, v2 in zip(
            min_pressure_cutoffs_mb[:-1], min_pressure_cutoffs_mb[1:]
        )
    ]
    pressure_description_strings[0] = '< {0:.1f}'.format(
        min_pressure_cutoffs_mb[1]
    )
    pressure_description_strings[-1] = '>= {0:.1f}'.format(
        min_pressure_cutoffs_mb[-2]
    )

    etbpll = eval_table_by_pressure_listlist

    for metric_name in scalar_eval_plotting.BASIC_METRIC_NAMES:
        for target_field_name in scalar_eval_plotting.BASIC_TARGET_FIELD_NAMES:
            metric_matrix = numpy.full(
                (num_latitude_bins, num_pressure_bins, num_bootstrap_reps),
                numpy.nan
            )

            for j in range(num_latitude_bins):
                for i in range(num_pressure_bins):
                    k = numpy.where(
                        etbpll[i][j].coords[
                            scalar_evaluation.TARGET_FIELD_DIM
                        ].values
                        == target_field_name
                    )[0][0]

                    metric_matrix[j, i, :] = (
                        etbpll[i][j][metric_name].values[k, :]
                    )

            if split_into_2d_bins:
                figure_object = (
                    scalar_eval_plotting.plot_metric_by_2categories(
                        metric_matrix=numpy.nanmean(metric_matrix, axis=-1),
                        metric_name=metric_name,
                        target_field_name=target_field_name,
                        y_category_description_strings=
                        latitude_description_strings,
                        y_label_string=r'TC-center latitude ($^{\circ}$N)',
                        x_category_description_strings=
                        pressure_description_strings,
                        x_label_string='Minimum central pressure (mb)',
                        colour_map_name=(
                            BIAS_COLOUR_MAP_NAME
                            if metric_name == scalar_evaluation.BIAS_KEY
                            else MAIN_COLOUR_MAP_NAME
                        ),
                        min_colour_percentile=MIN_COLOUR_PERCENTILE,
                        max_colour_percentile=MAX_COLOUR_PERCENTILE
                    )[0]
                )
            else:
                figure_object = scalar_eval_plotting.plot_metric_by_category(
                    metric_matrix=metric_matrix,
                    metric_name=metric_name,
                    target_field_name=target_field_name,
                    category_description_strings=pressure_description_strings,
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
            (num_latitude_bins, num_pressure_bins, num_bootstrap_reps),
            numpy.nan
        )

        for j in range(num_latitude_bins):
            for i in range(num_pressure_bins):
                metric_matrix[j, i, :] = etbpll[i][j][metric_name].values[:]

        if split_into_2d_bins:
            figure_object = (
                scalar_eval_plotting.plot_metric_by_2categories(
                    metric_matrix=numpy.nanmean(metric_matrix, axis=-1),
                    metric_name=metric_name,
                    target_field_name=
                    scalar_evaluation.OFFSET_DISTANCE_NAME,
                    y_category_description_strings=
                    latitude_description_strings,
                    y_label_string=r'TC-center latitude ($^{\circ}$N)',
                    x_category_description_strings=
                    pressure_description_strings,
                    x_label_string='Minimum central pressure (mb)',
                    colour_map_name=(
                        BIAS_COLOUR_MAP_NAME
                        if metric_name == scalar_evaluation.BIAS_KEY
                        else MAIN_COLOUR_MAP_NAME
                    ),
                    min_colour_percentile=MIN_COLOUR_PERCENTILE,
                    max_colour_percentile=MAX_COLOUR_PERCENTILE
                )[0]
            )
        else:
            figure_object = scalar_eval_plotting.plot_metric_by_category(
                metric_matrix=metric_matrix,
                metric_name=metric_name,
                target_field_name=scalar_evaluation.OFFSET_DISTANCE_NAME,
                category_description_strings=pressure_description_strings,
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
        tc_center_latitude_cutoffs_deg_n=numpy.array(
            getattr(INPUT_ARG_OBJECT, LATITUDE_CUTOFFS_ARG_NAME),
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
