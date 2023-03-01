"""Plots evaluation metrics as a function of TC intensity."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import matplotlib.colors
import matplotlib.patches
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4tccf.utils import evaluation_sans_uq
from ml4tccf.utils import misc_utils

MAX_WIND_KT = 250.
MAX_PRESSURE_MB = 1030.

METRES_TO_KM = 0.001

BASIC_TARGET_FIELD_NAMES = [
    evaluation_sans_uq.X_OFFSET_NAME,
    evaluation_sans_uq.Y_OFFSET_NAME,
    evaluation_sans_uq.OFFSET_DIRECTION_NAME
]

BASIC_METRIC_NAMES = [
    evaluation_sans_uq.MEAN_SQUARED_ERROR_KEY,
    evaluation_sans_uq.MSE_SKILL_SCORE_KEY,
    evaluation_sans_uq.MEAN_ABSOLUTE_ERROR_KEY,
    evaluation_sans_uq.MAE_SKILL_SCORE_KEY,
    evaluation_sans_uq.BIAS_KEY,
    evaluation_sans_uq.CORRELATION_KEY,
    evaluation_sans_uq.KGE_KEY
]

ADVANCED_METRIC_NAMES = [
    evaluation_sans_uq.MEAN_DISTANCE_KEY,
    evaluation_sans_uq.MEAN_DIST_SKILL_SCORE_KEY,
    evaluation_sans_uq.MEAN_SQUARED_DISTANCE_KEY,
    evaluation_sans_uq.MEAN_SQ_DIST_SKILL_SCORE_KEY
]

TARGET_FIELD_TO_CONV_RATIO = {
    evaluation_sans_uq.X_OFFSET_NAME: METRES_TO_KM,
    evaluation_sans_uq.Y_OFFSET_NAME: METRES_TO_KM,
    evaluation_sans_uq.OFFSET_DIRECTION_NAME: 1.,
    evaluation_sans_uq.OFFSET_DISTANCE_NAME: METRES_TO_KM
}

TARGET_FIELD_TO_FANCY_NAME = {
    evaluation_sans_uq.X_OFFSET_NAME: r' for $x$-position',
    evaluation_sans_uq.Y_OFFSET_NAME: r' for $y$-position',
    evaluation_sans_uq.OFFSET_DIRECTION_NAME: ' for offset direction',
    evaluation_sans_uq.OFFSET_DISTANCE_NAME: ''
}

TARGET_FIELD_TO_UNIT_STRING = {
    evaluation_sans_uq.X_OFFSET_NAME: 'km',
    evaluation_sans_uq.Y_OFFSET_NAME: 'km',
    evaluation_sans_uq.OFFSET_DIRECTION_NAME: 'deg',
    evaluation_sans_uq.OFFSET_DISTANCE_NAME: 'km'
}

METRIC_TO_UNIT_EXPONENT = {
    evaluation_sans_uq.MEAN_SQUARED_ERROR_KEY: 2,
    evaluation_sans_uq.MSE_SKILL_SCORE_KEY: 0,
    evaluation_sans_uq.MEAN_ABSOLUTE_ERROR_KEY: 1,
    evaluation_sans_uq.MAE_SKILL_SCORE_KEY: 0,
    evaluation_sans_uq.BIAS_KEY: 1,
    evaluation_sans_uq.CORRELATION_KEY: 0,
    evaluation_sans_uq.KGE_KEY: 0,
    evaluation_sans_uq.MEAN_DISTANCE_KEY: 1,
    evaluation_sans_uq.MEAN_DIST_SKILL_SCORE_KEY: 0,
    evaluation_sans_uq.MEAN_SQUARED_DISTANCE_KEY: 2,
    evaluation_sans_uq.MEAN_SQ_DIST_SKILL_SCORE_KEY: 0
}

METRIC_TO_FANCY_NAME = {
    evaluation_sans_uq.MEAN_SQUARED_ERROR_KEY: 'root mean squared error',
    evaluation_sans_uq.MSE_SKILL_SCORE_KEY: 'Mean-squared-error skill score',
    evaluation_sans_uq.MEAN_ABSOLUTE_ERROR_KEY: 'mean absolute error',
    evaluation_sans_uq.MAE_SKILL_SCORE_KEY: 'Mean-absolute-error skill score',
    evaluation_sans_uq.BIAS_KEY: 'bias',
    evaluation_sans_uq.CORRELATION_KEY: 'correlation',
    evaluation_sans_uq.KGE_KEY: 'Kling-Gupta efficiency',
    evaluation_sans_uq.MEAN_DISTANCE_KEY: 'mean Euclidean distance',
    evaluation_sans_uq.MEAN_DIST_SKILL_SCORE_KEY:
        'mean-Euclidean-distance skill score',
    evaluation_sans_uq.MEAN_SQUARED_DISTANCE_KEY:
        'root mean squared Euclidean distance',
    evaluation_sans_uq.MEAN_SQ_DIST_SKILL_SCORE_KEY:
        'mean-squared-Euclidean-distance skill score'
}

LINE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
LINE_WIDTH = 3
MARKER_TYPE = 'o'
MARKER_SIZE = 16
MARKER_COLOUR = numpy.full(3, 0.)
POLYGON_OPACITY = 0.5

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

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
    '`evaluation_sans_uq.read_file`).  This list should have length N + 1, '
    'where N = length of {0:s}.'
).format(MAX_WIND_CUTOFFS_ARG_NAME)

MIN_PRESSURE_BASED_FILES_HELP_STRING = (
    'List of paths to min-pressure-specific evaluation files (each will be '
    'read by `evaluation_sans_uq.read_file`).  This list should have length '
    'N + 1, where N = length of {0:s}.'
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


def _plot_one_metric(
        metric_matrix, metric_name, target_field_name,
        category_description_strings, x_label_string, confidence_level,
        output_file_name):
    """Plots one metric across intensity categories.

    C = number of categories
    B = number of bootstrap replicates

    :param metric_matrix: C-by-B numpy array of metric values.
    :param metric_name: Name of error metric.
    :param target_field_name: Name of target variable.
    :param category_description_strings: length-C list of category descriptions.
    :param x_label_string: Label for entire x-axis (all categories).
    :param confidence_level: See documentation at top of file.
    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    conv_ratio = (
        TARGET_FIELD_TO_CONV_RATIO[target_field_name] **
        METRIC_TO_UNIT_EXPONENT[metric_name]
    )
    metric_matrix *= conv_ratio

    if METRIC_TO_UNIT_EXPONENT[metric_name] == 2:
        metric_matrix = numpy.sqrt(metric_matrix)

    num_bootstrap_reps = metric_matrix.shape[1]
    num_categories = metric_matrix.shape[0]
    category_indices = 0.5 + numpy.linspace(
        0, num_categories - 1, num=num_categories, dtype=float
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    if num_bootstrap_reps > 1 and not numpy.all(numpy.isnan(metric_matrix)):
        polygon_coord_matrix = misc_utils.confidence_interval_to_polygon(
            x_value_matrix=
            numpy.expand_dims(category_indices.astype(float), axis=-1),
            y_value_matrix=metric_matrix,
            confidence_level=confidence_level,
            same_order=True
        )

        polygon_colour = matplotlib.colors.to_rgba(LINE_COLOUR, POLYGON_OPACITY)
        patch_object = matplotlib.patches.Polygon(
            polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
        )
        axes_object.add_patch(patch_object)

    axes_object.plot(
        category_indices, numpy.nanmedian(metric_matrix, axis=-1),
        color=LINE_COLOUR, linewidth=LINE_WIDTH, linestyle='solid',
        marker=MARKER_TYPE, markersize=MARKER_SIZE,
        markerfacecolor=MARKER_COLOUR, markeredgecolor=MARKER_COLOUR,
        markeredgewidth=0
    )

    axes_object.set_xticks(category_indices)
    axes_object.set_xticklabels(category_description_strings, rotation=90.)
    axes_object.set_xlabel(x_label_string)

    title_string = '{0:s}{1:s}{2:s}'.format(
        METRIC_TO_FANCY_NAME[metric_name][0].upper(),
        METRIC_TO_FANCY_NAME[metric_name][1:],
        TARGET_FIELD_TO_FANCY_NAME[target_field_name]
    )

    unit_exponent = METRIC_TO_UNIT_EXPONENT[metric_name]
    if unit_exponent > 0:
        title_string += ' ({0:s})'.format(
            TARGET_FIELD_TO_UNIT_STRING[target_field_name]
        )

    axes_object.set_title(title_string)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


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

    num_max_wind_categories = len(max_wind_cutoffs_kt) - 1
    eval_table_by_max_wind_cat = [None] * num_max_wind_categories
    num_bootstrap_reps = -1

    for i in range(num_max_wind_categories):
        print('Reading data from: "{0:s}"...'.format(
            max_wind_based_eval_file_names[i]
        ))
        eval_table_by_max_wind_cat[i] = evaluation_sans_uq.read_file(
            max_wind_based_eval_file_names[i]
        )

        ets = eval_table_by_max_wind_cat

        num_bootstrap_reps = len(
            ets[0].coords[evaluation_sans_uq.BOOTSTRAP_REP_DIM].values
        )
        this_num_bootstrap_reps = len(
            ets[i].coords[evaluation_sans_uq.BOOTSTRAP_REP_DIM].values
        )
        assert num_bootstrap_reps == this_num_bootstrap_reps

    category_description_strings = [
        '[{0:.1f}, {1:.1f})'.format(v1, v2)
        for v1, v2 in zip(max_wind_cutoffs_kt[:-1], max_wind_cutoffs_kt[1:])
    ]
    category_description_strings[0] = '< {0:.1f}'.format(max_wind_cutoffs_kt[1])
    category_description_strings[-1] = '>= {0:.1f}'.format(
        max_wind_cutoffs_kt[-2]
    )

    ets = eval_table_by_max_wind_cat

    for metric_name in BASIC_METRIC_NAMES:
        for target_field_name in BASIC_TARGET_FIELD_NAMES:
            metric_matrix = numpy.full(
                (num_max_wind_categories, num_bootstrap_reps), numpy.nan
            )

            for i in range(num_max_wind_categories):
                j = numpy.where(
                    ets[i].coords[evaluation_sans_uq.TARGET_FIELD_DIM].values
                    == target_field_name
                )[0][0]

                metric_matrix[i, :] = ets[i][metric_name].values[j, :]

            output_file_name = '{0:s}/{1:s}_{2:s}_by-max-wind.jpg'.format(
                output_dir_name,
                target_field_name.replace('_', '-'),
                metric_name.replace('_', '-')
            )

            _plot_one_metric(
                metric_matrix=metric_matrix,
                metric_name=metric_name,
                target_field_name=target_field_name,
                category_description_strings=category_description_strings,
                x_label_string='Max sustained wind (kt)',
                confidence_level=confidence_level,
                output_file_name=output_file_name
            )

    for metric_name in ADVANCED_METRIC_NAMES:
        metric_matrix = numpy.full(
            (num_max_wind_categories, num_bootstrap_reps), numpy.nan
        )

        for i in range(num_max_wind_categories):
            metric_matrix[i, :] = ets[i][metric_name].values[:]

        output_file_name = '{0:s}/{1:s}_by-max-wind.jpg'.format(
            output_dir_name,
            metric_name.replace('_', '-')
        )

        _plot_one_metric(
            metric_matrix=metric_matrix,
            metric_name=metric_name,
            target_field_name=evaluation_sans_uq.OFFSET_DISTANCE_NAME,
            category_description_strings=category_description_strings,
            x_label_string='Max sustained wind (kt)',
            confidence_level=confidence_level,
            output_file_name=output_file_name
        )

    # TODO(thunderhoser): Next steps are to make sure this works, then make it work for min-pressure-based categories.


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
