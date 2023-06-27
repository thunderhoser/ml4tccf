"""Plots error metrics as a function of hyperparams for Experiment 6."""

import os
import sys
import argparse
from itertools import combinations
import numpy
from scipy.stats import rankdata
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import gg_plotting_utils
import scalar_evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
METRES_TO_KM = 0.001

UNIQUE_WAVELENGTHS_MICRONS = numpy.array([
    3.9, 6.185, 6.95, 7.34, 8.5, 9.61, 10.35, 11.2, 12.3, 13.3
])
COMBINATION_OBJECT = combinations(UNIQUE_WAVELENGTHS_MICRONS, 3)

WAVELENGTH_GROUP_STRINGS_MICRONS = []
WAVELENGTH_GROUP_STRINGS_MICRONS_NICE = []

for this_array in list(COMBINATION_OBJECT):
    this_string = '-'.join(['{0:.3f}'.format(w) for w in this_array])
    WAVELENGTH_GROUP_STRINGS_MICRONS.append(this_string)

    this_string = '{0:.1f}, {1:.1f},\n{2:.1f}'.format(
        this_array[0], this_array[1], this_array[2]
    )
    WAVELENGTH_GROUP_STRINGS_MICRONS_NICE.append(this_string)

NUM_GRID_ROWS = 10
NUM_GRID_COLUMNS = 12

MEAN_DISTANCE_NAME = 'mean_distance_km'
MEDIAN_DISTANCE_NAME = 'median_distance_km'
ROOT_MEAN_SQUARED_DISTANCE_NAME = 'root_mean_squared_distance_km'
RELIABILITY_NAME = 'reliability_km2'

METRIC_NAMES = [
    MEAN_DISTANCE_NAME, MEDIAN_DISTANCE_NAME,
    ROOT_MEAN_SQUARED_DISTANCE_NAME, RELIABILITY_NAME
]
METRIC_NAMES_FANCY = [
    'Mean Euclidean distance',
    'Median Euclidean distance',
    'Root mean squared Euclidean distance',
    'Coord-averaged reliability'
]
METRIC_UNITS = ['km', 'km', 'km', r'km$^2$']

BEST_MARKER_TYPE = '*'
BEST_MARKER_SIZE_GRID_CELLS = 0.175
WHITE_COLOUR = numpy.full(3, 1.)
BLACK_COLOUR = numpy.full(3, 0.)

SELECTED_MARKER_TYPE = 'o'
SELECTED_MARKER_SIZE_GRID_CELLS = 0.175
SELECTED_MARKER_INDEX = 0

MAIN_COLOUR_MAP_OBJECT = pyplot.get_cmap(name='viridis', lut=20)
NAN_COLOUR = numpy.full(3, 0.)
MAIN_COLOUR_MAP_OBJECT.set_bad(NAN_COLOUR)

FONT_SIZE = 26
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

EXPERIMENT_DIR_ARG_NAME = 'experiment_dir_name'
EXPERIMENT_DIR_HELP_STRING = 'Name of top-level directory with models.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXPERIMENT_DIR_ARG_NAME, type=str, required=True,
    help=EXPERIMENT_DIR_HELP_STRING
)


def _finite_percentile(input_array, percentile_level):
    """Takes percentile of input array, considering only finite values.

    :param input_array: numpy array.
    :param percentile_level: Percentile level, ranging from 0...100.
    :return: output_percentile: Percentile value.
    """

    return numpy.percentile(
        input_array[numpy.isfinite(input_array)], percentile_level
    )


def _plot_scores_2d(
        score_matrix, colour_map_object, colour_norm_object, x_tick_labels,
        y_tick_labels):
    """Plots scores on 2-D grid.

    M = number of rows in grid
    N = number of columns in grid

    :param score_matrix: M-by-N numpy array of scores.
    :param colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :param colour_norm_object: Colour-normalizer (maps from data space to
        colour-bar space, which goes from 0...1).  This is an instance of
        `matplotlib.colors.Normalize`.
    :param x_tick_labels: length-N list of tick labels.
    :param y_tick_labels: length-M list of tick labels.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    score_matrix_to_plot = numpy.ma.masked_where(
        numpy.isnan(score_matrix), score_matrix
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.imshow(
        score_matrix_to_plot, cmap=colour_map_object, norm=colour_norm_object,
        origin='lower'
    )

    x_tick_values = numpy.linspace(
        0, score_matrix.shape[1] - 1, num=score_matrix.shape[1], dtype=float
    )
    y_tick_values = numpy.linspace(
        0, score_matrix.shape[0] - 1, num=score_matrix.shape[0], dtype=float
    )
    pyplot.xticks(x_tick_values, x_tick_labels)
    pyplot.yticks(y_tick_values, y_tick_labels)

    colour_bar_object = gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=score_matrix[numpy.invert(numpy.isnan(score_matrix))],
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', extend_min=False, extend_max=False,
        font_size=FONT_SIZE, fraction_of_axis_length=0.6
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.2g}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    return figure_object, axes_object


def _read_metrics_one_model(model_dir_name):
    """Reads metrics for one model.

    :param model_dir_name: Name of directory with trained model and validation
        data.
    :return: metric_dict: Dictionary, where each key is a string from the list
        `METRIC_NAMES` and each value is a scalar.
    """

    metric_dict = {}
    for this_metric_name in METRIC_NAMES:
        metric_dict[this_metric_name] = numpy.nan

    validation_dir_name = '{0:s}/validation'.format(model_dir_name)
    if not os.path.isdir(validation_dir_name):
        return metric_dict

    this_file_name = '{0:s}/evaluation.nc'.format(validation_dir_name)
    if not os.path.isfile(this_file_name):
        return metric_dict

    print('Reading data from: "{0:s}"...'.format(this_file_name))
    this_eval_table_xarray = scalar_evaluation.read_file(this_file_name)
    et = this_eval_table_xarray

    metric_dict[MEAN_DISTANCE_NAME] = METRES_TO_KM * numpy.nanmean(
        et[scalar_evaluation.MEAN_DISTANCE_KEY].values
    )
    metric_dict[MEDIAN_DISTANCE_NAME] = METRES_TO_KM * numpy.nanmean(
        et[scalar_evaluation.MEDIAN_DISTANCE_KEY].values
    )
    metric_dict[ROOT_MEAN_SQUARED_DISTANCE_NAME] = METRES_TO_KM * numpy.sqrt(
        numpy.nanmean(et[scalar_evaluation.MEAN_SQUARED_DISTANCE_KEY].values)
    )

    target_field_names = (
        et.coords[scalar_evaluation.TARGET_FIELD_DIM].values.tolist()
    )
    xy_indices = numpy.array([
        target_field_names.index(scalar_evaluation.X_OFFSET_NAME),
        target_field_names.index(scalar_evaluation.Y_OFFSET_NAME)
    ], dtype=int)

    metric_dict[RELIABILITY_NAME] = (METRES_TO_KM ** 2) * numpy.mean(
        numpy.nanmean(
            et[scalar_evaluation.RELIABILITY_KEY].values[xy_indices, :], axis=1
        )
    )

    return metric_dict


def _print_ranking_all_metrics(metric_matrix, main_metric_name):
    """Prints ranking for all metrics.

    A = length of first hyperparam axis
    M = number of metrics

    :param metric_matrix: A-by-M numpy array of metric values.
    :param main_metric_name: Name of main metric.
    """

    main_metric_index = METRIC_NAMES.index(main_metric_name)
    values_1d = metric_matrix[:, main_metric_index] + 0.
    values_1d[numpy.isnan(values_1d)] = numpy.inf

    i_sort_indices = numpy.argsort(values_1d)
    metric_rank_matrix = numpy.full(metric_matrix.shape, numpy.nan)

    for m in range(len(METRIC_NAMES)):
        these_values = metric_matrix[:, m] + 0.
        these_values[numpy.isnan(these_values)] = numpy.inf
        metric_rank_matrix[:, m] = rankdata(these_values, method='average')

    names = METRIC_NAMES
    mrm = metric_rank_matrix

    for m in range(len(i_sort_indices)):
        i = i_sort_indices[m]

        print((
            'Wavelengths = {0:s} microns ... '
            'Euc-distance-based ranks (mean, median, RMS) = '
            '{1:.1f}, {2:.1f}, {3:.1f} ... '
            'reliability rank = {4:.1f}'
        ).format(
            WAVELENGTH_GROUP_STRINGS_MICRONS[i].replace('-', ', '),
            mrm[i, names.index(MEAN_DISTANCE_NAME)],
            mrm[i, names.index(MEDIAN_DISTANCE_NAME)],
            mrm[i, names.index(ROOT_MEAN_SQUARED_DISTANCE_NAME)],
            mrm[i, names.index(RELIABILITY_NAME)]
        ))


def _print_ranking_one_metric(metric_matrix, metric_index):
    """Prints ranking for one metric.

    A = length of first hyperparam axis
    M = number of metrics

    :param metric_matrix: A-by-M numpy array of metric values.
    :param metric_index: Will print ranking for [i]th metric, where
        i = `metric_index`.
    """

    values_1d = metric_matrix[:, metric_index] + 0.
    values_1d[numpy.isnan(values_1d)] = numpy.inf
    i_sort_indices = numpy.argsort(values_1d)

    for m in range(len(i_sort_indices)):
        i = i_sort_indices[m]

        print((
            '{0:d}th-best {1:s} = {2:.3g} {3:s} ... '
            'wavelengths = {4:s} microns'
        ).format(
            m + 1, METRIC_NAMES_FANCY[metric_index],
            metric_matrix[i, metric_index], METRIC_UNITS[metric_index],
            WAVELENGTH_GROUP_STRINGS_MICRONS[i].replace('-', ', ')
        ))


def _run(experiment_dir_name):
    """Plots error metrics as a function of hyperparams for Experiment 6.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of file.
    """

    length_axis1 = len(WAVELENGTH_GROUP_STRINGS_MICRONS)
    num_metrics = len(METRIC_NAMES)

    y_axis_label = 'Wavelengths (microns)'
    x_axis_label = 'Wavelengths (microns)'
    metric_matrix = numpy.full((length_axis1, num_metrics), numpy.nan)

    for i in range(length_axis1):
        this_model_dir_name = '{0:s}/wavelengths-microns={1:s}'.format(
            experiment_dir_name,
            WAVELENGTH_GROUP_STRINGS_MICRONS[i]
        )

        this_metric_dict = _read_metrics_one_model(this_model_dir_name)
        for m in range(num_metrics):
            metric_matrix[i, m] = this_metric_dict[METRIC_NAMES[m]]

    print(SEPARATOR_STRING)

    for m in range(num_metrics):
        _print_ranking_one_metric(metric_matrix=metric_matrix, metric_index=m)
        print(SEPARATOR_STRING)

    _print_ranking_all_metrics(
        metric_matrix=metric_matrix, main_metric_name=MEAN_DISTANCE_NAME
    )
    print(SEPARATOR_STRING)

    output_dir_name = '{0:s}/hyperparam_grids'.format(experiment_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    for m in range(num_metrics):
        if numpy.all(numpy.isnan(metric_matrix[..., m])):
            continue

        max_colour_value = _finite_percentile(
            numpy.absolute(metric_matrix[..., m]), 95
        )
        min_colour_value = _finite_percentile(
            numpy.absolute(metric_matrix[..., m]), 0
        )

        colour_norm_object = matplotlib.colors.Normalize(
            vmin=min_colour_value, vmax=max_colour_value, clip=False
        )
        colour_map_object = MAIN_COLOUR_MAP_OBJECT

        best_linear_index = numpy.nanargmin(metric_matrix[:, m])
        marker_colour = WHITE_COLOUR

        figure_object, axes_object = _plot_scores_2d(
            score_matrix=numpy.reshape(
                metric_matrix[:, m], (NUM_GRID_ROWS, NUM_GRID_COLUMNS)
            ),
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            x_tick_labels=[' '] * NUM_GRID_COLUMNS,
            y_tick_labels=[' '] * NUM_GRID_ROWS
        )

        for i in range(NUM_GRID_ROWS):
            for j in range(NUM_GRID_COLUMNS):
                linear_index = numpy.ravel_multi_index(
                    (i, j), (NUM_GRID_ROWS, NUM_GRID_COLUMNS)
                )
                axes_object.text(
                    j, i, WAVELENGTH_GROUP_STRINGS_MICRONS_NICE[linear_index],
                    color=marker_colour, fontsize=20,
                    horizontalalignment='center', verticalalignment='center'
                )

        best_indices = numpy.unravel_index(
            best_linear_index, (NUM_GRID_ROWS, NUM_GRID_COLUMNS)
        )
        selected_indices = numpy.unravel_index(
            SELECTED_MARKER_INDEX, (NUM_GRID_ROWS, NUM_GRID_COLUMNS)
        )

        figure_width_px = (
            figure_object.get_size_inches()[0] * figure_object.dpi
        )
        marker_size_px = figure_width_px * (
            BEST_MARKER_SIZE_GRID_CELLS / metric_matrix.shape[1]
        )

        axes_object.plot(
            best_indices[1], best_indices[0],
            linestyle='None', marker=BEST_MARKER_TYPE,
            markersize=marker_size_px, markeredgewidth=0,
            markerfacecolor=marker_colour,
            markeredgecolor=marker_colour
        )
        axes_object.plot(
            selected_indices[1], selected_indices[0],
            linestyle='None', marker=SELECTED_MARKER_TYPE,
            markersize=marker_size_px, markeredgewidth=0,
            markerfacecolor=marker_colour,
            markeredgecolor=marker_colour
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)

        output_file_name = '{0:s}/{1:s}.jpg'.format(
            output_dir_name, METRIC_NAMES[m]
        )

        print('Saving figure to: "{0:s}"...'.format(output_file_name))
        figure_object.savefig(
            output_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        experiment_dir_name=getattr(INPUT_ARG_OBJECT, EXPERIMENT_DIR_ARG_NAME)
    )
