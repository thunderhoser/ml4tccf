"""Plots error metrics as a function of hyperparams for Experiment 7."""

import os
import sys
import argparse
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
import imagemagick_utils
import scalar_evaluation
import pit_utils
import spread_skill_utils as ss_utils
import discard_test_utils as dt_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

METRES_TO_KM = 0.001

WAVELENGTH_GROUP_STRINGS_AXIS1 = [
    '8.500-9.610-12.300', '3.900-7.340-13.300', '3.900-6.185-6.950',
    '6.950-10.350-11.200',
    '3.900-6.185-6.950-7.340-8.500-9.610-10.350-11.200-12.300-13.300'
]
BATCHES_PER_EPOCH_COUNTS_AXIS2 = numpy.array([24, 24, 36, 36], dtype=int)
BATCHES_PER_UPDATE_COUNTS_AXIS2 = numpy.array([1, 2, 1, 2], dtype=int)
SPECTRAL_COMPLEXITIES_AXIS3 = numpy.array([16, 24, 32, 40], dtype=int)

MEAN_DISTANCE_NAME = 'mean_distance_km'
MEDIAN_DISTANCE_NAME = 'median_distance_km'
ROOT_MEAN_SQUARED_DISTANCE_NAME = 'root_mean_squared_distance_km'
RELIABILITY_NAME = 'reliability_km2'
CRPS_NAME = 'crps_km'
SPREAD_SKILL_RELIABILITY_NAME = 'ssrel_km'
SPREAD_SKILL_RATIO_NAME = 'ssrat'
MONO_FRACTION_NAME = 'mono_fraction'
PIT_DEVIATION_NAME = 'pit_deviation'

METRIC_NAMES = [
    MEAN_DISTANCE_NAME, MEDIAN_DISTANCE_NAME,
    ROOT_MEAN_SQUARED_DISTANCE_NAME, RELIABILITY_NAME,
    CRPS_NAME, SPREAD_SKILL_RELIABILITY_NAME,
    SPREAD_SKILL_RATIO_NAME, MONO_FRACTION_NAME, PIT_DEVIATION_NAME
]

METRIC_NAMES_FANCY = [
    'Mean Euclidean distance',
    'Median Euclidean distance',
    'Root mean squared Euclidean distance',
    'Coord-averaged reliability',
    'Coord-averaged CRPS',
    'Coord-averaged SSREL',
    'Coord-averaged SSRAT',
    'Coord-averaged monotonicity fraction',
    'Coord-averaged PIT deviation'
]

METRIC_UNITS = [
    'km', 'km', 'km', r'km$^2$', 'km', 'km', 'unitless', 'unitless', 'unitless'
]

METRIC_CONVERSION_FACTORS = numpy.array([
    0.001, 0.001, 0.001, 1e-6, 0.001, 0.001, 1, 1, 1
])

BEST_MARKER_TYPE = '*'
BEST_MARKER_SIZE_GRID_CELLS = 0.175
WHITE_COLOUR = numpy.full(3, 1.)
BLACK_COLOUR = numpy.full(3, 0.)

SELECTED_MARKER_TYPE = 'o'
SELECTED_MARKER_SIZE_GRID_CELLS = 0.175
SELECTED_MARKER_INDICES = numpy.array([0, 0, 0], dtype=int)

MAIN_COLOUR_MAP_OBJECT = pyplot.get_cmap(name='viridis', lut=20)
MONO_FRACTION_COLOUR_MAP_OBJECT = pyplot.get_cmap(name='cividis', lut=20)
SSRAT_COLOUR_MAP_NAME = 'seismic'

NAN_COLOUR = numpy.full(3, 0.)
MAIN_COLOUR_MAP_OBJECT.set_bad(NAN_COLOUR)
MONO_FRACTION_COLOUR_MAP_OBJECT.set_bad(NAN_COLOUR)

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


def _get_ssrat_colour_scheme(max_colour_value):
    """Returns colour scheme for spread-skill ratio (SSRAT).

    :param max_colour_value: Max value in colour scheme.
    :return: colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :return: colour_norm_object: Colour-normalizer (maps from data space to
        colour-bar space, which goes from 0...1).  This is an instance of
        `matplotlib.colors.Normalize`.
    """

    orig_colour_map_object = pyplot.get_cmap(SSRAT_COLOUR_MAP_NAME)

    negative_values = numpy.linspace(0, 1, num=1001, dtype=float)
    positive_values = numpy.linspace(1, max_colour_value, num=1001, dtype=float)
    bias_values = numpy.concatenate((negative_values, positive_values))

    normalized_values = numpy.linspace(0, 1, num=len(bias_values), dtype=float)
    rgb_matrix = orig_colour_map_object(normalized_values)[:, :-1]

    colour_map_object = matplotlib.colors.ListedColormap(rgb_matrix)
    colour_map_object.set_bad(NAN_COLOUR)
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        bias_values, colour_map_object.N
    )

    return colour_map_object, colour_norm_object


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
        font_size=FONT_SIZE, fraction_of_axis_length=1.
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
    metric_dict[CRPS_NAME] = METRES_TO_KM * numpy.mean(numpy.nanmean(
        et[scalar_evaluation.CRPS_KEY].values[xy_indices, :], axis=1
    ))

    this_file_name = '{0:s}/spread_vs_skill.nc'.format(validation_dir_name)

    print('Reading data from: "{0:s}"...'.format(this_file_name))
    this_ss_table_xarray = ss_utils.read_results(this_file_name)
    sst = this_ss_table_xarray

    target_field_names = (
        sst.coords[ss_utils.TARGET_FIELD_DIM].values.tolist()
    )
    xy_indices = numpy.array([
        target_field_names.index(ss_utils.X_OFFSET_NAME),
        target_field_names.index(ss_utils.Y_OFFSET_NAME)
    ], dtype=int)

    metric_dict[SPREAD_SKILL_RELIABILITY_NAME] = METRES_TO_KM * numpy.mean(
        sst[ss_utils.XY_SSREL_KEY].values[xy_indices]
    )
    metric_dict[SPREAD_SKILL_RATIO_NAME] = numpy.mean(
        sst[ss_utils.XY_SSRAT_KEY].values[xy_indices]
    )

    this_file_name = '{0:s}/pit_histograms.nc'.format(validation_dir_name)

    print('Reading data from: "{0:s}"...'.format(this_file_name))
    this_pit_table_xarray = pit_utils.read_results(this_file_name)
    pitt = this_pit_table_xarray

    target_field_names = (
        pitt.coords[pit_utils.TARGET_FIELD_DIM].values.tolist()
    )
    xy_indices = numpy.array([
        target_field_names.index(pit_utils.X_OFFSET_NAME),
        target_field_names.index(pit_utils.Y_OFFSET_NAME)
    ], dtype=int)

    metric_dict[PIT_DEVIATION_NAME] = numpy.mean(
        pitt[pit_utils.PIT_DEVIATION_KEY].values[xy_indices]
    )

    this_file_name = '{0:s}/discard_test.nc'.format(validation_dir_name)

    print('Reading data from: "{0:s}"...'.format(this_file_name))
    this_dt_table_xarray = dt_utils.read_results(this_file_name)
    dtt = this_dt_table_xarray

    target_field_names = (
        dtt.coords[dt_utils.TARGET_FIELD_DIM].values.tolist()
    )
    xy_indices = numpy.array([
        target_field_names.index(dt_utils.X_OFFSET_NAME),
        target_field_names.index(dt_utils.Y_OFFSET_NAME)
    ], dtype=int)

    metric_dict[MONO_FRACTION_NAME] = numpy.mean(
        dtt[dt_utils.MONO_FRACTION_KEY].values[xy_indices]
    )

    return metric_dict


def _print_ranking_all_metrics(metric_matrix, main_metric_name):
    """Prints ranking for all metrics.

    A = length of first hyperparam axis
    B = length of second hyperparam axis
    C = length of third hyperparam axis
    M = number of metrics

    :param metric_matrix: A-by-B-by-C-by-M numpy array of metric values.
    :param main_metric_name: Name of main metric.
    """

    main_metric_index = METRIC_NAMES.index(main_metric_name)
    values_1d = numpy.ravel(metric_matrix[..., main_metric_index])

    if 'mono_fraction' in main_metric_name:
        values_1d[numpy.isnan(values_1d)] = -numpy.inf
        sort_indices_1d = numpy.argsort(-values_1d)
    elif 'ssrat' in main_metric_name:
        values_1d[numpy.isnan(values_1d)] = numpy.inf
        sort_indices_1d = numpy.argsort(numpy.absolute(1. - values_1d))
    else:
        values_1d[numpy.isnan(values_1d)] = numpy.inf
        sort_indices_1d = numpy.argsort(values_1d)

    i_sort_indices, j_sort_indices, k_sort_indices = numpy.unravel_index(
        sort_indices_1d, metric_matrix.shape[:-1]
    )

    metric_rank_matrix = numpy.full(metric_matrix.shape, numpy.nan)

    for m in range(len(METRIC_NAMES)):
        these_values = numpy.ravel(metric_matrix[..., m])

        if 'mono_fraction' in main_metric_name:
            these_values = -1 * these_values
            these_values[numpy.isnan(these_values)] = -numpy.inf
        elif 'ssrat' in main_metric_name:
            these_values = numpy.absolute(1. - these_values)
            these_values[numpy.isnan(these_values)] = numpy.inf
        else:
            these_values[numpy.isnan(these_values)] = numpy.inf

        metric_rank_matrix[..., m] = numpy.reshape(
            rankdata(these_values, method='average'),
            metric_rank_matrix.shape[:-1]
        )

    names = METRIC_NAMES
    mrm = metric_rank_matrix

    for m in range(len(i_sort_indices)):
        i = i_sort_indices[m]
        j = j_sort_indices[m]
        k = k_sort_indices[m]

        print((
            'Wavelengths = {0:s} microns ... '
            'batches per epoch/update = {1:d}/{2:d} ... '
            'spectral complexity = {3:d} ... '
            'Euc-distance-based ranks (mean, median, RMS) = '
            '{4:.1f}, {5:.1f}, {6:.1f} ... '
            'reliability rank = {7:.1f} ... CRPS rank = {8:.1f} ... '
            'other UQ-based ranks (SSREL, SSRAT, PITD, MF) = '
            '{9:.1f}, {10:.1f}, {11:.1f}, {12:.1f}'
        ).format(
            WAVELENGTH_GROUP_STRINGS_AXIS1[i].replace('-', ', '),
            BATCHES_PER_EPOCH_COUNTS_AXIS2[j],
            BATCHES_PER_UPDATE_COUNTS_AXIS2[j],
            SPECTRAL_COMPLEXITIES_AXIS3[k],
            mrm[i, j, k, names.index(MEAN_DISTANCE_NAME)],
            mrm[i, j, k, names.index(MEDIAN_DISTANCE_NAME)],
            mrm[i, j, k, names.index(ROOT_MEAN_SQUARED_DISTANCE_NAME)],
            mrm[i, j, k, names.index(RELIABILITY_NAME)],
            mrm[i, j, k, names.index(CRPS_NAME)],
            mrm[i, j, k, names.index(SPREAD_SKILL_RELIABILITY_NAME)],
            mrm[i, j, k, names.index(SPREAD_SKILL_RATIO_NAME)],
            mrm[i, j, k, names.index(MONO_FRACTION_NAME)],
            mrm[i, j, k, names.index(PIT_DEVIATION_NAME)]
        ))


def _print_ranking_one_metric(metric_matrix, metric_index):
    """Prints ranking for one metric.

    A = length of first hyperparam axis
    B = length of second hyperparam axis
    C = length of third hyperparam axis
    M = number of metrics

    :param metric_matrix: A-by-B-by-C-by-M numpy array of metric values.
    :param metric_index: Will print ranking for [i]th metric, where
        i = `metric_index`.
    """

    values_1d = numpy.ravel(metric_matrix[..., metric_index])

    if 'mono_fraction' in METRIC_NAMES[metric_index]:
        values_1d[numpy.isnan(values_1d)] = -numpy.inf
        sort_indices_1d = numpy.argsort(-values_1d)
    elif 'ssrat' in METRIC_NAMES[metric_index]:
        values_1d[numpy.isnan(values_1d)] = numpy.inf
        sort_indices_1d = numpy.argsort(numpy.absolute(1. - values_1d))
    else:
        values_1d[numpy.isnan(values_1d)] = numpy.inf
        sort_indices_1d = numpy.argsort(values_1d)

    i_sort_indices, j_sort_indices, k_sort_indices = numpy.unravel_index(
        sort_indices_1d, metric_matrix.shape[:-1]
    )

    for m in range(len(i_sort_indices)):
        i = i_sort_indices[m]
        j = j_sort_indices[m]
        k = k_sort_indices[m]

        print((
            '{0:d}th-best {1:s} = {2:.3g} {3:s} ... '
            'wavelengths = {4:s} microns ... '
            'batches per epoch/update = {5:d}/{6:d} ... '
            'spectral complexity = {7:d} ... '
        ).format(
            m + 1, METRIC_NAMES_FANCY[metric_index],
            metric_matrix[i, j, k, metric_index], METRIC_UNITS[metric_index],
            WAVELENGTH_GROUP_STRINGS_AXIS1[i].replace('-', ', '),
            BATCHES_PER_EPOCH_COUNTS_AXIS2[j],
            BATCHES_PER_UPDATE_COUNTS_AXIS2[j],
            SPECTRAL_COMPLEXITIES_AXIS3[k]
        ))


def _run(experiment_dir_name):
    """Plots error metrics as a function of hyperparams for Experiment 7.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of file.
    """

    length_axis1 = len(WAVELENGTH_GROUP_STRINGS_AXIS1)
    length_axis2 = len(BATCHES_PER_UPDATE_COUNTS_AXIS2)
    length_axis3 = len(SPECTRAL_COMPLEXITIES_AXIS3)
    num_metrics = len(METRIC_NAMES)

    y_tick_labels = [
        g.replace('-', ', ') for g in WAVELENGTH_GROUP_STRINGS_AXIS1
    ]
    y_tick_labels = ['All' if len(l) > 30 else l for l in y_tick_labels]
    x_tick_labels = [
        '{0:d}/{1:d}'.format(e, u) for e, u in zip(
            BATCHES_PER_EPOCH_COUNTS_AXIS2, BATCHES_PER_UPDATE_COUNTS_AXIS2
        )
    ]

    y_axis_label = 'Input wavelengths (microns)'
    x_axis_label = 'Num batches per epoch/update'

    metric_matrix = numpy.full(
        (length_axis1, length_axis2, length_axis3, num_metrics),
        numpy.nan
    )

    for i in range(length_axis1):
        for j in range(length_axis2):
            for k in range(length_axis3):
                this_model_dir_name = (
                    '{0:s}/wavelengths-microns={1:s}_'
                    'num-first-layer-filters={2:02d}_'
                    'num-batches-per-epoch={3:02d}_'
                    'num-batches-per-update={4:1d}'
                ).format(
                    experiment_dir_name,
                    WAVELENGTH_GROUP_STRINGS_AXIS1[i],
                    SPECTRAL_COMPLEXITIES_AXIS3[k],
                    BATCHES_PER_EPOCH_COUNTS_AXIS2[j],
                    BATCHES_PER_UPDATE_COUNTS_AXIS2[j]
                )

                this_metric_dict = _read_metrics_one_model(this_model_dir_name)
                for m in range(num_metrics):
                    metric_matrix[i, j, k, m] = this_metric_dict[
                        METRIC_NAMES[m]
                    ]

    print(SEPARATOR_STRING)

    for m in range(num_metrics):
        _print_ranking_one_metric(metric_matrix=metric_matrix, metric_index=m)
        print(SEPARATOR_STRING)

    _print_ranking_all_metrics(
        metric_matrix=metric_matrix, main_metric_name=MEAN_DISTANCE_NAME
    )
    print(SEPARATOR_STRING)

    _print_ranking_all_metrics(
        metric_matrix=metric_matrix, main_metric_name=CRPS_NAME
    )
    print(SEPARATOR_STRING)

    _print_ranking_all_metrics(
        metric_matrix=metric_matrix,
        main_metric_name=SPREAD_SKILL_RELIABILITY_NAME
    )
    print(SEPARATOR_STRING)

    output_dir_name = '{0:s}/hyperparam_grids'.format(experiment_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    for m in range(num_metrics):
        if numpy.all(numpy.isnan(metric_matrix[..., m])):
            continue

        panel_file_names = [''] * length_axis3

        for k in range(length_axis3):
            if 'mono_fraction' in METRIC_NAMES[m]:
                max_colour_value = _finite_percentile(
                    numpy.absolute(metric_matrix[..., m]), 100
                )
                min_colour_value = _finite_percentile(
                    numpy.absolute(metric_matrix[..., m]), 0
                )
                min_colour_value = min([min_colour_value, 18. / 19])

                colour_norm_object = matplotlib.colors.Normalize(
                    vmin=min_colour_value, vmax=max_colour_value, clip=False
                )
                colour_map_object = MONO_FRACTION_COLOUR_MAP_OBJECT

                best_linear_index = numpy.nanargmax(
                    numpy.ravel(metric_matrix[..., m])
                )
                marker_colour = BLACK_COLOUR

            elif 'ssrat' in METRIC_NAMES[m]:
                if not numpy.any(metric_matrix[..., m] > 1):
                    max_colour_value = _finite_percentile(
                        numpy.absolute(metric_matrix[..., m]), 100
                    )
                    min_colour_value = _finite_percentile(
                        numpy.absolute(metric_matrix[..., m]), 0
                    )

                    colour_norm_object = matplotlib.colors.Normalize(
                        vmin=min_colour_value, vmax=max_colour_value, clip=False
                    )
                    colour_map_object = MONO_FRACTION_COLOUR_MAP_OBJECT
                else:
                    this_offset = _finite_percentile(
                        numpy.absolute(metric_matrix[..., m] - 1.), 100
                    )
                    colour_map_object, colour_norm_object = (
                        _get_ssrat_colour_scheme(
                            max_colour_value=1. + this_offset
                        )
                    )

                best_linear_index = numpy.nanargmin(
                    numpy.absolute(numpy.ravel(metric_matrix[..., m]) - 1.)
                )
                marker_colour = BLACK_COLOUR

            else:
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

                best_linear_index = numpy.nanargmin(
                    numpy.ravel(metric_matrix[..., m])
                )
                marker_colour = WHITE_COLOUR

            figure_object, axes_object = _plot_scores_2d(
                score_matrix=metric_matrix[..., k, m],
                colour_map_object=colour_map_object,
                colour_norm_object=colour_norm_object,
                x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
            )

            best_indices = numpy.unravel_index(
                best_linear_index, metric_matrix[..., m].shape
            )

            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS / metric_matrix.shape[2]
            )

            if best_indices[2] == k:
                axes_object.plot(
                    best_indices[1], best_indices[0],
                    linestyle='None', marker=BEST_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=marker_colour,
                    markeredgecolor=marker_colour
                )

            if SELECTED_MARKER_INDICES[2] == k:
                axes_object.plot(
                    SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                    linestyle='None', marker=SELECTED_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=marker_colour,
                    markeredgecolor=marker_colour
                )

            axes_object.set_xlabel(x_axis_label)
            axes_object.set_ylabel(y_axis_label)
            axes_object.set_title('Spectral complexity = {0:d}'.format(
                SPECTRAL_COMPLEXITIES_AXIS3[k]
            ))

            panel_file_names[k] = (
                '{0:s}/{1:s}_spectral-complexity={2:02d}.jpg'
            ).format(
                output_dir_name,
                METRIC_NAMES[m].replace('_', '-'),
                SPECTRAL_COMPLEXITIES_AXIS3[k]
            )

            print('Saving figure to: "{0:s}"...'.format(panel_file_names[k]))
            figure_object.savefig(
                panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

        num_panel_rows = int(numpy.floor(
            numpy.sqrt(length_axis3)
        ))
        num_panel_columns = int(numpy.ceil(
            float(length_axis3) / num_panel_rows
        ))
        concat_figure_file_name = '{0:s}/{1:s}.jpg'.format(
            output_dir_name, METRIC_NAMES[m]
        )

        print('Concatenating panels to: "{0:s}"...'.format(
            concat_figure_file_name
        ))
        imagemagick_utils.concatenate_images(
            input_file_names=panel_file_names,
            output_file_name=concat_figure_file_name,
            num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
        )
        imagemagick_utils.resize_image(
            input_file_name=concat_figure_file_name,
            output_file_name=concat_figure_file_name,
            output_size_pixels=int(1e7)
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        experiment_dir_name=getattr(INPUT_ARG_OBJECT, EXPERIMENT_DIR_ARG_NAME)
    )