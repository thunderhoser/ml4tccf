"""Plotting methods for evaluation of scalar predictions."""

import os
import sys
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
import matplotlib.patches
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import grids
import error_checking
import gg_plotting_utils
import misc_utils
import scalar_evaluation
import taylor_diagram

TOLERANCE = 1e-6
METRES_TO_KM = 0.001

NUM_EXAMPLES_KEY = 'num_examples'

QUASI_SKILL_SCORE_NAMES = [
    scalar_evaluation.MSE_SKILL_SCORE_KEY,
    scalar_evaluation.MAE_SKILL_SCORE_KEY,
    scalar_evaluation.KGE_KEY,
    scalar_evaluation.MEAN_DIST_SKILL_SCORE_KEY,
    scalar_evaluation.MEAN_SQ_DIST_SKILL_SCORE_KEY
]

BASIC_TARGET_FIELD_NAMES = [
    scalar_evaluation.X_OFFSET_NAME,
    scalar_evaluation.Y_OFFSET_NAME,
    scalar_evaluation.OFFSET_DIRECTION_NAME
]

BASIC_METRIC_NAMES = [
    scalar_evaluation.CRPS_KEY,
    scalar_evaluation.MEAN_SQUARED_ERROR_KEY,
    scalar_evaluation.MSE_SKILL_SCORE_KEY,
    scalar_evaluation.MEAN_ABSOLUTE_ERROR_KEY,
    scalar_evaluation.MAE_SKILL_SCORE_KEY,
    scalar_evaluation.BIAS_KEY,
    scalar_evaluation.CORRELATION_KEY,
    scalar_evaluation.KGE_KEY
]

ADVANCED_METRIC_NAMES = [
    scalar_evaluation.MEAN_DISTANCE_KEY,
    scalar_evaluation.MEDIAN_DISTANCE_KEY,
    scalar_evaluation.MEAN_DIST_SKILL_SCORE_KEY,
    scalar_evaluation.MEAN_SQUARED_DISTANCE_KEY,
    scalar_evaluation.MEAN_SQ_DIST_SKILL_SCORE_KEY,
    NUM_EXAMPLES_KEY
]

TARGET_FIELD_TO_CONV_RATIO = {
    scalar_evaluation.X_OFFSET_NAME: METRES_TO_KM,
    scalar_evaluation.Y_OFFSET_NAME: METRES_TO_KM,
    scalar_evaluation.OFFSET_DIRECTION_NAME: 1.,
    scalar_evaluation.OFFSET_DISTANCE_NAME: METRES_TO_KM
}

TARGET_FIELD_TO_FANCY_NAME = {
    scalar_evaluation.X_OFFSET_NAME: r' for $x$-coord',
    scalar_evaluation.Y_OFFSET_NAME: r' for $y$-coord',
    scalar_evaluation.OFFSET_DIRECTION_NAME: ' for correction direction',
    scalar_evaluation.OFFSET_DISTANCE_NAME: ''
}

TARGET_FIELD_TO_UNIT_STRING = {
    scalar_evaluation.X_OFFSET_NAME: 'km',
    scalar_evaluation.Y_OFFSET_NAME: 'km',
    scalar_evaluation.OFFSET_DIRECTION_NAME: 'deg',
    scalar_evaluation.OFFSET_DISTANCE_NAME: 'km'
}

METRIC_TO_UNIT_EXPONENT = {
    scalar_evaluation.CRPS_KEY: 1,
    scalar_evaluation.MEAN_SQUARED_ERROR_KEY: 2,
    scalar_evaluation.MSE_SKILL_SCORE_KEY: 0,
    scalar_evaluation.MEAN_ABSOLUTE_ERROR_KEY: 1,
    scalar_evaluation.MAE_SKILL_SCORE_KEY: 0,
    scalar_evaluation.BIAS_KEY: 1,
    scalar_evaluation.CORRELATION_KEY: 0,
    scalar_evaluation.KGE_KEY: 0,
    scalar_evaluation.MEAN_DISTANCE_KEY: 1,
    scalar_evaluation.MEDIAN_DISTANCE_KEY: 1,
    scalar_evaluation.MEAN_DIST_SKILL_SCORE_KEY: 0,
    scalar_evaluation.MEAN_SQUARED_DISTANCE_KEY: 2,
    scalar_evaluation.MEAN_SQ_DIST_SKILL_SCORE_KEY: 0,
    NUM_EXAMPLES_KEY: 0
}

METRIC_TO_FANCY_NAME = {
    scalar_evaluation.CRPS_KEY: 'CRPS',
    scalar_evaluation.MEAN_SQUARED_ERROR_KEY: 'RMSE',
    scalar_evaluation.MSE_SKILL_SCORE_KEY: 'MSE skill score',
    scalar_evaluation.MEAN_ABSOLUTE_ERROR_KEY: 'MAE',
    scalar_evaluation.MAE_SKILL_SCORE_KEY: 'MAE skill score',
    scalar_evaluation.BIAS_KEY: 'bias',
    scalar_evaluation.CORRELATION_KEY: 'correlation',
    scalar_evaluation.KGE_KEY: 'KGE',
    scalar_evaluation.MEAN_DISTANCE_KEY: 'mean Euclidean distance',
    scalar_evaluation.MEDIAN_DISTANCE_KEY: 'median Euclidean distance',
    scalar_evaluation.MEAN_DIST_SKILL_SCORE_KEY:
        'MED skill score',
    scalar_evaluation.MEAN_SQUARED_DISTANCE_KEY:
        'RMS Euclidean distance',
    scalar_evaluation.MEAN_SQ_DIST_SKILL_SCORE_KEY:
        'MSED skill score',
    NUM_EXAMPLES_KEY: 'number of TC samples'
}

RELIABILITY_LINE_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
RELIABILITY_LINE_WIDTH = 3.

REFERENCE_LINE_COLOUR = numpy.full(3, 152. / 255)
REFERENCE_LINE_WIDTH = 2.

CLIMO_LINE_COLOUR = numpy.full(3, 152. / 255)
CLIMO_LINE_WIDTH = 2.

ZERO_SKILL_LINE_COLOUR = numpy.array([31, 120, 180], dtype=float) / 255
ZERO_SKILL_LINE_WIDTH = 2.
POSITIVE_SKILL_AREA_OPACITY = 0.2

HISTOGRAM_FACE_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
HISTOGRAM_FONT_SIZE = 21

TAYLOR_TARGET_MARKER_TYPE = '*'
TAYLOR_TARGET_MARKER_SIZE = 24
TAYLOR_PREDICTION_MARKER_TYPE = 'o'
TAYLOR_PREDICTION_MARKER_SIZE = 20

METRIC_LINE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
METRIC_LINE_WIDTH = 3
METRIC_MARKER_TYPE = 'o'
METRIC_MARKER_SIZE = 16
METRIC_MARKER_COLOUR = numpy.full(3, 0.)
POLYGON_OPACITY = 0.5

SATELLITE_SUBPOINT_MARKER = 'X'
SATELLITE_SUBPOINT_MARKER_SIZE = 24

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

FONT_SIZE = 36
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)


def _plot_reliability_curve(
        axes_object, mean_predictions, mean_observations, min_value_to_plot,
        max_value_to_plot, line_colour=RELIABILITY_LINE_COLOUR,
        line_style='solid', line_width=RELIABILITY_LINE_WIDTH):
    """Plots reliability curve.

    B = number of bins

    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param mean_predictions: length-B numpy array of mean predicted values.
    :param mean_observations: length-B numpy array of mean observed values.
    :param min_value_to_plot: See doc for `plot_attributes_diagram`.
    :param max_value_to_plot: Same.
    :param line_colour: Line colour (in any format accepted by matplotlib).
    :param line_style: Line style (in any format accepted by matplotlib).
    :param line_width: Line width (in any format accepted by matplotlib).
    :return: main_line_handle: Handle for main line (reliability curve).
    """

    perfect_x_coords = numpy.array([min_value_to_plot, max_value_to_plot])
    perfect_y_coords = numpy.array([min_value_to_plot, max_value_to_plot])

    axes_object.plot(
        perfect_x_coords, perfect_y_coords, color=REFERENCE_LINE_COLOUR,
        linestyle='dashed', linewidth=REFERENCE_LINE_WIDTH
    )

    nan_flags = numpy.logical_or(
        numpy.isnan(mean_predictions), numpy.isnan(mean_observations)
    )

    if numpy.all(nan_flags):
        main_line_handle = None
    else:
        real_indices = numpy.where(numpy.invert(nan_flags))[0]

        main_line_handle = axes_object.plot(
            mean_predictions[real_indices], mean_observations[real_indices],
            color=line_colour, linestyle=line_style, linewidth=line_width,
            marker='o', markersize=12, markeredgewidth=0,
            markerfacecolor=line_colour, markeredgecolor=line_colour
        )[0]

    axes_object.set_xlabel('Prediction')
    axes_object.set_ylabel('Conditional mean observation')
    axes_object.set_xlim(min_value_to_plot, max_value_to_plot)
    axes_object.set_ylim(min_value_to_plot, max_value_to_plot)

    return main_line_handle


def _get_positive_skill_area(mean_value_in_training, min_value_in_plot,
                             max_value_in_plot):
    """Returns positive-skill area (where BSS > 0) for attributes diagram.

    :param mean_value_in_training: Mean of target variable in training data.
    :param min_value_in_plot: Minimum value in plot (for both x- and y-axes).
    :param max_value_in_plot: Max value in plot (for both x- and y-axes).
    :return: x_coords_left: length-5 numpy array of x-coordinates for left part
        of positive-skill area.
    :return: y_coords_left: Same but for y-coordinates.
    :return: x_coords_right: length-5 numpy array of x-coordinates for right
        part of positive-skill area.
    :return: y_coords_right: Same but for y-coordinates.
    """

    x_coords_left = numpy.array([
        min_value_in_plot, mean_value_in_training, mean_value_in_training,
        min_value_in_plot, min_value_in_plot
    ])
    y_coords_left = numpy.array([
        min_value_in_plot, min_value_in_plot, mean_value_in_training,
        (min_value_in_plot + mean_value_in_training) / 2, min_value_in_plot
    ])

    x_coords_right = numpy.array([
        mean_value_in_training, max_value_in_plot, max_value_in_plot,
        mean_value_in_training, mean_value_in_training
    ])
    y_coords_right = numpy.array([
        mean_value_in_training,
        (max_value_in_plot + mean_value_in_training) / 2,
        max_value_in_plot, max_value_in_plot, mean_value_in_training
    ])

    return x_coords_left, y_coords_left, x_coords_right, y_coords_right


def _get_zero_skill_line(mean_value_in_training, min_value_in_plot,
                         max_value_in_plot):
    """Returns zero-skill line (where BSS = 0) for attributes diagram.

    :param mean_value_in_training: Mean of target variable in training data.
    :param min_value_in_plot: Minimum value in plot (for both x- and y-axes).
    :param max_value_in_plot: Max value in plot (for both x- and y-axes).
    :return: x_coords: length-2 numpy array of x-coordinates.
    :return: y_coords: Same but for y-coordinates.
    """

    x_coords = numpy.array([min_value_in_plot, max_value_in_plot], dtype=float)
    y_coords = 0.5 * (mean_value_in_training + x_coords)

    return x_coords, y_coords


def _plot_attr_diagram_background(
        axes_object, mean_value_in_training, min_value_in_plot,
        max_value_in_plot):
    """Plots background (reference lines and polygons) of attributes diagram.

    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param mean_value_in_training: Mean of target variable in training data.
    :param min_value_in_plot: Minimum value in plot (for both x- and y-axes).
    :param max_value_in_plot: Max value in plot (for both x- and y-axes).
    """

    x_coords_left, y_coords_left, x_coords_right, y_coords_right = (
        _get_positive_skill_area(
            mean_value_in_training=mean_value_in_training,
            min_value_in_plot=min_value_in_plot,
            max_value_in_plot=max_value_in_plot
        )
    )

    skill_area_colour = matplotlib.colors.to_rgba(
        ZERO_SKILL_LINE_COLOUR, POSITIVE_SKILL_AREA_OPACITY
    )

    left_polygon_coord_matrix = numpy.transpose(numpy.vstack((
        x_coords_left, y_coords_left
    )))
    left_patch_object = matplotlib.patches.Polygon(
        left_polygon_coord_matrix, lw=0,
        ec=skill_area_colour, fc=skill_area_colour
    )
    axes_object.add_patch(left_patch_object)

    right_polygon_coord_matrix = numpy.transpose(numpy.vstack((
        x_coords_right, y_coords_right
    )))
    right_patch_object = matplotlib.patches.Polygon(
        right_polygon_coord_matrix, lw=0,
        ec=skill_area_colour, fc=skill_area_colour
    )
    axes_object.add_patch(right_patch_object)

    no_skill_x_coords, no_skill_y_coords = _get_zero_skill_line(
        mean_value_in_training=mean_value_in_training,
        min_value_in_plot=min_value_in_plot,
        max_value_in_plot=max_value_in_plot
    )

    axes_object.plot(
        no_skill_x_coords, no_skill_y_coords, color=ZERO_SKILL_LINE_COLOUR,
        linestyle='solid', linewidth=ZERO_SKILL_LINE_WIDTH
    )

    climo_x_coords = numpy.full(2, mean_value_in_training)
    climo_y_coords = numpy.array([min_value_in_plot, max_value_in_plot])
    axes_object.plot(
        climo_x_coords, climo_y_coords, color=CLIMO_LINE_COLOUR,
        linestyle='dashed', linewidth=CLIMO_LINE_WIDTH
    )

    axes_object.plot(
        climo_y_coords, climo_x_coords, color=CLIMO_LINE_COLOUR,
        linestyle='dashed', linewidth=CLIMO_LINE_WIDTH
    )


def _metric_value_to_label(metric_value):
    """Converts metric value to string label.

    :param metric_value: Metric value (scalar float).
    :return: label_string: String.
    """

    label_string = '{0:.2f}'.format(metric_value).lstrip('0').replace('-0', '-')
    if len(label_string.replace('-', '')) <= 3:
        return label_string

    label_string = '{0:.1f}'.format(metric_value).lstrip('0').replace('-0', '-')
    if len(label_string.replace('-', '')) <= 3:
        return label_string

    return '{0:.0f}'.format(metric_value)


def plot_metric_by_latlng(
        axes_object, metric_matrix, metric_name, target_field_name,
        grid_edge_latitudes_deg_n, grid_edge_longitudes_deg_e,
        colour_map_name, min_colour_percentile, max_colour_percentile,
        label_font_size=30, plot_satellite_subpoints=True):
    """Plots one metric as a function of lat-long.

    M = number of rows in grid
    N = number of columns in grid

    :param axes_object: Will plot on this set of axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param metric_matrix: M-by-N numpy array of metric values.
    :param metric_name: Name of error metric.
    :param target_field_name: Name of target variable.
    :param grid_edge_latitudes_deg_n: length-(M + 1) numpy array of latitudes
        (deg north).
    :param grid_edge_longitudes_deg_e: length-(N + 1) numpy array of longitudes
        (deg east).
    :param colour_map_name: Name of colour scheme (must be accepted by
        `matplotlib.pyplot.get_cmap`).
    :param min_colour_percentile: Determines minimum value in colour scheme.
    :param max_colour_percentile: Determines max value in colour scheme.
    :param label_font_size: Font size for text labels.  If you do not want
        labels, make this None.
    :param plot_satellite_subpoints: Boolean flag.
    """

    # Check input args.
    error_checking.assert_is_valid_lat_numpy_array(grid_edge_latitudes_deg_n)
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(grid_edge_latitudes_deg_n), 0.
    )

    error_checking.assert_is_valid_lng_numpy_array(
        grid_edge_longitudes_deg_e, negative_in_west_flag=True
    )
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(grid_edge_longitudes_deg_e), 0.
    )

    error_checking.assert_is_numpy_array(metric_matrix, num_dimensions=2)
    num_grid_rows = metric_matrix.shape[0]
    num_grid_columns = metric_matrix.shape[1]

    error_checking.assert_is_numpy_array(
        grid_edge_latitudes_deg_n,
        exact_dimensions=numpy.array([num_grid_rows + 1], dtype=int)
    )
    error_checking.assert_is_numpy_array(
        grid_edge_longitudes_deg_e,
        exact_dimensions=numpy.array([num_grid_columns + 1], dtype=int)
    )

    error_checking.assert_is_geq(min_colour_percentile, 0.)
    error_checking.assert_is_leq(min_colour_percentile, 5.)
    error_checking.assert_is_geq(max_colour_percentile, 95.)
    error_checking.assert_is_leq(max_colour_percentile, 100.)
    error_checking.assert_is_boolean(plot_satellite_subpoints)

    colour_map_object = pyplot.get_cmap(colour_map_name)

    # Convert to display units.
    conv_ratio = (
        TARGET_FIELD_TO_CONV_RATIO[target_field_name] **
        METRIC_TO_UNIT_EXPONENT[metric_name]
    )
    metric_matrix_to_plot = metric_matrix * conv_ratio

    if METRIC_TO_UNIT_EXPONENT[metric_name] == 2:
        metric_matrix_to_plot = numpy.sqrt(metric_matrix_to_plot)

    if not numpy.any(numpy.isfinite(metric_matrix_to_plot)):
        min_colour_value = 0.
        max_colour_value = 1.
    else:
        if metric_name == scalar_evaluation.BIAS_KEY:
            max_colour_value = numpy.nanpercentile(
                numpy.absolute(metric_matrix_to_plot), max_colour_percentile
            )
            max_colour_value = max([max_colour_value, TOLERANCE])
            min_colour_value = -1 * max_colour_value
        else:
            min_colour_value = numpy.percentile(
                metric_matrix_to_plot[numpy.isfinite(metric_matrix_to_plot)],
                min_colour_percentile
            )
            max_colour_value = numpy.percentile(
                metric_matrix_to_plot[numpy.isfinite(metric_matrix_to_plot)],
                max_colour_percentile
            )

            if metric_name in QUASI_SKILL_SCORE_NAMES:
                min_colour_value = max([min_colour_value, -10.])

            max_colour_value = max([
                max_colour_value, min_colour_value + TOLERANCE
            ])

    metric_matrix_to_plot = grids.latlng_field_grid_points_to_edges(
        field_matrix=metric_matrix_to_plot,
        min_latitude_deg=1., min_longitude_deg=1.,
        lat_spacing_deg=1e-6, lng_spacing_deg=1e-6
    )[0]
    metric_matrix_unmasked = metric_matrix_to_plot + 0.

    # Do actual plotting.
    metric_matrix_to_plot = numpy.ma.masked_where(
        numpy.isnan(metric_matrix_to_plot), metric_matrix_to_plot
    )
    colour_norm_object = pyplot.Normalize(
        vmin=min_colour_value, vmax=max_colour_value
    )

    try:
        axes_object.pcolormesh(
            grid_edge_longitudes_deg_e, grid_edge_latitudes_deg_n,
            metric_matrix_to_plot,
            cmap=colour_map_object, norm=colour_norm_object,
            shading='flat', edgecolors='None', zorder=-1e11
        )
    except:
        axes_object.pcolormesh(
            grid_edge_longitudes_deg_e, grid_edge_latitudes_deg_n,
            metric_matrix_to_plot[:-1, :-1],
            cmap=colour_map_object, norm=colour_norm_object,
            shading='flat', edgecolors='None', zorder=-1e11
        )

    if plot_satellite_subpoints:
        if metric_name == scalar_evaluation.BIAS_KEY:
            marker_colour = matplotlib.colors.to_rgba(
                c=numpy.array([27, 158, 119], dtype=float) / 255,
                alpha=1.
            )
        else:
            marker_colour = matplotlib.colors.to_rgba(
                c=numpy.array([217, 95, 2], dtype=float) / 255,
                alpha=1.
            )

        for cyclone_id_string in ['2021AL01', '2021EP01', '2021WP01']:
            subpoint_longitude_deg_e = (
                misc_utils.cyclone_id_to_satellite_metadata(
                    cyclone_id_string
                )[0]
            )

            axes_object.plot(
                subpoint_longitude_deg_e, 0., linestyle='None',
                marker=SATELLITE_SUBPOINT_MARKER,
                markersize=SATELLITE_SUBPOINT_MARKER_SIZE,
                markerfacecolor=marker_colour,
                markeredgecolor=marker_colour,
                markeredgewidth=0
            )

    if label_font_size is not None:
        if metric_name == scalar_evaluation.BIAS_KEY:
            median_colour_value = 0.5 * max_colour_value
        else:
            median_colour_value = 0.5 * (min_colour_value + max_colour_value)

        for i in range(num_grid_rows):
            for j in range(num_grid_columns):
                if numpy.isnan(metric_matrix_unmasked[i, j]):
                    continue

                this_string = _metric_value_to_label(
                    metric_matrix_to_plot[i, j]
                )

                if metric_name == scalar_evaluation.BIAS_KEY:
                    if (
                            numpy.absolute(metric_matrix_to_plot[i, j]) >
                            median_colour_value
                    ):
                        this_colour = numpy.full(3, 1.)
                    else:
                        this_colour = numpy.full(3, 0.)
                else:
                    if metric_matrix_to_plot[i, j] > median_colour_value:
                        this_colour = numpy.full(3, 0.)
                    else:
                        this_colour = numpy.full(3, 1.)

                axes_object.text(
                    numpy.mean(grid_edge_longitudes_deg_e[j:(j + 2)]),
                    numpy.mean(grid_edge_latitudes_deg_n[i:(i + 2)]),
                    this_string, color=this_colour, rotation=90.,
                    fontsize=label_font_size, fontweight='bold',
                    verticalalignment='center', horizontalalignment='center'
                )

    title_string = '{0:s}{1:s}{2:s}'.format(
        METRIC_TO_FANCY_NAME[metric_name][0].upper(),
        METRIC_TO_FANCY_NAME[metric_name][1:],
        TARGET_FIELD_TO_FANCY_NAME[target_field_name]
    )

    unit_exponent = METRIC_TO_UNIT_EXPONENT[metric_name]
    if unit_exponent == 1 or unit_exponent == 2:
        title_string += ' ({0:s})'.format(
            TARGET_FIELD_TO_UNIT_STRING[target_field_name]
        )

    axes_object.set_title(title_string)

    gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=metric_matrix_to_plot,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical'
    )


def plot_metric_by_2categories(
        metric_matrix, metric_name, target_field_name,
        y_category_description_strings, y_label_string,
        x_category_description_strings, x_label_string,
        colour_map_name, min_colour_percentile, max_colour_percentile,
        label_font_size=30, cbar_fraction_of_axis_length=1.):
    """Plots one evaluation metric across 2 categories (stratified evaluation).

    M = number of categories along y-axis
    N = number of categories along x-axis

    :param metric_matrix: M-by-N numpy array of metric values.
    :param metric_name: Name of error metric.
    :param target_field_name: Name of target variable.
    :param y_category_description_strings: length-M list of category
        descriptions.
    :param y_label_string: Label for entire y-axis (all categories along
        y-axis).
    :param x_category_description_strings: length-N list of category
        descriptions.
    :param x_label_string: Label for entire x-axis (all categories along
        x-axis).
    :param colour_map_name: Name of colour scheme (must be accepted by
        `matplotlib.pyplot.get_cmap`).
    :param min_colour_percentile: Determines minimum value in colour scheme.
    :param max_colour_percentile: Determines max value in colour scheme.
    :param label_font_size: Font size for text labels.  If you do not want
        labels, make this None.
    :param cbar_fraction_of_axis_length: Fraction of axis length for colour bar.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    # Check input args.
    error_checking.assert_is_numpy_array(metric_matrix, num_dimensions=2)
    error_checking.assert_is_string(y_label_string)
    error_checking.assert_is_string(x_label_string)

    num_grid_rows = metric_matrix.shape[0]
    num_grid_columns = metric_matrix.shape[1]

    error_checking.assert_is_string_list(y_category_description_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(y_category_description_strings),
        exact_dimensions=numpy.array([num_grid_rows], dtype=int)
    )

    error_checking.assert_is_string_list(x_category_description_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(x_category_description_strings),
        exact_dimensions=numpy.array([num_grid_columns], dtype=int)
    )

    error_checking.assert_is_geq(min_colour_percentile, 0.)
    error_checking.assert_is_leq(min_colour_percentile, 5.)
    error_checking.assert_is_geq(max_colour_percentile, 95.)
    error_checking.assert_is_leq(max_colour_percentile, 100.)
    error_checking.assert_is_greater(cbar_fraction_of_axis_length, 0.)
    error_checking.assert_is_leq(cbar_fraction_of_axis_length, 1.)

    colour_map_object = pyplot.get_cmap(colour_map_name)

    # Convert to display units.
    conv_ratio = (
        TARGET_FIELD_TO_CONV_RATIO[target_field_name] **
        METRIC_TO_UNIT_EXPONENT[metric_name]
    )
    metric_matrix_to_plot = metric_matrix * conv_ratio

    if METRIC_TO_UNIT_EXPONENT[metric_name] == 2:
        metric_matrix_to_plot = numpy.sqrt(metric_matrix_to_plot)

    if not numpy.any(numpy.isfinite(metric_matrix_to_plot)):
        min_colour_value = 0.
        max_colour_value = 1.
    else:
        if metric_name == scalar_evaluation.BIAS_KEY:
            max_colour_value = numpy.nanpercentile(
                numpy.absolute(metric_matrix_to_plot), max_colour_percentile
            )
            max_colour_value = max([max_colour_value, TOLERANCE])
            min_colour_value = -1 * max_colour_value
        else:
            min_colour_value = numpy.percentile(
                metric_matrix_to_plot[numpy.isfinite(metric_matrix_to_plot)],
                min_colour_percentile
            )
            max_colour_value = numpy.percentile(
                metric_matrix_to_plot[numpy.isfinite(metric_matrix_to_plot)],
                max_colour_percentile
            )

            if metric_name in QUASI_SKILL_SCORE_NAMES:
                min_colour_value = max([min_colour_value, -10.])

            max_colour_value = max([
                max_colour_value, min_colour_value + TOLERANCE
            ])

    # Do actual stuff.
    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    colour_norm_object = pyplot.Normalize(
        vmin=min_colour_value, vmax=max_colour_value
    )
    axes_object.imshow(
        metric_matrix_to_plot, origin='lower',
        cmap=colour_map_object, norm=colour_norm_object
    )

    x_coords = numpy.linspace(
        0, num_grid_columns - 1, num=num_grid_columns, dtype=float
    )
    y_coords = numpy.linspace(
        0, num_grid_rows - 1, num=num_grid_rows, dtype=float
    )

    axes_object.set_xticks(x_coords)
    axes_object.set_xticklabels(x_category_description_strings, rotation=90.)
    axes_object.set_xlabel(x_label_string)

    axes_object.set_yticks(y_coords)
    axes_object.set_yticklabels(y_category_description_strings)
    axes_object.set_ylabel(y_label_string)

    if label_font_size is not None:
        if metric_name == scalar_evaluation.BIAS_KEY:
            median_colour_value = 0.5 * max_colour_value
        else:
            median_colour_value = 0.5 * (min_colour_value + max_colour_value)

        for i in range(num_grid_rows):
            for j in range(num_grid_columns):
                if numpy.isnan(metric_matrix_to_plot[i, j]):
                    continue

                this_string = _metric_value_to_label(
                    metric_matrix_to_plot[i, j]
                )

                if metric_name == scalar_evaluation.BIAS_KEY:
                    if (
                            numpy.absolute(metric_matrix_to_plot[i, j]) >
                            median_colour_value
                    ):
                        this_colour = numpy.full(3, 1.)
                    else:
                        this_colour = numpy.full(3, 0.)
                else:
                    if metric_matrix_to_plot[i, j] > median_colour_value:
                        this_colour = numpy.full(3, 0.)
                    else:
                        this_colour = numpy.full(3, 1.)

                axes_object.text(
                    j, i, this_string, color=this_colour,
                    fontsize=label_font_size, fontweight='bold',
                    verticalalignment='center', horizontalalignment='center'
                )

    title_string = '{0:s}{1:s}{2:s}'.format(
        METRIC_TO_FANCY_NAME[metric_name][0].upper(),
        METRIC_TO_FANCY_NAME[metric_name][1:],
        TARGET_FIELD_TO_FANCY_NAME[target_field_name]
    )

    unit_exponent = METRIC_TO_UNIT_EXPONENT[metric_name]
    if unit_exponent == 1 or unit_exponent == 2:
        title_string += ' ({0:s})'.format(
            TARGET_FIELD_TO_UNIT_STRING[target_field_name]
        )

    axes_object.set_title(title_string)

    gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=metric_matrix_to_plot,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical',
        fraction_of_axis_length=cbar_fraction_of_axis_length
    )

    return figure_object, axes_object


def plot_metric_by_category(
        metric_matrix, metric_name, target_field_name,
        category_description_strings, x_label_string, confidence_level):
    """Plots one evaluation metric across categories (stratified evaluation).

    C = number of categories
    B = number of bootstrap replicates

    :param metric_matrix: C-by-B numpy array of metric values.
    :param metric_name: Name of error metric.
    :param target_field_name: Name of target variable.
    :param category_description_strings: length-C list of category descriptions.
    :param x_label_string: Label for entire x-axis (all categories).
    :param confidence_level: See documentation at top of file.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    # Check input args.
    error_checking.assert_is_numpy_array(metric_matrix, num_dimensions=2)
    error_checking.assert_is_string(x_label_string)

    num_categories = metric_matrix.shape[0]
    num_bootstrap_reps = metric_matrix.shape[1]
    category_indices = 0.5 + numpy.linspace(
        0, num_categories - 1, num=num_categories, dtype=float
    )

    error_checking.assert_is_string_list(category_description_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(category_description_strings),
        exact_dimensions=numpy.array([num_categories], dtype=int)
    )

    # Convert to display units.
    conv_ratio = (
        TARGET_FIELD_TO_CONV_RATIO[target_field_name] **
        METRIC_TO_UNIT_EXPONENT[metric_name]
    )
    metric_matrix_to_plot = metric_matrix * conv_ratio

    if METRIC_TO_UNIT_EXPONENT[metric_name] == 2:
        metric_matrix_to_plot = numpy.sqrt(metric_matrix_to_plot)

    # Do actual stuff.
    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    if (
            num_bootstrap_reps > 1 and
            not numpy.all(numpy.isnan(metric_matrix_to_plot))
    ):
        polygon_coord_matrix = misc_utils.confidence_interval_to_polygon(
            x_value_matrix=
            numpy.expand_dims(category_indices.astype(float), axis=-1),
            y_value_matrix=metric_matrix_to_plot,
            confidence_level=confidence_level,
            same_order=True
        )

        polygon_colour = matplotlib.colors.to_rgba(
            METRIC_LINE_COLOUR, POLYGON_OPACITY
        )
        patch_object = matplotlib.patches.Polygon(
            polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
        )
        axes_object.add_patch(patch_object)

    axes_object.plot(
        category_indices, numpy.nanmedian(metric_matrix_to_plot, axis=-1),
        color=METRIC_LINE_COLOUR, linewidth=METRIC_LINE_WIDTH,
        linestyle='solid',
        marker=METRIC_MARKER_TYPE, markersize=METRIC_MARKER_SIZE,
        markerfacecolor=METRIC_MARKER_COLOUR,
        markeredgecolor=METRIC_MARKER_COLOUR,
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
    if unit_exponent == 1 or unit_exponent == 2:
        title_string += ' ({0:s})'.format(
            TARGET_FIELD_TO_UNIT_STRING[target_field_name]
        )

    axes_object.set_title(title_string)

    return figure_object, axes_object


def plot_inset_histogram(
        figure_object, bin_centers, bin_counts, has_predictions,
        bar_colour=HISTOGRAM_FACE_COLOUR):
    """Plots histogram as inset in attributes diagram.

    B = number of bins

    :param figure_object: Will plot on this figure (instance of
        `matplotlib.figure.Figure`).
    :param bin_centers: length-B numpy array with value at center of each bin.
        These values will be plotted on the x-axis.
    :param bin_counts: length-B numpy array with number of examples in each bin.
        These values will be plotted on the y-axis.
    :param has_predictions: Boolean flag.  If True, histogram will contain
        prediction frequencies.  If False, will contain observation frequencies.
    :param bar_colour: Bar colour (in any format accepted by matplotlib).
    """

    error_checking.assert_is_numpy_array(bin_centers, num_dimensions=1)
    error_checking.assert_is_boolean(has_predictions)

    num_bins = len(bin_centers)
    expected_dim = numpy.array([num_bins], dtype=int)

    error_checking.assert_is_integer_numpy_array(bin_counts)
    error_checking.assert_is_geq_numpy_array(bin_counts, 0)
    error_checking.assert_is_numpy_array(
        bin_counts, exact_dimensions=expected_dim
    )

    bin_frequencies = bin_counts.astype(float) / numpy.sum(bin_counts)

    if has_predictions:
        inset_axes_object = figure_object.add_axes([0.675, 0.2, 0.2, 0.2])
    else:
        inset_axes_object = figure_object.add_axes([0.2, 0.65, 0.2, 0.2])

    fake_bin_centers = (
        0.5 + numpy.linspace(0, num_bins - 1, num=num_bins, dtype=float)
    )

    real_indices = numpy.where(numpy.invert(numpy.isnan(bin_centers)))[0]

    inset_axes_object.bar(
        fake_bin_centers[real_indices], bin_frequencies[real_indices], 1.,
        color=bar_colour, linewidth=0
    )
    inset_axes_object.set_ylim(bottom=0.)

    this_spacing = int(numpy.floor(
        (1. / 6) * len(real_indices)
    ))
    this_spacing = max([this_spacing, 1])

    tick_indices = numpy.concatenate((
        real_indices[::this_spacing], real_indices[[-1]]
    ))
    tick_indices = numpy.unique(tick_indices)
    if numpy.diff(tick_indices[-2:])[0] < 2:
        tick_indices = tick_indices[:-1]

    x_tick_values = fake_bin_centers[tick_indices]
    x_tick_labels = ['{0:.1f}'.format(b) for b in bin_centers[tick_indices]]
    inset_axes_object.set_xticks(x_tick_values)
    inset_axes_object.set_xticklabels(x_tick_labels)

    for this_tick_object in inset_axes_object.xaxis.get_major_ticks():
        this_tick_object.label.set_fontsize(HISTOGRAM_FONT_SIZE)
        this_tick_object.label.set_rotation('vertical')

    for this_tick_object in inset_axes_object.yaxis.get_major_ticks():
        this_tick_object.label.set_fontsize(HISTOGRAM_FONT_SIZE)

    inset_axes_object.set_title(
        'Prediction frequency' if has_predictions else 'Observation frequency',
        fontsize=HISTOGRAM_FONT_SIZE
    )


def plot_attributes_diagram(
        figure_object, axes_object, mean_predictions, mean_observations,
        mean_value_in_training, min_value_to_plot, max_value_to_plot,
        line_colour=RELIABILITY_LINE_COLOUR, line_style='solid',
        line_width=RELIABILITY_LINE_WIDTH, example_counts=None,
        inv_mean_observations=None, inv_example_counts=None):
    """Plots attributes diagram.

    If `example_counts is None`, will not plot histogram of predicted values.

    If `inv_mean_observations is None` and `inv_example_counts is None`, will
    not plot histogram of observed values.

    B = number of bins

    :param figure_object: Will plot on this figure (instance of
        `matplotlib.figure.Figure`).
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param mean_predictions: length-B numpy array of mean predicted values.
    :param mean_observations: length-B numpy array of mean observed values.
    :param mean_value_in_training: Mean of target variable in training data.
    :param min_value_to_plot: Minimum value in plot (for both x- and y-axes).
    :param max_value_to_plot: Max value in plot (for both x- and y-axes).
        If None, will be determined automatically.
    :param line_colour: See doc for `_plot_reliability_curve`.
    :param line_width: Same.
    :param line_style: Same.
    :param example_counts: length-B numpy array with number of examples in each
        bin.
    :param inv_mean_observations: length-B numpy array of mean observed values
        for inverted reliability curve.
    :param inv_example_counts: length-B numpy array of example counts for
        inverted reliability curve.
    :return: main_line_handle: See doc for `_plot_reliability_curve`.
    """

    # Check input args.
    error_checking.assert_is_numpy_array(mean_predictions, num_dimensions=1)

    num_bins = len(mean_predictions)
    expected_dim = numpy.array([num_bins], dtype=int)
    error_checking.assert_is_numpy_array(
        mean_observations, exact_dimensions=expected_dim
    )

    plot_prediction_histogram = example_counts is not None

    if plot_prediction_histogram:
        error_checking.assert_is_integer_numpy_array(example_counts)
        error_checking.assert_is_geq_numpy_array(example_counts, 0)
        error_checking.assert_is_numpy_array(
            example_counts, exact_dimensions=expected_dim
        )

    error_checking.assert_is_not_nan(mean_value_in_training)
    error_checking.assert_is_geq(max_value_to_plot, min_value_to_plot)
    if max_value_to_plot == min_value_to_plot:
        max_value_to_plot = min_value_to_plot + 1.

    plot_obs_histogram = not(
        inv_mean_observations is None and inv_example_counts is None
    )

    if plot_obs_histogram:
        error_checking.assert_is_numpy_array(
            inv_mean_observations, exact_dimensions=expected_dim
        )

        error_checking.assert_is_integer_numpy_array(inv_example_counts)
        error_checking.assert_is_geq_numpy_array(inv_example_counts, 0)
        error_checking.assert_is_numpy_array(
            inv_example_counts, exact_dimensions=expected_dim
        )

    _plot_attr_diagram_background(
        axes_object=axes_object, mean_value_in_training=mean_value_in_training,
        min_value_in_plot=min_value_to_plot, max_value_in_plot=max_value_to_plot
    )

    if plot_prediction_histogram:
        plot_inset_histogram(
            figure_object=figure_object, bin_centers=mean_predictions,
            bin_counts=example_counts, has_predictions=True,
            bar_colour=line_colour
        )

    if plot_obs_histogram:
        plot_inset_histogram(
            figure_object=figure_object, bin_centers=inv_mean_observations,
            bin_counts=inv_example_counts, has_predictions=False,
            bar_colour=line_colour
        )

    return _plot_reliability_curve(
        axes_object=axes_object, mean_predictions=mean_predictions,
        mean_observations=mean_observations,
        min_value_to_plot=min_value_to_plot,
        max_value_to_plot=max_value_to_plot,
        line_colour=line_colour, line_style=line_style, line_width=line_width
    )


def plot_taylor_diagram(target_stdev, prediction_stdev, correlation,
                        marker_colour, figure_object):
    """Plots Taylor diagram.

    :param target_stdev: Standard deviation of target (actual) values.
    :param prediction_stdev: Standard deviation of predicted values.
    :param correlation: Correlation between actual and predicted values.
    :param marker_colour: Colour for markers (in any format accepted by
        matplotlib).
    :param figure_object: Will plot on this figure (instance of
        `matplotlib.figure.Figure`).
    :return: taylor_diagram_object: Handle for Taylor diagram (instance of
        `taylor_diagram.TaylorDiagram`).
    """

    error_checking.assert_is_geq(target_stdev, 0.)
    error_checking.assert_is_geq(prediction_stdev, 0.)
    error_checking.assert_is_geq(correlation, -1., allow_nan=True)
    error_checking.assert_is_leq(correlation, 1., allow_nan=True)

    taylor_diagram_object = taylor_diagram.TaylorDiagram(
        refstd=target_stdev, fig=figure_object, srange=(0, 2), extend=False
    )

    target_marker_object = taylor_diagram_object.samplePoints[0]
    target_marker_object.set_marker(TAYLOR_TARGET_MARKER_TYPE)
    target_marker_object.set_markersize(TAYLOR_TARGET_MARKER_SIZE)
    target_marker_object.set_markerfacecolor(marker_colour)
    target_marker_object.set_markeredgewidth(0)

    if not numpy.isnan(correlation):
        taylor_diagram_object.add_sample(
            stddev=prediction_stdev, corrcoef=correlation
        )

        prediction_marker_object = taylor_diagram_object.samplePoints[-1]
        prediction_marker_object.set_marker(TAYLOR_PREDICTION_MARKER_TYPE)
        prediction_marker_object.set_markersize(TAYLOR_PREDICTION_MARKER_SIZE)
        prediction_marker_object.set_markerfacecolor(marker_colour)
        prediction_marker_object.set_markeredgewidth(0)

    crmse_contour_object = taylor_diagram_object.add_contours(
        levels=5, colors='0.5'
    )
    pyplot.clabel(crmse_contour_object, inline=1, fmt='%.0f')

    taylor_diagram_object.add_grid()
    taylor_diagram_object._ax.axis[:].major_ticks.set_tick_out(True)
    taylor_diagram_object._ax.axis['left'].label.set_text('')

    return taylor_diagram_object
