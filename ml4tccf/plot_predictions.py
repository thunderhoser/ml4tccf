"""Plots predictions."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import matplotlib.colors
from scipy.interpolate import interp1d, interp2d

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import gg_general_utils
import time_conversion
import longitude_conversion as lng_conversion
import file_system_utils
import error_checking
import imagemagick_utils
import prediction_io
import misc_utils
import scalar_prediction_utils
import gridded_prediction_utils
import border_io
import neural_net_utils as nn_utils
import neural_net_training_cira_ir as nn_training_cira_ir
import neural_net_training_simple as nn_training_simple
import neural_net_training_fancy as nn_training_fancy
import plotting_utils
import satellite_plotting

TOLERANCE = 1e-6
SENTINEL_VALUE = -9999.
TIME_FORMAT = '%Y-%m-%d-%H%M'

PREDICTED_CENTER_MARKER = 'o'
PREDICTED_CENTER_MARKER_COLOUR = numpy.full(3, 0.)
PREDICTED_CENTER_MARKER_SIZE = 12
PREDICTED_CENTER_MARKER_EDGE_WIDTH = 0
PREDICTED_CENTER_MARKER_EDGE_COLOUR = numpy.full(3, 0.)

ACTUAL_CENTER_MARKER = '*'
ACTUAL_CENTER_MARKER_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
ACTUAL_CENTER_MARKER_SIZE = 16
ACTUAL_CENTER_MARKER_EDGE_WIDTH = 2
ACTUAL_CENTER_MARKER_EDGE_COLOUR = numpy.full(3, 0.)

IMAGE_CENTER_MARKER = 's'
IMAGE_CENTER_MARKER_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
IMAGE_CENTER_MARKER_SIZE = 12
IMAGE_CENTER_MARKER_EDGE_WIDTH = 2
IMAGE_CENTER_MARKER_EDGE_COLOUR = numpy.full(3, 0.)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300
PANEL_SIZE_PX = int(2.5e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
SATELLITE_DIR_ARG_NAME = 'input_satellite_dir_name'
ARE_DATA_NORMALIZED_ARG_NAME = 'are_data_normalized'
PLOT_ONE_LAG_TIME_ARG_NAME = 'plot_one_lag_time'
MIN_SEMANTIC_SEG_PROB_ARG_NAME = 'min_semantic_seg_prob'
MAX_SEMANTIC_SEG_PROB_ARG_NAME = 'max_semantic_seg_prob'
MIN_SEMANTIC_SEG_PROB_PERCENTILE_ARG_NAME = 'min_semantic_seg_prob_percentile'
MAX_SEMANTIC_SEG_PROB_PERCENTILE_ARG_NAME = 'max_semantic_seg_prob_percentile'
PROB_COLOUR_MAP_ARG_NAME = 'prob_colour_map_name'
POINTS_TO_LINE_CONTOURS_ARG_NAME = 'convert_points_to_line_contours'
LINE_CONTOUR_SMOOTH_RAD_ARG_NAME = 'line_contour_smoothing_radius_px'
POINT_PREDICTION_OPACITY_ARG_NAME = 'point_prediction_opacity'
PLOT_MEAN_POINT_PREDICTION_ARG_NAME = 'plot_mean_point_prediction'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTION_FILE_HELP_STRING = (
    'Path to prediction file, containing predictions and targets for one '
    'cyclone.  Will be read by `prediction_io.read_file`.  One figure will be '
    'created for each data sample in this file.'
)
SATELLITE_DIR_HELP_STRING = (
    'Name of directory with satellite data (predictors).  Files therein will '
    'be found by `satellite_io.find_file` and read by `satellite_io.read_file`.'
)
ARE_DATA_NORMALIZED_HELP_STRING = (
    'Boolean flag.  If 1 (0), plotting code will assume that satellite data '
    'are (un)normalized.'
)
PLOT_ONE_LAG_TIME_HELP_STRING = (
    'Boolean flag.  If 1 (0), will plot predictors only at most recent lag '
    'time (at all lag times).'
)
MIN_SEMANTIC_SEG_PROB_HELP_STRING = (
    '[used only if model does semantic segmentation] Minimum probability in '
    'colour scheme.  If you want to specify colour limits with percentiles '
    'instead, leave this argument alone.'
)
MAX_SEMANTIC_SEG_PROB_HELP_STRING = 'Same as `{0:s}` but for max.'.format(
    MIN_SEMANTIC_SEG_PROB_HELP_STRING
)
MIN_SEMANTIC_SEG_PROB_PERCENTILE_HELP_STRING = (
    '[used only if model does semantic segmentation] Minimum probability in '
    'colour scheme, stated as a percentile (from 0...100) over all values in '
    'the grid.  If you want to specify colour limits with raw probabilities '
    'instead, leave this argument alone.'
)
MAX_SEMANTIC_SEG_PROB_PERCENTILE_HELP_STRING = (
    'Same as `{0:s}` but for max.'
).format(MIN_SEMANTIC_SEG_PROB_PERCENTILE_HELP_STRING)

PROB_COLOUR_MAP_HELP_STRING = (
    '[used if model does semantic segmentation or {0:s} == 1] Name of colour '
    'scheme used for probabilities.  Must be accepted by `pyplot.get_cmap`.'
).format(POINTS_TO_LINE_CONTOURS_ARG_NAME)

POINTS_TO_LINE_CONTOURS_HELP_STRING = (
    '[used only if model predicts scalar coordinates] Boolean flag.  If 1, '
    'will convert each ensemble of predictions to a probability grid, then '
    'plot line contours over the grid.  If 0, will plot each ensemble of '
    'predictions as points.'
)
LINE_CONTOUR_SMOOTH_RAD_HELP_STRING = (
    '[used only if {0:s} == 1] Smoothing radius for probability contours.  If '
    'you do not want to smooth, leave this argument alone.'
).format(POINTS_TO_LINE_CONTOURS_ARG_NAME)

POINT_PREDICTION_OPACITY_HELP_STRING = (
    '[used only if model predicts scalar coordinates and {0:s} == 0] Opacity, '
    'in range 0...1, for point predictions (black dots).'
).format(POINTS_TO_LINE_CONTOURS_ARG_NAME)

PLOT_MEAN_POINT_PREDICTION_HELP_STRING = (
    '[used only if model predicts scalar coordinates and {0:s} == 0] Boolean '
    'flag.  If 1 (0), will plot only the mean prediction (every ensemble '
    'member).'
).format(POINTS_TO_LINE_CONTOURS_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=PREDICTION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SATELLITE_DIR_ARG_NAME, type=str, required=True,
    help=SATELLITE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ARE_DATA_NORMALIZED_ARG_NAME, type=int, required=True,
    help=ARE_DATA_NORMALIZED_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_ONE_LAG_TIME_ARG_NAME, type=int, required=True,
    help=PLOT_ONE_LAG_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_SEMANTIC_SEG_PROB_ARG_NAME, type=float, required=False,
    default=1, help=MIN_SEMANTIC_SEG_PROB_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_SEMANTIC_SEG_PROB_ARG_NAME, type=float, required=False,
    default=0, help=MAX_SEMANTIC_SEG_PROB_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_SEMANTIC_SEG_PROB_PERCENTILE_ARG_NAME, type=float,
    required=False, default=100,
    help=MIN_SEMANTIC_SEG_PROB_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_SEMANTIC_SEG_PROB_PERCENTILE_ARG_NAME, type=float,
    required=False, default=0,
    help=MAX_SEMANTIC_SEG_PROB_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PROB_COLOUR_MAP_ARG_NAME, type=str, required=False,
    default='BuGn', help=PROB_COLOUR_MAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + POINTS_TO_LINE_CONTOURS_ARG_NAME, type=int, required=False,
    default=0, help=POINTS_TO_LINE_CONTOURS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LINE_CONTOUR_SMOOTH_RAD_ARG_NAME, type=int, required=False,
    default=-1, help=LINE_CONTOUR_SMOOTH_RAD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + POINT_PREDICTION_OPACITY_ARG_NAME, type=float, required=False,
    default=0.5, help=POINT_PREDICTION_OPACITY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_MEAN_POINT_PREDICTION_ARG_NAME, type=int, required=False,
    default=0, help=PLOT_MEAN_POINT_PREDICTION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _get_probability_colour_map(
        base_colour_map_name, min_value, max_value, percent_flag):
    """Returns colour scheme for gridded probabilities.

    :param base_colour_map_name: Name of base colour map (must be accepted by
        `matplotlib.pyplot.get_cmap`).
    :param min_value: Minimum probability in colour scheme.
    :param max_value: Max probability in colour scheme.
    :param percent_flag: Boolean flag.  If True, will use percentage units.  If
        False, will use 0...1 units.
    :return: colour_map_object: Colour scheme (instance of
        `matplotlib.colors.ListedColormap`).
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`,
        defining the scale of the colour map.
    """

    num_colours = 1001
    prob_values = numpy.linspace(
        min_value, max_value, num=num_colours, dtype=float
    )
    if percent_flag:
        prob_values *= 100

    opacity_values = numpy.full(num_colours, 0.8)
    opacity_values[:100] = 0.
    opacity_values[-100:] = 1.

    this_colour_map_object = pyplot.get_cmap(base_colour_map_name)
    this_colour_norm_object = matplotlib.colors.BoundaryNorm(
        prob_values, this_colour_map_object.N
    )

    rgba_matrix = this_colour_map_object(this_colour_norm_object(prob_values))
    colour_list = [
        matplotlib.colors.to_rgba(
            c=rgba_matrix[i, ..., :-1], alpha=opacity_values[i]
        )
        for i in range(num_colours)
    ]

    colour_map_object = matplotlib.colors.ListedColormap(colour_list)
    colour_map_object.set_under(
        matplotlib.colors.to_rgba(c=numpy.full(3, 1.), alpha=0.)
    )
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        prob_values, colour_map_object.N
    )

    return colour_map_object, colour_norm_object


def _plot_data_one_channel(
        predictor_matrix, are_data_normalized, plotting_brightness_temp,
        border_latitudes_deg_n, border_longitudes_deg_e,
        predictor_grid_latitudes_deg_n, predictor_grid_longitudes_deg_e,
        actual_center_x_coord, actual_center_y_coord, coord_transform_string,
        probability_matrix,
        prediction_grid_latitudes_deg_n, prediction_grid_longitudes_deg_e,
        prob_colour_map_object, prob_colour_norm_object, prob_contour_levels,
        predicted_x_coords, predicted_y_coords, prediction_opacity,
        title_string, output_file_name):
    """Plots satellite data and predictions for one channel.

    One channel = one wavelength at one lag time

    M = number of rows in predictor grid
    N = number of columns in predictor grid
    P = number of points in border set
    m = number of rows in prediction grid
    n = number of columns in prediction grid
    S = ensemble size

    If plotting gridded predictions, you will need the following input args:

    - probability_matrix
    - prediction_grid_latitudes_deg_n
    - prediction_grid_longitudes_deg_e
    - prob_colour_map_object
    - prob_colour_norm_object
    - prob_contour_levels if you want line contours

    If plotting point predictions, you will need the following input args:

    - predicted_x_coords
    - predicted_y_coords
    - prediction_opacity

    :param predictor_matrix: M-by-N numpy array of predictor values.
    :param are_data_normalized: Boolean flag.
    :param plotting_brightness_temp: Boolean flag.
    :param border_latitudes_deg_n: length-P numpy array of latitudes (deg
        north).
    :param border_longitudes_deg_e: length-P numpy array of longitudes (deg
        east).
    :param predictor_grid_latitudes_deg_n: numpy array of latitudes (deg north).
        Array shape is length-M if regular grid, M-by-N otherwise.
    :param predictor_grid_longitudes_deg_e: numpy array of longitudes (deg
        east).  Array shape is length-N if regular grid, M-by-N otherwise.
    :param actual_center_x_coord: x-coordinate of actual TC center.
    :param actual_center_y_coord: y-coordinate of actual TC center.
    :param coord_transform_string: Either "transAxes" or "transData".
    :param probability_matrix: m-by-n numpy array of probabilities.
    :param prediction_grid_latitudes_deg_n: numpy array of latitudes (deg
        north).  Array shape is length-m if regular grid, m-by-n otherwise.
    :param prediction_grid_longitudes_deg_e: numpy array of longitudes (deg
        east).  Array shape is length-n if regular grid, m-by-n otherwise.
    :param prob_colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :param prob_colour_norm_object: Colour-normalizer (maps from data space to
        colour-bar space, which goes from 0...1).  This is an instance of
        `matplotlib.colors.Normalize`.
    :param prob_contour_levels: 1-D numpy array of probabilities for which to
        plot contour lines.
    :param predicted_x_coords: length-S numpy array of x-coordinates.
    :param predicted_y_coords: length-S numpy array of y-coordinates.
    :param prediction_opacity: Opacity (from 0...1) for point predictions.
    :param title_string: Title.
    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    if are_data_normalized:
        colour_map_object = pyplot.get_cmap('seismic', lut=1001)
        colour_norm_object = matplotlib.colors.Normalize(vmin=-3., vmax=3.)
    else:
        colour_map_object, colour_norm_object = (
            satellite_plotting.get_colour_scheme_for_bdrf()
        )

    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object
    )

    plot_filled_contours = (
        probability_matrix is not None and prob_contour_levels is None
    )
    plot_line_contours = (
        probability_matrix is not None and prob_contour_levels is not None
    )

    satellite_plotting.plot_2d_grid_latlng(
        data_matrix=predictor_matrix,
        axes_object=axes_object,
        latitude_array_deg_n=predictor_grid_latitudes_deg_n,
        longitude_array_deg_e=predictor_grid_longitudes_deg_e,
        plotting_brightness_temp=plotting_brightness_temp,
        cbar_orientation_string=None,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        opacity=0.5 if plot_filled_contours else 1.
    )

    if plot_filled_contours:
        satellite_plotting.plot_2d_grid_latlng(
            data_matrix=probability_matrix,
            axes_object=axes_object,
            latitude_array_deg_n=prediction_grid_latitudes_deg_n,
            longitude_array_deg_e=prediction_grid_longitudes_deg_e,
            plotting_brightness_temp=False,
            cbar_orientation_string=None,
            colour_map_object=prob_colour_map_object,
            colour_norm_object=prob_colour_norm_object,
            use_contourf=True
        )

    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=numpy.ravel(predictor_grid_latitudes_deg_n),
        plot_longitudes_deg_e=numpy.ravel(predictor_grid_longitudes_deg_e),
        axes_object=axes_object,
        parallel_spacing_deg=2., meridian_spacing_deg=2.
    )

    coord_transform_object = (
        axes_object.transAxes if coord_transform_string == 'transAxes'
        else axes_object.transData
    )
    axes_object.plot(
        actual_center_x_coord, actual_center_y_coord, linestyle='None',
        marker=ACTUAL_CENTER_MARKER,
        markersize=ACTUAL_CENTER_MARKER_SIZE,
        markerfacecolor=ACTUAL_CENTER_MARKER_COLOUR,
        markeredgecolor=ACTUAL_CENTER_MARKER_EDGE_COLOUR,
        markeredgewidth=ACTUAL_CENTER_MARKER_EDGE_WIDTH,
        transform=coord_transform_object, zorder=3e12
    )

    axes_object.plot(
        0.5, 0.5, linestyle='None',
        marker=IMAGE_CENTER_MARKER, markersize=IMAGE_CENTER_MARKER_SIZE,
        markerfacecolor=IMAGE_CENTER_MARKER_COLOUR,
        markeredgecolor=IMAGE_CENTER_MARKER_EDGE_COLOUR,
        markeredgewidth=IMAGE_CENTER_MARKER_EDGE_WIDTH,
        transform=axes_object.transAxes, zorder=2e12
    )

    if plot_line_contours:
        axes_object.contour(
            prediction_grid_longitudes_deg_e, prediction_grid_latitudes_deg_n,
            probability_matrix, prob_contour_levels,
            cmap=prob_colour_map_object, norm=prob_colour_norm_object,
            linewidths=4, linestyles='solid', zorder=1e12
        )
    elif not plot_filled_contours:
        ensemble_size = len(predicted_x_coords)
        this_colour = matplotlib.colors.to_rgba(
            c=PREDICTED_CENTER_MARKER_COLOUR, alpha=prediction_opacity
        )

        for k in range(ensemble_size):
            axes_object.plot(
                predicted_x_coords[k], predicted_y_coords[k], linestyle='None',
                marker=PREDICTED_CENTER_MARKER,
                markersize=PREDICTED_CENTER_MARKER_SIZE,
                markerfacecolor=this_colour,
                markeredgecolor=PREDICTED_CENTER_MARKER_EDGE_COLOUR,
                markeredgewidth=PREDICTED_CENTER_MARKER_EDGE_WIDTH,
                transform=coord_transform_object, zorder=1e10
            )

    axes_object.set_title(title_string)
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name,
        dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.resize_image(
        input_file_name=output_file_name, output_file_name=output_file_name,
        output_size_pixels=PANEL_SIZE_PX
    )


def _plot_data_one_example(
        predictor_matrices, scalar_target_values, prediction_matrix,
        model_metadata_dict, cyclone_id_string,
        low_res_latitudes_deg_n, low_res_longitudes_deg_e,
        high_res_latitudes_deg_n, high_res_longitudes_deg_e,
        are_data_normalized, border_latitudes_deg_n, border_longitudes_deg_e,
        output_file_name, prob_colour_map_name=None,
        min_semantic_seg_prob=None, max_semantic_seg_prob=None,
        convert_points_to_line_contours=False,
        line_contour_smoothing_radius_px=None,
        point_prediction_opacity=None):
    """Plots satellite data for one example.

    P = number of points in border set
    S = ensemble size
    M = number of rows
    N = number of columns

    :param predictor_matrices: Same as output from `nn_training*.create_data`
        but without first axis.
    :param scalar_target_values: Same as output from `nn_training*.create_data`
        but without first axis.
    :param prediction_matrix: If predictions are scalar...

        2-by-S numpy array of predictions.  prediction_matrix[0, :] contains
        predicted row positions of TC centers, and prediction_matrix[1, :]
        contains predicted column positions of TC centers.

    If predictions are gridded...

    M-by-N numpy array of predicted probabilities.

    :param model_metadata_dict: Dictionary with model metadata, in format
        returned by `nn_utils.read_metafile`.
    :param cyclone_id_string: Cyclone ID.
    :param low_res_latitudes_deg_n: Same as output from
        `nn_training*.create_data` but without first or last axis.
    :param low_res_longitudes_deg_e: Same as output from
        `nn_training*.create_data` but without first or last axis.
    :param high_res_latitudes_deg_n: Same as output from
        `nn_training*.create_data` but without first or last axis.
    :param high_res_longitudes_deg_e: Same as output from
        `nn_training*.create_data` but without first or last axis.
    :param are_data_normalized: See documentation at top of file.
    :param border_latitudes_deg_n: length-P numpy array of latitudes
        (deg north).  If None, will plot without coords.
    :param border_longitudes_deg_e: length-P numpy array of longitudes
        (deg east).  If None, will plot without coords.
    :param output_file_name: Path to output file.  Figure will be saved here.
    :param prob_colour_map_name: : Name of base colour map for
        probabilities (must be accepted by `matplotlib.pyplot.get_cmap`).
    :param min_semantic_seg_prob: Minimum probability in colour scheme.
    :param max_semantic_seg_prob: Max probability in colour scheme.
    :param convert_points_to_line_contours: See documentation at top of file.
    :param line_contour_smoothing_radius_px: Same.
    :param point_prediction_opacity: Same.
    """

    # Housekeeping for coordinates.
    low_res_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
        low_res_longitudes_deg_e
    )
    longitude_range_deg = (
        numpy.max(low_res_longitudes_deg_e) -
        numpy.min(low_res_longitudes_deg_e)
    )
    if longitude_range_deg > 100:
        low_res_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
            low_res_longitudes_deg_e
        )

    brightness_temp_matrix = nn_utils.get_low_res_data_from_predictors(
        predictor_matrices
    )
    bidirectional_reflectance_matrix = (
        nn_utils.get_high_res_data_from_predictors(predictor_matrices)
    )

    num_grid_rows_low_res = brightness_temp_matrix.shape[0]
    num_grid_columns_low_res = brightness_temp_matrix.shape[1]
    regular_grids = len(low_res_latitudes_deg_n.shape) == 1

    center_row_index_low_res = int(numpy.round(
        float(num_grid_rows_low_res) / 2 - 1
    ))
    center_column_index_low_res = int(numpy.round(
        float(num_grid_columns_low_res) / 2 - 1
    ))

    row_indices_low_res = numpy.linspace(
        0, num_grid_rows_low_res - 1, num=num_grid_rows_low_res, dtype=float
    )
    column_indices_low_res = numpy.linspace(
        0, num_grid_columns_low_res - 1, num=num_grid_columns_low_res,
        dtype=float
    )

    # TODO(thunderhoser): This will not handle wrap-around at International Date
    # Line.
    if regular_grids:
        low_res_latitude_interp_object = interp1d(
            x=row_indices_low_res, y=low_res_latitudes_deg_n, kind='linear',
            bounds_error=True
        )
        low_res_longitude_interp_object = interp1d(
            x=column_indices_low_res, y=low_res_longitudes_deg_e, kind='linear',
            bounds_error=True
        )
    else:
        low_res_latitude_interp_object = interp2d(
            x=column_indices_low_res, y=row_indices_low_res,
            z=low_res_latitudes_deg_n, kind='linear', bounds_error=True
        )
        low_res_longitude_interp_object = interp2d(
            x=column_indices_low_res, y=row_indices_low_res,
            z=low_res_longitudes_deg_e, kind='linear', bounds_error=True
        )

    # Housekeeping for prediction format (points, filled contours, or line
    # contours).
    plot_filled_contours = prediction_matrix.shape[0] > 2
    ensemble_size = 1 if plot_filled_contours else prediction_matrix.shape[1]

    probability_matrix = None
    prob_colour_map_object = None
    prob_colour_norm_object = None
    prob_contour_levels = None

    if plot_filled_contours:
        prob_colour_map_object, prob_colour_norm_object = (
            _get_probability_colour_map(
                base_colour_map_name=prob_colour_map_name,
                min_value=min_semantic_seg_prob,
                max_value=max_semantic_seg_prob,
                percent_flag=False
            )
        )

        convert_points_to_line_contours = False
    elif convert_points_to_line_contours:
        if regular_grids:
            point_latitudes_deg_n = numpy.array([
                low_res_latitude_interp_object(
                    center_row_index_low_res + prediction_matrix[0, k]
                )
                for k in range(ensemble_size)
            ])

            point_longitudes_deg_e = numpy.array([
                low_res_longitude_interp_object(
                    center_column_index_low_res + prediction_matrix[1, k]
                )
                for k in range(ensemble_size)
            ])
        else:
            point_latitudes_deg_n = numpy.array([
                low_res_latitude_interp_object(
                    center_column_index_low_res + prediction_matrix[1, k],
                    center_row_index_low_res + prediction_matrix[0, k]
                )
                for k in range(ensemble_size)
            ])

            point_longitudes_deg_e = numpy.array([
                low_res_longitude_interp_object(
                    center_column_index_low_res + prediction_matrix[1, k],
                    center_row_index_low_res + prediction_matrix[0, k]
                )
                for k in range(ensemble_size)
            ])

        probability_matrix = misc_utils.latlng_points_to_probability_grid(
            point_latitudes_deg_n=point_latitudes_deg_n,
            point_longitudes_deg_e=point_longitudes_deg_e,
            grid_latitude_array_deg_n=low_res_latitudes_deg_n,
            grid_longitude_array_deg_e=low_res_longitudes_deg_e
        )

        prob_colour_map_object = pyplot.get_cmap(prob_colour_map_name)

        if line_contour_smoothing_radius_px is not None:
            print((
                'Applying Gaussian smoother with {0:d}-pixel e-folding radius '
                'to newly gridded probabilities...'
            ).format(
                2 * line_contour_smoothing_radius_px + 1
            ))

            probability_matrix = gg_general_utils.apply_gaussian_filter(
                input_matrix=probability_matrix,
                e_folding_radius_grid_cells=line_contour_smoothing_radius_px
            )
            probability_matrix = (
                probability_matrix / numpy.sum(probability_matrix)
            )

        if numpy.any(probability_matrix > TOLERANCE):
            min_colour_value = numpy.min(
                probability_matrix[probability_matrix > TOLERANCE]
            )
        else:
            min_colour_value = 0.

        max_colour_value = numpy.max(probability_matrix)
        prob_colour_norm_object = pyplot.Normalize(
            vmin=min_colour_value, vmax=max_colour_value
        )
        prob_contour_levels = numpy.linspace(
            min_colour_value, max_colour_value, num=10
        )

    plot_line_contours = convert_points_to_line_contours

    validation_option_dict = model_metadata_dict[
        nn_utils.VALIDATION_OPTIONS_KEY
    ]
    d = validation_option_dict
    high_res_wavelengths_microns = d[nn_utils.HIGH_RES_WAVELENGTHS_KEY]
    low_res_wavelengths_microns = d[nn_utils.LOW_RES_WAVELENGTHS_KEY]
    lag_times_minutes = d[nn_utils.LAG_TIMES_KEY]

    # brightness_temp_matrix = nn_utils.separate_lag_times_and_wavelengths(
    #     satellite_data_matrix=numpy.expand_dims(brightness_temp_matrix, axis=0),
    #     num_lag_times=len(lag_times_minutes)
    # )[0, ...]
    #
    # bidirectional_reflectance_matrix = (
    #     nn_utils.separate_lag_times_and_wavelengths(
    #         satellite_data_matrix=
    #         numpy.expand_dims(bidirectional_reflectance_matrix, axis=0),
    #         num_lag_times=len(lag_times_minutes)
    #     )[0, ...]
    # )

    if regular_grids:
        actual_center_x_coord = (
            0.5 + scalar_target_values[1] / num_grid_columns_low_res
        )
        actual_center_y_coord = (
            0.5 + scalar_target_values[0] / num_grid_rows_low_res
        )
        coord_transform_string = 'transAxes'
    else:
        actual_center_y_coord = low_res_latitude_interp_object(
            center_column_index_low_res + scalar_target_values[1],
            center_row_index_low_res + scalar_target_values[0]
        )
        actual_center_x_coord = low_res_longitude_interp_object(
            center_column_index_low_res + scalar_target_values[1],
            center_row_index_low_res + scalar_target_values[0]
        )
        coord_transform_string = 'transData'

    if plot_line_contours or plot_filled_contours:
        predicted_x_coords = None
        predicted_y_coords = None
    else:
        predicted_x_coords = numpy.full(ensemble_size, numpy.nan)
        predicted_y_coords = numpy.full(ensemble_size, numpy.nan)

        for k in range(ensemble_size):
            if regular_grids:
                predicted_x_coords[k] = (
                    0.5 + prediction_matrix[1, k] / num_grid_columns_low_res
                )
                predicted_y_coords[k] = (
                    0.5 + prediction_matrix[0, k] / num_grid_rows_low_res
                )
            else:
                predicted_y_coords[k] = low_res_latitude_interp_object(
                    center_column_index_low_res +
                    prediction_matrix[1, k],
                    center_row_index_low_res + prediction_matrix[0, k]
                )
                predicted_x_coords[k] = low_res_longitude_interp_object(
                    center_column_index_low_res +
                    prediction_matrix[1, k],
                    center_row_index_low_res + prediction_matrix[0, k]
                )

    num_lag_times = len(lag_times_minutes)
    num_high_res_wavelengths = len(high_res_wavelengths_microns)
    num_low_res_wavelengths = len(low_res_wavelengths_microns)

    high_res_panel_file_name_matrix = numpy.full(
        (num_lag_times, num_high_res_wavelengths), '', dtype=object
    )
    low_res_panel_file_name_matrix = numpy.full(
        (num_lag_times, num_low_res_wavelengths), '', dtype=object
    )

    for i in range(num_lag_times):
        for j in range(num_high_res_wavelengths):
            title_string = (
                '{0:.3f}-micron BDRF for {1:s} at {2:d}-minute lag'
            ).format(
                high_res_wavelengths_microns[j],
                cyclone_id_string,
                int(numpy.round(lag_times_minutes[i]))
            )

            high_res_panel_file_name_matrix[i, j] = (
                '{0:s}_{1:04d}minutes_{2:06.3f}microns.{3:s}'
            ).format(
                os.path.splitext(output_file_name)[0],
                int(numpy.round(lag_times_minutes[i])),
                high_res_wavelengths_microns[j],
                'png' if plot_filled_contours else 'jpg'
            )

            _plot_data_one_channel(
                predictor_matrix=bidirectional_reflectance_matrix[..., i, j],
                are_data_normalized=are_data_normalized,
                plotting_brightness_temp=False,
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                predictor_grid_latitudes_deg_n=high_res_latitudes_deg_n,
                predictor_grid_longitudes_deg_e=high_res_longitudes_deg_e,
                actual_center_x_coord=actual_center_x_coord,
                actual_center_y_coord=actual_center_y_coord,
                coord_transform_string=coord_transform_string,
                probability_matrix=probability_matrix,
                prediction_grid_latitudes_deg_n=low_res_latitudes_deg_n,
                prediction_grid_longitudes_deg_e=low_res_longitudes_deg_e,
                prob_colour_map_object=prob_colour_map_object,
                prob_colour_norm_object=prob_colour_norm_object,
                prob_contour_levels=prob_contour_levels,
                predicted_x_coords=predicted_x_coords,
                predicted_y_coords=predicted_y_coords,
                prediction_opacity=point_prediction_opacity,
                title_string=title_string,
                output_file_name=high_res_panel_file_name_matrix[i, j]
            )

    for i in range(num_lag_times):
        for j in range(num_low_res_wavelengths):
            title_string = (
                r'{0:.3f}-micron $T_b$ for {1:s} at {2:d}-minute lag'
            ).format(
                low_res_wavelengths_microns[j],
                cyclone_id_string,
                int(numpy.round(lag_times_minutes[i]))
            )

            low_res_panel_file_name_matrix[i, j] = (
                '{0:s}_{1:04d}minutes_{2:06.3f}microns.{3:s}'
            ).format(
                os.path.splitext(output_file_name)[0],
                int(numpy.round(lag_times_minutes[i])),
                low_res_wavelengths_microns[j],
                'png' if plot_filled_contours else 'jpg'
            )

            _plot_data_one_channel(
                predictor_matrix=brightness_temp_matrix[..., i, j],
                are_data_normalized=are_data_normalized,
                plotting_brightness_temp=True,
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                predictor_grid_latitudes_deg_n=low_res_latitudes_deg_n,
                predictor_grid_longitudes_deg_e=low_res_longitudes_deg_e,
                actual_center_x_coord=actual_center_x_coord,
                actual_center_y_coord=actual_center_y_coord,
                coord_transform_string=coord_transform_string,
                probability_matrix=probability_matrix,
                prediction_grid_latitudes_deg_n=low_res_latitudes_deg_n,
                prediction_grid_longitudes_deg_e=low_res_longitudes_deg_e,
                prob_colour_map_object=prob_colour_map_object,
                prob_colour_norm_object=prob_colour_norm_object,
                prob_contour_levels=prob_contour_levels,
                predicted_x_coords=predicted_x_coords,
                predicted_y_coords=predicted_y_coords,
                prediction_opacity=point_prediction_opacity,
                title_string=title_string,
                output_file_name=low_res_panel_file_name_matrix[i, j]
            )

    num_panels_total = num_lag_times * (
        num_high_res_wavelengths + num_low_res_wavelengths
    )
    concat_figure_file_names = []

    if num_panels_total > 15:
        for i in range(num_lag_times):
            these_file_names = numpy.concatenate((
                high_res_panel_file_name_matrix[i, :],
                low_res_panel_file_name_matrix[i, :]
            )).tolist()

            concat_figure_file_names.append(
                '{0:s}_{1:04d}minute-lag.{2:s}'.format(
                    os.path.splitext(output_file_name)[0],
                    int(numpy.round(lag_times_minutes[i])),
                    'png' if plot_filled_contours else 'jpg'
                )
            )

            plotting_utils.concat_panels(
                panel_file_names=these_file_names,
                concat_figure_file_name=concat_figure_file_names[-1]
            )
    else:
        these_file_names = []

        for i in range(num_lag_times):
            these_file_names += numpy.concatenate((
                high_res_panel_file_name_matrix[i, :],
                low_res_panel_file_name_matrix[i, :]
            )).tolist()

        concat_figure_file_names.append(output_file_name)
        plotting_utils.concat_panels(
            panel_file_names=these_file_names,
            num_panel_rows=num_lag_times,
            concat_figure_file_name=concat_figure_file_names[-1]
        )

    for this_figure_file_name in concat_figure_file_names:
        if num_high_res_wavelengths > 0:
            if are_data_normalized:
                colour_map_object = pyplot.get_cmap('seismic', lut=1001)
                colour_norm_object = matplotlib.colors.Normalize(
                    vmin=-3., vmax=3.
                )
            else:
                colour_map_object, colour_norm_object = (
                    satellite_plotting.get_colour_scheme_for_bdrf()
                )

            plotting_utils.add_colour_bar(
                figure_file_name=this_figure_file_name,
                colour_map_object=colour_map_object,
                colour_norm_object=colour_norm_object,
                orientation_string='vertical', font_size=20,
                cbar_label_string=(
                    r'BDRF ($z$-score)' if are_data_normalized
                    else 'BDRF (unitless)'
                ),
                tick_label_format_string='{0:.2f}', log_space=False,
                temporary_cbar_file_name='{0:s}_cbar.{1:s}'.format(
                    this_figure_file_name[:-4],
                    'png' if plot_filled_contours else 'jpg'
                )
            )

        if are_data_normalized:
            colour_map_object = pyplot.get_cmap('seismic', lut=1001)
            colour_norm_object = matplotlib.colors.Normalize(vmin=-3., vmax=3.)
        else:
            colour_map_object, colour_norm_object = (
                satellite_plotting.get_colour_scheme_for_brightness_temp()
            )

        plotting_utils.add_colour_bar(
            figure_file_name=this_figure_file_name,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            orientation_string='vertical', font_size=20,
            cbar_label_string=(
                r'$T_b$ ($z-score$)' if are_data_normalized
                else r'$T_b$ (Kelvins)'
            ),
            tick_label_format_string='{0:.0f}', log_space=False,
            temporary_cbar_file_name='{0:s}_cbar.{1:s}'.format(
                this_figure_file_name[:-4],
                'png' if plot_filled_contours else 'jpg'
            )
        )

        if plot_filled_contours:
            prob_colour_map_object, prob_colour_norm_object = (
                _get_probability_colour_map(
                    base_colour_map_name=prob_colour_map_name,
                    min_value=min_semantic_seg_prob,
                    max_value=max_semantic_seg_prob,
                    percent_flag=True
                )
            )

            plotting_utils.add_colour_bar(
                figure_file_name=this_figure_file_name,
                colour_map_object=prob_colour_map_object,
                colour_norm_object=prob_colour_norm_object,
                orientation_string='vertical', font_size=20,
                cbar_label_string='Probability (%)',
                tick_label_format_string='{0:.2f}', log_space=False,
                temporary_cbar_file_name='{0:s}_cbar.{1:s}'.format(
                    this_figure_file_name[:-4],
                    'png' if plot_filled_contours else 'jpg'
                )
            )


def _run(prediction_file_name, satellite_dir_name,
         are_data_normalized, plot_one_lag_time, prob_colour_map_name,
         min_semantic_seg_prob, max_semantic_seg_prob,
         min_semantic_seg_prob_percentile, max_semantic_seg_prob_percentile,
         convert_points_to_line_contours, line_contour_smoothing_radius_px,
         point_prediction_opacity, plot_mean_point_prediction,
         output_dir_name):
    """Plots predictions.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param satellite_dir_name: Same.
    :param are_data_normalized: Same.
    :param plot_one_lag_time: Same.
    :param prob_colour_map_name: Same.
    :param min_semantic_seg_prob: Same.
    :param max_semantic_seg_prob: Same.
    :param min_semantic_seg_prob_percentile: Same.
    :param max_semantic_seg_prob_percentile: Same.
    :param convert_points_to_line_contours: Same.
    :param line_contour_smoothing_radius_px: Same.
    :param point_prediction_opacity: Same.
    :param plot_mean_point_prediction: Same.
    :param output_dir_name: Same.
    """

    # Check and process input args.
    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_table_xarray = prediction_io.read_file(prediction_file_name)

    semantic_seg_flag = (
        scalar_prediction_utils.PREDICTED_ROW_OFFSET_KEY
        not in prediction_table_xarray
    )

    if semantic_seg_flag:
        if min_semantic_seg_prob >= max_semantic_seg_prob:
            min_semantic_seg_prob = None
            max_semantic_seg_prob = None

            error_checking.assert_is_geq(min_semantic_seg_prob_percentile, 50.)
            error_checking.assert_is_leq(max_semantic_seg_prob_percentile, 100.)
            error_checking.assert_is_greater(
                max_semantic_seg_prob_percentile,
                min_semantic_seg_prob_percentile
            )
        else:
            error_checking.assert_is_greater(min_semantic_seg_prob, 0.)
            error_checking.assert_is_greater(max_semantic_seg_prob, 0.)
    else:
        pt = prediction_table_xarray
        ensemble_size = len(
            pt.coords[scalar_prediction_utils.ENSEMBLE_MEMBER_DIM_KEY].values
        )
        convert_points_to_line_contours = (
            convert_points_to_line_contours and ensemble_size > 1
        )

        if line_contour_smoothing_radius_px < 1:
            line_contour_smoothing_radius_px = None

    error_checking.assert_is_greater(point_prediction_opacity, 0.)
    error_checking.assert_is_leq(point_prediction_opacity, 1.)

    # Read model metadata.
    model_file_name = (
        prediction_table_xarray.attrs[scalar_prediction_utils.MODEL_FILE_KEY]
    )
    model_metafile_name = nn_utils.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = nn_utils.read_metafile(model_metafile_name)
    validation_option_dict = (
        model_metadata_dict[nn_utils.VALIDATION_OPTIONS_KEY]
    )
    validation_option_dict[nn_utils.SATELLITE_DIRECTORY_KEY] = (
        satellite_dir_name
    )
    validation_option_dict[nn_utils.SENTINEL_VALUE_KEY] = SENTINEL_VALUE

    # TODO(thunderhoser): This will not work if I ever have multiple cyclones in
    # one prediction file.
    pt = prediction_table_xarray
    cyclone_id_string = (
        pt[scalar_prediction_utils.CYCLONE_ID_KEY].values[0].decode('utf-8')
    )
    target_times_unix_sec = pt[scalar_prediction_utils.TARGET_TIME_KEY].values

    # Find actual TC centers.
    if semantic_seg_flag:
        num_examples = len(target_times_unix_sec)
        actual_row_offsets = numpy.full(num_examples, 0, dtype=int)
        actual_column_offsets = numpy.full(num_examples, 0, dtype=int)

        for i in range(num_examples):
            (
                actual_row_offsets[i], actual_column_offsets[i]
            ) = misc_utils.target_matrix_to_centroid(
                pt[gridded_prediction_utils.TARGET_MATRIX_KEY].values[i, ...]
            )
    else:
        actual_row_offsets = numpy.round(
            pt[scalar_prediction_utils.ACTUAL_ROW_OFFSET_KEY].values
        ).astype(int)

        actual_column_offsets = numpy.round(
            pt[scalar_prediction_utils.ACTUAL_COLUMN_OFFSET_KEY].values
        ).astype(int)

    validation_option_dict[nn_utils.SEMANTIC_SEG_FLAG_KEY] = False
    if plot_one_lag_time:
        validation_option_dict[nn_utils.LAG_TIMES_KEY] = numpy.array([
            numpy.min(validation_option_dict[nn_utils.LAG_TIMES_KEY])
        ])

    model_metadata_dict[nn_utils.VALIDATION_OPTIONS_KEY] = (
        validation_option_dict
    )

    # Read predictor data.
    data_type_string = model_metadata_dict[nn_utils.DATA_TYPE_KEY]

    if data_type_string == nn_utils.CIRA_IR_DATA_TYPE_STRING:
        data_dict = nn_training_cira_ir.create_data_specific_trans(
            option_dict=validation_option_dict,
            cyclone_id_string=cyclone_id_string,
            target_times_unix_sec=target_times_unix_sec,
            row_translations_low_res_px=actual_row_offsets,
            column_translations_low_res_px=actual_column_offsets
        )
    elif data_type_string == nn_utils.RG_SIMPLE_DATA_TYPE_STRING:
        data_dict = nn_training_simple.create_data_specific_trans(
            option_dict=validation_option_dict,
            cyclone_id_string=cyclone_id_string,
            target_times_unix_sec=target_times_unix_sec,
            row_translations_low_res_px=actual_row_offsets,
            column_translations_low_res_px=actual_column_offsets
        )
    else:
        data_dict = nn_training_fancy.create_data_specific_trans(
            option_dict=validation_option_dict,
            cyclone_id_string=cyclone_id_string,
            target_times_unix_sec=target_times_unix_sec,
            row_translations_low_res_px=actual_row_offsets,
            column_translations_low_res_px=actual_column_offsets
        )

    if data_dict is None:
        return

    predictor_matrices = data_dict[nn_utils.PREDICTOR_MATRICES_KEY]
    scalar_target_matrix = data_dict[nn_utils.TARGET_MATRIX_KEY]
    low_res_latitude_matrix_deg_n = data_dict[nn_utils.LOW_RES_LATITUDES_KEY]
    low_res_longitude_matrix_deg_e = (
        data_dict[nn_utils.LOW_RES_LONGITUDES_KEY]
    )
    high_res_latitude_matrix_deg_n = (
        data_dict[nn_utils.HIGH_RES_LATITUDES_KEY]
    )
    high_res_longitude_matrix_deg_e = (
        data_dict[nn_utils.HIGH_RES_LONGITUDES_KEY]
    )

    for k in range(len(predictor_matrices)):
        predictor_matrices[k] = predictor_matrices[k].astype(numpy.float64)
        predictor_matrices[k][predictor_matrices[k] < SENTINEL_VALUE + 1] = (
            numpy.nan
        )

    # Do the plotting.
    if semantic_seg_flag:
        pt = gridded_prediction_utils.get_ensemble_mean(pt)
        prediction_matrix = (
            pt[gridded_prediction_utils.PREDICTION_MATRIX_KEY].values[..., 0]
        )
    else:
        prediction_matrix = numpy.stack((
            pt[scalar_prediction_utils.PREDICTED_ROW_OFFSET_KEY].values,
            pt[scalar_prediction_utils.PREDICTED_COLUMN_OFFSET_KEY].values
        ), axis=-2)

        if plot_mean_point_prediction:
            prediction_matrix = numpy.mean(
                prediction_matrix, axis=-1, keepdims=True
            )

    num_examples = predictor_matrices[0].shape[0]
    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    for i in range(num_examples):
        same_time_indices = numpy.where(
            target_times_unix_sec == target_times_unix_sec[i]
        )[0]
        this_index = numpy.where(same_time_indices == i)[0][0]

        output_file_name = '{0:s}/{1:s}_{2:s}_{3:03d}th.{4:s}'.format(
            output_dir_name,
            cyclone_id_string,
            time_conversion.unix_sec_to_string(
                target_times_unix_sec[i], TIME_FORMAT
            ),
            this_index,
            'png' if semantic_seg_flag else 'jpg'
        )

        if semantic_seg_flag:
            if min_semantic_seg_prob is None:
                this_min_prob = numpy.percentile(
                    prediction_matrix[i, ...], min_semantic_seg_prob_percentile
                )
                this_max_prob = numpy.percentile(
                    prediction_matrix[i, ...], max_semantic_seg_prob_percentile
                )

                if this_max_prob - this_min_prob < 0.01:
                    new_max = this_min_prob + 0.01
                    if new_max > 1:
                        this_min_prob = this_max_prob - 0.01
                    else:
                        this_max_prob = this_min_prob + 0.01
            else:
                this_min_prob = min_semantic_seg_prob + 0.
                this_max_prob = max_semantic_seg_prob + 0.
        else:
            this_min_prob = None
            this_max_prob = None

        _plot_data_one_example(
            predictor_matrices=[p[i, ...] for p in predictor_matrices],
            scalar_target_values=scalar_target_matrix[i, ...],
            prediction_matrix=prediction_matrix[i, ...],
            model_metadata_dict=model_metadata_dict,
            cyclone_id_string=cyclone_id_string,
            low_res_latitudes_deg_n=low_res_latitude_matrix_deg_n[i, ..., 0],
            low_res_longitudes_deg_e=low_res_longitude_matrix_deg_e[i, ..., 0],
            high_res_latitudes_deg_n=(
                None if high_res_latitude_matrix_deg_n is None
                else high_res_latitude_matrix_deg_n[i, ..., 0]
            ),
            high_res_longitudes_deg_e=(
                None if high_res_longitude_matrix_deg_e is None
                else high_res_longitude_matrix_deg_e[i, ..., 0]
            ),
            are_data_normalized=are_data_normalized,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            output_file_name=output_file_name,
            prob_colour_map_name=prob_colour_map_name,
            min_semantic_seg_prob=this_min_prob,
            max_semantic_seg_prob=this_max_prob,
            convert_points_to_line_contours=convert_points_to_line_contours,
            line_contour_smoothing_radius_px=line_contour_smoothing_radius_px,
            point_prediction_opacity=point_prediction_opacity
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        satellite_dir_name=getattr(INPUT_ARG_OBJECT, SATELLITE_DIR_ARG_NAME),
        are_data_normalized=bool(getattr(
            INPUT_ARG_OBJECT, ARE_DATA_NORMALIZED_ARG_NAME
        )),
        plot_one_lag_time=bool(getattr(
            INPUT_ARG_OBJECT, PLOT_ONE_LAG_TIME_ARG_NAME
        )),
        min_semantic_seg_prob=getattr(
            INPUT_ARG_OBJECT, MIN_SEMANTIC_SEG_PROB_ARG_NAME
        ),
        max_semantic_seg_prob=getattr(
            INPUT_ARG_OBJECT, MAX_SEMANTIC_SEG_PROB_ARG_NAME
        ),
        min_semantic_seg_prob_percentile=getattr(
            INPUT_ARG_OBJECT, MIN_SEMANTIC_SEG_PROB_PERCENTILE_ARG_NAME
        ),
        max_semantic_seg_prob_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_SEMANTIC_SEG_PROB_PERCENTILE_ARG_NAME
        ),
        prob_colour_map_name=getattr(
            INPUT_ARG_OBJECT, PROB_COLOUR_MAP_ARG_NAME
        ),
        convert_points_to_line_contours=bool(getattr(
            INPUT_ARG_OBJECT, POINTS_TO_LINE_CONTOURS_ARG_NAME
        )),
        line_contour_smoothing_radius_px=getattr(
            INPUT_ARG_OBJECT, LINE_CONTOUR_SMOOTH_RAD_ARG_NAME
        ),
        point_prediction_opacity=getattr(
            INPUT_ARG_OBJECT, POINT_PREDICTION_OPACITY_ARG_NAME
        ),
        plot_mean_point_prediction=bool(getattr(
            INPUT_ARG_OBJECT, PLOT_MEAN_POINT_PREDICTION_ARG_NAME
        )),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
