"""Plots predicted TC centers on top of satellite images for 2024 WAF paper.

WAF = Weather and Forecasting
"""

import os
import argparse
import numpy
import xarray
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot
from scipy.interpolate import interp1d
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import general_utils as gg_general_utils
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import imagemagick_utils
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from ml4tccf.io import border_io
from ml4tccf.io import prediction_io
from ml4tccf.utils import misc_utils
from ml4tccf.utils import normalization
from ml4tccf.utils import satellite_utils
from ml4tccf.utils import spread_skill_utils as ss_utils
from ml4tccf.utils import scalar_prediction_utils as prediction_utils
from ml4tccf.machine_learning import neural_net_utils as nn_utils
from ml4tccf.machine_learning import \
    neural_net_training_cira_ir as nn_training_cira_ir
from ml4tccf.machine_learning import \
    neural_net_training_simple as nn_training_simple
from ml4tccf.machine_learning import \
    neural_net_training_fancy as nn_training_fancy
from ml4tccf.plotting import plotting_utils
from ml4tccf.plotting import satellite_plotting

TOLERANCE = 1e-6
INPUT_ARG_TIME_FORMAT = '%Y-%m-%d-%H%M'
TIME_FORMAT = '%Y-%m-%d-%H%M'

MICRONS_TO_METRES = 1e-6

GOES_WAVELENGTHS_METRES = 1e-6 * numpy.array([
    3.9, 6.185, 6.95, 7.34, 8.5, 9.61, 10.35, 11.2, 12.3, 13.3
])
HIMAWARI_WAVELENGTHS_METRES = 1e-6 * numpy.array([
    3.9, 6.200, 6.90, 7.30, 8.6, 9.60, 10.40, 11.2, 12.4, 13.3
])

PREDICTED_CENTER_MARKER = 'o'
PREDICTED_CENTER_MARKER_COLOUR = numpy.full(3, 0.)
PREDICTED_CENTER_MARKER_EDGE_WIDTH = 0
PREDICTED_CENTER_MARKER_EDGE_COLOUR = numpy.full(3, 0.)

ACTUAL_CENTER_MARKER = '*'
ACTUAL_CENTER_MARKER_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
ACTUAL_CENTER_MARKER_SIZE_MULT = 4. / 3
ACTUAL_CENTER_MARKER_EDGE_WIDTH = 2
ACTUAL_CENTER_MARKER_EDGE_COLOUR = numpy.full(3, 0.)

IMAGE_CENTER_MARKER = 's'
IMAGE_CENTER_MARKER_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
IMAGE_CENTER_MARKER_SIZE_MULT = 1.
IMAGE_CENTER_MARKER_EDGE_WIDTH = 2
IMAGE_CENTER_MARKER_EDGE_COLOUR = numpy.full(3, 0.)

TITLE_FONT_SIZE = 50
TICK_LABEL_FONT_SIZE = 40

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300
PANEL_SIZE_PX = int(2.5e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

PROB_CONTOURS_FORMAT_STRING = 'probability_contours'
MEAN_POINT_FORMAT_STRING = 'one_point_for_ensemble_mean'
MANY_POINTS_FORMAT_STRING = 'one_point_per_ensemble_member'
VALID_PRED_PLOTTING_FORMAT_STRINGS = [
    PROB_CONTOURS_FORMAT_STRING,
    MEAN_POINT_FORMAT_STRING,
    MANY_POINTS_FORMAT_STRING
]

PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
SATELLITE_DIR_ARG_NAME = 'input_satellite_dir_name'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
TARGET_TIMES_ARG_NAME = 'target_time_strings'
NUM_SAMPLES_PER_TIME_ARG_NAME = 'num_samples_per_target_time'
LAG_TIMES_ARG_NAME = 'lag_times_minutes'
WAVELENGTHS_ARG_NAME = 'wavelengths_microns'
PREDICTION_PLOTTING_FORMAT_ARG_NAME = 'prediction_plotting_format_string'
PROB_COLOUR_MAP_ARG_NAME = 'prob_colour_map_name'
PROB_CONTOUR_SMOOTHING_RADIUS_ARG_NAME = 'prob_contour_smoothing_radius_px'
POINT_PREDICTION_MARKER_SIZE_ARG_NAME = 'point_prediction_marker_size'
POINT_PREDICTION_OPACITY_ARG_NAME = 'point_prediction_opacity'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTION_FILE_HELP_STRING = (
    'Path to prediction file, containing predictions and targets for one TC.  '
    'This file will be read by `prediction_io.read_file`, and one figure will '
    'be created for each TC sample (time step) in the file.'
)
SATELLITE_DIR_HELP_STRING = (
    'Path to directory with satellite data (predictors).  Files therein will '
    'be found by `satellite_io.find_file` and read by `satellite_io.read_file`.'
)
NORMALIZATION_FILE_HELP_STRING = (
    'Path to file with normalization parameters.  If the satellite data are '
    'normalized, this file will be used to convert back to physical units '
    'before plotting.'
)
TARGET_TIMES_HELP_STRING = (
    '1-D list of target times (format "yyyy-mm-dd-HH") -- will plot only TC '
    'samples with these target times.  If you do not want to subset by time, '
    'leave this argument alone.'
)
NUM_SAMPLES_PER_TIME_HELP_STRING = (
    'Will plot this many (randomly selected) TC samples per target time.  Keep '
    'in mind that in general there are many TC samples per target time, each '
    'corresponding to a different random translation.  If you do not want to '
    'subset by random translations, leave this argument alone.'
)
LAG_TIMES_HELP_STRING = (
    '1-D list of lag times at which to plot predictor (satellite) data.'
)
WAVELENGTHS_HELP_STRING = (
    '1-D list of wavelengths at which to plot predictor (satellite) data.'
)
PREDICTION_PLOTTING_FORMAT_HELP_STRING = (
    'Plotting format for predictions.  Options are listed below:\n{0:s}'
).format(str(VALID_PRED_PLOTTING_FORMAT_STRINGS))

PROB_COLOUR_MAP_HELP_STRING = (
    '[used only if predictions are plotted as probability contours] Name of '
    'colour scheme for contours.  Must be accepted by '
    '`matplotlib.pyplot.get_cmap`.'
)
PROB_CONTOUR_SMOOTHING_RADIUS_HELP_STRING = (
    '[used only if predictions are plotted as probability contours] Smoothing '
    'radius for probability grid -- will be applied before plotting contours.'
)
POINT_PREDICTION_MARKER_SIZE_HELP_STRING = (
    '[used only if predictions are plotted as point(s)] Marker size -- will be '
    'applied to each point.'
)
POINT_PREDICTION_OPACITY_HELP_STRING = (
    '[used only if predictions are plotted as point(s)] Marker opacity -- will '
    'be applied to each point.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.'
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
    '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=False, default='',
    help=NORMALIZATION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_TIMES_ARG_NAME, type=str, nargs='+', required=False,
    default=[''], help=TARGET_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_SAMPLES_PER_TIME_ARG_NAME, type=int, required=False,
    default=-1, help=NUM_SAMPLES_PER_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAG_TIMES_ARG_NAME, type=int, nargs='+', required=True,
    help=LAG_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + WAVELENGTHS_ARG_NAME, type=float, nargs='+', required=True,
    help=WAVELENGTHS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_PLOTTING_FORMAT_ARG_NAME, type=str, required=True,
    help=PREDICTION_PLOTTING_FORMAT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PROB_COLOUR_MAP_ARG_NAME, type=str, required=False, default='BuGn',
    help=PROB_COLOUR_MAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PROB_CONTOUR_SMOOTHING_RADIUS_ARG_NAME, type=float, required=False,
    default=-1., help=PROB_CONTOUR_SMOOTHING_RADIUS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + POINT_PREDICTION_MARKER_SIZE_ARG_NAME, type=int, required=False,
    default=12, help=POINT_PREDICTION_MARKER_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + POINT_PREDICTION_OPACITY_ARG_NAME, type=float, required=False,
    default=0.5, help=POINT_PREDICTION_OPACITY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _check_prediction_plotting_format(plotting_format_string, ensemble_size):
    """Verifies prediction-plotting format.

    :param plotting_format_string: Plotting format.
    :param ensemble_size: Number of ensemble members.
    :return: plotting_format_string: Plotting format (may be changed from
        input).
    :raises: ValueError: if `plotting_format_string not in
        VALID_PRED_PLOTTING_FORMAT_STRINGS`.
    """

    if plotting_format_string not in VALID_PRED_PLOTTING_FORMAT_STRINGS:
        error_string = (
            'Plotting format ("{0:s}") is not in the following list of '
            'accepted formats:\n{1:s}'
        ).format(
            plotting_format_string, str(VALID_PRED_PLOTTING_FORMAT_STRINGS)
        )

        raise ValueError(error_string)

    if ensemble_size == 1:
        plotting_format_string = MEAN_POINT_FORMAT_STRING

    return plotting_format_string


def _plot_predictors_1panel(
        brightness_temp_matrix_kelvins, grid_latitudes_deg_n,
        grid_longitudes_deg_e, border_latitudes_deg_n, border_longitudes_deg_e,
        predicted_center_marker_size, title_string):
    """Plots one 2-D image of predictors (satellite data).

    This image will be one panel in a final, concatenated figure.

    M = number of rows in grid
    N = number of columns in grid
    P = number of points in border collection

    :param brightness_temp_matrix_kelvins: M-by-N numpy array of brightness
        temperatures.
    :param grid_latitudes_deg_n: length-M numpy array of latitudes (deg north).
    :param grid_longitudes_deg_e: length-N numpy array of longitudes (deg east).
    :param border_latitudes_deg_n: length-P numpy array of latitudes (deg
        north).
    :param border_longitudes_deg_e: length-P numpy array of longitudes (deg
        east).
    :param predicted_center_marker_size: Marker size for predicted TC center.
        Even though this method does not plot predictions, this size will be
        used to determine the sizes of other markers.
    :param title_string: Title.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object
    )
    colour_map_object, colour_norm_object = (
        satellite_plotting.get_colour_scheme_for_brightness_temp()
    )
    satellite_plotting.plot_2d_grid_latlng(
        data_matrix=brightness_temp_matrix_kelvins,
        axes_object=axes_object,
        latitude_array_deg_n=grid_latitudes_deg_n,
        longitude_array_deg_e=grid_longitudes_deg_e,
        plotting_brightness_temp=True,
        cbar_orientation_string=None,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object
    )

    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=numpy.ravel(grid_latitudes_deg_n),
        plot_longitudes_deg_e=numpy.ravel(grid_longitudes_deg_e),
        axes_object=axes_object,
        parallel_spacing_deg=2., meridian_spacing_deg=2.
    )
    axes_object.set_xticklabels(
        axes_object.get_xticklabels(),
        fontsize=TICK_LABEL_FONT_SIZE, rotation=90.
    )
    axes_object.set_yticklabels(
        axes_object.get_yticklabels(), fontsize=TICK_LABEL_FONT_SIZE
    )

    axes_object.plot(
        0.5, 0.5, linestyle='None',
        marker=IMAGE_CENTER_MARKER,
        markersize=IMAGE_CENTER_MARKER_SIZE_MULT * predicted_center_marker_size,
        markerfacecolor=IMAGE_CENTER_MARKER_COLOUR,
        markeredgecolor=IMAGE_CENTER_MARKER_EDGE_COLOUR,
        markeredgewidth=IMAGE_CENTER_MARKER_EDGE_WIDTH,
        transform=axes_object.transAxes, zorder=2e12
    )

    axes_object.set_title(title_string, fontsize=TITLE_FONT_SIZE)
    return figure_object, axes_object


def _plot_prob_contours_1panel(
        figure_object, axes_object, prob_matrix,
        grid_latitudes_deg_n, grid_longitudes_deg_e,
        colour_map_object, colour_norm_object, contour_levels,
        output_file_name):
    """Plots predictions as probability contours for one panel.

    One panel = one pair of lag time and wavelength

    M = number of rows in probability grid
    N = number of columns in probability grid

    :param figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :param axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param prob_matrix: M-by-N numpy array of probabilities.
    :param grid_latitudes_deg_n: length-M numpy array of latitudes (deg north).
    :param grid_longitudes_deg_e: length-N numpy array of longitudes (deg east).
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :param colour_norm_object: Colour-normalizer (maps from data space to
        0...1).  This is an instance of `matplotlib.colors.Normalize`.
    :param contour_levels: 1-D numpy array of probabilities corresponding to
        contour lines.
    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    axes_object.contour(
        grid_longitudes_deg_e, grid_latitudes_deg_n, prob_matrix,
        contour_levels,
        cmap=colour_map_object, norm=colour_norm_object,
        linewidths=4, linestyles='solid', zorder=1e12
    )

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


def _plot_point_predictions_1panel(
        figure_object, axes_object, x_centers_transaxes, y_centers_transaxes,
        marker_size, opacity, output_file_name):
    """Plots point predictions for one panel.

    S = ensemble size

    :param figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :param axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param x_centers_transaxes: length-S numpy array of predicted x-coords in
        0...1 space.
    :param y_centers_transaxes: length-S numpy array of predicted y-coords in
        0...1 space.
    :param marker_size: Marker size.
    :param opacity: Opacity.
    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    marker_colour = matplotlib.colors.to_rgba(
        c=PREDICTED_CENTER_MARKER_COLOUR, alpha=opacity
    )
    ensemble_size = len(x_centers_transaxes)

    for k in range(ensemble_size):
        axes_object.plot(
            x_centers_transaxes[k], y_centers_transaxes[k],
            linestyle='None', marker=PREDICTED_CENTER_MARKER,
            markersize=marker_size, markerfacecolor=marker_colour,
            markeredgecolor=PREDICTED_CENTER_MARKER_EDGE_COLOUR,
            markeredgewidth=PREDICTED_CENTER_MARKER_EDGE_WIDTH,
            transform=axes_object.transAxes, zorder=1e10
        )

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


def _make_figure_one_example(
        brightness_temp_matrix_kelvins, target_values, prediction_matrix,
        model_metadata_dict,
        btemp_latitude_matrix_deg_n, btemp_longitude_matrix_deg_e,
        border_latitudes_deg_n, border_longitudes_deg_e,
        prediction_plotting_format_string, prob_colour_map_name,
        prob_contour_smoothing_radius_px, point_prediction_marker_size,
        point_prediction_opacity, output_file_name):
    """Makes complete figure for one TC sample.

    M = number of rows in grid
    N = number of columns in grid
    L = number of lag times
    W = number of wavelengths
    S = ensemble size
    P = number of points in border collection

    :param brightness_temp_matrix_kelvins: M-by-N-by-L-by-W numpy array of
        brightness temperatures.
    :param target_values: length-4 numpy array with [true row offset,
        true column offset, grid spacing in km, true TC-center latitude in deg N].
    :param prediction_matrix: 2-by-S numpy array, where prediction_matrix[:, 0]
        contains predicted row offsets and prediction_matrix[:, 1] contains
        predicted column offsets.
    :param model_metadata_dict: Dictionary in format returned by
        `neural_net_utils.read_metafile`.
    :param btemp_latitude_matrix_deg_n: M-by-L numpy array of latitudes (deg
        north).
    :param btemp_longitude_matrix_deg_e: N-by-L numpy array of longitudes (deg
        east).
    :param border_latitudes_deg_n: length-P numpy array of latitudes (deg
        north).
    :param border_longitudes_deg_e: length-P numpy array of longitudes (deg
        east).
    :param prediction_plotting_format_string: See documentation at top of this
        script.
    :param prob_colour_map_name: Same.
    :param prob_contour_smoothing_radius_px: Same.
    :param point_prediction_marker_size: Same.
    :param point_prediction_opacity: Same.
    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    # Housekeeping.
    btemp_longitude_matrix_deg_e = lng_conversion.convert_lng_negative_in_west(
        btemp_longitude_matrix_deg_e
    )
    longitude_range_deg = (
        numpy.max(btemp_longitude_matrix_deg_e) -
        numpy.min(btemp_longitude_matrix_deg_e)
    )
    if longitude_range_deg > 100:
        btemp_longitude_matrix_deg_e = (
            lng_conversion.convert_lng_positive_in_west(
                btemp_longitude_matrix_deg_e
            )
        )

    num_grid_rows = brightness_temp_matrix_kelvins.shape[0]
    num_grid_columns = brightness_temp_matrix_kelvins.shape[1]
    actual_x_center_transaxes = 0.5 + target_values[1] / num_grid_columns
    actual_y_center_transaxes = 0.5 + target_values[0] / num_grid_rows

    # TODO(thunderhoser): Is this correct?
    center_row_index = 0.5 * float(num_grid_rows + 1)
    center_column_index = 0.5 * float(num_grid_columns + 1)

    all_row_indices = numpy.linspace(
        0, num_grid_rows - 1, num=num_grid_rows, dtype=float
    )
    all_column_indices = numpy.linspace(
        0, num_grid_columns - 1, num=num_grid_columns, dtype=float
    )

    if prediction_plotting_format_string == PROB_CONTOURS_FORMAT_STRING:
        prob_colour_map_object = pyplot.get_cmap(prob_colour_map_name)
    else:
        prob_colour_map_object = None

    grid_spacing_km = target_values[2]
    y_error_km = grid_spacing_km * (
        numpy.mean(prediction_matrix[0, :]) - target_values[0]
    )
    x_error_km = grid_spacing_km * (
        numpy.mean(prediction_matrix[1, :]) - target_values[1]
    )
    euclidean_error_km = numpy.sqrt(y_error_km ** 2 + x_error_km ** 2)

    y_stdev_km = grid_spacing_km * numpy.std(prediction_matrix[0, :], ddof=1)
    x_stdev_km = grid_spacing_km * numpy.std(prediction_matrix[1, :], ddof=1)
    euclidean_stdev_km = ss_utils._get_predictive_stdistdevs(
        grid_spacing_km * numpy.expand_dims(prediction_matrix, axis=0)
    )[0]

    if prediction_plotting_format_string == MEAN_POINT_FORMAT_STRING:
        prediction_matrix = numpy.mean(
            prediction_matrix, axis=-1, keepdims=True
        )

    # Convert predictions to whatever format is needed for plotting.
    ensemble_size = prediction_matrix.shape[1]

    if prediction_plotting_format_string == PROB_CONTOURS_FORMAT_STRING:

        # TODO(thunderhoser): Does not handle issues with Int'l Date Line.
        latitude_interp_object = interp1d(
            x=all_row_indices, y=btemp_latitude_matrix_deg_n[:, -1],
            kind='linear', bounds_error=True
        )
        longitude_interp_object = interp1d(
            x=all_column_indices, y=btemp_longitude_matrix_deg_e[:, -1],
            kind='linear', bounds_error=True
        )

        ensemble_latitudes_deg_n = numpy.array([
            latitude_interp_object(center_row_index + prediction_matrix[0, k])
            for k in range(ensemble_size)
        ])

        ensemble_longitudes_deg_e = numpy.array([
            longitude_interp_object(
                center_column_index + prediction_matrix[1, k]
            )
            for k in range(ensemble_size)
        ])

        prob_matrix = misc_utils.latlng_points_to_probability_grid(
            point_latitudes_deg_n=ensemble_latitudes_deg_n,
            point_longitudes_deg_e=ensemble_longitudes_deg_e,
            grid_latitude_array_deg_n=btemp_latitude_matrix_deg_n[:, -1],
            grid_longitude_array_deg_e=btemp_longitude_matrix_deg_e[:, -1]
        )

        if prob_contour_smoothing_radius_px is not None:
            print((
                'Smoothing gridded probs with {0:.1f}-pixel e-folding '
                'radius...'
            ).format(
                prob_contour_smoothing_radius_px
            ))

            prob_matrix = gg_general_utils.apply_gaussian_filter(
                input_matrix=prob_matrix,
                e_folding_radius_grid_cells=prob_contour_smoothing_radius_px
            )
            prob_matrix = prob_matrix / numpy.sum(prob_matrix)

        prob_matrix = 100 * prob_matrix

        if numpy.any(prob_matrix > TOLERANCE):
            min_colour_value = numpy.min(prob_matrix[prob_matrix > TOLERANCE])
        else:
            min_colour_value = 0.

        max_colour_value = numpy.max(prob_matrix)
        prob_colour_norm_object = pyplot.Normalize(
            vmin=min_colour_value, vmax=max_colour_value
        )
        prob_contour_levels = numpy.linspace(
            min_colour_value, max_colour_value, num=10
        )

        predicted_x_centers_transaxes = None
        predicted_y_centers_transaxes = None
    else:
        predicted_x_centers_transaxes = (
            0.5 + prediction_matrix[1, :] / num_grid_columns
        )
        predicted_y_centers_transaxes = (
            0.5 + prediction_matrix[0, :] / num_grid_rows
        )

        prob_matrix = None
        prob_colour_norm_object = None
        prob_contour_levels = None

    # Do actual stuff.
    vod = model_metadata_dict[nn_utils.VALIDATION_OPTIONS_KEY]
    lag_times_minutes = numpy.sort(vod[nn_utils.LAG_TIMES_KEY])[::-1]
    wavelengths_microns = vod[nn_utils.LOW_RES_WAVELENGTHS_KEY]

    num_lag_times = len(lag_times_minutes)
    num_wavelengths = len(wavelengths_microns)
    panel_file_names = []
    panel_letter = None

    for i in range(num_lag_times):
        for j in range(num_wavelengths):
            if panel_letter is None:
                panel_letter = 'a'
            else:
                panel_letter = chr(ord(panel_letter) + 1)

            title_string = r'{0:.3f}-micron $T_b$ at {1:d}-min lag'.format(
                wavelengths_microns[j],
                int(numpy.round(lag_times_minutes[i]))
            )

            if lag_times_minutes[i] == 0 and j == num_wavelengths - 1:
                title_string += (
                    '\nErr (x/y/Euc) = {0:.1f}/{1:.1f}/{2:.1f} km'
                    '\nStd (x/y/Euc) = {3:.1f}/{4:.1f}/{5:.1f} km'
                ).format(
                    x_error_km, y_error_km, euclidean_error_km,
                    x_stdev_km, y_stdev_km, euclidean_stdev_km
                )

            this_file_name = (
                '{0:s}_{1:04d}minutes_{2:06.3f}microns.jpg'
            ).format(
                os.path.splitext(output_file_name)[0],
                int(numpy.round(lag_times_minutes[i])),
                wavelengths_microns[j]
            )

            panel_file_names.append(this_file_name)
            file_system_utils.mkdir_recursive_if_necessary(
                file_name=panel_file_names[-1]
            )

            figure_object, axes_object = _plot_predictors_1panel(
                brightness_temp_matrix_kelvins=
                brightness_temp_matrix_kelvins[..., i, j],
                grid_latitudes_deg_n=btemp_latitude_matrix_deg_n[:, i],
                grid_longitudes_deg_e=btemp_longitude_matrix_deg_e[:, i],
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                predicted_center_marker_size=point_prediction_marker_size,
                title_string=title_string
            )

            if lag_times_minutes[i] == 0 and j == num_wavelengths - 1:
                gg_plotting_utils.label_axes(
                    axes_object=axes_object,
                    label_string='({0:s})'.format(panel_letter),
                    y_coord_normalized=1.13
                )
            else:
                gg_plotting_utils.label_axes(
                    axes_object=axes_object,
                    label_string='({0:s})'.format(panel_letter),
                    y_coord_normalized=1.
                )

            if lag_times_minutes[i] != 0:
                print('Saving figure to file: "{0:s}"...'.format(
                    panel_file_names[-1]
                ))
                figure_object.savefig(
                    panel_file_names[-1],
                    dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
                )
                pyplot.close(figure_object)

                imagemagick_utils.resize_image(
                    input_file_name=panel_file_names[-1],
                    output_file_name=panel_file_names[-1],
                    output_size_pixels=PANEL_SIZE_PX
                )

                continue

            axes_object.plot(
                actual_x_center_transaxes, actual_y_center_transaxes,
                linestyle='None', marker=ACTUAL_CENTER_MARKER,
                markersize=
                ACTUAL_CENTER_MARKER_SIZE_MULT * point_prediction_marker_size,
                markerfacecolor=ACTUAL_CENTER_MARKER_COLOUR,
                markeredgecolor=ACTUAL_CENTER_MARKER_EDGE_COLOUR,
                markeredgewidth=ACTUAL_CENTER_MARKER_EDGE_WIDTH,
                transform=axes_object.transAxes, zorder=3e12
            )

            if prediction_plotting_format_string == PROB_CONTOURS_FORMAT_STRING:
                _plot_prob_contours_1panel(
                    figure_object=figure_object,
                    axes_object=axes_object,
                    prob_matrix=prob_matrix,
                    grid_latitudes_deg_n=btemp_latitude_matrix_deg_n[:, i],
                    grid_longitudes_deg_e=btemp_longitude_matrix_deg_e[:, i],
                    colour_map_object=prob_colour_map_object,
                    colour_norm_object=prob_colour_norm_object,
                    contour_levels=prob_contour_levels,
                    output_file_name=panel_file_names[-1]
                )
            else:
                _plot_point_predictions_1panel(
                    figure_object=figure_object,
                    axes_object=axes_object,
                    x_centers_transaxes=predicted_x_centers_transaxes,
                    y_centers_transaxes=predicted_y_centers_transaxes,
                    marker_size=point_prediction_marker_size,
                    opacity=point_prediction_opacity,
                    output_file_name=panel_file_names[-1]
                )

    if num_lag_times == 1:
        num_panel_rows = int(numpy.ceil(
            numpy.sqrt(num_wavelengths)
        ))
        plotting_utils.concat_panels(
            panel_file_names=panel_file_names,
            num_panel_rows=num_panel_rows,
            concat_figure_file_name=output_file_name
        )
    else:
        plotting_utils.concat_panels(
            panel_file_names=panel_file_names,
            num_panel_rows=num_lag_times,
            concat_figure_file_name=output_file_name
        )

    colour_map_object, colour_norm_object = (
        satellite_plotting.get_colour_scheme_for_brightness_temp()
    )
    plotting_utils.add_colour_bar(
        figure_file_name=output_file_name,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical',
        font_size=20,
        cbar_label_string=r'$T_b$ (Kelvins)',
        tick_label_format_string='{0:.0f}',
        log_space=False,
        temporary_cbar_file_name='{0:s}_cbar.jpg'.format(output_file_name[:-4])
    )

    if prediction_plotting_format_string != PROB_CONTOURS_FORMAT_STRING:
        return

    plotting_utils.add_colour_bar(
        figure_file_name=output_file_name,
        colour_map_object=prob_colour_map_object,
        colour_norm_object=prob_colour_norm_object,
        orientation_string='vertical',
        font_size=20,
        cbar_label_string='Probability (%)',
        tick_label_format_string='{0:.2f}',
        log_space=False,
        temporary_cbar_file_name='{0:s}_cbar.jpg'.format(output_file_name[:-4])
    )


def _subset_random_translations(target_times_unix_sec,
                                num_samples_per_target_time):
    """Randomly subsets translation vectors.

    E = number of examples (or "TC samples")

    :param target_times_unix_sec: length-E numpy array of target times, which
        are not necessarily unique.
    :param num_samples_per_target_time: Number of samples to keep for each
        target time.
    :return: selected_indices: 1-D numpy array of selected example indices.
    """

    selected_indices = numpy.array([], dtype=int)

    for this_time_unix_sec in numpy.unique(target_times_unix_sec):
        these_indices = numpy.where(
            target_times_unix_sec == this_time_unix_sec
        )[0]

        if len(these_indices) > num_samples_per_target_time:
            these_indices = numpy.random.choice(
                these_indices, size=num_samples_per_target_time, replace=False
            )

        selected_indices = numpy.concatenate((selected_indices, these_indices))

    return selected_indices


def _run(prediction_file_name, satellite_dir_name, normalization_file_name,
         target_time_strings, num_samples_per_target_time,
         lag_times_minutes, wavelengths_microns,
         prediction_plotting_format_string, prob_colour_map_name,
         prob_contour_smoothing_radius_px, point_prediction_marker_size,
         point_prediction_opacity, output_dir_name):
    """Plots predicted TC centers on top of satellite images for 2024 WAF paper.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of this script.
    :param satellite_dir_name: Same.
    :param normalization_file_name: Same.
    :param target_time_strings: Same.
    :param num_samples_per_target_time: Same.
    :param lag_times_minutes: Same.
    :param wavelengths_microns: Same.
    :param prediction_plotting_format_string: Same.
    :param prob_colour_map_name: Same.
    :param prob_contour_smoothing_radius_px: Same.
    :param point_prediction_marker_size: Same.
    :param point_prediction_opacity: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_table_xarray = prediction_io.read_file(prediction_file_name)
    assert prediction_utils.PREDICTED_ROW_OFFSET_KEY in prediction_table_xarray

    ptx = prediction_table_xarray
    ensemble_size = len(
        ptx.coords[prediction_utils.ENSEMBLE_MEMBER_DIM_KEY].values
    )
    prediction_plotting_format_string = _check_prediction_plotting_format(
        plotting_format_string=prediction_plotting_format_string,
        ensemble_size=ensemble_size
    )

    if prediction_plotting_format_string == PROB_CONTOURS_FORMAT_STRING:
        if prob_contour_smoothing_radius_px <= 0:
            prob_contour_smoothing_radius_px = None
    else:
        error_checking.assert_is_greater(point_prediction_marker_size, 0)
        error_checking.assert_is_greater(point_prediction_opacity, 0.)
        error_checking.assert_is_leq(point_prediction_opacity, 1.)

    assert 0 in lag_times_minutes

    if len(target_time_strings) == 1 and target_time_strings[0] == '':
        target_time_strings = None
        target_times_unix_sec = None
    else:
        target_times_unix_sec = numpy.array([
            time_conversion.string_to_unix_sec(t, INPUT_ARG_TIME_FORMAT)
            for t in target_time_strings
        ], dtype=int)

    if num_samples_per_target_time < 1:
        num_samples_per_target_time = None

    if target_times_unix_sec is not None:
        found_time_flags = numpy.isin(
            element=target_times_unix_sec,
            test_elements=ptx[prediction_utils.TARGET_TIME_KEY].values
        )
        missing_time_indices = numpy.where(numpy.invert(found_time_flags))[0]

        if len(missing_time_indices) > 0:
            error_string = (
                'Could not find the following target times in the prediction '
                'file:\n{0:s}'
            ).format(
                str([target_time_strings[k] for k in missing_time_indices])
            )

            raise ValueError(error_string)

        keep_flags = numpy.isin(
            element=ptx[prediction_utils.TARGET_TIME_KEY].values,
            test_elements=target_times_unix_sec
        )
        keep_indices = numpy.where(keep_flags)[0]
        ptx = ptx.isel({prediction_utils.EXAMPLE_DIM_KEY: keep_indices})

    if num_samples_per_target_time is not None:
        keep_indices = _subset_random_translations(
            target_times_unix_sec=ptx[prediction_utils.TARGET_TIME_KEY].values,
            num_samples_per_target_time=num_samples_per_target_time
        )
        ptx = ptx.isel({prediction_utils.EXAMPLE_DIM_KEY: keep_indices})

    # Read model metadata.
    model_file_name = ptx.attrs[prediction_utils.MODEL_FILE_KEY]
    model_metafile_name = nn_utils.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = nn_utils.read_metafile(model_metafile_name)

    vod = model_metadata_dict[nn_utils.VALIDATION_OPTIONS_KEY]
    vod[nn_utils.SATELLITE_DIRECTORY_KEY] = satellite_dir_name
    vod[nn_utils.LAG_TIMES_KEY] = lag_times_minutes
    vod[nn_utils.LOW_RES_WAVELENGTHS_KEY] = wavelengths_microns
    vod[nn_utils.HIGH_RES_WAVELENGTHS_KEY] = numpy.array([])
    # vod[nn_utils.SENTINEL_VALUE_KEY] = SENTINEL_VALUE

    validation_option_dict = vod
    model_metadata_dict[nn_utils.VALIDATION_OPTIONS_KEY] = (
        validation_option_dict
    )

    # TODO(thunderhoser): This works as long as I have one TC per pred'n file.
    target_times_unix_sec = ptx[prediction_utils.TARGET_TIME_KEY].values
    cyclone_id_string = ptx[prediction_utils.CYCLONE_ID_KEY].values[0]

    try:
        cyclone_id_string = cyclone_id_string.decode('utf-8')
    except:
        pass

    # Find actual TC centers.
    actual_row_offsets = numpy.round(
        ptx[prediction_utils.ACTUAL_ROW_OFFSET_KEY].values
    ).astype(int)

    actual_column_offsets = numpy.round(
        ptx[prediction_utils.ACTUAL_COLUMN_OFFSET_KEY].values
    ).astype(int)

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

    target_matrix = data_dict[nn_utils.TARGET_MATRIX_KEY]
    btemp_latitude_matrix_deg_n = data_dict[nn_utils.LOW_RES_LATITUDES_KEY]
    btemp_longitude_matrix_deg_e = (
        data_dict[nn_utils.LOW_RES_LONGITUDES_KEY]
    )
    predictor_matrices = data_dict[nn_utils.PREDICTOR_MATRICES_KEY]
    brightness_temp_matrix_kelvins = predictor_matrices[0].astype(numpy.float32)

    if len(brightness_temp_matrix_kelvins.shape) != 5:
        brightness_temp_matrix_kelvins = (
            nn_utils.separate_lag_times_and_wavelengths(
                brightness_temp_matrix_kelvins,
                num_lag_times=len(lag_times_minutes)
            )
        )

    # Denormalize data if necessary.
    are_data_normalized = not numpy.any(brightness_temp_matrix_kelvins > 20.)

    if are_data_normalized:
        print('Reading normalization params from: "{0:s}"...'.format(
            normalization_file_name
        ))
        norm_param_table_xarray = normalization.read_file(
            normalization_file_name
        )

        actual_dim_keys = (
            satellite_utils.TIME_DIM,
            satellite_utils.LOW_RES_ROW_DIM,
            satellite_utils.LOW_RES_COLUMN_DIM,
            satellite_utils.LOW_RES_WAVELENGTH_DIM
        )
        dummy_dim_keys = (
            satellite_utils.TIME_DIM,
            satellite_utils.HIGH_RES_ROW_DIM,
            satellite_utils.HIGH_RES_COLUMN_DIM,
            satellite_utils.HIGH_RES_WAVELENGTH_DIM
        )

        num_examples = brightness_temp_matrix_kelvins.shape[0]

        for i in range(num_examples):
            this_bt_matrix_kelvins = numpy.swapaxes(
                brightness_temp_matrix_kelvins[i, ...], 1, 2
            )
            this_bt_matrix_kelvins = numpy.swapaxes(
                this_bt_matrix_kelvins, 0, 1
            )

            dummy_dims = (
                this_bt_matrix_kelvins.shape[0],
                this_bt_matrix_kelvins.shape[1] * 2,
                this_bt_matrix_kelvins.shape[2] * 2,
                0
            )
            this_dummy_bdrf_matrix = numpy.full(dummy_dims, numpy.nan)

            main_data_dict = {
                satellite_utils.BRIGHTNESS_TEMPERATURE_KEY: (
                    actual_dim_keys, this_bt_matrix_kelvins
                ),
                satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY: (
                    dummy_dim_keys, this_dummy_bdrf_matrix
                )
            }
            coord_dict = {
                satellite_utils.HIGH_RES_WAVELENGTH_DIM: numpy.array([]),
                satellite_utils.LOW_RES_WAVELENGTH_DIM:
                    MICRONS_TO_METRES * wavelengths_microns
            }
            satellite_table_xarray = xarray.Dataset(
                data_vars=main_data_dict, coords=coord_dict
            )

            satellite_table_xarray = normalization.denormalize_data(
                satellite_table_xarray=satellite_table_xarray,
                normalization_param_table_xarray=norm_param_table_xarray
            )
            this_bt_matrix_kelvins = satellite_table_xarray[
                satellite_utils.BRIGHTNESS_TEMPERATURE_KEY
            ].values

            this_bt_matrix_kelvins = numpy.swapaxes(
                this_bt_matrix_kelvins, 0, 1
            )
            brightness_temp_matrix_kelvins[i, ...] = numpy.swapaxes(
                this_bt_matrix_kelvins, 1, 2
            )

    prediction_matrix = numpy.stack((
        ptx[prediction_utils.PREDICTED_ROW_OFFSET_KEY].values,
        ptx[prediction_utils.PREDICTED_COLUMN_OFFSET_KEY].values
    ), axis=-2)

    # if prediction_plotting_format_string == MEAN_POINT_FORMAT_STRING:
    #     prediction_matrix = numpy.mean(
    #         prediction_matrix, axis=-1, keepdims=True
    #     )

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()
    num_examples = brightness_temp_matrix_kelvins.shape[0]

    basin_id_string = misc_utils.parse_cyclone_id(cyclone_id_string)[1]
    if basin_id_string == misc_utils.NORTHWEST_PACIFIC_ID_STRING:
        vod = model_metadata_dict[nn_utils.VALIDATION_OPTIONS_KEY]
        wavelengths_microns = vod[nn_utils.LOW_RES_WAVELENGTHS_KEY]

        wavelengths_microns = numpy.array([
            HIMAWARI_WAVELENGTHS_METRES[
                numpy.argmin(numpy.absolute(GOES_WAVELENGTHS_METRES - w))
            ]
            for w in wavelengths_microns
        ])

        vod[nn_utils.LOW_RES_WAVELENGTHS_KEY] = wavelengths_microns
        model_metadata_dict[nn_utils.VALIDATION_OPTIONS_KEY] = vod

    for i in range(num_examples):
        same_time_indices = numpy.where(
            target_times_unix_sec == target_times_unix_sec[i]
        )[0]
        this_index = 1 + numpy.where(same_time_indices == i)[0][0]

        output_file_name = '{0:s}/{1:s}_{2:s}_{3:03d}th.jpg'.format(
            output_dir_name,
            cyclone_id_string,
            time_conversion.unix_sec_to_string(
                target_times_unix_sec[i], TIME_FORMAT
            ),
            this_index
        )

        _make_figure_one_example(
            brightness_temp_matrix_kelvins=
            brightness_temp_matrix_kelvins[i, ...],
            target_values=target_matrix[i, ...],
            prediction_matrix=prediction_matrix[i, ...],
            model_metadata_dict=model_metadata_dict,
            btemp_latitude_matrix_deg_n=btemp_latitude_matrix_deg_n[i, ...],
            btemp_longitude_matrix_deg_e=btemp_longitude_matrix_deg_e[i, ...],
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            prediction_plotting_format_string=prediction_plotting_format_string,
            prob_colour_map_name=prob_colour_map_name,
            prob_contour_smoothing_radius_px=prob_contour_smoothing_radius_px,
            point_prediction_marker_size=point_prediction_marker_size,
            point_prediction_opacity=point_prediction_opacity,
            output_file_name=output_file_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        satellite_dir_name=getattr(INPUT_ARG_OBJECT, SATELLITE_DIR_ARG_NAME),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        target_time_strings=getattr(INPUT_ARG_OBJECT, TARGET_TIMES_ARG_NAME),
        num_samples_per_target_time=getattr(
            INPUT_ARG_OBJECT, NUM_SAMPLES_PER_TIME_ARG_NAME
        ),
        lag_times_minutes=numpy.array(
            getattr(INPUT_ARG_OBJECT, LAG_TIMES_ARG_NAME), dtype=int
        ),
        wavelengths_microns=numpy.array(
            getattr(INPUT_ARG_OBJECT, WAVELENGTHS_ARG_NAME), dtype=float
        ),
        prediction_plotting_format_string=getattr(
            INPUT_ARG_OBJECT, PREDICTION_PLOTTING_FORMAT_ARG_NAME
        ),
        prob_colour_map_name=getattr(
            INPUT_ARG_OBJECT, PROB_COLOUR_MAP_ARG_NAME
        ),
        prob_contour_smoothing_radius_px=getattr(
            INPUT_ARG_OBJECT, PROB_CONTOUR_SMOOTHING_RADIUS_ARG_NAME
        ),
        point_prediction_marker_size=getattr(
            INPUT_ARG_OBJECT, POINT_PREDICTION_MARKER_SIZE_ARG_NAME
        ),
        point_prediction_opacity=getattr(
            INPUT_ARG_OBJECT, POINT_PREDICTION_OPACITY_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
