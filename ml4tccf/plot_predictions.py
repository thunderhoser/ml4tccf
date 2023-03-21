"""Plots predictions."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import matplotlib.colors

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import file_system_utils
import error_checking
import imagemagick_utils
import prediction_io
import misc_utils
import scalar_prediction_utils
import gridded_prediction_utils
import border_io
import neural_net
import plotting_utils
import satellite_plotting

SENTINEL_VALUE = -9999.
TIME_FORMAT = '%Y-%m-%d-%H%M'

PREDICTED_CENTER_MARKER = 'o'
PREDICTED_CENTER_MARKER_COLOUR = numpy.full(3, 0.)
PREDICTED_CENTER_MARKER_COLOUR = matplotlib.colors.to_rgba(
    PREDICTED_CENTER_MARKER_COLOUR, alpha=0.5
)
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
MIN_GRIDDED_PROB_ARG_NAME = 'min_gridded_prob'
MAX_GRIDDED_PROB_ARG_NAME = 'max_gridded_prob'
MIN_GRIDDED_PROB_PERCENTILE_ARG_NAME = 'min_gridded_prob_percentile'
MAX_GRIDDED_PROB_PERCENTILE_ARG_NAME = 'max_gridded_prob_percentile'
PROB_COLOUR_MAP_ARG_NAME = 'prob_colour_map_name'
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
MIN_GRIDDED_PROB_HELP_STRING = (
    '[used only if predictions are gridded] Minimum probability to show in '
    'colour scheme.  If you want to specify colour limits with percentiles '
    'instead, leave this argument alone.'
)
MAX_GRIDDED_PROB_HELP_STRING = 'Same as `{0:s}` but for max.'.format(
    MIN_GRIDDED_PROB_HELP_STRING
)
MIN_GRIDDED_PROB_PERCENTILE_HELP_STRING = (
    '[used only if predictions are gridded] Minimum probability to show in '
    'colour scheme, stated as a percentile (from 0...100) over all values in '
    'the grid.  If you want to specify colour limits with raw probabilities '
    'instead, leave this argument alone.'
)
MAX_GRIDDED_PROB_PERCENTILE_HELP_STRING = 'Same as `{0:s}` but for max.'.format(
    MIN_GRIDDED_PROB_PERCENTILE_HELP_STRING
)
PROB_COLOUR_MAP_HELP_STRING = (
    '[used only if predictions are gridded] Name of colour scheme used for '
    'probabilities.  Must be accepted by `pyplot.get_cmap`.'
)
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
    '--' + MIN_GRIDDED_PROB_ARG_NAME, type=float, required=False, default=1,
    help=MIN_GRIDDED_PROB_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_GRIDDED_PROB_ARG_NAME, type=float, required=False, default=0,
    help=MAX_GRIDDED_PROB_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_GRIDDED_PROB_PERCENTILE_ARG_NAME, type=float, required=False,
    default=100, help=MIN_GRIDDED_PROB_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_GRIDDED_PROB_PERCENTILE_ARG_NAME, type=float, required=False,
    default=0, help=MAX_GRIDDED_PROB_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PROB_COLOUR_MAP_ARG_NAME, type=str, required=False,
    default='BuGn', help=PROB_COLOUR_MAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _get_colour_map_for_gridded_probs(
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


def _plot_data_one_example(
        predictor_matrices, scalar_target_values, prediction_matrix,
        model_metadata_dict, cyclone_id_string, target_time_unix_sec,
        low_res_latitudes_deg_n, low_res_longitudes_deg_e,
        high_res_latitudes_deg_n, high_res_longitudes_deg_e,
        are_data_normalized, border_latitudes_deg_n, border_longitudes_deg_e,
        output_file_name, min_gridded_prob=None, max_gridded_prob=None,
        prob_colour_map_name=None):
    """Plots satellite data for one example.

    P = number of points in border set
    S = ensemble size
    M = number of rows
    N = number of columns

    :param predictor_matrices: Same as output from `neural_net.create_data` but
        without first axis.
    :param scalar_target_values: Same as output from `neural_net.create_data`
        but without first axis.
    :param prediction_matrix: If predictions are scalar...

        2-by-S numpy array of predictions.  prediction_matrix[0, :] contains
        predicted row positions of TC centers, and prediction_matrix[1, :]
        contains predicted column positions of TC centers.

    If predictions are gridded...

    M-by-N numpy array of predicted probabilities.

    :param model_metadata_dict: Dictionary with model metadata, in format
        returned by `neural_net.read_metafile`.
    :param cyclone_id_string: Cyclone ID.
    :param target_time_unix_sec: Target time.
    :param low_res_latitudes_deg_n: Same as output from `neural_net.create_data`
        but without first or last axis.
    :param low_res_longitudes_deg_e: Same as output from
        `neural_net.create_data` but without first or last axis.
    :param high_res_latitudes_deg_n: Same as output from
        `neural_net.create_data` but without first or last axis.
    :param high_res_longitudes_deg_e: Same as output from
        `neural_net.create_data` but without first or last axis.
    :param are_data_normalized: See documentation at top of file.
    :param border_latitudes_deg_n: length-P numpy array of latitudes
        (deg north).  If None, will plot without coords.
    :param border_longitudes_deg_e: length-P numpy array of longitudes
        (deg east).  If None, will plot without coords.
    :param output_file_name: Path to output file.  Figure will be saved here.
    :param min_gridded_prob: Minimum probability in colour scheme.
    :param max_gridded_prob: Max probability in colour scheme.
    :param prob_colour_map_name: Name of base colour map for probabilities (must
        be accepted by `matplotlib.pyplot.get_cmap`).
    """

    num_grid_rows_low_res = predictor_matrices[-1].shape[0]
    num_grid_columns_low_res = predictor_matrices[-1].shape[1]
    are_predictions_gridded = prediction_matrix.shape[0] > 2
    ensemble_size = 1 if are_predictions_gridded else prediction_matrix.shape[1]

    if are_predictions_gridded:
        prob_colour_map_object, prob_colour_norm_object = (
            _get_colour_map_for_gridded_probs(
                base_colour_map_name=prob_colour_map_name,
                min_value=min_gridded_prob, max_value=max_gridded_prob,
                percent_flag=False
            )
        )
    else:
        prob_colour_map_object = None
        prob_colour_norm_object = None

    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    d = training_option_dict
    high_res_wavelengths_microns = d[neural_net.HIGH_RES_WAVELENGTHS_KEY]
    low_res_wavelengths_microns = d[neural_net.LOW_RES_WAVELENGTHS_KEY]
    lag_times_minutes = d[neural_net.LAG_TIMES_KEY]

    predictor_matrices = [
        neural_net.separate_lag_times_and_wavelengths(
            satellite_data_matrix=numpy.expand_dims(p, axis=0),
            num_lag_times=len(lag_times_minutes)
        )[0, ...] for p in predictor_matrices
    ]

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
            figure_object, axes_object = pyplot.subplots(
                1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
            )

            if are_data_normalized:
                colour_map_object = pyplot.get_cmap('seismic', lut=1001)
                colour_norm_object = matplotlib.colors.Normalize(
                    vmin=-3., vmax=3.
                )
            else:
                colour_map_object, colour_norm_object = (
                    satellite_plotting.get_colour_scheme_for_bdrf()
                )

            plotting_utils.plot_borders(
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                axes_object=axes_object
            )

            satellite_plotting.plot_2d_grid_latlng(
                data_matrix=predictor_matrices[0][..., i, j],
                axes_object=axes_object,
                latitude_array_deg_n=high_res_latitudes_deg_n,
                longitude_array_deg_e=high_res_longitudes_deg_e,
                plotting_brightness_temp=False,
                cbar_orientation_string=None,
                colour_map_object=colour_map_object,
                colour_norm_object=colour_norm_object,
                opacity=0.5 if are_predictions_gridded else 1.
            )

            if are_predictions_gridded:
                satellite_plotting.plot_2d_grid_latlng(
                    data_matrix=prediction_matrix,
                    axes_object=axes_object,
                    latitude_array_deg_n=low_res_latitudes_deg_n,
                    longitude_array_deg_e=low_res_longitudes_deg_e,
                    plotting_brightness_temp=False,
                    cbar_orientation_string=None,
                    colour_map_object=prob_colour_map_object,
                    colour_norm_object=prob_colour_norm_object,
                    use_contourf=True
                )

            plotting_utils.plot_grid_lines(
                plot_latitudes_deg_n=numpy.ravel(high_res_latitudes_deg_n),
                plot_longitudes_deg_e=numpy.ravel(high_res_longitudes_deg_e),
                axes_object=axes_object,
                parallel_spacing_deg=2., meridian_spacing_deg=2.
            )

            axes_object.plot(
                0.5 + scalar_target_values[1] / num_grid_columns_low_res,
                0.5 + scalar_target_values[0] / num_grid_rows_low_res,
                linestyle='None', marker=ACTUAL_CENTER_MARKER,
                markersize=ACTUAL_CENTER_MARKER_SIZE,
                markerfacecolor=ACTUAL_CENTER_MARKER_COLOUR,
                markeredgecolor=ACTUAL_CENTER_MARKER_EDGE_COLOUR,
                markeredgewidth=ACTUAL_CENTER_MARKER_EDGE_WIDTH,
                transform=axes_object.transAxes, zorder=1e10
            )

            axes_object.plot(
                0.5, 0.5, linestyle='None',
                marker=IMAGE_CENTER_MARKER, markersize=IMAGE_CENTER_MARKER_SIZE,
                markerfacecolor=IMAGE_CENTER_MARKER_COLOUR,
                markeredgecolor=IMAGE_CENTER_MARKER_EDGE_COLOUR,
                markeredgewidth=IMAGE_CENTER_MARKER_EDGE_WIDTH,
                transform=axes_object.transAxes, zorder=1e10
            )

            if not are_predictions_gridded:
                for k in range(ensemble_size):
                    axes_object.plot(
                        0.5 + prediction_matrix[1, k] / num_grid_columns_low_res,
                        0.5 + prediction_matrix[0, k] / num_grid_rows_low_res,
                        linestyle='None',
                        marker=PREDICTED_CENTER_MARKER,
                        markersize=PREDICTED_CENTER_MARKER_SIZE,
                        markerfacecolor=PREDICTED_CENTER_MARKER_COLOUR,
                        markeredgecolor=PREDICTED_CENTER_MARKER_EDGE_COLOUR,
                        markeredgewidth=PREDICTED_CENTER_MARKER_EDGE_WIDTH,
                        transform=axes_object.transAxes, zorder=1e10
                    )

            title_string = (
                '{0:.3f}-micron BDRF for {1:s} at {2:d}-minute lag'
            ).format(
                high_res_wavelengths_microns[j],
                cyclone_id_string,
                int(numpy.round(lag_times_minutes[i]))
            )
            axes_object.set_title(title_string)

            high_res_panel_file_name_matrix[i, j] = (
                '{0:s}_{1:04d}minutes_{2:06.3f}microns.{3:s}'
            ).format(
                os.path.splitext(output_file_name)[0],
                int(numpy.round(lag_times_minutes[i])),
                high_res_wavelengths_microns[j],
                'png' if are_predictions_gridded else 'jpg'
            )

            file_system_utils.mkdir_recursive_if_necessary(
                file_name=high_res_panel_file_name_matrix[i, j]
            )

            print('Saving figure to file: "{0:s}"...'.format(
                high_res_panel_file_name_matrix[i, j]
            ))
            figure_object.savefig(
                high_res_panel_file_name_matrix[i, j],
                dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

            imagemagick_utils.resize_image(
                input_file_name=high_res_panel_file_name_matrix[i, j],
                output_file_name=high_res_panel_file_name_matrix[i, j],
                output_size_pixels=PANEL_SIZE_PX
            )

    for i in range(num_lag_times):
        for j in range(num_low_res_wavelengths):
            figure_object, axes_object = pyplot.subplots(
                1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
            )

            if are_data_normalized:
                colour_map_object = pyplot.get_cmap('seismic', lut=1001)
                colour_norm_object = matplotlib.colors.Normalize(
                    vmin=-3., vmax=3.
                )
            else:
                colour_map_object, colour_norm_object = (
                    satellite_plotting.get_colour_scheme_for_brightness_temp()
                )

            plotting_utils.plot_borders(
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                axes_object=axes_object
            )

            satellite_plotting.plot_2d_grid_latlng(
                data_matrix=predictor_matrices[-1][..., i, j],
                axes_object=axes_object,
                latitude_array_deg_n=low_res_latitudes_deg_n,
                longitude_array_deg_e=low_res_longitudes_deg_e,
                plotting_brightness_temp=True,
                cbar_orientation_string=None,
                colour_map_object=colour_map_object,
                colour_norm_object=colour_norm_object,
                opacity=0.5 if are_predictions_gridded else 1.
            )

            if are_predictions_gridded:
                satellite_plotting.plot_2d_grid_latlng(
                    data_matrix=prediction_matrix,
                    axes_object=axes_object,
                    latitude_array_deg_n=low_res_latitudes_deg_n,
                    longitude_array_deg_e=low_res_longitudes_deg_e,
                    plotting_brightness_temp=False,
                    cbar_orientation_string=None,
                    colour_map_object=prob_colour_map_object,
                    colour_norm_object=prob_colour_norm_object,
                    use_contourf=True
                )

            plotting_utils.plot_grid_lines(
                plot_latitudes_deg_n=numpy.ravel(low_res_latitudes_deg_n),
                plot_longitudes_deg_e=numpy.ravel(low_res_longitudes_deg_e),
                axes_object=axes_object,
                parallel_spacing_deg=2., meridian_spacing_deg=2.
            )

            axes_object.plot(
                0.5 + scalar_target_values[1] / num_grid_columns_low_res,
                0.5 + scalar_target_values[0] / num_grid_rows_low_res,
                linestyle='None', marker=ACTUAL_CENTER_MARKER,
                markersize=ACTUAL_CENTER_MARKER_SIZE,
                markerfacecolor=ACTUAL_CENTER_MARKER_COLOUR,
                markeredgecolor=ACTUAL_CENTER_MARKER_EDGE_COLOUR,
                markeredgewidth=ACTUAL_CENTER_MARKER_EDGE_WIDTH,
                transform=axes_object.transAxes, zorder=1e10
            )

            axes_object.plot(
                0.5, 0.5, linestyle='None',
                marker=IMAGE_CENTER_MARKER, markersize=IMAGE_CENTER_MARKER_SIZE,
                markerfacecolor=IMAGE_CENTER_MARKER_COLOUR,
                markeredgecolor=IMAGE_CENTER_MARKER_EDGE_COLOUR,
                markeredgewidth=IMAGE_CENTER_MARKER_EDGE_WIDTH,
                transform=axes_object.transAxes, zorder=1e10
            )

            if not are_predictions_gridded:
                for k in range(ensemble_size):
                    axes_object.plot(
                        0.5 + prediction_matrix[1, k] / num_grid_columns_low_res,
                        0.5 + prediction_matrix[0, k] / num_grid_rows_low_res,
                        linestyle='None',
                        marker=PREDICTED_CENTER_MARKER,
                        markersize=PREDICTED_CENTER_MARKER_SIZE,
                        markerfacecolor=PREDICTED_CENTER_MARKER_COLOUR,
                        markeredgecolor=PREDICTED_CENTER_MARKER_EDGE_COLOUR,
                        markeredgewidth=PREDICTED_CENTER_MARKER_EDGE_WIDTH,
                        transform=axes_object.transAxes, zorder=1e10
                    )

            title_string = (
                r'{0:.3f}-micron $T_b$ for {1:s} at {2:d}-minute lag'
            ).format(
                low_res_wavelengths_microns[j],
                cyclone_id_string,
                int(numpy.round(lag_times_minutes[i]))
            )
            axes_object.set_title(title_string)

            low_res_panel_file_name_matrix[i, j] = (
                '{0:s}_{1:04d}minutes_{2:06.3f}microns.{3:s}'
            ).format(
                os.path.splitext(output_file_name)[0],
                int(numpy.round(lag_times_minutes[i])),
                low_res_wavelengths_microns[j],
                'png' if are_predictions_gridded else 'jpg'
            )

            file_system_utils.mkdir_recursive_if_necessary(
                file_name=low_res_panel_file_name_matrix[i, j]
            )

            print('Saving figure to file: "{0:s}"...'.format(
                low_res_panel_file_name_matrix[i, j]
            ))
            figure_object.savefig(
                low_res_panel_file_name_matrix[i, j],
                dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

            imagemagick_utils.resize_image(
                input_file_name=low_res_panel_file_name_matrix[i, j],
                output_file_name=low_res_panel_file_name_matrix[i, j],
                output_size_pixels=PANEL_SIZE_PX
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
                    'png' if are_predictions_gridded else 'jpg'
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
                    'png' if are_predictions_gridded else 'jpg'
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
                'png' if are_predictions_gridded else 'jpg'
            )
        )

        if are_predictions_gridded:
            prob_colour_map_object, prob_colour_norm_object = (
                _get_colour_map_for_gridded_probs(
                    base_colour_map_name=prob_colour_map_name,
                    min_value=min_gridded_prob, max_value=max_gridded_prob,
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
                    'png' if are_predictions_gridded else 'jpg'
                )
            )


def _run(prediction_file_name, satellite_dir_name, are_data_normalized,
         min_gridded_prob, max_gridded_prob, min_gridded_prob_percentile,
         max_gridded_prob_percentile, prob_colour_map_name, output_dir_name):
    """Plots predictions.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param satellite_dir_name: Same.
    :param are_data_normalized: Same.
    :param min_gridded_prob: Same.
    :param max_gridded_prob: Same.
    :param min_gridded_prob_percentile: Same.
    :param max_gridded_prob_percentile: Same.
    :param prob_colour_map_name: Same.
    :param output_dir_name: Same.
    """

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_table_xarray = prediction_io.read_file(prediction_file_name)

    are_predictions_gridded = (
        scalar_prediction_utils.PREDICTED_ROW_OFFSET_KEY
        not in prediction_table_xarray
    )

    if are_predictions_gridded:
        if min_gridded_prob >= max_gridded_prob:
            min_gridded_prob = None
            max_gridded_prob = None

            error_checking.assert_is_geq(min_gridded_prob_percentile, 50.)
            error_checking.assert_is_leq(max_gridded_prob_percentile, 100.)
            error_checking.assert_is_greater(
                max_gridded_prob_percentile, min_gridded_prob_percentile
            )
        else:
            error_checking.assert_is_greater(min_gridded_prob, 0.)
            error_checking.assert_is_greater(max_gridded_prob, 0.)

    model_file_name = (
        prediction_table_xarray.attrs[scalar_prediction_utils.MODEL_FILE_KEY]
    )
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )
    validation_option_dict[neural_net.SATELLITE_DIRECTORY_KEY] = (
        satellite_dir_name
    )
    validation_option_dict[neural_net.SENTINEL_VALUE_KEY] = SENTINEL_VALUE

    prediction_table_xarray = prediction_table_xarray.isel(indexers={gridded_prediction_utils.EXAMPLE_DIM: numpy.array([0], dtype=int)})
    pt = prediction_table_xarray

    # TODO(thunderhoser): This will not work if I ever have multiple cyclones in
    # one prediction file.
    cyclone_id_string = (
        pt[scalar_prediction_utils.CYCLONE_ID_KEY].values[0].decode('utf-8')
    )
    target_times_unix_sec = pt[scalar_prediction_utils.TARGET_TIME_KEY].values

    if are_predictions_gridded:
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

    validation_option_dict[neural_net.SEMANTIC_SEG_FLAG_KEY] = False
    data_dict = neural_net.create_data_specific_trans(
        option_dict=validation_option_dict,
        cyclone_id_string=cyclone_id_string,
        target_times_unix_sec=target_times_unix_sec,
        row_translations_low_res_px=actual_row_offsets,
        column_translations_low_res_px=actual_column_offsets
    )

    if data_dict is None:
        return

    predictor_matrices = data_dict[neural_net.PREDICTOR_MATRICES_KEY]
    scalar_target_matrix = data_dict[neural_net.TARGET_MATRIX_KEY]
    low_res_latitude_matrix_deg_n = data_dict[neural_net.LOW_RES_LATITUDES_KEY]
    low_res_longitude_matrix_deg_e = (
        data_dict[neural_net.LOW_RES_LONGITUDES_KEY]
    )
    high_res_latitude_matrix_deg_n = (
        data_dict[neural_net.HIGH_RES_LATITUDES_KEY]
    )
    high_res_longitude_matrix_deg_e = (
        data_dict[neural_net.HIGH_RES_LONGITUDES_KEY]
    )

    for k in range(len(predictor_matrices)):
        predictor_matrices[k] = predictor_matrices[k].astype(numpy.float64)
        predictor_matrices[k][
            predictor_matrices[k] < SENTINEL_VALUE + 1
        ] = numpy.nan

    if are_predictions_gridded:
        pt = gridded_prediction_utils.get_ensemble_mean(pt)
        prediction_matrix = (
            pt[gridded_prediction_utils.PREDICTION_MATRIX_KEY].values[..., 0]
        )
    else:
        prediction_matrix = numpy.stack((
            pt[scalar_prediction_utils.PREDICTED_ROW_OFFSET_KEY].values,
            pt[scalar_prediction_utils.PREDICTED_COLUMN_OFFSET_KEY].values
        ), axis=-2)

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
            'png' if are_predictions_gridded else 'jpg'
        )

        if are_predictions_gridded:
            if min_gridded_prob is None:
                this_min_gridded_prob = numpy.percentile(
                    prediction_matrix[i, ...], min_gridded_prob_percentile
                )
                this_max_gridded_prob = numpy.percentile(
                    prediction_matrix[i, ...], max_gridded_prob_percentile
                )

                print('SUM OF PREDICTIONS OVER GRID = {0:f}'.format(numpy.sum(prediction_matrix[i, ...])))

                if this_max_gridded_prob - this_min_gridded_prob < 0.01:
                    new_max = this_min_gridded_prob + 0.01
                    if new_max > 1:
                        this_min_gridded_prob = this_max_gridded_prob - 0.01
                    else:
                        this_max_gridded_prob = this_min_gridded_prob + 0.01
            else:
                this_min_gridded_prob = min_gridded_prob + 0.
                this_max_gridded_prob = max_gridded_prob + 0.
        else:
            this_min_gridded_prob = None
            this_max_gridded_prob = None

        _plot_data_one_example(
            predictor_matrices=[p[i, ...] for p in predictor_matrices],
            scalar_target_values=scalar_target_matrix[i, ...],
            prediction_matrix=prediction_matrix[i, ...],
            model_metadata_dict=model_metadata_dict,
            cyclone_id_string=cyclone_id_string,
            target_time_unix_sec=target_times_unix_sec[i],
            low_res_latitudes_deg_n=low_res_latitude_matrix_deg_n[i, :, 0],
            low_res_longitudes_deg_e=low_res_longitude_matrix_deg_e[i, :, 0],
            high_res_latitudes_deg_n=(
                None if high_res_latitude_matrix_deg_n is None
                else high_res_latitude_matrix_deg_n[i, :, 0]
            ),
            high_res_longitudes_deg_e=(
                None if high_res_longitude_matrix_deg_e is None
                else high_res_longitude_matrix_deg_e[i, :, 0]
            ),
            are_data_normalized=are_data_normalized,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            output_file_name=output_file_name,
            min_gridded_prob=this_min_gridded_prob,
            max_gridded_prob=this_max_gridded_prob,
            prob_colour_map_name=prob_colour_map_name
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
        min_gridded_prob=getattr(INPUT_ARG_OBJECT, MIN_GRIDDED_PROB_ARG_NAME),
        max_gridded_prob=getattr(INPUT_ARG_OBJECT, MAX_GRIDDED_PROB_ARG_NAME),
        min_gridded_prob_percentile=getattr(
            INPUT_ARG_OBJECT, MIN_GRIDDED_PROB_PERCENTILE_ARG_NAME
        ),
        max_gridded_prob_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_GRIDDED_PROB_PERCENTILE_ARG_NAME
        ),
        prob_colour_map_name=getattr(
            INPUT_ARG_OBJECT, PROB_COLOUR_MAP_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
