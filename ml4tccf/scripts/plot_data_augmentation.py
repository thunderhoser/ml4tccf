"""Plots data augmentation."""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import matplotlib.colors
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import imagemagick_utils
from ml4tccf.io import border_io
from ml4tccf.machine_learning import neural_net
from ml4tccf.plotting import plotting_utils
from ml4tccf.plotting import satellite_plotting

TIME_FORMAT = '%Y-%m-%d-%H%M'

LAG_TIMES_MINUTES = numpy.array([0], dtype=int)
LOW_RES_WAVELENGTHS_MICRONS = numpy.array([
    3.9, 6.185, 6.95, 7.34, 8.5, 9.61, 10.35, 11.2, 12.3, 13.3
])
HIGH_RES_WAVELENGTHS_MICRONS = numpy.array([0.64])

LAG_TIME_TOLERANCE_SEC = 900
MAX_NUM_MISSING_LAG_TIMES = 1
MAX_INTERP_GAP_SEC = 3600
SENTINEL_VALUE = -9999.

IMAGE_CENTER_MARKER = 'o'
IMAGE_CENTER_MARKER_COLOUR = numpy.full(3, 0.)
IMAGE_CENTER_MARKER_SIZE = 18
CYCLONE_CENTER_MARKER = '*'
CYCLONE_CENTER_MARKER_COLOUR = numpy.full(3, 0.)
CYCLONE_CENTER_MARKER_SIZE = 36

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300
PANEL_SIZE_PX = int(2.5e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

INPUT_DIR_ARG_NAME = 'input_satellite_dir_name'
CYCLONE_ID_ARG_NAME = 'cyclone_id_string'
NUM_TARGET_TIMES_ARG_NAME = 'num_target_times'
NUM_GRID_ROWS_ARG_NAME = 'num_grid_rows_low_res'
NUM_GRID_COLUMNS_ARG_NAME = 'num_grid_columns_low_res'
NUM_TRANSLATIONS_ARG_NAME = 'num_translations'
MEAN_TRANSLATION_ARG_NAME = 'mean_translation_low_res_px'
STDEV_TRANSLATION_ARG_NAME = 'stdev_translation_low_res_px'
ARE_DATA_NORMALIZED_ARG_NAME = 'are_data_normalized'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing satellite data.  Files therein will '
    'be found by `satellite_io.find_file` and read by `satellite_io.read_file`.'
)
CYCLONE_ID_HELP_STRING = (
    'Cyclone ID in format "yyyyBBnn".  Will plot data augmentation only for '
    'this cyclone.'
)
NUM_TARGET_TIMES_HELP_STRING = (
    'Will plot data augmentation at this many target times for the given '
    'cyclone.  Number of figures will be {0:s} * {1:s}.'
).format(NUM_TARGET_TIMES_ARG_NAME, NUM_TRANSLATIONS_ARG_NAME)

NUM_GRID_ROWS_HELP_STRING = (
    'Number of grid rows to retain in low-resolution (infrared) data.'
)
NUM_GRID_COLUMNS_HELP_STRING = (
    'Number of grid columns to retain in low-resolution (infrared) data.'
)
NUM_TRANSLATIONS_HELP_STRING = (
    'Number of translations (i.e., augmentations) for each cyclone.'
)
MEAN_TRANSLATION_HELP_STRING = (
    'Mean translation distance (in units of low-resolution pixels) for data '
    'augmentation.'
)
STDEV_TRANSLATION_HELP_STRING = (
    'Standard deviation of translation distance (in units of low-resolution '
    'pixels) for data augmentation.'
)
ARE_DATA_NORMALIZED_HELP_STRING = (
    'Boolean flag.  If 1 (0), plotting code will assume that satellite data '
    'are (un)normalized.'
)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CYCLONE_ID_ARG_NAME, type=str, required=True,
    help=CYCLONE_ID_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TARGET_TIMES_ARG_NAME, type=int, required=True,
    help=NUM_TARGET_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_GRID_ROWS_ARG_NAME, type=int, required=True,
    help=NUM_GRID_ROWS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_GRID_COLUMNS_ARG_NAME, type=int, required=True,
    help=NUM_GRID_COLUMNS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TRANSLATIONS_ARG_NAME, type=int, required=False, default=1,
    help=NUM_TRANSLATIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MEAN_TRANSLATION_ARG_NAME, type=float, required=False,
    default=15, help=MEAN_TRANSLATION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + STDEV_TRANSLATION_ARG_NAME, type=float, required=False,
    default=7.5, help=STDEV_TRANSLATION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ARE_DATA_NORMALIZED_ARG_NAME, type=int, required=True,
    help=ARE_DATA_NORMALIZED_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_data_one_example(
        predictor_matrices, target_values, cyclone_id_string,
        target_time_unix_sec, low_res_latitudes_deg_n, low_res_longitudes_deg_e,
        high_res_latitudes_deg_n, high_res_longitudes_deg_e,
        are_data_normalized, border_latitudes_deg_n, border_longitudes_deg_e,
        output_file_name):
    """Plots satellite data for one example.

    P = number of points in border set

    :param predictor_matrices: Same as output from `neural_net.create_data` but
        without first axis.
    :param target_values: Same as output from `neural_net.create_data` but
        without first axis.
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
    """

    num_grid_rows_low_res = predictor_matrices[1].shape[1]
    num_grid_columns_low_res = predictor_matrices[1].shape[2]

    num_panels = (
        len(HIGH_RES_WAVELENGTHS_MICRONS) + len(LOW_RES_WAVELENGTHS_MICRONS)
    )
    panel_file_names = [''] * num_panels
    panel_index = -1

    for j in range(len(HIGH_RES_WAVELENGTHS_MICRONS)):
        panel_index += 1

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

        satellite_plotting.plot_2d_grid_latlng(
            data_matrix=predictor_matrices[0][..., j],
            axes_object=axes_object,
            latitude_array_deg_n=high_res_latitudes_deg_n,
            longitude_array_deg_e=high_res_longitudes_deg_e,
            plotting_brightness_temp=False,
            cbar_orientation_string=None,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object
        )
        plotting_utils.plot_grid_lines(
            plot_latitudes_deg_n=numpy.ravel(high_res_latitudes_deg_n),
            plot_longitudes_deg_e=numpy.ravel(high_res_longitudes_deg_e),
            axes_object=axes_object,
            parallel_spacing_deg=2., meridian_spacing_deg=2.
        )

        axes_object.plot(
            0.5, 0.5, linestyle='None',
            marker=IMAGE_CENTER_MARKER, markersize=IMAGE_CENTER_MARKER_SIZE,
            markerfacecolor=IMAGE_CENTER_MARKER_COLOUR,
            markeredgecolor=IMAGE_CENTER_MARKER_COLOUR,
            markeredgewidth=0,
            transform=axes_object.transAxes, zorder=1e10
        )

        axes_object.plot(
            0.5 + target_values[1] / num_grid_columns_low_res,
            0.5 + target_values[0] / num_grid_rows_low_res,
            linestyle='None',
            marker=CYCLONE_CENTER_MARKER, markersize=CYCLONE_CENTER_MARKER_SIZE,
            markerfacecolor=CYCLONE_CENTER_MARKER_COLOUR,
            markeredgecolor=CYCLONE_CENTER_MARKER_COLOUR,
            markeredgewidth=0,
            transform=axes_object.transAxes, zorder=1e10
        )

        title_string = '{0:.3f}-micron BDRF for {1:s} at {2:s}'.format(
            HIGH_RES_WAVELENGTHS_MICRONS[j],
            cyclone_id_string,
            time_conversion.unix_sec_to_string(
                target_time_unix_sec, TIME_FORMAT
            )
        )
        axes_object.set_title(title_string)

        panel_file_names[panel_index] = '{0:s}_{1:06.3f}microns.jpg'.format(
            os.path.splitext(output_file_name)[0],
            HIGH_RES_WAVELENGTHS_MICRONS[j]
        )

        file_system_utils.mkdir_recursive_if_necessary(
            file_name=panel_file_names[panel_index]
        )

        print('Saving figure to file: "{0:s}"...'.format(
            panel_file_names[panel_index]
        ))
        figure_object.savefig(
            panel_file_names[panel_index],
            dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    for j in range(len(LOW_RES_WAVELENGTHS_MICRONS)):
        panel_index += 1

        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        if are_data_normalized:
            colour_map_object = pyplot.get_cmap('seismic', lut=1001)
            colour_norm_object = matplotlib.colors.Normalize(vmin=-3., vmax=3.)
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
            data_matrix=predictor_matrices[1][..., j],
            axes_object=axes_object,
            latitude_array_deg_n=low_res_latitudes_deg_n,
            longitude_array_deg_e=low_res_longitudes_deg_e,
            plotting_brightness_temp=True,
            cbar_orientation_string=None,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object
        )
        plotting_utils.plot_grid_lines(
            plot_latitudes_deg_n=numpy.ravel(low_res_latitudes_deg_n),
            plot_longitudes_deg_e=numpy.ravel(low_res_longitudes_deg_e),
            axes_object=axes_object,
            parallel_spacing_deg=2., meridian_spacing_deg=2.
        )

        axes_object.plot(
            0.5, 0.5, linestyle='None',
            marker=IMAGE_CENTER_MARKER, markersize=IMAGE_CENTER_MARKER_SIZE,
            markerfacecolor=IMAGE_CENTER_MARKER_COLOUR,
            markeredgecolor=IMAGE_CENTER_MARKER_COLOUR,
            markeredgewidth=0,
            transform=axes_object.transAxes, zorder=1e10
        )

        axes_object.plot(
            0.5 + target_values[1] / num_grid_columns_low_res,
            0.5 + target_values[0] / num_grid_rows_low_res,
            linestyle='None',
            marker=CYCLONE_CENTER_MARKER, markersize=CYCLONE_CENTER_MARKER_SIZE,
            markerfacecolor=CYCLONE_CENTER_MARKER_COLOUR,
            markeredgecolor=CYCLONE_CENTER_MARKER_COLOUR,
            markeredgewidth=0,
            transform=axes_object.transAxes, zorder=1e10
        )

        title_string = r'{0:.3f}-micron $T_b$ for {1:s} at {2:s}'.format(
            LOW_RES_WAVELENGTHS_MICRONS[j],
            cyclone_id_string,
            time_conversion.unix_sec_to_string(
                target_time_unix_sec, TIME_FORMAT
            )
        )
        axes_object.set_title(title_string)

        panel_file_names[panel_index] = '{0:s}_{1:06.3f}microns.jpg'.format(
            os.path.splitext(output_file_name)[0],
            LOW_RES_WAVELENGTHS_MICRONS[j]
        )

        file_system_utils.mkdir_recursive_if_necessary(
            file_name=panel_file_names[panel_index]
        )

        print('Saving figure to file: "{0:s}"...'.format(
            panel_file_names[panel_index]
        ))
        figure_object.savefig(
            panel_file_names[panel_index],
            dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    for this_file_name in panel_file_names:
        imagemagick_utils.resize_image(
            input_file_name=this_file_name, output_file_name=this_file_name,
            output_size_pixels=PANEL_SIZE_PX
        )

    plotting_utils.concat_panels(
        panel_file_names=panel_file_names,
        concat_figure_file_name=output_file_name
    )

    if are_data_normalized:
        colour_map_object = pyplot.get_cmap('seismic', lut=1001)
        colour_norm_object = matplotlib.colors.Normalize(vmin=-3., vmax=3.)
    else:
        colour_map_object, colour_norm_object = (
            satellite_plotting.get_colour_scheme_for_bdrf()
        )

    plotting_utils.add_colour_bar(
        figure_file_name=output_file_name,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', font_size=20,
        cbar_label_string=(
            r'BDRF ($z$-score)' if are_data_normalized
            else 'BDRF (unitless)'
        ),
        tick_label_format_string='{0:.2f}', log_space=False,
        temporary_cbar_file_name='{0:s}_cbar.jpg'.format(output_file_name[:-4])
    )

    if are_data_normalized:
        colour_map_object = pyplot.get_cmap('seismic', lut=1001)
        colour_norm_object = matplotlib.colors.Normalize(vmin=-3., vmax=3.)
    else:
        colour_map_object, colour_norm_object = (
            satellite_plotting.get_colour_scheme_for_brightness_temp()
        )

    plotting_utils.add_colour_bar(
        figure_file_name=output_file_name,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', font_size=20,
        cbar_label_string=(
            r'$T_b$ ($z-score$)' if are_data_normalized
            else r'$T_b$ (Kelvins)'
        ),
        tick_label_format_string='{0:.0f}', log_space=False,
        temporary_cbar_file_name='{0:s}_cbar.jpg'.format(output_file_name[:-4])
    )


def _run(satellite_dir_name, cyclone_id_string, num_target_times,
         num_grid_rows_low_res, num_grid_columns_low_res, num_translations,
         mean_translation_low_res_px, stdev_translation_low_res_px,
         are_data_normalized, output_dir_name):
    """Plots data augmentation.

    This is effectively the main method.

    :param satellite_dir_name: See documentation at top of file.
    :param cyclone_id_string: Same.
    :param num_target_times: Same.
    :param num_grid_rows_low_res: Same.
    :param num_grid_columns_low_res: Same.
    :param num_translations: Same.
    :param mean_translation_low_res_px: Same.
    :param stdev_translation_low_res_px: Same.
    :param are_data_normalized: Same.
    :param output_dir_name: Same.
    """

    option_dict = {
        neural_net.SATELLITE_DIRECTORY_KEY: satellite_dir_name,
        neural_net.YEARS_KEY: numpy.array([2000], dtype=int),
        neural_net.LAG_TIMES_KEY: LAG_TIMES_MINUTES,
        neural_net.HIGH_RES_WAVELENGTHS_KEY: HIGH_RES_WAVELENGTHS_MICRONS,
        neural_net.LOW_RES_WAVELENGTHS_KEY: LOW_RES_WAVELENGTHS_MICRONS,
        neural_net.BATCH_SIZE_KEY: 1,
        neural_net.MAX_EXAMPLES_PER_CYCLONE_KEY: 1,
        neural_net.NUM_GRID_ROWS_KEY: num_grid_rows_low_res,
        neural_net.NUM_GRID_COLUMNS_KEY: num_grid_columns_low_res,
        neural_net.DATA_AUG_NUM_TRANS_KEY: num_translations,
        neural_net.DATA_AUG_MEAN_TRANS_KEY: mean_translation_low_res_px,
        neural_net.DATA_AUG_STDEV_TRANS_KEY: stdev_translation_low_res_px,
        neural_net.LAG_TIME_TOLERANCE_KEY: LAG_TIME_TOLERANCE_SEC,
        neural_net.MAX_MISSING_LAG_TIMES_KEY: MAX_NUM_MISSING_LAG_TIMES,
        neural_net.MAX_INTERP_GAP_KEY: MAX_INTERP_GAP_SEC,
        neural_net.SENTINEL_VALUE_KEY: SENTINEL_VALUE
    }

    data_dict = neural_net.create_data(
        option_dict=option_dict, cyclone_id_string=cyclone_id_string,
        num_target_times=num_target_times
    )

    if data_dict is None:
        return

    predictor_matrices = data_dict[neural_net.PREDICTOR_MATRICES_KEY]
    target_matrix = data_dict[neural_net.TARGET_MATRIX_KEY]
    target_times_unix_sec = data_dict[neural_net.TARGET_TIMES_KEY]
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

    num_examples = predictor_matrices[0].shape[0]
    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    for i in range(num_examples):
        _plot_data_one_example(
            predictor_matrices=[p[i, ...] for p in predictor_matrices],
            target_values=target_matrix[i, ...],
            cyclone_id_string=cyclone_id_string,
            target_time_unix_sec=target_times_unix_sec[i],
            low_res_latitudes_deg_n=low_res_latitude_matrix_deg_n[i, :, 0],
            low_res_longitudes_deg_e=low_res_longitude_matrix_deg_e[i, :, 0],
            high_res_latitudes_deg_n=high_res_latitude_matrix_deg_n[i, :, 0],
            high_res_longitudes_deg_e=high_res_longitude_matrix_deg_e[i, :, 0],
            are_data_normalized=are_data_normalized,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            output_file_name='{0:s}/example{1:05d}.jpg'.format(
                output_dir_name, i
            )
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        satellite_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        cyclone_id_string=getattr(INPUT_ARG_OBJECT, CYCLONE_ID_ARG_NAME),
        num_target_times=getattr(INPUT_ARG_OBJECT, NUM_TARGET_TIMES_ARG_NAME),
        num_grid_rows_low_res=getattr(INPUT_ARG_OBJECT, NUM_GRID_ROWS_ARG_NAME),
        num_grid_columns_low_res=getattr(
            INPUT_ARG_OBJECT, NUM_GRID_COLUMNS_ARG_NAME
        ),
        num_translations=getattr(INPUT_ARG_OBJECT, NUM_TRANSLATIONS_ARG_NAME),
        mean_translation_low_res_px=getattr(
            INPUT_ARG_OBJECT, MEAN_TRANSLATION_ARG_NAME
        ),
        stdev_translation_low_res_px=getattr(
            INPUT_ARG_OBJECT, STDEV_TRANSLATION_ARG_NAME
        ),
        are_data_normalized=bool(getattr(
            INPUT_ARG_OBJECT, ARE_DATA_NORMALIZED_ARG_NAME
        )),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
