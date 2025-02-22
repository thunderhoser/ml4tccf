"""Plots data augmentation."""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import matplotlib.colors
from scipy.interpolate import interp2d
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import imagemagick_utils
from ml4tccf.io import border_io
from ml4tccf.machine_learning import neural_net_utils as nn_utils
from ml4tccf.machine_learning import \
    neural_net_training_simple as nn_training_simple
from ml4tccf.machine_learning import \
    neural_net_training_fancy as nn_training_fancy
from ml4tccf.machine_learning import \
    neural_net_training_cira_ir as nn_training_cira_ir
from ml4tccf.plotting import plotting_utils
from ml4tccf.plotting import satellite_plotting

TOLERANCE = 1e-6
TIME_FORMAT = '%Y-%m-%d-%H%M'

LAG_TIMES_MINUTES = numpy.array([0], dtype=int)
LOW_RES_WAVELENGTHS_RG_MICRONS = numpy.array([
    3.9, 6.185, 6.95, 7.34, 8.5, 9.61, 10.35, 11.2, 12.3, 13.3
])
HIGH_RES_WAVELENGTHS_RG_MICRONS = numpy.array([0.64])
LOW_RES_WAVELENGTHS_CIRA_IR_MICRONS = numpy.array([11.2])
HIGH_RES_WAVELENGTHS_CIRA_IR_MICRONS = numpy.array([])

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
DATA_TYPE_ARG_NAME = 'data_type_string'
CYCLONE_ID_ARG_NAME = 'cyclone_id_string'
NUM_TARGET_TIMES_ARG_NAME = 'num_target_times'
NUM_GRID_ROWS_ARG_NAME = 'num_grid_rows_low_res'
NUM_GRID_COLUMNS_ARG_NAME = 'num_grid_columns_low_res'
LOW_RES_WAVELENGTHS_ARG_NAME = 'low_res_wavelengths_microns'
HIGH_RES_WAVELENGTHS_ARG_NAME = 'high_res_wavelengths_microns'
NUM_TRANSLATIONS_ARG_NAME = 'num_translations'
MEAN_TRANSLATION_ARG_NAME = 'mean_translation_low_res_px'
STDEV_TRANSLATION_ARG_NAME = 'stdev_translation_low_res_px'
ARE_DATA_NORMALIZED_ARG_NAME = 'are_data_normalized'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing satellite data.  Files therein will '
    'be found by `satellite_io.find_file` and read by `satellite_io.read_file`.'
)
DATA_TYPE_HELP_STRING = (
    'Data type.  Must be one of the following:\n{0:s}'
).format(str(nn_utils.VALID_DATA_TYPE_STRINGS))

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
LOW_RES_WAVELENGTHS_HELP_STRING = (
    'Low-resolution wavelengths to plot.  To use the default (depending on '
    'whether the data source is CIRA IR or Robert/Galina), leave this argument '
    'alone.'
)
HIGH_RES_WAVELENGTHS_HELP_STRING = (
    'High-resolution wavelengths to plot.  To use the default (depending on '
    'whether the data source is CIRA IR or Robert/Galina), leave this argument '
    'alone.  To omit high-res data completely, make this a one-item list with '
    'zero only.'
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
    '--' + DATA_TYPE_ARG_NAME, type=str, required=True,
    help=DATA_TYPE_HELP_STRING
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
    '--' + LOW_RES_WAVELENGTHS_ARG_NAME, type=float, nargs='+', required=False,
    default=[-1], help=LOW_RES_WAVELENGTHS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + HIGH_RES_WAVELENGTHS_ARG_NAME, type=float, nargs='+', required=False,
    default=[-1], help=HIGH_RES_WAVELENGTHS_HELP_STRING
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
        target_time_unix_sec, low_res_wavelengths_microns,
        low_res_latitudes_deg_n, low_res_longitudes_deg_e,
        high_res_wavelengths_microns,
        high_res_latitudes_deg_n, high_res_longitudes_deg_e,
        are_data_normalized, border_latitudes_deg_n, border_longitudes_deg_e,
        output_file_name):
    """Plots satellite data for one example.

    P = number of points in border set

    :param predictor_matrices: Same as output from `nn_training*.create_data`
        but without first axis.
    :param target_values: Same as output from `nn_training*.create_data` but
        without first axis.
    :param cyclone_id_string: Cyclone ID.
    :param target_time_unix_sec: Target time.
    :param low_res_wavelengths_microns: 1-D numpy array of wavelengths for low-
        resolution data.
    :param low_res_latitudes_deg_n: Same as output from
        `nn_training*.create_data` but without first or last axis.
    :param low_res_longitudes_deg_e: Same as output from
        `nn_training*.create_data` but without first or last axis.
    :param high_res_wavelengths_microns: 1-D numpy array of wavelengths for
        high-resolution data.
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
    """

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

    regular_grids = len(low_res_latitudes_deg_n.shape) == 1

    # TODO(thunderhoser): This will not handle wrap-around at International Date
    # Line.
    if regular_grids:
        low_res_latitude_interp_object = None
        low_res_longitude_interp_object = None
    else:
        low_res_latitude_interp_object = interp2d(
            x=column_indices_low_res, y=row_indices_low_res,
            z=low_res_latitudes_deg_n, kind='linear', bounds_error=True
        )
        low_res_longitude_interp_object = interp2d(
            x=column_indices_low_res, y=row_indices_low_res,
            z=low_res_longitudes_deg_e, kind='linear', bounds_error=True
        )

    num_panels = (
        len(high_res_wavelengths_microns) + len(low_res_wavelengths_microns)
    )
    panel_file_names = [''] * num_panels
    panel_index = -1

    for j in range(len(high_res_wavelengths_microns)):
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
            data_matrix=bidirectional_reflectance_matrix[..., j],
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

        if regular_grids:
            x_coord = 0.5 + target_values[1] / num_grid_columns_low_res
            y_coord = 0.5 + target_values[0] / num_grid_rows_low_res
            transform_object = axes_object.transAxes
        else:
            y_coord = low_res_latitude_interp_object(
                center_column_index_low_res + target_values[1],
                center_row_index_low_res + target_values[0]
            )
            x_coord = low_res_longitude_interp_object(
                center_column_index_low_res + target_values[1],
                center_row_index_low_res + target_values[0]
            )
            transform_object = axes_object.transData

        axes_object.plot(
            x_coord, y_coord, linestyle='None',
            marker=CYCLONE_CENTER_MARKER, markersize=CYCLONE_CENTER_MARKER_SIZE,
            markerfacecolor=CYCLONE_CENTER_MARKER_COLOUR,
            markeredgecolor=CYCLONE_CENTER_MARKER_COLOUR,
            markeredgewidth=0,
            transform=transform_object, zorder=1e10
        )

        title_string = '{0:.3f}-micron BDRF for {1:s} at {2:s}'.format(
            high_res_wavelengths_microns[j],
            cyclone_id_string,
            time_conversion.unix_sec_to_string(
                target_time_unix_sec, TIME_FORMAT
            )
        )
        axes_object.set_title(title_string)

        panel_file_names[panel_index] = '{0:s}_{1:06.3f}microns.jpg'.format(
            os.path.splitext(output_file_name)[0],
            high_res_wavelengths_microns[j]
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

    for j in range(len(low_res_wavelengths_microns)):
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
            data_matrix=brightness_temp_matrix[..., j],
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

        if regular_grids:
            x_coord = 0.5 + target_values[1] / num_grid_columns_low_res
            y_coord = 0.5 + target_values[0] / num_grid_rows_low_res
            transform_object = axes_object.transAxes
        else:
            y_coord = low_res_latitude_interp_object(
                center_column_index_low_res + target_values[1],
                center_row_index_low_res + target_values[0]
            )
            x_coord = low_res_longitude_interp_object(
                center_column_index_low_res + target_values[1],
                center_row_index_low_res + target_values[0]
            )
            transform_object = axes_object.transData

        axes_object.plot(
            x_coord, y_coord, linestyle='None',
            marker=CYCLONE_CENTER_MARKER, markersize=CYCLONE_CENTER_MARKER_SIZE,
            markerfacecolor=CYCLONE_CENTER_MARKER_COLOUR,
            markeredgecolor=CYCLONE_CENTER_MARKER_COLOUR,
            markeredgewidth=0,
            transform=transform_object, zorder=1e10
        )

        title_string = r'{0:.3f}-micron $T_b$ for {1:s} at {2:s}'.format(
            low_res_wavelengths_microns[j],
            cyclone_id_string,
            time_conversion.unix_sec_to_string(
                target_time_unix_sec, TIME_FORMAT
            )
        )
        axes_object.set_title(title_string)

        panel_file_names[panel_index] = '{0:s}_{1:06.3f}microns.jpg'.format(
            os.path.splitext(output_file_name)[0],
            low_res_wavelengths_microns[j]
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

    if len(high_res_wavelengths_microns) > 0:
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
            temporary_cbar_file_name='{0:s}_cbar.jpg'.format(
                output_file_name[:-4]
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


def _run(satellite_dir_name, data_type_string, cyclone_id_string,
         num_target_times, num_grid_rows_low_res, num_grid_columns_low_res,
         low_res_wavelengths_microns, high_res_wavelengths_microns,
         num_translations, mean_translation_low_res_px,
         stdev_translation_low_res_px, are_data_normalized, output_dir_name):
    """Plots data augmentation.

    This is effectively the main method.

    :param satellite_dir_name: See documentation at top of file.
    :param data_type_string: Same.
    :param cyclone_id_string: Same.
    :param num_target_times: Same.
    :param num_grid_rows_low_res: Same.
    :param num_grid_columns_low_res: Same.
    :param low_res_wavelengths_microns: Same.
    :param high_res_wavelengths_microns: Same.
    :param num_translations: Same.
    :param mean_translation_low_res_px: Same.
    :param stdev_translation_low_res_px: Same.
    :param are_data_normalized: Same.
    :param output_dir_name: Same.
    """

    if data_type_string == nn_utils.CIRA_IR_DATA_TYPE_STRING:
        low_res_wavelengths_microns = LOW_RES_WAVELENGTHS_CIRA_IR_MICRONS
        high_res_wavelengths_microns = HIGH_RES_WAVELENGTHS_CIRA_IR_MICRONS

    if (
            len(low_res_wavelengths_microns) == 1
            and low_res_wavelengths_microns[0] < 0
    ):
        low_res_wavelengths_microns = None

    if len(high_res_wavelengths_microns) == 1:
        if numpy.isclose(high_res_wavelengths_microns[0], 0, atol=TOLERANCE):
            high_res_wavelengths_microns = numpy.array([])
        elif high_res_wavelengths_microns[0] < 0:
            high_res_wavelengths_microns = None

    if low_res_wavelengths_microns is None:
        low_res_wavelengths_microns = LOW_RES_WAVELENGTHS_RG_MICRONS
    if high_res_wavelengths_microns is None:
        high_res_wavelengths_microns = HIGH_RES_WAVELENGTHS_RG_MICRONS

    option_dict = {
        nn_utils.SATELLITE_DIRECTORY_KEY: satellite_dir_name,
        nn_utils.YEARS_KEY: numpy.array([2000], dtype=int),
        nn_utils.LAG_TIMES_KEY: LAG_TIMES_MINUTES,
        nn_utils.HIGH_RES_WAVELENGTHS_KEY: high_res_wavelengths_microns,
        nn_utils.LOW_RES_WAVELENGTHS_KEY: low_res_wavelengths_microns,
        nn_utils.BATCH_SIZE_KEY: 1,
        nn_utils.MAX_EXAMPLES_PER_CYCLONE_KEY: 1,
        nn_utils.NUM_GRID_ROWS_KEY: num_grid_rows_low_res,
        nn_utils.NUM_GRID_COLUMNS_KEY: num_grid_columns_low_res,
        nn_utils.DATA_AUG_NUM_TRANS_KEY: num_translations,
        nn_utils.DATA_AUG_MEAN_TRANS_KEY: mean_translation_low_res_px,
        nn_utils.DATA_AUG_STDEV_TRANS_KEY: stdev_translation_low_res_px,
        nn_utils.LAG_TIME_TOLERANCE_KEY: LAG_TIME_TOLERANCE_SEC,
        nn_utils.MAX_MISSING_LAG_TIMES_KEY: MAX_NUM_MISSING_LAG_TIMES,
        nn_utils.MAX_INTERP_GAP_KEY: MAX_INTERP_GAP_SEC,
        nn_utils.SENTINEL_VALUE_KEY: SENTINEL_VALUE,
        nn_utils.SEMANTIC_SEG_FLAG_KEY: False,
        nn_utils.TARGET_SMOOOTHER_STDEV_KEY: 1e-6
    }

    if data_type_string == nn_utils.CIRA_IR_DATA_TYPE_STRING:
        data_dict = nn_training_cira_ir.create_data(
            option_dict=option_dict, cyclone_id_string=cyclone_id_string,
            num_target_times=num_target_times
        )
    elif data_type_string == nn_utils.RG_SIMPLE_DATA_TYPE_STRING:
        data_dict = nn_training_simple.create_data(
            option_dict=option_dict, cyclone_id_string=cyclone_id_string,
            num_target_times=num_target_times
        )
    else:
        data_dict = nn_training_fancy.create_data(
            option_dict=option_dict, cyclone_id_string=cyclone_id_string,
            num_target_times=num_target_times
        )

    if data_dict is None:
        return

    predictor_matrices = data_dict[nn_utils.PREDICTOR_MATRICES_KEY]
    target_matrix = data_dict[nn_utils.TARGET_MATRIX_KEY]
    target_times_unix_sec = data_dict[nn_utils.TARGET_TIMES_KEY]
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

    num_examples = predictor_matrices[0].shape[0]
    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    for i in range(num_examples):
        this_index = i - numpy.where(
            target_times_unix_sec == target_times_unix_sec[i]
        )[0][0]

        output_file_name = '{0:s}/{1:s}_{2:s}_{3:03d}th.jpg'.format(
            output_dir_name,
            cyclone_id_string,
            time_conversion.unix_sec_to_string(
                target_times_unix_sec[i], TIME_FORMAT
            ),
            this_index
        )

        _plot_data_one_example(
            predictor_matrices=[p[i, ...] for p in predictor_matrices],
            target_values=target_matrix[i, ...],
            cyclone_id_string=cyclone_id_string,
            target_time_unix_sec=target_times_unix_sec[i],
            low_res_wavelengths_microns=low_res_wavelengths_microns,
            low_res_latitudes_deg_n=low_res_latitude_matrix_deg_n[i, ..., 0],
            low_res_longitudes_deg_e=low_res_longitude_matrix_deg_e[i, ..., 0],
            high_res_wavelengths_microns=high_res_wavelengths_microns,
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
            output_file_name=output_file_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        satellite_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        data_type_string=getattr(INPUT_ARG_OBJECT, DATA_TYPE_ARG_NAME),
        cyclone_id_string=getattr(INPUT_ARG_OBJECT, CYCLONE_ID_ARG_NAME),
        num_target_times=getattr(INPUT_ARG_OBJECT, NUM_TARGET_TIMES_ARG_NAME),
        num_grid_rows_low_res=getattr(INPUT_ARG_OBJECT, NUM_GRID_ROWS_ARG_NAME),
        num_grid_columns_low_res=getattr(
            INPUT_ARG_OBJECT, NUM_GRID_COLUMNS_ARG_NAME
        ),
        low_res_wavelengths_microns=numpy.array(
            getattr(INPUT_ARG_OBJECT, LOW_RES_WAVELENGTHS_ARG_NAME),
            dtype=float
        ),
        high_res_wavelengths_microns=numpy.array(
            getattr(INPUT_ARG_OBJECT, HIGH_RES_WAVELENGTHS_ARG_NAME),
            dtype=float
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
