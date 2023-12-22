"""Plots satellite images for one cyclone at the given times."""

import os
import sys
import argparse
import numpy
import xarray
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import matplotlib.colors

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import longitude_conversion as lng_conversion
import file_system_utils
import error_checking
import imagemagick_utils
import border_io
import satellite_io
import normalization
import satellite_utils
import plotting_utils
import satellite_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
TIME_FORMAT = '%Y-%m-%d-%H%M'
DATE_FORMAT = '%Y-%m-%d'

METRES_TO_MICRONS = 1e6

IMAGE_CENTER_MARKER = 'o'
IMAGE_CENTER_MARKER_COLOUR = numpy.full(3, 0.)
IMAGE_CENTER_MARKER_SIZE = 9

IMAGE_CENTER_LABEL_FONT_SIZE = 32
IMAGE_CENTER_LABEL_BBOX_DICT = {
    'alpha': 0.5,
    'edgecolor': numpy.full(3, 0.),
    'linewidth': 1,
    'facecolor': numpy.full(3, 1.)
}

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300
PANEL_SIZE_PX = int(2.5e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

INPUT_DIR_ARG_NAME = 'input_satellite_dir_name'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
CYCLONE_ID_ARG_NAME = 'cyclone_id_string'
VALID_TIMES_ARG_NAME = 'valid_time_strings'
NUM_TIMES_ARG_NAME = 'num_times'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
PLOT_LATLNG_ARG_NAME = 'plot_latlng_coords'
LOW_RES_WAVELENGTHS_ARG_NAME = 'low_res_wavelengths_microns'
HIGH_RES_WAVELENGTHS_ARG_NAME = 'high_res_wavelengths_microns'
NUM_GRID_ROWS_ARG_NAME = 'num_grid_rows_low_res'
NUM_GRID_COLUMNS_ARG_NAME = 'num_grid_columns_low_res'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of directory with satellite data.  Files for the relevant cyclone '
    'will be found by `satellite_io.find_file` and read by '
    '`satellite_io.read_file`.'
)
NORMALIZATION_FILE_HELP_STRING = (
    'Path to file with normalization parameters.  If the satellite data are '
    'normalized, this file will be used to convert back to physical units '
    'before plotting.  You can also leave this empty, if you just want to plot '
    'in z-score units.'
)
CYCLONE_ID_HELP_STRING = 'Cyclone ID, in format "yyyyBBnn".'
VALID_TIMES_HELP_STRING = (
    'List of valid times, in format "yyyy-mm-dd-HHMM".  Will plot data only at '
    'these times.  If you instead want to plot at random times, use the '
    'argument {0:s} and leave this argument alone.'
).format(NUM_TIMES_ARG_NAME)

NUM_TIMES_HELP_STRING = (
    'Will plot data at this many randomly selected times.  If you instead want '
    'to plot at specific times, use the argument {0:s} and leave this argument '
    'alone.'
).format(VALID_TIMES_ARG_NAME)

DATE_HELP_STRING = (
    '[optional, used only if {0:s} is filled] Will select N times from period '
    '{1:s}...{2:s}, where N = {0:s}.'
).format(NUM_TIMES_ARG_NAME, FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

PLOT_LATLNG_HELP_STRING = (
    'Boolean flag.  If 1 (0), will plot with(out) lat-long coordinates.'
)
LOW_RES_WAVELENGTHS_HELP_STRING = (
    'Will plot low-resolution data (brightness temperature) at these '
    'wavelengths.'
)
HIGH_RES_WAVELENGTHS_HELP_STRING = (
    'Will plot high-resolution data (bidirectional reflectance) at these '
    'wavelengths.'
)
NUM_GRID_ROWS_HELP_STRING = (
    'Number of rows in low-resolution (brightness-temperature) grid.'
)
NUM_GRID_COLUMNS_HELP_STRING = (
    'Number of columns in low-resolution (brightness-temperature) grid.'
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
    '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=False, default='',
    help=NORMALIZATION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CYCLONE_ID_ARG_NAME, type=str, required=True,
    help=CYCLONE_ID_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + VALID_TIMES_ARG_NAME, type=str, nargs='+', required=False,
    default=[], help=VALID_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TIMES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=False, default='',
    help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=False, default='',
    help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_LATLNG_ARG_NAME, type=int, required=True,
    help=PLOT_LATLNG_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LOW_RES_WAVELENGTHS_ARG_NAME, type=float, nargs='+',
    required=False,
    default=[3.9, 6.185, 6.95, 7.34, 8.5, 9.61, 10.35, 11.2, 12.3, 13.3],
    help=LOW_RES_WAVELENGTHS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + HIGH_RES_WAVELENGTHS_ARG_NAME, type=float, nargs='+',
    required=False, default=[0.64], help=HIGH_RES_WAVELENGTHS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_GRID_ROWS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_GRID_ROWS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_GRID_COLUMNS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_GRID_COLUMNS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def plot_data_one_time(
        satellite_table_xarray, time_index, border_latitudes_deg_n,
        border_longitudes_deg_e, are_data_normalized, output_dir_name):
    """Plots satellite data at one time.

    P = number of points in border set

    :param satellite_table_xarray: xarray table in format returned by
        `satellite_io.read_file`.
    :param time_index: Index of time to plot.
    :param border_latitudes_deg_n: length-P numpy array of latitudes
        (deg north).  If None, will plot without coords.
    :param border_longitudes_deg_e: length-P numpy array of longitudes
        (deg east).  If None, will plot without coords.
    :param are_data_normalized: Boolean flag.
    :param output_dir_name: Name of output directory.  Image will be saved here.
    :return: output_file_name: Path to output file, where image was saved.
    """

    plot_with_coords = not (
        border_latitudes_deg_n is None or border_longitudes_deg_e is None
    )

    t = satellite_table_xarray
    cyclone_id_string = t[satellite_utils.CYCLONE_ID_KEY].values[time_index]
    if not isinstance(cyclone_id_string, str):
        cyclone_id_string = cyclone_id_string.decode('utf-8')

    valid_time_unix_sec = t.coords[satellite_utils.TIME_DIM].values[time_index]
    valid_time_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, TIME_FORMAT
    )

    high_res_wavelengths_microns = (
        METRES_TO_MICRONS *
        t.coords[satellite_utils.HIGH_RES_WAVELENGTH_DIM].values
    )
    low_res_wavelengths_microns = (
        METRES_TO_MICRONS *
        t.coords[satellite_utils.LOW_RES_WAVELENGTH_DIM].values
    )

    grid_latitudes_deg_n = (
        t[satellite_utils.LATITUDE_LOW_RES_KEY].values[time_index, :]
    )
    num_grid_rows = len(grid_latitudes_deg_n)
    i_start = int(numpy.round(
        float(num_grid_rows) / 2
    ))
    i_end = i_start + 2
    center_latitude_deg_n = numpy.mean(grid_latitudes_deg_n[i_start:i_end])

    # TODO(thunderhoser): This does not handle wrap-around at International
    # Date Line.
    grid_longitudes_deg_e = (
        t[satellite_utils.LONGITUDE_LOW_RES_KEY].values[time_index, :]
    )
    num_grid_columns = len(grid_longitudes_deg_e)
    j_start = int(numpy.round(
        float(num_grid_columns) / 2
    ))
    j_end = j_start + 2
    center_longitude_deg_e = numpy.mean(
        grid_longitudes_deg_e[j_start:j_end]
    )
    center_longitude_deg_e = lng_conversion.convert_lng_negative_in_west(
        center_longitude_deg_e
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

        if plot_with_coords:
            plotting_utils.plot_borders(
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                axes_object=axes_object
            )

            satellite_plotting.plot_2d_grid_latlng(
                data_matrix=t[
                    satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY
                ].values[time_index, ..., j],
                axes_object=axes_object,
                latitude_array_deg_n=grid_latitudes_deg_n,
                longitude_array_deg_e=grid_longitudes_deg_e,
                plotting_brightness_temp=False,
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
        else:
            satellite_plotting.plot_2d_grid_no_coords(
                data_matrix=t[
                    satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY
                ].values[time_index, ..., j],
                axes_object=axes_object,
                plotting_brightness_temp=False,
                cbar_orientation_string=None,
                colour_map_object=colour_map_object,
                colour_norm_object=colour_norm_object
            )

        axes_object.plot(
            0.5, 0.5, linestyle='None',
            marker=IMAGE_CENTER_MARKER, markersize=IMAGE_CENTER_MARKER_SIZE,
            markerfacecolor=IMAGE_CENTER_MARKER_COLOUR,
            markeredgecolor=IMAGE_CENTER_MARKER_COLOUR,
            markeredgewidth=0,
            transform=axes_object.transAxes, zorder=1e10
        )

        # label_string = (
        #     '{0:.4f}'.format(center_latitude_deg_n) + r' $^{\circ}$N' +
        #     '\n{0:.4f}'.format(center_longitude_deg_e) + r' $^{\circ}$E'
        # )
        # axes_object.text(
        #     0.55, 0.5, label_string, color=numpy.full(3, 0.),
        #     fontsize=IMAGE_CENTER_LABEL_FONT_SIZE,
        #     bbox=IMAGE_CENTER_LABEL_BBOX_DICT,
        #     horizontalalignment='left', verticalalignment='center',
        #     transform=axes_object.transAxes, zorder=1e10
        # )

        title_string = '{0:.3f}-micron BDRF for {1:s} at {2:s}'.format(
            high_res_wavelengths_microns[j], cyclone_id_string,
            valid_time_string
        )
        axes_object.set_title(title_string)

        panel_file_names[panel_index] = (
            '{0:s}/{1:s}_{2:s}_{3:06.3f}microns.jpg'
        ).format(
            output_dir_name, cyclone_id_string, valid_time_string,
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

        if plot_with_coords:
            plotting_utils.plot_borders(
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                axes_object=axes_object
            )

            grid_latitudes_deg_n = (
                t[satellite_utils.LATITUDE_LOW_RES_KEY].values[time_index, :]
            )
            grid_longitudes_deg_e = (
                t[satellite_utils.LONGITUDE_LOW_RES_KEY].values[time_index, :]
            )

            satellite_plotting.plot_2d_grid_latlng(
                data_matrix=t[
                    satellite_utils.BRIGHTNESS_TEMPERATURE_KEY
                ].values[time_index, ..., j],
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
        else:
            satellite_plotting.plot_2d_grid_no_coords(
                data_matrix=t[
                    satellite_utils.BRIGHTNESS_TEMPERATURE_KEY
                ].values[time_index, ..., j],
                axes_object=axes_object,
                plotting_brightness_temp=True,
                cbar_orientation_string=None,
                colour_map_object=colour_map_object,
                colour_norm_object=colour_norm_object
            )

        axes_object.plot(
            0.5, 0.5, linestyle='None',
            marker=IMAGE_CENTER_MARKER, markersize=IMAGE_CENTER_MARKER_SIZE,
            markerfacecolor=IMAGE_CENTER_MARKER_COLOUR,
            markeredgecolor=IMAGE_CENTER_MARKER_COLOUR,
            markeredgewidth=0,
            transform=axes_object.transAxes, zorder=1e10
        )

        # label_string = (
        #     '{0:.4f}'.format(center_latitude_deg_n) + r' $^{\circ}$N' +
        #     '\n{0:.4f}'.format(center_longitude_deg_e) + r' $^{\circ}$E'
        # )
        # axes_object.text(
        #     0.55, 0.5, label_string, color=numpy.full(3, 0.),
        #     fontsize=IMAGE_CENTER_LABEL_FONT_SIZE,
        #     bbox=IMAGE_CENTER_LABEL_BBOX_DICT,
        #     horizontalalignment='left', verticalalignment='center',
        #     transform=axes_object.transAxes, zorder=1e10
        # )

        title_string = r'{0:.3f}-micron $T_b$ for {1:s} at {2:s}'.format(
            low_res_wavelengths_microns[j], cyclone_id_string,
            valid_time_string
        )
        axes_object.set_title(title_string)

        panel_file_names[panel_index] = (
            '{0:s}/{1:s}_{2:s}_{3:06.3f}microns.jpg'
        ).format(
            output_dir_name, cyclone_id_string, valid_time_string,
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

    concat_figure_file_name = '{0:s}/{1:s}_{2:s}.jpg'.format(
        output_dir_name, cyclone_id_string, valid_time_string
    )
    plotting_utils.concat_panels(
        panel_file_names=panel_file_names,
        concat_figure_file_name=concat_figure_file_name
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
            figure_file_name=concat_figure_file_name,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            orientation_string='vertical', font_size=20,
            cbar_label_string=(
                r'BDRF ($z$-score)' if are_data_normalized
                else 'BDRF (unitless)'
            ),
            tick_label_format_string='{0:.2f}', log_space=False,
            temporary_cbar_file_name='{0:s}_cbar.jpg'.format(
                concat_figure_file_name[:-4]
            )
        )

    if len(low_res_wavelengths_microns) > 0:
        if are_data_normalized:
            colour_map_object = pyplot.get_cmap('seismic', lut=1001)
            colour_norm_object = matplotlib.colors.Normalize(vmin=-3., vmax=3.)
        else:
            colour_map_object, colour_norm_object = (
                satellite_plotting.get_colour_scheme_for_brightness_temp()
            )

        plotting_utils.add_colour_bar(
            figure_file_name=concat_figure_file_name,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            orientation_string='vertical', font_size=20,
            cbar_label_string=(
                r'$T_b$ ($z-score$)' if are_data_normalized
                else r'$T_b$ (Kelvins)'
            ),
            tick_label_format_string='{0:.0f}', log_space=False,
            temporary_cbar_file_name='{0:s}_cbar.jpg'.format(
                concat_figure_file_name[:-4]
            )
        )


def _run(satellite_dir_name, normalization_file_name, cyclone_id_string,
         valid_time_strings, num_times, first_date_string, last_date_string,
         plot_latlng_coords,
         low_res_wavelengths_microns, high_res_wavelengths_microns,
         num_grid_rows_low_res, num_grid_columns_low_res, output_dir_name):
    """Plots satellite images for one cyclone at the given times.

    This is effectively the main method.

    :param satellite_dir_name: See documentation at top of file.
    :param normalization_file_name: Same.
    :param cyclone_id_string: Same.
    :param valid_time_strings: Same.
    :param num_times: Same.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param plot_latlng_coords: Same.
    :param low_res_wavelengths_microns: Same.
    :param high_res_wavelengths_microns: Same.
    :param num_grid_rows_low_res: Same.
    :param num_grid_columns_low_res: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if `first_date_string` and `last_date_string` result in
        no files being found.
    """

    # Check input args.
    if (
            len(high_res_wavelengths_microns) == 1
            and high_res_wavelengths_microns[0] < 0
    ):
        high_res_wavelengths_microns = numpy.array([])

    if normalization_file_name == '':
        normalization_file_name = None

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    if len(valid_time_strings) == 0:
        valid_time_strings = None
        valid_times_unix_sec = None

        error_checking.assert_is_greater(num_times, 0)

        if first_date_string == '' or last_date_string == '':
            first_date_string = None
            last_date_string = None
        else:
            first_date_unix_sec = time_conversion.string_to_unix_sec(
                first_date_string, DATE_FORMAT
            )
            last_date_unix_sec = time_conversion.string_to_unix_sec(
                last_date_string, DATE_FORMAT
            )
            error_checking.assert_is_geq(
                last_date_unix_sec, first_date_unix_sec
            )
    else:
        valid_times_unix_sec = numpy.array([
            time_conversion.string_to_unix_sec(t, TIME_FORMAT)
            for t in valid_time_strings
        ], dtype=int)

        first_date_string = None
        last_date_string = None
        first_date_unix_sec = None
        last_date_unix_sec = None

    if num_grid_rows_low_res <= 0 or num_grid_columns_low_res <= 0:
        num_grid_rows_low_res = None
        num_grid_columns_low_res = None

    # Do actual stuff.
    satellite_file_names = satellite_io.find_files_one_cyclone(
        directory_name=satellite_dir_name, cyclone_id_string=cyclone_id_string,
        raise_error_if_all_missing=True
    )

    if first_date_string is not None:
        file_date_strings = [
            satellite_io.file_name_to_date(f) for f in satellite_file_names
        ]
        file_dates_unix_sec = numpy.array([
            time_conversion.string_to_unix_sec(d, satellite_io.DATE_FORMAT)
            for d in file_date_strings
        ], dtype=int)

        good_indices = numpy.where(numpy.logical_and(
            file_dates_unix_sec >= first_date_unix_sec,
            file_dates_unix_sec <= last_date_unix_sec
        ))[0]

        if len(good_indices) == 0:
            error_string = (
                'Cannot find data between dates {0:s} and {1:s}.  Found the '
                'following files:\n{2:s}'
            ).format(
                first_date_string, last_date_string, str(satellite_file_names)
            )

            raise ValueError(error_string)

        satellite_file_names = [satellite_file_names[k] for k in good_indices]
        del file_date_strings
        del file_dates_unix_sec

    all_times_unix_sec = numpy.concatenate([
        xarray.open_zarr(f).coords[satellite_utils.TIME_DIM].values
        for f in satellite_file_names
    ])

    if valid_times_unix_sec is None:
        valid_times_unix_sec = all_times_unix_sec + 0

        if num_times < len(valid_times_unix_sec):
            valid_times_unix_sec = numpy.random.choice(
                valid_times_unix_sec, size=num_times, replace=False
            )
    else:
        assert numpy.all(
            numpy.isin(valid_times_unix_sec, test_elements=all_times_unix_sec)
        )

    satellite_tables_xarray = []
    are_data_normalized = False

    for this_file_name in satellite_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        satellite_table_xarray = satellite_io.read_file(this_file_name)

        these_flags = numpy.isin(
            valid_times_unix_sec,
            test_elements=
            satellite_table_xarray.coords[satellite_utils.TIME_DIM].values
        )

        if not numpy.any(these_flags):
            continue

        satellite_table_xarray = (
            satellite_utils.subset_to_multiple_time_windows(
                satellite_table_xarray=satellite_table_xarray,
                start_times_unix_sec=valid_times_unix_sec[these_flags],
                end_times_unix_sec=valid_times_unix_sec[these_flags]
            )
        )

        stx = satellite_table_xarray
        are_data_normalized = not numpy.any(
            stx[satellite_utils.BRIGHTNESS_TEMPERATURE_KEY].values > 20.
        )

        satellite_table_xarray = satellite_utils.subset_wavelengths(
            satellite_table_xarray=satellite_table_xarray,
            wavelengths_to_keep_microns=low_res_wavelengths_microns,
            for_high_res=False
        )

        try:
            satellite_table_xarray = satellite_utils.subset_wavelengths(
                satellite_table_xarray=satellite_table_xarray,
                wavelengths_to_keep_microns=high_res_wavelengths_microns,
                for_high_res=True
            )
        except KeyError:
            pass

        if num_grid_rows_low_res is not None:
            satellite_table_xarray = satellite_utils.subset_grid(
                satellite_table_xarray=satellite_table_xarray,
                num_rows_to_keep=num_grid_rows_low_res,
                num_columns_to_keep=num_grid_columns_low_res,
                for_high_res=False
            )

            if len(high_res_wavelengths_microns) > 0:
                satellite_table_xarray = satellite_utils.subset_grid(
                    satellite_table_xarray=satellite_table_xarray,
                    num_rows_to_keep=4 * num_grid_rows_low_res,
                    num_columns_to_keep=4 * num_grid_columns_low_res,
                    for_high_res=True
                )

        satellite_tables_xarray.append(satellite_table_xarray)

    satellite_table_xarray = satellite_utils.concat_over_time(
        satellite_tables_xarray
    )
    del satellite_tables_xarray

    if are_data_normalized and normalization_file_name is not None:
        print('Reading data from: "{0:s}"...'.format(normalization_file_name))
        norm_param_table_xarray = normalization.read_file(
            normalization_file_name
        )

        satellite_table_xarray = normalization.denormalize_data(
            satellite_table_xarray=satellite_table_xarray,
            normalization_param_table_xarray=norm_param_table_xarray
        )
        are_data_normalized = False

    if plot_latlng_coords:
        border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()
    else:
        border_latitudes_deg_n = None
        border_longitudes_deg_e = None

    num_times = len(
        satellite_table_xarray.coords[satellite_utils.TIME_DIM].values
    )

    for i in range(num_times):
        plot_data_one_time(
            satellite_table_xarray=satellite_table_xarray, time_index=i,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            are_data_normalized=are_data_normalized,
            output_dir_name=output_dir_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        satellite_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        cyclone_id_string=getattr(INPUT_ARG_OBJECT, CYCLONE_ID_ARG_NAME),
        valid_time_strings=getattr(INPUT_ARG_OBJECT, VALID_TIMES_ARG_NAME),
        num_times=getattr(INPUT_ARG_OBJECT, NUM_TIMES_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        plot_latlng_coords=bool(
            getattr(INPUT_ARG_OBJECT, PLOT_LATLNG_ARG_NAME)
        ),
        low_res_wavelengths_microns=numpy.array(
            getattr(INPUT_ARG_OBJECT, LOW_RES_WAVELENGTHS_ARG_NAME), dtype=float
        ),
        high_res_wavelengths_microns=numpy.array(
            getattr(INPUT_ARG_OBJECT, HIGH_RES_WAVELENGTHS_ARG_NAME),
            dtype=float
        ),
        num_grid_rows_low_res=getattr(INPUT_ARG_OBJECT, NUM_GRID_ROWS_ARG_NAME),
        num_grid_columns_low_res=getattr(
            INPUT_ARG_OBJECT, NUM_GRID_COLUMNS_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
