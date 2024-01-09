"""Plots translation schematic for 2024 WAF paper.

WAF = Weather and Forecasting
"""

import copy
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from ml4tccf.io import satellite_io
from ml4tccf.utils import satellite_utils
from ml4tccf.utils import normalization
from ml4tccf.machine_learning import data_augmentation
from ml4tccf.plotting import plotting_utils
from ml4tccf.plotting import satellite_plotting

TIME_FORMAT = '%Y-%m-%d-%H%M'

WAVELENGTH_MICRONS = 8.5
NUM_GRID_ROWS = 300
NUM_GRID_COLUMNS = 300

MEAN_TRANSLATION_DISTANCE_PX = 24.
STDEV_TRANSLATION_DISTANCE_PX = 12.
SAMPLE_SIZE_FOR_DISTRIBUTION = int(5e6)
GRID_SPACING_KM = 2.

ACTUAL_COLUMN_TRANSLATION_DISTANCE_PX = -40
ACTUAL_ROW_TRANSLATION_DISTANCE_PX = 30

LOWER_BIN_EDGE_KM = -100.
UPPER_BIN_EDGE_KM = 100.
NUM_BINS = 100

DEFAULT_FONT_SIZE = 40
IMAGE_CENTER_MARKER = '*'
IMAGE_CENTER_MARKER_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
IMAGE_CENTER_MARKER_SIZE = 75

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

PANEL_SIZE_PX = int(5e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

SATELLITE_DIR_ARG_NAME = 'input_satellite_dir_name'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
CYCLONE_ID_ARG_NAME = 'cyclone_id_string'
VALID_TIME_ARG_NAME = 'valid_time_string'
OUTPUT_DIR_ARG_NAME = 'output_figure_dir_name'

SATELLITE_DIR_HELP_STRING = (
    'Name of directory with satellite data.  Files therein will be found by '
    '`satellite_io.find_file` and read by `satellite_io.read_file`.'
)
NORMALIZATION_FILE_HELP_STRING = (
    'Path to file with normalization parameters.  If the satellite data are '
    'normalized, this file will be used to convert back to physical units '
    'before plotting.'
)
CYCLONE_ID_HELP_STRING = 'Cyclone ID.'
VALID_TIME_HELP_STRING = (
    'Valid time shown in figure (format "yyyy-mm-dd-HHMM").'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SATELLITE_DIR_ARG_NAME, type=str, required=True,
    help=SATELLITE_DIR_HELP_STRING
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
    '--' + VALID_TIME_ARG_NAME, type=str, required=True,
    help=VALID_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(satellite_dir_name, normalization_file_name, cyclone_id_string,
         valid_time_string, output_dir_name):
    """Plots translation schematic for 2024 WAF paper.

    This is effectively the main method.

    :param satellite_dir_name: See documentation at top of file.
    :param normalization_file_name: Same.
    :param cyclone_id_string: Same.
    :param valid_time_string: Same.
    :param output_dir_name: Same.
    """

    # TODO(thunderhoser): Remember to say in caption that same thing is done for all lag times and channels.

    # Check input args.
    valid_time_unix_sec = time_conversion.string_to_unix_sec(
        valid_time_string, TIME_FORMAT
    )
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Do actual stuff.
    valid_date_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, '%Y-%m-%d'
    )
    satellite_file_name = satellite_io.find_file(
        directory_name=satellite_dir_name,
        cyclone_id_string=cyclone_id_string,
        valid_date_string=valid_date_string,
        raise_error_if_missing=True
    )

    print('Reading data from: "{0:s}"...'.format(satellite_file_name))
    satellite_table_xarray = satellite_io.read_file(satellite_file_name)

    satellite_table_xarray = satellite_utils.subset_to_multiple_time_windows(
        satellite_table_xarray=satellite_table_xarray,
        start_times_unix_sec=numpy.array([valid_time_unix_sec], dtype=int),
        end_times_unix_sec=numpy.array([valid_time_unix_sec], dtype=int)
    )
    satellite_table_xarray = satellite_utils.subset_wavelengths(
        satellite_table_xarray=satellite_table_xarray,
        wavelengths_to_keep_microns=numpy.array([WAVELENGTH_MICRONS]),
        for_high_res=False
    )

    num_extra_rowcols = 2 * int(numpy.ceil(
        MEAN_TRANSLATION_DISTANCE_PX +
        5 * STDEV_TRANSLATION_DISTANCE_PX
    ))
    num_grid_rows_with_extra = NUM_GRID_ROWS + num_extra_rowcols
    num_grid_columns_with_extra = NUM_GRID_COLUMNS + num_extra_rowcols

    satellite_table_with_extra_xarray = satellite_utils.subset_grid(
        satellite_table_xarray=satellite_table_xarray,
        num_rows_to_keep=num_grid_rows_with_extra,
        num_columns_to_keep=num_grid_columns_with_extra,
        for_high_res=False
    )
    satellite_table_xarray = satellite_utils.subset_grid(
        satellite_table_xarray=copy.deepcopy(satellite_table_with_extra_xarray),
        num_rows_to_keep=NUM_GRID_ROWS,
        num_columns_to_keep=NUM_GRID_COLUMNS,
        for_high_res=False
    )

    stx = satellite_table_xarray
    are_data_normalized = not numpy.any(
        stx[satellite_utils.BRIGHTNESS_TEMPERATURE_KEY].values > 20.
    )

    if are_data_normalized:
        print('Reading data from: "{0:s}"...'.format(normalization_file_name))
        norm_param_table_xarray = normalization.read_file(
            normalization_file_name
        )

        satellite_table_xarray = normalization.denormalize_data(
            satellite_table_xarray=satellite_table_xarray,
            normalization_param_table_xarray=norm_param_table_xarray
        )
        stx = satellite_table_xarray
    else:
        norm_param_table_xarray = None

    letter_label = None
    panel_file_names = []

    row_translations_px, column_translations_px = (
        data_augmentation.get_translation_distances(
            mean_translation_px=MEAN_TRANSLATION_DISTANCE_PX,
            stdev_translation_px=STDEV_TRANSLATION_DISTANCE_PX,
            num_translations=SAMPLE_SIZE_FOR_DISTRIBUTION
        )
    )

    actual_x_offsets_km = GRID_SPACING_KM * column_translations_px
    actual_y_offsets_km = GRID_SPACING_KM * row_translations_px
    predicted_x_offsets_km = numpy.full(actual_x_offsets_km.shape, 0.)
    predicted_y_offsets_km = numpy.full(actual_y_offsets_km.shape, 0.)

    bin_edges_km = numpy.linspace(
        LOWER_BIN_EDGE_KM, UPPER_BIN_EDGE_KM, num=NUM_BINS + 1, dtype=float
    )
    bin_centers_km = bin_edges_km[:-1] + numpy.diff(bin_edges_km) / 2

    bin_count_matrix = grids.count_events_on_equidistant_grid(
        event_x_coords_metres=predicted_x_offsets_km - actual_x_offsets_km,
        event_y_coords_metres=predicted_y_offsets_km - actual_y_offsets_km,
        grid_point_x_coords_metres=bin_centers_km,
        grid_point_y_coords_metres=bin_centers_km
    )[0]
    bin_frequency_matrix = (
        bin_count_matrix.astype(float) / numpy.sum(bin_count_matrix)
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    min_colour_value = numpy.min(bin_frequency_matrix)
    max_colour_value = numpy.percentile(bin_frequency_matrix, 99.5)
    colour_norm_object = pyplot.Normalize(
        vmin=min_colour_value, vmax=max_colour_value
    )
    axes_object.imshow(
        bin_frequency_matrix, origin='lower',
        cmap=pyplot.get_cmap('cividis'), norm=colour_norm_object
    )

    axes_object.set_title(
        'Distribution of translation vectors\n'
        '(= error dist without NN correction)',
        fontsize=DEFAULT_FONT_SIZE
    )
    axes_object.set_xlabel(r'$x$-translation distance (km)')
    axes_object.set_ylabel(r'$y$-translation distance (km)')

    tick_coords_km = numpy.linspace(
        LOWER_BIN_EDGE_KM, UPPER_BIN_EDGE_KM, num=9, dtype=float
    )
    tick_coord_labels = ['{0:.0f}'.format(t) for t in tick_coords_km]

    plot_grid_spacing_km = numpy.diff(bin_centers_km)[0]
    tick_coords_px = -0.5 + (
        (tick_coords_km - tick_coords_km[0]) / plot_grid_spacing_km
    )

    axes_object.set_xticks(tick_coords_px)
    axes_object.set_xticklabels(tick_coord_labels)
    axes_object.set_yticks(tick_coords_px)
    axes_object.set_yticklabels(tick_coord_labels)
    axes_object.grid(
        b=True, which='major', axis='both', linestyle='--',
        linewidth=2, color=numpy.full(3, 0.)
    )

    if letter_label is None:
        letter_label = 'a'
    else:
        letter_label = chr(ord(letter_label) + 1)

    gg_plotting_utils.label_axes(
        axes_object=axes_object, label_string='({0:s})'.format(letter_label)
    )

    panel_file_names.append(
        '{0:s}/translation_vector_distribution.jpg'.format(output_dir_name)
    )

    print('Saving figure to file: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    plotting_utils.add_colour_bar(
        figure_file_name=panel_file_names[-1],
        colour_map_object=pyplot.get_cmap('cividis'),
        colour_norm_object=colour_norm_object,
        orientation_string='vertical',
        font_size=30,
        cbar_label_string='Frequency',
        tick_label_format_string='{0:.2g}',
        log_space=False,
        temporary_cbar_file_name=
        '{0:s}/translation_vector_dist_cbar.jpg'.format(output_dir_name)
    )

    brightness_temp_matrix_kelvins = (
        stx[satellite_utils.BRIGHTNESS_TEMPERATURE_KEY].values[0, ..., 0]
    ).astype(numpy.float64)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    satellite_plotting.plot_2d_grid_no_coords(
        data_matrix=brightness_temp_matrix_kelvins,
        axes_object=axes_object,
        plotting_brightness_temp=True,
        cbar_orientation_string=None,
        font_size=DEFAULT_FONT_SIZE
    )
    axes_object.plot(
        0.5, 0.5, linestyle='None',
        marker=IMAGE_CENTER_MARKER, markersize=IMAGE_CENTER_MARKER_SIZE,
        markerfacecolor=IMAGE_CENTER_MARKER_COLOUR,
        markeredgecolor=IMAGE_CENTER_MARKER_COLOUR,
        markeredgewidth=0,
        transform=axes_object.transAxes, zorder=1e10
    )

    if letter_label is None:
        letter_label = 'a'
    else:
        letter_label = chr(ord(letter_label) + 1)

    gg_plotting_utils.label_axes(
        axes_object=axes_object, label_string='({0:s})'.format(letter_label)
    )
    title_string = 'Original image, {0:d} x {1:d}'.format(
        brightness_temp_matrix_kelvins.shape[0],
        brightness_temp_matrix_kelvins.shape[1]
    )
    axes_object.set_title(title_string, fontsize=DEFAULT_FONT_SIZE)

    panel_file_names.append(
        '{0:s}/original_image_small.jpg'.format(output_dir_name)
    )

    print('Saving figure to file: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    brightness_temp_matrix_kelvins = numpy.expand_dims(
        brightness_temp_matrix_kelvins, axis=0
    )
    brightness_temp_matrix_kelvins = numpy.expand_dims(
        brightness_temp_matrix_kelvins, axis=-1
    )
    brightness_temp_matrix_kelvins = (
        data_augmentation.augment_data_specific_trans(
            bidirectional_reflectance_matrix=None,
            brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
            row_translations_low_res_px=
            numpy.array([ACTUAL_ROW_TRANSLATION_DISTANCE_PX], dtype=int),
            column_translations_low_res_px=
            numpy.array([ACTUAL_COLUMN_TRANSLATION_DISTANCE_PX], dtype=int),
            sentinel_value=0.
        )[1][0, ..., 0]
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    satellite_plotting.plot_2d_grid_no_coords(
        data_matrix=brightness_temp_matrix_kelvins,
        axes_object=axes_object,
        plotting_brightness_temp=True,
        cbar_orientation_string=None,
        font_size=DEFAULT_FONT_SIZE
    )
    axes_object.plot(
        0.5, 0.5, linestyle='None',
        marker=IMAGE_CENTER_MARKER, markersize=IMAGE_CENTER_MARKER_SIZE,
        markerfacecolor=IMAGE_CENTER_MARKER_COLOUR,
        markeredgecolor=IMAGE_CENTER_MARKER_COLOUR,
        markeredgewidth=0,
        transform=axes_object.transAxes, zorder=1e10
    )

    if letter_label is None:
        letter_label = 'a'
    else:
        letter_label = chr(ord(letter_label) + 1)

    gg_plotting_utils.label_axes(
        axes_object=axes_object, label_string='({0:s})'.format(letter_label)
    )
    title_string = 'Translated image, {0:d} x {1:d}'.format(
        brightness_temp_matrix_kelvins.shape[0],
        brightness_temp_matrix_kelvins.shape[1]
    )
    axes_object.set_title(title_string, fontsize=DEFAULT_FONT_SIZE)

    panel_file_names.append(
        '{0:s}/translated_image_small.jpg'.format(output_dir_name)
    )

    print('Saving figure to file: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    stx = satellite_table_with_extra_xarray
    if are_data_normalized:
        satellite_table_with_extra_xarray = normalization.denormalize_data(
            satellite_table_xarray=satellite_table_with_extra_xarray,
            normalization_param_table_xarray=norm_param_table_xarray
        )
        stx = satellite_table_with_extra_xarray

    brightness_temp_matrix_kelvins = (
        stx[satellite_utils.BRIGHTNESS_TEMPERATURE_KEY].values[0, ..., 0]
    ).astype(numpy.float64)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    satellite_plotting.plot_2d_grid_no_coords(
        data_matrix=brightness_temp_matrix_kelvins,
        axes_object=axes_object,
        plotting_brightness_temp=True,
        cbar_orientation_string=None,
        font_size=DEFAULT_FONT_SIZE
    )
    axes_object.plot(
        0.5, 0.5, linestyle='None',
        marker=IMAGE_CENTER_MARKER, markersize=IMAGE_CENTER_MARKER_SIZE,
        markerfacecolor=IMAGE_CENTER_MARKER_COLOUR,
        markeredgecolor=IMAGE_CENTER_MARKER_COLOUR,
        markeredgewidth=0,
        transform=axes_object.transAxes, zorder=1e10
    )

    if letter_label is None:
        letter_label = 'a'
    else:
        letter_label = chr(ord(letter_label) + 1)

    gg_plotting_utils.label_axes(
        axes_object=axes_object, label_string='({0:s})'.format(letter_label)
    )
    title_string = 'Original image, {0:d} x {1:d}'.format(
        brightness_temp_matrix_kelvins.shape[0],
        brightness_temp_matrix_kelvins.shape[1]
    )
    axes_object.set_title(title_string, fontsize=DEFAULT_FONT_SIZE)

    panel_file_names.append(
        '{0:s}/original_image_large.jpg'.format(output_dir_name)
    )

    print('Saving figure to file: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    brightness_temp_matrix_kelvins = numpy.expand_dims(
        brightness_temp_matrix_kelvins, axis=0
    )
    brightness_temp_matrix_kelvins = numpy.expand_dims(
        brightness_temp_matrix_kelvins, axis=-1
    )
    brightness_temp_matrix_kelvins = (
        data_augmentation.augment_data_specific_trans(
            bidirectional_reflectance_matrix=None,
            brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
            row_translations_low_res_px=
            numpy.array([ACTUAL_ROW_TRANSLATION_DISTANCE_PX], dtype=int),
            column_translations_low_res_px=
            numpy.array([ACTUAL_COLUMN_TRANSLATION_DISTANCE_PX], dtype=int),
            sentinel_value=0.
        )[1][0, ..., 0]
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    satellite_plotting.plot_2d_grid_no_coords(
        data_matrix=brightness_temp_matrix_kelvins,
        axes_object=axes_object,
        plotting_brightness_temp=True,
        cbar_orientation_string=None,
        font_size=DEFAULT_FONT_SIZE
    )
    axes_object.plot(
        0.5, 0.5, linestyle='None',
        marker=IMAGE_CENTER_MARKER, markersize=IMAGE_CENTER_MARKER_SIZE,
        markerfacecolor=IMAGE_CENTER_MARKER_COLOUR,
        markeredgecolor=IMAGE_CENTER_MARKER_COLOUR,
        markeredgewidth=0,
        transform=axes_object.transAxes, zorder=1e10
    )

    if letter_label is None:
        letter_label = 'a'
    else:
        letter_label = chr(ord(letter_label) + 1)

    gg_plotting_utils.label_axes(
        axes_object=axes_object, label_string='({0:s})'.format(letter_label)
    )
    title_string = 'Translated image, {0:d} x {1:d}'.format(
        brightness_temp_matrix_kelvins.shape[0],
        brightness_temp_matrix_kelvins.shape[1]
    )
    axes_object.set_title(title_string, fontsize=DEFAULT_FONT_SIZE)

    panel_file_names.append(
        '{0:s}/translated_image_large.jpg'.format(output_dir_name)
    )

    print('Saving figure to file: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    brightness_temp_matrix_kelvins = numpy.expand_dims(
        brightness_temp_matrix_kelvins, axis=0
    )
    brightness_temp_matrix_kelvins = numpy.expand_dims(
        brightness_temp_matrix_kelvins, axis=-1
    )
    brightness_temp_matrix_kelvins = (
        data_augmentation.subset_grid_after_data_aug(
            data_matrix=brightness_temp_matrix_kelvins,
            num_rows_to_keep=NUM_GRID_ROWS,
            num_columns_to_keep=NUM_GRID_COLUMNS,
            for_high_res=False
        )[0, ..., 0]
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    satellite_plotting.plot_2d_grid_no_coords(
        data_matrix=brightness_temp_matrix_kelvins,
        axes_object=axes_object,
        plotting_brightness_temp=True,
        cbar_orientation_string=None,
        font_size=DEFAULT_FONT_SIZE
    )
    axes_object.plot(
        0.5, 0.5, linestyle='None',
        marker=IMAGE_CENTER_MARKER, markersize=IMAGE_CENTER_MARKER_SIZE,
        markerfacecolor=IMAGE_CENTER_MARKER_COLOUR,
        markeredgecolor=IMAGE_CENTER_MARKER_COLOUR,
        markeredgewidth=0,
        transform=axes_object.transAxes, zorder=1e10
    )

    if letter_label is None:
        letter_label = 'a'
    else:
        letter_label = chr(ord(letter_label) + 1)

    gg_plotting_utils.label_axes(
        axes_object=axes_object, label_string='({0:s})'.format(letter_label)
    )
    title_string = 'Translated/cropped image, {0:d} x {1:d}'.format(
        brightness_temp_matrix_kelvins.shape[0],
        brightness_temp_matrix_kelvins.shape[1]
    )
    axes_object.set_title(title_string, fontsize=DEFAULT_FONT_SIZE)

    panel_file_names.append(
        '{0:s}/cropped_image_large.jpg'.format(output_dir_name)
    )

    print('Saving figure to file: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    for this_file_name in panel_file_names:
        imagemagick_utils.resize_image(
            input_file_name=this_file_name,
            output_file_name=this_file_name,
            output_size_pixels=PANEL_SIZE_PX
        )

    concat_figure_file_name = '{0:s}/translation_schematic.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=2, num_panel_columns=3
    )

    colour_map_object, colour_norm_object = (
        satellite_plotting.get_colour_scheme_for_brightness_temp()
    )
    plotting_utils.add_colour_bar(
        figure_file_name=concat_figure_file_name,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical',
        font_size=30,
        cbar_label_string='Brightness temperature (K)',
        tick_label_format_string='{0:.0f}',
        log_space=False,
        temporary_cbar_file_name=
        '{0:s}/translation_schematic_cbar.jpg'.format(output_dir_name)
    )

    imagemagick_utils.resize_image(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        satellite_dir_name=getattr(INPUT_ARG_OBJECT, SATELLITE_DIR_ARG_NAME),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        cyclone_id_string=getattr(INPUT_ARG_OBJECT, CYCLONE_ID_ARG_NAME),
        valid_time_string=getattr(INPUT_ARG_OBJECT, VALID_TIME_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
