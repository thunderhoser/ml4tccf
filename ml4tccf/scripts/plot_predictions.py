"""Plots predictions."""

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
from ml4tccf.io import prediction_io
from ml4tccf.utils import scalar_prediction_utils
from ml4tccf.io import border_io
from ml4tccf.machine_learning import neural_net
from ml4tccf.plotting import plotting_utils
from ml4tccf.plotting import satellite_plotting

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
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_data_one_example(
        predictor_matrices, target_values, prediction_matrix,
        model_metadata_dict, cyclone_id_string, target_time_unix_sec,
        low_res_latitudes_deg_n, low_res_longitudes_deg_e,
        high_res_latitudes_deg_n, high_res_longitudes_deg_e,
        are_data_normalized, border_latitudes_deg_n, border_longitudes_deg_e,
        output_file_name):
    """Plots satellite data for one example.

    P = number of points in border set
    S = ensemble size

    :param predictor_matrices: Same as output from `neural_net.create_data` but
        without first axis.
    :param target_values: Same as output from `neural_net.create_data` but
        without first axis.
    :param prediction_matrix: 2-by-S numpy array of predictions.
        prediction_matrix[0, :] contains predicted row positions of TC centers,
        and prediction_matrix[1, :] contains predicted column positions of TC
        centers.
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
    """

    num_grid_rows_low_res = predictor_matrices[-1].shape[0]
    num_grid_columns_low_res = predictor_matrices[-1].shape[1]
    ensemble_size = prediction_matrix.shape[1]

    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    d = training_option_dict
    high_res_wavelengths_microns = d[neural_net.HIGH_RES_WAVELENGTHS_KEY]
    low_res_wavelengths_microns = d[neural_net.LOW_RES_WAVELENGTHS_KEY]

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
            0.5 + target_values[1] / num_grid_columns_low_res,
            0.5 + target_values[0] / num_grid_rows_low_res,
            linestyle='None',
            marker=ACTUAL_CENTER_MARKER, markersize=ACTUAL_CENTER_MARKER_SIZE,
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
            data_matrix=predictor_matrices[-1][..., j],
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
            0.5 + target_values[1] / num_grid_columns_low_res,
            0.5 + target_values[0] / num_grid_rows_low_res,
            linestyle='None',
            marker=ACTUAL_CENTER_MARKER, markersize=ACTUAL_CENTER_MARKER_SIZE,
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


def _run(prediction_file_name, satellite_dir_name, are_data_normalized,
         output_dir_name):
    """Plots predictions.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param satellite_dir_name: Same.
    :param are_data_normalized: Same.
    :param output_dir_name: Same.
    """

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_table_xarray = prediction_io.read_file(prediction_file_name)

    are_predictions_gridded = (
        scalar_prediction_utils.PREDICTED_ROW_OFFSET_KEY
        not in prediction_table_xarray
    )
    if are_predictions_gridded:
        raise ValueError(
            'This script does not yet work for gridded predictions.'
        )

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

    pt = prediction_table_xarray

    # TODO(thunderhoser): This will not work if I ever have multiple cyclones in
    # one prediction file.
    cyclone_id_string = pt[scalar_prediction_utils.CYCLONE_ID_KEY].values[0]
    target_times_unix_sec = pt[scalar_prediction_utils.TARGET_TIME_KEY].values

    data_dict = neural_net.create_data_specific_trans(
        option_dict=validation_option_dict,
        cyclone_id_string=cyclone_id_string,
        target_times_unix_sec=target_times_unix_sec,
        row_translations_low_res_px=numpy.round(
            pt[scalar_prediction_utils.ACTUAL_ROW_OFFSET_KEY].values
        ).astype(int),
        column_translations_low_res_px=numpy.round(
            pt[scalar_prediction_utils.ACTUAL_COLUMN_OFFSET_KEY].values
        ).astype(int)
    )

    if data_dict is None:
        return

    predictor_matrices = data_dict[neural_net.PREDICTOR_MATRICES_KEY]
    target_matrix = data_dict[neural_net.TARGET_MATRIX_KEY]
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

    # target_matrix = numpy.transpose(numpy.vstack((
    #     pt[prediction_utils.ACTUAL_ROW_OFFSET_KEY].values,
    #     pt[prediction_utils.ACTUAL_COLUMN_OFFSET_KEY].values,
    #     pt[prediction_utils.GRID_SPACING_KEY].values,
    #     pt[prediction_utils.ACTUAL_CENTER_LATITUDE_KEY].values
    # )))

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
            output_file_name=output_file_name
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
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
