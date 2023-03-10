"""Plots heat map of model errors."""

import os
import glob
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4tccf.io import prediction_io
from ml4tccf.utils import misc_utils
from ml4tccf.utils import scalar_prediction_utils
from ml4tccf.utils import gridded_prediction_utils
from ml4tccf.machine_learning import neural_net
from ml4tccf.plotting import plotting_utils

METRES_TO_KM = 0.001
SAMPLE_SIZE_FOR_DATA_AUG = int(1e7)
USE_TOP_N_GRIDDED_PROBS = 1000

GRID_LINE_WIDTH = 2
GRID_LINE_COLOUR = numpy.full(3, 0.)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

INPUT_FILE_PATTERN_ARG_NAME = 'input_prediction_file_pattern'
NUM_XY_BINS_ARG_NAME = 'num_xy_error_bins'
XY_ERROR_LIMITS_ARG_NAME = 'xy_error_limits_metres'
COLOUR_MAP_ARG_NAME = 'colour_map_name'
COLOUR_MAP_LIMITS_ARG_NAME = 'colour_map_limits'
PLOT_ORIG_ERRORS_ARG_NAME = 'plot_orig_errors'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_PATTERN_HELP_STRING = (
    'Glob pattern for input files.  Each will be read by '
    '`prediction_io.read_file`, and predictions in all these files will be '
    'evaluated together.'
)
NUM_XY_BINS_HELP_STRING = 'Number of bins in each direction.'
XY_ERROR_LIMITS_HELP_STRING = (
    'Limits of grid in each direction (list of two values).'
)
COLOUR_MAP_HELP_STRING = (
    'Name of colour scheme for heat map.  Must be accepted by '
    '`pyplot.get_cmap`.'
)
COLOUR_MAP_LIMITS_HELP_STRING = (
    'Min and max values (frequencies) represented in colour scheme.  Keep in '
    'mind that the value plotted in each grid cell G is the fraction of '
    'examples with error in G.  If you want the min and max to be determined '
    'by the data, leave this argument alone.'
)
PLOT_ORIG_ERRORS_HELP_STRING = (
    'Boolean flag.  If 1, will plot "original" errors -- i.e., errors created '
    'by data augmentation during training.  This heat map will represent the '
    'errors of a completely uninformative model.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_PATTERN_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_PATTERN_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_XY_BINS_ARG_NAME, type=int, required=False, default=100,
    help=NUM_XY_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + XY_ERROR_LIMITS_ARG_NAME, type=float, nargs=2, required=False,
    default=[-1e5, 1e5], help=XY_ERROR_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + COLOUR_MAP_ARG_NAME, type=str, required=False, default='cividis',
    help=COLOUR_MAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + COLOUR_MAP_LIMITS_ARG_NAME, type=int, nargs='+', required=False,
    default=[1, -1], help=COLOUR_MAP_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_ORIG_ERRORS_ARG_NAME, type=int, required=False, default=1,
    help=PLOT_ORIG_ERRORS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _prediction_grid_to_xy_errors(prediction_table_xarray, example_index):
    """Converts one prediction grid to a set of x- and y-distance errors.

    G = number of grid points at which distance errors are computed

    :param prediction_table_xarray: xarray table with gridded predictions.
    :param example_index: Will compute x- and y-distance errors only for the
        [i]th grid, where i = `example_index`.
    :return: x_errors_km: length-G numpy array of x-distance errors.
    :return: y_errors_km: length-G numpy array of y-distance errors.
    :return: error_weights: length-G numpy array of error weights (predicted
        probabilities).
    """

    pt = prediction_table_xarray
    i = example_index

    target_row_offset_px, target_column_offset_px = (
        misc_utils.target_matrix_to_centroid(
            target_matrix=
            pt[gridded_prediction_utils.TARGET_MATRIX_KEY].values[i, ...]
        )
    )

    prediction_matrix = (
        pt[gridded_prediction_utils.PREDICTION_MATRIX_KEY].values[i, ..., 0]
    )
    grid_spacing_km = pt[gridded_prediction_utils.GRID_SPACING_KEY].values[i]

    num_grid_rows = prediction_matrix.shape[0]
    num_grid_columns = prediction_matrix.shape[1]
    image_center_row_index = 0.5 * num_grid_rows - 0.5
    image_center_column_index = 0.5 * num_grid_columns - 0.5

    linear_indices = numpy.argsort(-1 * numpy.ravel(prediction_matrix))[
        :USE_TOP_N_GRIDDED_PROBS
    ]

    predicted_row_indices, predicted_column_indices = numpy.unravel_index(
        linear_indices, prediction_matrix.shape
    )
    predicted_row_offsets_px = predicted_row_indices - image_center_row_index
    predicted_column_offsets_px = (
        predicted_column_indices - image_center_column_index
    )

    x_errors_km = (
        grid_spacing_km *
        (predicted_column_offsets_px - target_column_offset_px)
    )
    y_errors_km = (
        grid_spacing_km *
        (predicted_row_offsets_px - target_row_offset_px)
    )
    error_weights = prediction_matrix[
        predicted_row_indices, predicted_column_indices
    ]

    return x_errors_km, y_errors_km, error_weights


def _run(prediction_file_pattern, num_xy_error_bins, xy_error_limits_metres,
         colour_map_name, colour_map_limits, plot_orig_errors, output_dir_name):
    """Plots heat map of model errors.

    This is effectively the main method.

    :param prediction_file_pattern: See documentation at top of file.
    :param num_xy_error_bins: Same.
    :param xy_error_limits_metres: Same.
    :param colour_map_name: Same.
    :param colour_map_limits: Same.
    :param plot_orig_errors: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if no prediction files could be found.
    """

    # Check input args.
    error_checking.assert_is_geq(num_xy_error_bins, 2)
    error_checking.assert_is_numpy_array(
        xy_error_limits_metres,
        exact_dimensions=numpy.array([2], dtype=int)
    )
    error_checking.assert_is_greater(
        xy_error_limits_metres[1], xy_error_limits_metres[0]
    )

    colour_map_object = pyplot.get_cmap(colour_map_name)

    error_checking.assert_is_numpy_array(
        colour_map_limits,
        exact_dimensions=numpy.array([2], dtype=int)
    )

    if colour_map_limits[1] <= colour_map_limits[0]:
        colour_map_limits = None
    else:
        error_checking.assert_is_geq_numpy_array(colour_map_limits, 0.)
        error_checking.assert_is_leq_numpy_array(colour_map_limits, 0.5)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Read files.
    prediction_file_names = glob.glob(prediction_file_pattern)
    if len(prediction_file_names) == 0:
        error_string = (
            'Could not find any prediction file with the following pattern: '
            '"{0:s}"'
        ).format(prediction_file_pattern)

        raise ValueError(error_string)

    prediction_file_names.sort()

    num_files = len(prediction_file_names)
    prediction_tables_xarray = [None] * num_files
    are_predictions_gridded = False

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(prediction_file_names[i]))
        prediction_tables_xarray[i] = prediction_io.read_file(
            prediction_file_names[i]
        )

        are_predictions_gridded = (
            scalar_prediction_utils.PREDICTED_ROW_OFFSET_KEY
            not in prediction_tables_xarray[i]
        )

        if are_predictions_gridded:
            prediction_tables_xarray[i] = (
                gridded_prediction_utils.get_ensemble_mean(
                    prediction_tables_xarray[i]
                )
            )
        else:
            prediction_tables_xarray[i] = (
                scalar_prediction_utils.get_ensemble_mean(
                    prediction_tables_xarray[i]
                )
            )

    if are_predictions_gridded:
        prediction_table_xarray = gridded_prediction_utils.concat_over_examples(
            prediction_tables_xarray
        )
    else:
        prediction_table_xarray = scalar_prediction_utils.concat_over_examples(
            prediction_tables_xarray
        )

    pt = prediction_table_xarray

    # Do actual stuff.
    bin_edges_km = METRES_TO_KM * numpy.linspace(
        xy_error_limits_metres[0], xy_error_limits_metres[1],
        num=num_xy_error_bins + 1, dtype=float
    )
    bin_centers_km = bin_edges_km[:-1] + numpy.diff(bin_edges_km) / 2

    if are_predictions_gridded:
        grid_spacings_km = pt[gridded_prediction_utils.GRID_SPACING_KEY].values
        num_examples = len(grid_spacings_km)
        num_errors = num_examples * USE_TOP_N_GRIDDED_PROBS

        x_errors_km = numpy.full(num_errors, numpy.nan)
        y_errors_km = numpy.full(num_errors, numpy.nan)
        error_weights = numpy.full(num_errors, numpy.nan)

        for i in range(num_examples):
            first_index = i * USE_TOP_N_GRIDDED_PROBS
            last_index = first_index + USE_TOP_N_GRIDDED_PROBS

            (
                x_errors_km[first_index:last_index],
                y_errors_km[first_index:last_index],
                error_weights[first_index:last_index]
            ) = _prediction_grid_to_xy_errors(
                prediction_table_xarray=pt, example_index=i
            )

        bin_count_matrix = grids.count_events_on_equidistant_grid(
            event_x_coords_metres=x_errors_km,
            event_y_coords_metres=y_errors_km,
            grid_point_x_coords_metres=bin_centers_km,
            grid_point_y_coords_metres=bin_centers_km,
            event_weights=error_weights
        )[0]
    else:
        grid_spacings_km = pt[scalar_prediction_utils.GRID_SPACING_KEY].values
        x_errors_km = grid_spacings_km * (
            pt[scalar_prediction_utils.PREDICTED_COLUMN_OFFSET_KEY].values[:, 0]
            - pt[scalar_prediction_utils.ACTUAL_COLUMN_OFFSET_KEY].values
        )
        y_errors_km = grid_spacings_km * (
            pt[scalar_prediction_utils.PREDICTED_ROW_OFFSET_KEY].values[:, 0]
            - pt[scalar_prediction_utils.ACTUAL_ROW_OFFSET_KEY].values
        )

        bin_count_matrix = grids.count_events_on_equidistant_grid(
            event_x_coords_metres=x_errors_km,
            event_y_coords_metres=y_errors_km,
            grid_point_x_coords_metres=bin_centers_km,
            grid_point_y_coords_metres=bin_centers_km
        )[0]

    bin_frequency_matrix = (
        bin_count_matrix.astype(float) / numpy.sum(bin_count_matrix)
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    if colour_map_limits is None:
        these_colour_map_limits = numpy.array([
            numpy.min(bin_frequency_matrix),
            numpy.max(bin_frequency_matrix)
        ])
    else:
        these_colour_map_limits = colour_map_limits + 0.

    colour_norm_object = pyplot.Normalize(
        vmin=these_colour_map_limits[0], vmax=these_colour_map_limits[1]
    )
    axes_object.imshow(
        bin_frequency_matrix, origin='lower', cmap=colour_map_object,
        norm=colour_norm_object,
        vmin=these_colour_map_limits[0], vmax=these_colour_map_limits[1]
    )

    axes_object.set_title(
        'Heat map of model errors for TC center\n'
        'Actual center is at (0, 0) km'
    )
    axes_object.set_xlabel(r'Predicted $x$-position (km)')
    axes_object.set_ylabel(r'Predicted $y$-position (km)')

    tick_coords_km = METRES_TO_KM * numpy.linspace(
        xy_error_limits_metres[0], xy_error_limits_metres[1],
        num=9, dtype=float
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
        linewidth=GRID_LINE_WIDTH, color=GRID_LINE_COLOUR
    )

    output_file_name = '{0:s}/model_error_heat_map.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    plotting_utils.add_colour_bar(
        figure_file_name=output_file_name,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical',
        font_size=FONT_SIZE, cbar_label_string='Frequency',
        tick_label_format_string='{0:.2g}', log_space=False,
        temporary_cbar_file_name=
        '{0:s}/model_error_heat_map_cbar.jpg'.format(output_dir_name)
    )

    if not plot_orig_errors:
        return

    model_file_name = pt.attrs[scalar_prediction_utils.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    row_translations_px, column_translations_px = (
        neural_net.get_translation_distances(
            mean_translation_px=
            training_option_dict[neural_net.DATA_AUG_MEAN_TRANS_KEY],
            stdev_translation_px=
            training_option_dict[neural_net.DATA_AUG_STDEV_TRANS_KEY],
            num_translations=SAMPLE_SIZE_FOR_DATA_AUG
        )
    )

    actual_x_offsets_km = numpy.mean(grid_spacings_km) * column_translations_px
    actual_y_offsets_km = numpy.mean(grid_spacings_km) * row_translations_px
    predicted_x_offsets_km = numpy.full(actual_x_offsets_km.shape, 0.)
    predicted_y_offsets_km = numpy.full(actual_y_offsets_km.shape, 0.)

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

    if colour_map_limits is None:
        these_colour_map_limits = numpy.array([
            numpy.min(bin_frequency_matrix),
            numpy.percentile(bin_frequency_matrix, 99.5)
        ])
    else:
        these_colour_map_limits = colour_map_limits + 0.

    colour_norm_object = pyplot.Normalize(
        vmin=these_colour_map_limits[0], vmax=these_colour_map_limits[1]
    )
    axes_object.imshow(
        bin_frequency_matrix, origin='lower', cmap=colour_map_object,
        norm=colour_norm_object,
        vmin=these_colour_map_limits[0], vmax=these_colour_map_limits[1]
    )

    axes_object.set_title(
        'Heat map of original errors (created by data\n'
        'augmentation during training) for TC center'
    )
    axes_object.set_xlabel(r'Predicted $x$-position (km)')
    axes_object.set_ylabel(r'Predicted $y$-position (km)')

    axes_object.set_xticks(tick_coords_px)
    axes_object.set_xticklabels(tick_coord_labels)
    axes_object.set_yticks(tick_coords_px)
    axes_object.set_yticklabels(tick_coord_labels)
    axes_object.grid(
        b=True, which='major', axis='both', linestyle='--',
        linewidth=GRID_LINE_WIDTH, color=GRID_LINE_COLOUR
    )

    output_file_name = '{0:s}/original_error_heat_map.jpg'.format(
        output_dir_name
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    plotting_utils.add_colour_bar(
        figure_file_name=output_file_name,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical',
        font_size=FONT_SIZE, cbar_label_string='Frequency',
        tick_label_format_string='{0:.2g}', log_space=False,
        temporary_cbar_file_name=
        '{0:s}/original_error_heat_map_cbar.jpg'.format(output_dir_name)
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_pattern=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_PATTERN_ARG_NAME
        ),
        num_xy_error_bins=getattr(INPUT_ARG_OBJECT, NUM_XY_BINS_ARG_NAME),
        xy_error_limits_metres=numpy.array(
            getattr(INPUT_ARG_OBJECT, XY_ERROR_LIMITS_ARG_NAME),
            dtype=float
        ),
        colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        colour_map_limits=numpy.array(
            getattr(INPUT_ARG_OBJECT, COLOUR_MAP_LIMITS_ARG_NAME),
            dtype=float
        ),
        plot_orig_errors=bool(
            getattr(INPUT_ARG_OBJECT, PLOT_ORIG_ERRORS_ARG_NAME)
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
