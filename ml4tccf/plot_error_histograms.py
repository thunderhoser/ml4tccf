"""Plots error histograms."""

import os
import sys
import glob
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import histograms
import file_system_utils
import error_checking
import prediction_io
import prediction_utils
import evaluation_sans_uq
import neural_net

METRES_TO_KM = 0.001

MIN_ANGULAR_DIFF_DEG = -180.
MAX_ANGULAR_DIFF_DEG = 180.

SAMPLE_SIZE_FOR_DATA_AUG = int(1e7)

HISTOGRAM_FACE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
HISTOGRAM_EDGE_COLOUR = numpy.full(3, 0.)
HISTOGRAM_EDGE_WIDTH = 1.5

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
NUM_DISTANCE_BINS_ARG_NAME = 'num_distance_error_bins'
DISTANCE_ERROR_LIMITS_ARG_NAME = 'distance_error_limits_metres'
NUM_DIRECTION_BINS_ARG_NAME = 'num_direction_error_bins'
PLOT_ORIG_ERRORS_ARG_NAME = 'plot_orig_errors'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_PATTERN_HELP_STRING = (
    'Glob pattern for input files.  Each will be read by '
    '`prediction_io.read_file`, and predictions in all these files will be '
    'evaluated together.'
)
NUM_XY_BINS_HELP_STRING = (
    'Number of bins in x-error histogram and y-error histogram.'
)
XY_ERROR_LIMITS_HELP_STRING = (
    'Min and max xy-errors in histogram (list of two values).'
)
NUM_DISTANCE_BINS_HELP_STRING = (
    'Number of bins in error histogram for Euclidean distance.'
)
DISTANCE_ERROR_LIMITS_HELP_STRING = (
    'Min and max Euclidean distances in error histogram (list of two values).'
)
NUM_DIRECTION_BINS_HELP_STRING = (
    'Number of bins in error histogram for direction.'
)
PLOT_ORIG_ERRORS_HELP_STRING = (
    'Boolean flag.  If 1, will plot "original" errors -- i.e., errors created '
    'by data augmentation during training.  This set of histograms will '
    'represent the errors of a completely uninformative model.'
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
    '--' + NUM_DISTANCE_BINS_ARG_NAME, type=int, required=False, default=100,
    help=NUM_DISTANCE_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DISTANCE_ERROR_LIMITS_ARG_NAME, type=float, nargs=2, required=False,
    default=[0, 1e5], help=DISTANCE_ERROR_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_DIRECTION_BINS_ARG_NAME, type=int, required=False, default=72,
    help=NUM_DIRECTION_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_ORIG_ERRORS_ARG_NAME, type=int, required=False, default=1,
    help=PLOT_ORIG_ERRORS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_x_error_histogram(
        predicted_x_offsets_km, actual_x_offsets_km, num_xy_error_bins,
        xy_error_limits_metres, plotting_orig_error, output_dir_name):
    """Plots histogram for x-error.

    E = number of examples

    :param predicted_x_offsets_km: length-E numpy array of predicted offsets.
    :param actual_x_offsets_km: length-E numpy array of actual offsets.
    :param num_xy_error_bins: See documentation at top of file.
    :param xy_error_limits_metres: Same.
    :param plotting_orig_error: Boolean flag.  If True, plotting original error
        (created by data augmentation during training).  If False, plotting
        actual model error.
    :param output_dir_name: See documentation at top of file.
    """

    bin_counts = histograms.create_histogram(
        input_values=predicted_x_offsets_km - actual_x_offsets_km,
        num_bins=num_xy_error_bins,
        min_value=METRES_TO_KM * xy_error_limits_metres[0],
        max_value=METRES_TO_KM * xy_error_limits_metres[1]
    )[1]
    bin_frequencies = bin_counts.astype(float) / numpy.sum(bin_counts)

    bin_edges_km = METRES_TO_KM * numpy.linspace(
        xy_error_limits_metres[0], xy_error_limits_metres[1],
        num=num_xy_error_bins + 1, dtype=float
    )
    bin_centers_km = bin_edges_km[:-1] + numpy.diff(bin_edges_km) / 2

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    axes_object.bar(
        x=bin_centers_km, height=bin_frequencies,
        width=numpy.diff(bin_centers_km)[0],
        color=HISTOGRAM_FACE_COLOUR,
        edgecolor=HISTOGRAM_EDGE_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH
    )

    if plotting_orig_error:
        axes_object.set_title(
            'Histogram of original errors (created by data\n' +
            r'augmentation during training) for $x$-position'
        )
    else:
        axes_object.set_title(r'Histogram of model errors for $x$-position')

    axes_object.set_xlabel('Error (predicted minus actual; km)')
    axes_object.set_ylabel('Frequency')

    output_file_name = '{0:s}/{1:s}_error_histogram_x.jpg'.format(
        output_dir_name,
        'original' if plotting_orig_error else 'model'
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _plot_y_error_histogram(
        predicted_y_offsets_km, actual_y_offsets_km, num_xy_error_bins,
        xy_error_limits_metres, plotting_orig_error, output_dir_name):
    """Plots histogram for y-error.

    E = number of examples

    :param predicted_y_offsets_km: length-E numpy array of predicted offsets.
    :param actual_y_offsets_km: length-E numpy array of actual offsets.
    :param num_xy_error_bins: See doc for `_plot_x_error_histogram`.
    :param xy_error_limits_metres: Same.
    :param plotting_orig_error: Same.
    :param output_dir_name: Same.
    """

    bin_counts = histograms.create_histogram(
        input_values=predicted_y_offsets_km - actual_y_offsets_km,
        num_bins=num_xy_error_bins,
        min_value=METRES_TO_KM * xy_error_limits_metres[0],
        max_value=METRES_TO_KM * xy_error_limits_metres[1]
    )[1]
    bin_frequencies = bin_counts.astype(float) / numpy.sum(bin_counts)

    bin_edges_km = METRES_TO_KM * numpy.linspace(
        xy_error_limits_metres[0], xy_error_limits_metres[1],
        num=num_xy_error_bins + 1, dtype=float
    )
    bin_centers_km = bin_edges_km[:-1] + numpy.diff(bin_edges_km) / 2

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.bar(
        x=bin_centers_km, height=bin_frequencies,
        width=numpy.diff(bin_centers_km)[0],
        color=HISTOGRAM_FACE_COLOUR,
        edgecolor=HISTOGRAM_EDGE_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH
    )

    if plotting_orig_error:
        axes_object.set_title(
            'Histogram of original errors (created by data\n' +
            r'augmentation during training) for $y$-position'
        )
    else:
        axes_object.set_title(r'Histogram of model errors for $y$-position')

    axes_object.set_xlabel('Error (predicted minus actual; km)')
    axes_object.set_ylabel('Frequency')

    output_file_name = '{0:s}/{1:s}_error_histogram_y.jpg'.format(
        output_dir_name,
        'original' if plotting_orig_error else 'model'
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _plot_euclidean_error_histogram(
        predicted_x_offsets_km, actual_x_offsets_km,
        predicted_y_offsets_km, actual_y_offsets_km, num_distance_error_bins,
        distance_error_limits_metres, plotting_orig_error, output_dir_name):
    """Plots histogram for Euclidean error.

    E = number of examples

    :param predicted_x_offsets_km: length-E numpy array of predicted offsets in
        x-direction.
    :param actual_x_offsets_km: length-E numpy array of actual offsets in
        x-direction.
    :param predicted_y_offsets_km: length-E numpy array of predicted offsets in
        y-direction.
    :param actual_y_offsets_km: length-E numpy array of actual offsets in
        y-direction.
    :param num_distance_error_bins: See documentation at top of file.
    :param distance_error_limits_metres: Same.
    :param plotting_orig_error: Boolean flag.  If True, plotting original error
        (created by data augmentation during training).  If False, plotting
        actual model error.
    :param output_dir_name: See documentation at top of file.
    """

    euclidean_errors_km = numpy.sqrt(
        (predicted_y_offsets_km - actual_y_offsets_km) ** 2 +
        (predicted_x_offsets_km - actual_x_offsets_km) ** 2
    )

    bin_counts = histograms.create_histogram(
        input_values=euclidean_errors_km,
        num_bins=num_distance_error_bins,
        min_value=METRES_TO_KM * distance_error_limits_metres[0],
        max_value=METRES_TO_KM * distance_error_limits_metres[1]
    )[1]
    bin_frequencies = bin_counts.astype(float) / numpy.sum(bin_counts)

    bin_edges_km = METRES_TO_KM * numpy.linspace(
        distance_error_limits_metres[0], distance_error_limits_metres[1],
        num=num_distance_error_bins + 1, dtype=float
    )
    bin_centers_km = bin_edges_km[:-1] + numpy.diff(bin_edges_km) / 2

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.bar(
        x=bin_centers_km, height=bin_frequencies,
        width=numpy.diff(bin_centers_km)[0],
        color=HISTOGRAM_FACE_COLOUR,
        edgecolor=HISTOGRAM_EDGE_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH
    )

    if plotting_orig_error:
        axes_object.set_title(
            'Histogram of original Euclidean errors\n'
            '(created by data augmentation during training)'
        )
    else:
        axes_object.set_title("Histogram of model's Euclidean errors")

    axes_object.set_xlabel('Total distance between predicted and actual (km)')
    axes_object.set_ylabel('Frequency')

    output_file_name = '{0:s}/{1:s}_error_histogram_euclidean.jpg'.format(
        output_dir_name,
        'original' if plotting_orig_error else 'model'
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _plot_direction_error_histogram(
        predicted_x_offsets_km, actual_x_offsets_km,
        predicted_y_offsets_km, actual_y_offsets_km, num_direction_error_bins,
        plotting_orig_error, output_dir_name):
    """Plots histogram for direction error.

    :param predicted_x_offsets_km: See doc for
        `_plot_euclidean_error_histogram`.
    :param actual_x_offsets_km: Same.
    :param predicted_y_offsets_km: Same.
    :param actual_y_offsets_km: Same.
    :param num_direction_error_bins: Same.
    :param plotting_orig_error: Same.
    :param output_dir_name: Same.
    """

    predicted_offset_angles_deg = evaluation_sans_uq._get_offset_angles(
        x_offsets=predicted_x_offsets_km, y_offsets=predicted_y_offsets_km
    )
    actual_offset_angles_deg = evaluation_sans_uq._get_offset_angles(
        x_offsets=actual_x_offsets_km, y_offsets=actual_y_offsets_km
    )

    real_indices = numpy.where(numpy.invert(numpy.logical_or(
        numpy.isnan(predicted_offset_angles_deg),
        numpy.isnan(actual_offset_angles_deg)
    )))[0]

    angular_diffs_deg = evaluation_sans_uq._get_angular_diffs(
        target_angles_deg=actual_offset_angles_deg[real_indices],
        predicted_angles_deg=predicted_offset_angles_deg[real_indices]
    )

    bin_counts = histograms.create_histogram(
        input_values=angular_diffs_deg,
        num_bins=num_direction_error_bins,
        min_value=MIN_ANGULAR_DIFF_DEG,
        max_value=MAX_ANGULAR_DIFF_DEG
    )[1]
    bin_frequencies = bin_counts.astype(float) / numpy.sum(bin_counts)

    bin_edges_deg = numpy.linspace(
        MIN_ANGULAR_DIFF_DEG, MAX_ANGULAR_DIFF_DEG,
        num=num_direction_error_bins + 1, dtype=float
    )
    bin_centers_deg = bin_edges_deg[:-1] + numpy.diff(bin_edges_deg) / 2

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.bar(
        x=bin_centers_deg, height=bin_frequencies,
        width=numpy.diff(bin_centers_deg)[0],
        color=HISTOGRAM_FACE_COLOUR,
        edgecolor=HISTOGRAM_EDGE_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH
    )

    if plotting_orig_error:
        axes_object.set_title(
            'Histogram of original direction errors\n'
            '(created by data augmentation during training)'
        )
    else:
        axes_object.set_title("Histogram of model's direction errors")

    axes_object.set_xlabel('Error (predicted minus actual; CCW positive; deg)')
    axes_object.set_ylabel('Frequency')

    output_file_name = '{0:s}/{1:s}_error_histogram_direction.jpg'.format(
        output_dir_name,
        'original' if plotting_orig_error else 'model'
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(prediction_file_pattern, num_xy_error_bins, xy_error_limits_metres,
         num_distance_error_bins, distance_error_limits_metres,
         num_direction_error_bins, plot_orig_errors, output_dir_name):
    """Plots error histograms.

    This is effectively the main method.

    :param prediction_file_pattern: See documentation at top of file.
    :param num_xy_error_bins: Same.
    :param xy_error_limits_metres: Same.
    :param num_distance_error_bins: Same.
    :param distance_error_limits_metres: Same.
    :param num_direction_error_bins: Same.
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

    error_checking.assert_is_geq(num_distance_error_bins, 2)
    error_checking.assert_is_numpy_array(
        distance_error_limits_metres,
        exact_dimensions=numpy.array([2], dtype=int)
    )
    error_checking.assert_is_greater(
        distance_error_limits_metres[1], distance_error_limits_metres[0]
    )
    error_checking.assert_is_geq(distance_error_limits_metres[0], 0.)

    error_checking.assert_is_geq(num_direction_error_bins, 2)

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

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(prediction_file_names[i]))
        prediction_tables_xarray[i] = prediction_io.read_file(
            prediction_file_names[i]
        )
        prediction_tables_xarray[i] = prediction_utils.get_ensemble_mean(
            prediction_tables_xarray[i]
        )

    prediction_table_xarray = prediction_utils.concat_over_examples(
        prediction_tables_xarray
    )
    pt = prediction_table_xarray

    # Do actual stuff.
    grid_spacings_km = pt[prediction_utils.GRID_SPACING_KEY].values
    predicted_x_offsets_km = (
        grid_spacings_km *
        pt[prediction_utils.PREDICTED_COLUMN_OFFSET_KEY].values[:, 0]
    )
    actual_x_offsets_km = (
        grid_spacings_km *
        pt[prediction_utils.ACTUAL_COLUMN_OFFSET_KEY].values
    )

    _plot_x_error_histogram(
        predicted_x_offsets_km=predicted_x_offsets_km,
        actual_x_offsets_km=actual_x_offsets_km,
        num_xy_error_bins=num_xy_error_bins,
        xy_error_limits_metres=xy_error_limits_metres,
        plotting_orig_error=False,
        output_dir_name=output_dir_name
    )

    predicted_y_offsets_km = (
        grid_spacings_km *
        pt[prediction_utils.PREDICTED_ROW_OFFSET_KEY].values[:, 0]
    )
    actual_y_offsets_km = (
        grid_spacings_km *
        pt[prediction_utils.ACTUAL_ROW_OFFSET_KEY].values
    )
    _plot_y_error_histogram(
        predicted_y_offsets_km=predicted_y_offsets_km,
        actual_y_offsets_km=actual_y_offsets_km,
        num_xy_error_bins=num_xy_error_bins,
        xy_error_limits_metres=xy_error_limits_metres,
        plotting_orig_error=False,
        output_dir_name=output_dir_name
    )

    _plot_euclidean_error_histogram(
        predicted_x_offsets_km=predicted_x_offsets_km,
        actual_x_offsets_km=actual_x_offsets_km,
        predicted_y_offsets_km=predicted_y_offsets_km,
        actual_y_offsets_km=actual_y_offsets_km,
        num_distance_error_bins=num_distance_error_bins,
        distance_error_limits_metres=distance_error_limits_metres,
        plotting_orig_error=False,
        output_dir_name=output_dir_name
    )

    _plot_direction_error_histogram(
        predicted_x_offsets_km=predicted_x_offsets_km,
        actual_x_offsets_km=actual_x_offsets_km,
        predicted_y_offsets_km=predicted_y_offsets_km,
        actual_y_offsets_km=actual_y_offsets_km,
        num_direction_error_bins=num_direction_error_bins,
        plotting_orig_error=False,
        output_dir_name=output_dir_name
    )

    if not plot_orig_errors:
        return

    model_file_name = pt.attrs[prediction_utils.MODEL_FILE_KEY]
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

    _plot_x_error_histogram(
        predicted_x_offsets_km=predicted_x_offsets_km,
        actual_x_offsets_km=actual_x_offsets_km,
        num_xy_error_bins=num_xy_error_bins,
        xy_error_limits_metres=xy_error_limits_metres,
        plotting_orig_error=True,
        output_dir_name=output_dir_name
    )

    _plot_y_error_histogram(
        predicted_y_offsets_km=predicted_y_offsets_km,
        actual_y_offsets_km=actual_y_offsets_km,
        num_xy_error_bins=num_xy_error_bins,
        xy_error_limits_metres=xy_error_limits_metres,
        plotting_orig_error=True,
        output_dir_name=output_dir_name
    )

    _plot_euclidean_error_histogram(
        predicted_x_offsets_km=predicted_x_offsets_km,
        actual_x_offsets_km=actual_x_offsets_km,
        predicted_y_offsets_km=predicted_y_offsets_km,
        actual_y_offsets_km=actual_y_offsets_km,
        num_distance_error_bins=num_distance_error_bins,
        distance_error_limits_metres=distance_error_limits_metres,
        plotting_orig_error=True,
        output_dir_name=output_dir_name
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
        num_distance_error_bins=getattr(
            INPUT_ARG_OBJECT, NUM_DISTANCE_BINS_ARG_NAME
        ),
        distance_error_limits_metres=numpy.array(
            getattr(INPUT_ARG_OBJECT, DISTANCE_ERROR_LIMITS_ARG_NAME),
            dtype=float
        ),
        num_direction_error_bins=getattr(
            INPUT_ARG_OBJECT, NUM_DIRECTION_BINS_ARG_NAME
        ),
        plot_orig_errors=bool(
            getattr(INPUT_ARG_OBJECT, PLOT_ORIG_ERRORS_ARG_NAME)
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
