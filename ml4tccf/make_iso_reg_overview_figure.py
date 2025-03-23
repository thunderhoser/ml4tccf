"""Creates overview figure to explain isotonic regression."""

import os
import sys
import argparse
import numpy
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import imagemagick_utils
import prediction_io
import scalar_prediction_utils as prediction_utils

numpy.random.seed(666)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

RAW_PREDICTION_LINE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
CORR1_PREDICTION_LINE_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
CORR2_PREDICTION_LINE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
TARGET_LINE_COLOUR = numpy.full(3, 0.)
LINE_WIDTH = 3

PREDICTION_MARKER_TYPE = 'o'
PREDICTION_MARKER_SIZE = 16
TARGET_MARKER_TYPE = '*'
TARGET_MARKER_SIZE = 24

DEFAULT_FONT_SIZE = 30
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

RAW_FILE_ARG_NAME = 'input_raw_prediction_file_name'
FIRST_CORRECTED_FILE_ARG_NAME = 'input_corr1_prediction_file_name'
SECOND_CORRECTED_FILE_ARG_NAME = 'input_corr2_prediction_file_name'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
PLOT_X_COORD_ARG_NAME = 'plot_x_coord'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

RAW_FILE_HELP_STRING = (
    'Path to file with raw predictions (i.e., from base model).  Will be read '
    'by `prediction_io.read_file`.'
)
FIRST_CORRECTED_FILE_HELP_STRING = (
    'Path to file with bias-corrected ensemble means (i.e., from isotonic '
    'regression).  Will be read by `prediction_io.read_file`.'
)
SECOND_CORRECTED_FILE_HELP_STRING = (
    'Path to file with bias-corrected ensemble means and spreads (both from '
    'isotonic regression).  Will be read by `prediction_io.read_file`.'
)
NUM_EXAMPLES_HELP_STRING = 'Number of examples to use in plot.'
PLOT_X_COORD_HELP_STRING = (
    'Boolean flag.  If 1 (0), will plot predictions for x- (y-)coordinate.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + RAW_FILE_ARG_NAME, type=str, required=True,
    help=RAW_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_CORRECTED_FILE_ARG_NAME, type=str, required=True,
    help=FIRST_CORRECTED_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SECOND_CORRECTED_FILE_ARG_NAME, type=str, required=True,
    help=SECOND_CORRECTED_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=True,
    help=NUM_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_X_COORD_ARG_NAME, type=int, required=True,
    help=PLOT_X_COORD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(raw_prediction_file_name, corr1_prediction_file_name,
         corr2_prediction_file_name, num_examples, plot_x_coord,
         output_dir_name):
    """Creates overview figure to explain isotonic regression.

    This is effectively the main method.

    :param raw_prediction_file_name: See documentation at top of file.
    :param corr1_prediction_file_name: Same.
    :param corr2_prediction_file_name: Same.
    :param num_examples: Same.
    :param plot_x_coord: Same.
    :param output_dir_name: Same.
    """

    # Basic error-checking.
    error_checking.assert_is_geq(num_examples, 10)
    error_checking.assert_is_leq(num_examples, 1000)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Read files and subset to desired field.
    print('Reading data from: "{0:s}"...'.format(raw_prediction_file_name))
    raw_prediction_table_xarray = prediction_io.read_file(
        raw_prediction_file_name
    )

    print('Reading data from: "{0:s}"...'.format(corr1_prediction_file_name))
    corr1_prediction_table_xarray = prediction_io.read_file(
        corr1_prediction_file_name
    )

    print('Reading data from: "{0:s}"...'.format(corr2_prediction_file_name))
    corr2_prediction_table_xarray = prediction_io.read_file(
        corr2_prediction_file_name
    )

    # Make sure files have matching metadata.
    raw_ptx = raw_prediction_table_xarray
    corr1_ptx = corr1_prediction_table_xarray
    corr2_ptx = corr2_prediction_table_xarray

    num_examples_total = len(
        raw_ptx.coords[prediction_utils.EXAMPLE_DIM_KEY].values
    )
    ensemble_size = len(
        raw_ptx.coords[prediction_utils.ENSEMBLE_MEMBER_DIM_KEY].values
    )

    assert (
        num_examples_total ==
        len(corr1_ptx.coords[prediction_utils.EXAMPLE_DIM_KEY].values) ==
        len(corr2_ptx.coords[prediction_utils.EXAMPLE_DIM_KEY].values)
    )
    assert (
        ensemble_size ==
        len(corr1_ptx.coords[prediction_utils.ENSEMBLE_MEMBER_DIM_KEY].values) ==
        len(corr2_ptx.coords[prediction_utils.ENSEMBLE_MEMBER_DIM_KEY].values)
    )
    assert ensemble_size > 1

    assert numpy.array_equal(
        raw_ptx[prediction_utils.TARGET_TIME_KEY].values,
        corr1_ptx[prediction_utils.TARGET_TIME_KEY].values
    )
    assert numpy.array_equal(
        raw_ptx[prediction_utils.TARGET_TIME_KEY].values,
        corr2_ptx[prediction_utils.TARGET_TIME_KEY].values
    )
    assert numpy.array_equal(
        raw_ptx[prediction_utils.CYCLONE_ID_KEY].values,
        corr1_ptx[prediction_utils.CYCLONE_ID_KEY].values,
    )
    assert numpy.array_equal(
        raw_ptx[prediction_utils.CYCLONE_ID_KEY].values,
        corr2_ptx[prediction_utils.CYCLONE_ID_KEY].values,
    )
    assert numpy.allclose(
        raw_ptx[prediction_utils.ACTUAL_ROW_OFFSET_KEY].values,
        corr1_ptx[prediction_utils.ACTUAL_ROW_OFFSET_KEY].values,
        atol=1e-6
    )
    assert numpy.allclose(
        raw_ptx[prediction_utils.ACTUAL_ROW_OFFSET_KEY].values,
        corr2_ptx[prediction_utils.ACTUAL_ROW_OFFSET_KEY].values,
        atol=1e-6
    )
    assert numpy.allclose(
        raw_ptx[prediction_utils.ACTUAL_COLUMN_OFFSET_KEY].values,
        corr1_ptx[prediction_utils.ACTUAL_COLUMN_OFFSET_KEY].values,
        atol=1e-6
    )
    assert numpy.allclose(
        raw_ptx[prediction_utils.ACTUAL_COLUMN_OFFSET_KEY].values,
        corr2_ptx[prediction_utils.ACTUAL_COLUMN_OFFSET_KEY].values,
        atol=1e-6
    )

    # Plot stuff for correction #1 (of ensemble means).
    grid_spacings_km = raw_ptx[prediction_utils.GRID_SPACING_KEY].values

    if plot_x_coord:
        actual_offsets_km = (
            grid_spacings_km *
            raw_ptx[prediction_utils.ACTUAL_COLUMN_OFFSET_KEY].values
        )
        raw_predicted_offset_matrix_km = (
            numpy.expand_dims(grid_spacings_km, axis=-1) *
            raw_ptx[prediction_utils.PREDICTED_COLUMN_OFFSET_KEY].values
        )
        corr1_predicted_offset_matrix_km = (
            numpy.expand_dims(grid_spacings_km, axis=-1) *
            corr1_ptx[prediction_utils.PREDICTED_COLUMN_OFFSET_KEY].values
        )
    else:
        actual_offsets_km = (
            grid_spacings_km *
            raw_ptx[prediction_utils.ACTUAL_ROW_OFFSET_KEY].values
        )
        raw_predicted_offset_matrix_km = (
            numpy.expand_dims(grid_spacings_km, axis=-1) *
            raw_ptx[prediction_utils.PREDICTED_ROW_OFFSET_KEY].values
        )
        corr1_predicted_offset_matrix_km = (
            numpy.expand_dims(grid_spacings_km, axis=-1) *
            corr1_ptx[prediction_utils.PREDICTED_ROW_OFFSET_KEY].values
        )

    select_indices = numpy.linspace(
        0, num_examples_total - 1, num=num_examples_total, dtype=int
    )
    if num_examples_total > num_examples:
        select_indices = numpy.random.choice(
            select_indices, size=num_examples, replace=False
        )

    actual_offsets_km = actual_offsets_km[select_indices]
    raw_predicted_offset_matrix_km = (
        raw_predicted_offset_matrix_km[select_indices, :]
    )
    corr1_predicted_offset_matrix_km = (
        corr1_predicted_offset_matrix_km[select_indices, :]
    )

    # Create plot to show how correction #1 affects ensemble means.
    sort_indices = numpy.argsort(actual_offsets_km)
    actual_offsets_km = actual_offsets_km[sort_indices]
    raw_predicted_offset_matrix_km = (
        raw_predicted_offset_matrix_km[sort_indices, :]
    )
    corr1_predicted_offset_matrix_km = (
        corr1_predicted_offset_matrix_km[sort_indices, :]
    )

    num_examples = len(actual_offsets_km)
    example_indices = numpy.linspace(
        1, num_examples, num=num_examples, dtype=float
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    legend_handles = [None] * 3

    legend_handles[0] = axes_object.plot(
        example_indices, numpy.mean(raw_predicted_offset_matrix_km, axis=-1),
        color=RAW_PREDICTION_LINE_COLOUR,
        linestyle='solid',
        linewidth=LINE_WIDTH,
        marker=PREDICTION_MARKER_TYPE,
        markersize=PREDICTION_MARKER_SIZE,
        markerfacecolor=RAW_PREDICTION_LINE_COLOUR,
        markeredgecolor=RAW_PREDICTION_LINE_COLOUR,
        markeredgewidth=0
    )[0]
    legend_handles[1] = axes_object.plot(
        example_indices, numpy.mean(corr1_predicted_offset_matrix_km, axis=-1),
        color=CORR1_PREDICTION_LINE_COLOUR,
        linestyle='solid',
        linewidth=LINE_WIDTH,
        marker=PREDICTION_MARKER_TYPE,
        markersize=PREDICTION_MARKER_SIZE,
        markerfacecolor=CORR1_PREDICTION_LINE_COLOUR,
        markeredgecolor=CORR1_PREDICTION_LINE_COLOUR,
        markeredgewidth=0
    )[0]
    legend_handles[2] = axes_object.plot(
        example_indices, actual_offsets_km,
        color=TARGET_LINE_COLOUR,
        linestyle='solid',
        linewidth=LINE_WIDTH,
        marker=TARGET_MARKER_TYPE,
        markersize=TARGET_MARKER_SIZE,
        markerfacecolor=TARGET_LINE_COLOUR,
        markeredgecolor=TARGET_LINE_COLOUR,
        markeredgewidth=0
    )[0]

    raw_mean_abs_error_km = numpy.mean(numpy.absolute(
        numpy.mean(raw_predicted_offset_matrix_km, axis=-1) -
        actual_offsets_km
    ))
    corrected_mean_abs_error_km = numpy.mean(numpy.absolute(
        numpy.mean(corr1_predicted_offset_matrix_km, axis=-1) -
        actual_offsets_km
    ))

    legend_strings = [
        'Raw ens. mean\n(MAE = {0:.1f} km)'.format(raw_mean_abs_error_km),
        'Ens. mean after correction #1\n(MAE = {0:.1f} km)'.format(
            corrected_mean_abs_error_km
        ),
        'Actual'
    ]

    axes_object.legend(
        legend_handles, legend_strings, loc='upper left', fontsize=20,
        bbox_to_anchor=(0, 0.95), fancybox=True, shadow=False,
        facecolor='white', edgecolor='k', framealpha=0.5, ncol=1
    )

    axes_object.set_xticks([], [])
    axes_object.set_xlabel('Data sample')
    axes_object.set_ylabel('{0:s} correction distance (km)'.format(
        'Zonal' if plot_x_coord else 'Meridional'
    ))
    axes_object.set_title('(a) Effect of correction #1 on ensemble mean')

    panel_file_names = []
    panel_file_names.append('{0:s}/correction1_ensemble_mean.jpg'.format(
        output_dir_name
    ))

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Create plot to show how correction #1 affects ensemble distribution.
    select_index = numpy.random.choice(
        example_indices, size=1, replace=False
    )[0]
    select_index = int(select_index)

    raw_predicted_offsets_km = raw_predicted_offset_matrix_km[select_index, :]
    corr1_predicted_offsets_km = (
        corr1_predicted_offset_matrix_km[select_index, :]
    )
    actual_offset_km = actual_offsets_km[select_index]

    raw_kde_object = gaussian_kde(raw_predicted_offsets_km, bw_method='scott')
    raw_x_values = numpy.linspace(
        numpy.min(raw_predicted_offsets_km) - 2.,
        numpy.max(raw_predicted_offsets_km) + 2.,
        num=1000
    )
    raw_y_values = raw_kde_object(raw_x_values)

    corr1_kde_object = gaussian_kde(
        corr1_predicted_offsets_km, bw_method='scott'
    )
    corr1_x_values = numpy.linspace(
        numpy.min(corr1_predicted_offsets_km) - 2.,
        numpy.max(corr1_predicted_offsets_km) + 2.,
        num=1000
    )
    corr1_y_values = corr1_kde_object(corr1_x_values)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    legend_handles = [None] * 3

    legend_handles[0] = axes_object.plot(
        raw_x_values, raw_y_values,
        color=RAW_PREDICTION_LINE_COLOUR,
        linestyle='solid',
        linewidth=LINE_WIDTH
    )[0]
    axes_object.fill_between(
        raw_x_values, raw_y_values, color=RAW_PREDICTION_LINE_COLOUR, alpha=0.2
    )

    legend_handles[1] = axes_object.plot(
        corr1_x_values, corr1_y_values,
        color=CORR1_PREDICTION_LINE_COLOUR,
        linestyle='solid',
        linewidth=LINE_WIDTH
    )[0]
    axes_object.fill_between(
        corr1_x_values, corr1_y_values,
        color=CORR1_PREDICTION_LINE_COLOUR, alpha=0.2
    )

    y_max = axes_object.get_ylim()[1]

    legend_handles[2] = axes_object.plot(
        numpy.full(2, actual_offset_km), axes_object.get_ylim(),
        color=TARGET_LINE_COLOUR,
        linestyle='dashed',
        linewidth=0.5 * LINE_WIDTH
    )[0]

    legend_strings = [
        'Raw ensemble', 'Ensemble after correction #1', 'Actual value'
    ]
    axes_object.legend(
        legend_handles, legend_strings, loc='lower left', fontsize=20,
        bbox_to_anchor=(0, 0.1), fancybox=True, shadow=False,
        facecolor='white', edgecolor='k', framealpha=0.5, ncol=1
    )

    axes_object.set_ylabel('Probability density')
    axes_object.set_xlabel('{0:s} correction distance (km)'.format(
        'Zonal' if plot_x_coord else 'Meridional'
    ))

    x_min = min([numpy.min(raw_x_values), numpy.min(corr1_x_values)])
    x_max = max([numpy.max(raw_x_values), numpy.max(corr1_x_values)])
    axes_object.set_xlim([x_min, x_max])
    axes_object.set_ylim([0, y_max])
    axes_object.set_title('(b) Effect of correction #1 on full ensemble')

    panel_file_names.append(
        '{0:s}/correction1_ensemble_distribution.jpg'.format(output_dir_name)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot stuff for correction #2 (of ensemble spread).
    if plot_x_coord:
        actual_offsets_km = (
            grid_spacings_km *
            raw_ptx[prediction_utils.ACTUAL_COLUMN_OFFSET_KEY].values
        )
        corr1_predicted_offset_matrix_km = (
            numpy.expand_dims(grid_spacings_km, axis=-1) *
            corr1_ptx[prediction_utils.PREDICTED_COLUMN_OFFSET_KEY].values
        )
        corr2_predicted_offset_matrix_km = (
            numpy.expand_dims(grid_spacings_km, axis=-1) *
            corr2_ptx[prediction_utils.PREDICTED_COLUMN_OFFSET_KEY].values
        )
    else:
        actual_offsets_km = (
            grid_spacings_km *
            raw_ptx[prediction_utils.ACTUAL_ROW_OFFSET_KEY].values
        )
        corr1_predicted_offset_matrix_km = (
            numpy.expand_dims(grid_spacings_km, axis=-1) *
            corr1_ptx[prediction_utils.PREDICTED_ROW_OFFSET_KEY].values
        )
        corr2_predicted_offset_matrix_km = (
            numpy.expand_dims(grid_spacings_km, axis=-1) *
            corr2_ptx[prediction_utils.PREDICTED_ROW_OFFSET_KEY].values
        )

    actual_offsets_km = actual_offsets_km[select_indices]
    corr1_predicted_offset_matrix_km = (
        corr1_predicted_offset_matrix_km[select_indices, :]
    )
    corr2_predicted_offset_matrix_km = (
        corr2_predicted_offset_matrix_km[select_indices, :]
    )

    # Create plot to show how correction #2 affects ensemble variances.
    squared_errors_km2 = (
        actual_offsets_km -
        numpy.mean(corr1_predicted_offset_matrix_km, axis=-1)
    ) ** 2

    sort_indices = numpy.argsort(squared_errors_km2)
    corr1_predicted_offset_matrix_km = (
        corr1_predicted_offset_matrix_km[sort_indices, :]
    )
    corr2_predicted_offset_matrix_km = (
        corr2_predicted_offset_matrix_km[sort_indices, :]
    )

    num_examples = len(actual_offsets_km)
    example_indices = numpy.linspace(
        1, num_examples, num=num_examples, dtype=float
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    legend_handles = [None] * 3

    legend_handles[0] = axes_object.plot(
        example_indices, numpy.var(corr1_predicted_offset_matrix_km, axis=-1),
        color=CORR1_PREDICTION_LINE_COLOUR,
        linestyle='solid',
        linewidth=LINE_WIDTH,
        marker=PREDICTION_MARKER_TYPE,
        markersize=PREDICTION_MARKER_SIZE,
        markerfacecolor=CORR1_PREDICTION_LINE_COLOUR,
        markeredgecolor=CORR1_PREDICTION_LINE_COLOUR,
        markeredgewidth=0
    )[0]
    legend_handles[1] = axes_object.plot(
        example_indices, numpy.var(corr2_predicted_offset_matrix_km, axis=-1),
        color=CORR2_PREDICTION_LINE_COLOUR,
        linestyle='solid',
        linewidth=LINE_WIDTH,
        marker=PREDICTION_MARKER_TYPE,
        markersize=PREDICTION_MARKER_SIZE,
        markerfacecolor=CORR2_PREDICTION_LINE_COLOUR,
        markeredgecolor=CORR2_PREDICTION_LINE_COLOUR,
        markeredgewidth=0
    )[0]
    legend_handles[2] = axes_object.plot(
        example_indices, squared_errors_km2,
        color=TARGET_LINE_COLOUR,
        linestyle='solid',
        linewidth=LINE_WIDTH,
        marker=TARGET_MARKER_TYPE,
        markersize=TARGET_MARKER_SIZE,
        markerfacecolor=TARGET_LINE_COLOUR,
        markeredgecolor=TARGET_LINE_COLOUR,
        markeredgewidth=0
    )[0]

    corr1_spread_skill_ratio = (
        numpy.mean(numpy.std(corr1_predicted_offset_matrix_km, axis=-1)) /
        numpy.sqrt(numpy.mean(squared_errors_km2))
    )
    corrected_spread_skill_ratio = (
        numpy.mean(numpy.std(corr2_predicted_offset_matrix_km, axis=-1)) /
        numpy.sqrt(numpy.mean(squared_errors_km2))
    )

    legend_strings = [
        'Ens. variance after correction #1\n(SSRAT = {0:.2f})'.format(
            corr1_spread_skill_ratio
        ),
        'Ens. variance after correction #2\n(SSRAT = {0:.2f})'.format(
            corrected_spread_skill_ratio
        ),
        'Squared error'
    ]

    axes_object.legend(
        legend_handles, legend_strings, loc='upper left', fontsize=20,
        bbox_to_anchor=(0, 0.95), fancybox=True, shadow=False,
        facecolor='white', edgecolor='k', framealpha=0.5, ncol=1
    )

    axes_object.set_xticks([], [])
    axes_object.set_xlabel('Data sample')
    axes_object.set_ylabel('{0:s} correction distance (km)'.format(
        'Zonal' if plot_x_coord else 'Meridional'
    ))
    axes_object.set_title('(c) Effect of correction #2 on ensemble variance')

    panel_file_names.append('{0:s}/correction2_ensemble_variance.jpg'.format(
        output_dir_name
    ))

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Create plot to show how correction #1 affects ensemble distribution.
    corr1_predicted_offsets_km = (
        corr1_predicted_offset_matrix_km[select_index, :]
    )
    corr2_predicted_offsets_km = (
        corr2_predicted_offset_matrix_km[select_index, :]
    )
    actual_offset_km = actual_offsets_km[select_index]

    corr1_kde_object = gaussian_kde(
        corr1_predicted_offsets_km, bw_method='scott'
    )
    corr1_x_values = numpy.linspace(
        numpy.min(corr1_predicted_offsets_km) - 2.,
        numpy.max(corr1_predicted_offsets_km) + 2.,
        num=1000
    )
    corr1_y_values = corr1_kde_object(corr1_x_values)

    corr2_kde_object = gaussian_kde(
        corr2_predicted_offsets_km, bw_method='scott'
    )
    corr2_x_values = numpy.linspace(
        numpy.min(corr2_predicted_offsets_km) - 2.,
        numpy.max(corr2_predicted_offsets_km) + 2.,
        num=1000
    )
    corr2_y_values = corr2_kde_object(corr2_x_values)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    legend_handles = [None] * 3

    legend_handles[0] = axes_object.plot(
        corr1_x_values, corr1_y_values,
        color=CORR1_PREDICTION_LINE_COLOUR,
        linestyle='solid',
        linewidth=LINE_WIDTH
    )[0]
    axes_object.fill_between(
        corr1_x_values, corr1_y_values,
        color=CORR1_PREDICTION_LINE_COLOUR, alpha=0.2
    )

    legend_handles[1] = axes_object.plot(
        corr2_x_values, corr2_y_values,
        color=CORR2_PREDICTION_LINE_COLOUR,
        linestyle='solid',
        linewidth=LINE_WIDTH
    )[0]
    axes_object.fill_between(
        corr2_x_values, corr2_y_values,
        color=CORR2_PREDICTION_LINE_COLOUR, alpha=0.2
    )

    y_max = axes_object.get_ylim()[1]

    legend_handles[2] = axes_object.plot(
        numpy.full(2, actual_offset_km), axes_object.get_ylim(),
        color=TARGET_LINE_COLOUR,
        linestyle='dashed',
        linewidth=0.5 * LINE_WIDTH
    )[0]

    legend_strings = [
        'Ensemble after correction #1', 'Ensemble after correction #2',
        'Actual value'
    ]
    axes_object.legend(
        legend_handles, legend_strings, loc='lower left', fontsize=20,
        bbox_to_anchor=(0, 0.1), fancybox=True, shadow=False,
        facecolor='white', edgecolor='k', framealpha=0.5, ncol=1
    )

    axes_object.set_ylabel('Probability density')
    axes_object.set_xlabel('{0:s} correction distance (km)'.format(
        'Zonal' if plot_x_coord else 'Meridional'
    ))

    x_min = min([numpy.min(corr1_x_values), numpy.min(corr2_x_values)])
    x_max = max([numpy.max(corr1_x_values), numpy.max(corr2_x_values)])
    axes_object.set_xlim([x_min, x_max])
    axes_object.set_ylim([0, y_max])
    axes_object.set_title('(d) Effect of correction #2 on full ensemble')

    panel_file_names.append(
        '{0:s}/correction2_ensemble_distribution.jpg'.format(output_dir_name)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    concat_file_name = '{0:s}/iso_reg_overview.jpg'.format(output_dir_name)
    print('Concatenating panels to: "{0:s}"...'.format(concat_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_file_name,
        num_panel_rows=2,
        num_panel_columns=2
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        raw_prediction_file_name=getattr(INPUT_ARG_OBJECT, RAW_FILE_ARG_NAME),
        corr1_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, FIRST_CORRECTED_FILE_ARG_NAME
        ),
        corr2_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, SECOND_CORRECTED_FILE_ARG_NAME
        ),
        num_examples=getattr(
            INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME
        ),
        plot_x_coord=bool(getattr(INPUT_ARG_OBJECT, PLOT_X_COORD_ARG_NAME)),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
