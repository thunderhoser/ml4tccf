"""Plots evaluation metrics for scalar predictions."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import scalar_evaluation
import scalar_evaluation_plotting as evaluation_plotting

# TODO(thunderhoser): Need to allow confidence intervals in plotting.

METRES_TO_KM = 0.001

TAYLOR_MARKER_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

TARGET_FIELD_TO_VERBOSE_WITH_UNITS = {
    scalar_evaluation.X_OFFSET_NAME: r'$x$-offset (km)',
    scalar_evaluation.Y_OFFSET_NAME: r'$y$-offset (km)',
    scalar_evaluation.OFFSET_DISTANCE_NAME: 'Euclidean offset (km)',
    scalar_evaluation.OFFSET_DIRECTION_NAME: 'offset direction (deg)'
}

EVALUATION_FILE_ARG_NAME = 'input_evaluation_file_name'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

EVALUATION_FILE_HELP_STRING = (
    'Path to file with evaluation results.  Will be read by '
    '`scalar_evaluation.read_file`.'
)
CONFIDENCE_LEVEL_HELP_STRING = (
    'Level (in range 0...1) for confidence intervals based on bootstrapping.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EVALUATION_FILE_ARG_NAME, type=str, required=True,
    help=EVALUATION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(evaluation_file_name, confidence_level, output_dir_name):
    """Plots model evaluation.

    This is effectively the main method.

    :param evaluation_file_name: See documentation at top of file.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    """

    error_checking.assert_is_geq(confidence_level, 0.9)
    error_checking.assert_is_leq(confidence_level, 1.)
    min_percentile = 50. * (1. - confidence_level)
    max_percentile = 50. * (1. + confidence_level)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(evaluation_file_name))
    evaluation_table_xarray = scalar_evaluation.read_file(evaluation_file_name)
    t = evaluation_table_xarray

    target_field_names = (
        t.coords[scalar_evaluation.TARGET_FIELD_DIM].values.tolist()
    )
    xy_indices = numpy.array([
        target_field_names.index(scalar_evaluation.X_OFFSET_NAME),
        target_field_names.index(scalar_evaluation.Y_OFFSET_NAME)
    ], dtype=int)

    for j in xy_indices:
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        mean_predictions_km = METRES_TO_KM * numpy.nanmean(
            t[scalar_evaluation.XY_OFFSET_MEAN_PREDICTION_KEY].values[j, ...],
            axis=-1
        )
        mean_observations_km = METRES_TO_KM * numpy.nanmean(
            t[scalar_evaluation.XY_OFFSET_MEAN_OBSERVATION_KEY].values[j, ...],
            axis=-1
        )
        all_mean_values_km = numpy.concatenate((
            mean_predictions_km, mean_observations_km
        ))

        evaluation_plotting.plot_attributes_diagram(
            figure_object=figure_object, axes_object=axes_object,
            mean_predictions=mean_predictions_km,
            mean_observations=mean_observations_km,
            mean_value_in_training=0.,
            min_value_to_plot=numpy.nanmin(all_mean_values_km),
            max_value_to_plot=numpy.nanmax(all_mean_values_km)
        )

        evaluation_plotting.plot_inset_histogram(
            figure_object=figure_object,
            bin_centers=(
                METRES_TO_KM *
                t[scalar_evaluation.XY_OFFSET_BIN_CENTER_KEY].values[j, :]
            ),
            bin_counts=
            t[scalar_evaluation.XY_OFFSET_BIN_COUNT_KEY].values[j, :],
            has_predictions=True,
            bar_colour=evaluation_plotting.RELIABILITY_LINE_COLOUR
        )

        evaluation_plotting.plot_inset_histogram(
            figure_object=figure_object,
            # bin_centers=(
            #     METRES_TO_KM *
            #     t[scalar_evaluation.XY_OFFSET_INV_BIN_CENTER_KEY].values[j, :]
            # ),
            bin_centers=(
                METRES_TO_KM *
                t[scalar_evaluation.XY_OFFSET_BIN_CENTER_KEY].values[j, :]
            ),
            bin_counts=
            t[scalar_evaluation.XY_OFFSET_INV_BIN_COUNT_KEY].values[j, :],
            has_predictions=False,
            bar_colour=evaluation_plotting.RELIABILITY_LINE_COLOUR
        )

        mean_squared_errors = (METRES_TO_KM ** 2) * (
            t[scalar_evaluation.MEAN_SQUARED_ERROR_KEY].values[j, :]
        )
        mse_skill_scores = (
            t[scalar_evaluation.MSE_SKILL_SCORE_KEY].values[j, :]
        )
        reliabilities = (METRES_TO_KM ** 2) * (
            t[scalar_evaluation.RELIABILITY_KEY].values[j, :]
        )
        resolutions = (METRES_TO_KM ** 2) * (
            t[scalar_evaluation.RESOLUTION_KEY].values[j, :]
        )
        num_bootstrap_reps = len(mean_squared_errors)

        if num_bootstrap_reps == 1:
            title_string = (
                'Attributes diagram for {0:s}\n'
                'MSESS = {1:.3f}; REL = {2:.1f}'
            ).format(
                r'$x$-coord'
                if target_field_names[j] == scalar_evaluation.X_OFFSET_NAME
                else r'$y$-coord',
                mse_skill_scores[0],
                reliabilities[0]
            )
        else:
            title_string = (
                'Attributes diagram for {0:s}\n'
                'MSESS = [{1:.3f}, {2:.3f}]\n'
                'REL = [{3:.1f}, {4:.1f}]'
            ).format(
                r'$x$-coord'
                if target_field_names[j] == scalar_evaluation.X_OFFSET_NAME
                else r'$y$-coord',
                numpy.nanpercentile(mse_skill_scores, min_percentile),
                numpy.nanpercentile(mse_skill_scores, max_percentile),
                numpy.nanpercentile(reliabilities, min_percentile),
                numpy.nanpercentile(reliabilities, max_percentile)
            )

        title_string += r' km$^2$'
        print(title_string)
        axes_object.set_title(title_string)

        axes_object.set_xlabel('Prediction (km)')
        axes_object.set_ylabel('Conditional mean observation (km)')

        figure_file_name = '{0:s}/attributes_diagram_{1:s}.jpg'.format(
            output_dir_name,
            target_field_names[j].replace('_', '-')
        )

        print('Saving figure to file: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    distance_indices = numpy.array([
        target_field_names.index(scalar_evaluation.OFFSET_DISTANCE_NAME)
    ], dtype=int)

    for j in distance_indices:
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        mean_predictions_km = METRES_TO_KM * numpy.nanmean(
            t[scalar_evaluation.OFFSET_DIST_MEAN_PREDICTION_KEY].values,
            axis=-1
        )
        mean_observations_km = METRES_TO_KM * numpy.nanmean(
            t[scalar_evaluation.OFFSET_DIST_MEAN_OBSERVATION_KEY].values,
            axis=-1
        )
        all_mean_values_km = numpy.concatenate((
            mean_predictions_km, mean_observations_km
        ))
        climo_offset_distance_km = (
            METRES_TO_KM * t.attrs[scalar_evaluation.CLIMO_OFFSET_DISTANCE_KEY]
        )

        evaluation_plotting.plot_attributes_diagram(
            figure_object=figure_object, axes_object=axes_object,
            mean_predictions=mean_predictions_km,
            mean_observations=mean_observations_km,
            mean_value_in_training=climo_offset_distance_km,
            min_value_to_plot=numpy.nanmin(all_mean_values_km),
            max_value_to_plot=numpy.nanmax(all_mean_values_km)
        )

        evaluation_plotting.plot_inset_histogram(
            figure_object=figure_object,
            bin_centers=(
                METRES_TO_KM *
                t[scalar_evaluation.OFFSET_DIST_BIN_CENTER_KEY].values
            ),
            bin_counts=t[scalar_evaluation.OFFSET_DIST_BIN_COUNT_KEY].values,
            has_predictions=True,
            bar_colour=evaluation_plotting.RELIABILITY_LINE_COLOUR
        )

        evaluation_plotting.plot_inset_histogram(
            figure_object=figure_object,
            bin_centers=(
                METRES_TO_KM *
                t[scalar_evaluation.OFFSET_DIST_BIN_CENTER_KEY].values
            ),
            # bin_centers=(
            #     METRES_TO_KM *
            #     t[scalar_evaluation.OFFSET_DIST_INV_BIN_CENTER_KEY].values
            # ),
            bin_counts=
            t[scalar_evaluation.OFFSET_DIST_INV_BIN_COUNT_KEY].values,
            has_predictions=False,
            bar_colour=evaluation_plotting.RELIABILITY_LINE_COLOUR
        )

        mean_squared_errors = (METRES_TO_KM ** 2) * (
            t[scalar_evaluation.MEAN_SQUARED_ERROR_KEY].values[j, :]
        )
        mse_skill_scores = (
            t[scalar_evaluation.MSE_SKILL_SCORE_KEY].values[j, :]
        )
        reliabilities = (METRES_TO_KM ** 2) * (
            t[scalar_evaluation.RELIABILITY_KEY].values[j, :]
        )
        resolutions = (METRES_TO_KM ** 2) * (
            t[scalar_evaluation.RESOLUTION_KEY].values[j, :]
        )
        num_bootstrap_reps = len(mean_squared_errors)

        if num_bootstrap_reps == 1:
            title_string = (
                'Attributes diagram for total correction distance\n'
                'MSESS = {0:.3f}; REL = {1:.1f}'
            ).format(
                mse_skill_scores[0],
                reliabilities[0]
            )
        else:
            title_string = (
                'Attributes diagram for total correction distance\n'
                'MSESS = [{0:.3f}, {1:.3f}]\n'
                'REL = [{2:.1f}, {3:.1f}]'
            ).format(
                numpy.nanpercentile(mse_skill_scores, min_percentile),
                numpy.nanpercentile(mse_skill_scores, max_percentile),
                numpy.nanpercentile(reliabilities, min_percentile),
                numpy.nanpercentile(reliabilities, max_percentile)
            )

        title_string += r' km$^2$'
        print(title_string)
        axes_object.set_title(title_string)

        axes_object.set_xlabel('Prediction (km)')
        axes_object.set_ylabel('Conditional mean observation (km)')

        figure_file_name = '{0:s}/attributes_diagram_{1:s}.jpg'.format(
            output_dir_name,
            target_field_names[j].replace('_', '-')
        )

        print('Saving figure to file: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    direction_indices = numpy.array([
        target_field_names.index(scalar_evaluation.OFFSET_DIRECTION_NAME)
    ], dtype=int)

    for j in direction_indices:
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        mean_predictions_deg = numpy.nanmean(
            t[scalar_evaluation.OFFSET_DIR_MEAN_PREDICTION_KEY].values,
            axis=-1
        )
        mean_observations_deg = numpy.nanmean(
            t[scalar_evaluation.OFFSET_DIR_MEAN_OBSERVATION_KEY].values,
            axis=-1
        )
        all_mean_values_deg = numpy.concatenate((
            mean_predictions_deg, mean_observations_deg
        ))

        # TODO(thunderhoser): `mean_value_in_training` should actually be NaN.
        evaluation_plotting.plot_attributes_diagram(
            figure_object=figure_object, axes_object=axes_object,
            mean_predictions=mean_predictions_deg,
            mean_observations=mean_observations_deg,
            mean_value_in_training=180.,
            min_value_to_plot=numpy.nanmin(all_mean_values_deg),
            max_value_to_plot=numpy.nanmax(all_mean_values_deg)
        )

        evaluation_plotting.plot_inset_histogram(
            figure_object=figure_object,
            bin_centers=t[scalar_evaluation.OFFSET_DIR_BIN_CENTER_KEY].values,
            bin_counts=t[scalar_evaluation.OFFSET_DIR_BIN_COUNT_KEY].values,
            has_predictions=True,
            bar_colour=evaluation_plotting.RELIABILITY_LINE_COLOUR
        )

        evaluation_plotting.plot_inset_histogram(
            figure_object=figure_object,
            bin_centers=t[scalar_evaluation.OFFSET_DIR_BIN_CENTER_KEY].values,
            # bin_centers=
            # t[scalar_evaluation.OFFSET_DIR_INV_BIN_CENTER_KEY].values,
            bin_counts=
            t[scalar_evaluation.OFFSET_DIR_INV_BIN_COUNT_KEY].values,
            has_predictions=False,
            bar_colour=evaluation_plotting.RELIABILITY_LINE_COLOUR
        )

        mean_squared_errors = (
            t[scalar_evaluation.MEAN_SQUARED_ERROR_KEY].values[j, :]
        )
        mse_skill_scores = (
            t[scalar_evaluation.MSE_SKILL_SCORE_KEY].values[j, :]
        )
        reliabilities = t[scalar_evaluation.RELIABILITY_KEY].values[j, :]
        resolutions = t[scalar_evaluation.RESOLUTION_KEY].values[j, :]
        num_bootstrap_reps = len(mean_squared_errors)

        if num_bootstrap_reps == 1:
            title_string = (
                'Attributes diagram for correction direction\n'
                'MSESS = {0:.3f}; REL = {1:.1f}'
            ).format(
                mse_skill_scores[0],
                reliabilities[0]
            )
        else:
            title_string = (
                'Attributes diagram for correction direction\n'
                'MSESS = [{0:.3f}, {1:.3f}]\n'
                'REL = [{2:.1f}, {3:.1f}]'
            ).format(
                numpy.nanpercentile(mse_skill_scores, min_percentile),
                numpy.nanpercentile(mse_skill_scores, max_percentile),
                numpy.nanpercentile(reliabilities, min_percentile),
                numpy.nanpercentile(reliabilities, max_percentile)
            )

        title_string += r' deg$^2$'
        print(title_string)
        axes_object.set_title(title_string)

        axes_object.set_xlabel('Prediction (deg)')
        axes_object.set_ylabel('Conditional mean observation (deg)')

        figure_file_name = '{0:s}/attributes_diagram_{1:s}.jpg'.format(
            output_dir_name,
            target_field_names[j].replace('_', '-')
        )

        print('Saving figure to file: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    for j in range(len(target_field_names)):
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )
        axes_object.set_xticks([], [])
        axes_object.set_yticks([], [])

        if target_field_names[j] == scalar_evaluation.OFFSET_DIRECTION_NAME:
            conv_ratio = 1.
        else:
            conv_ratio = METRES_TO_KM

        evaluation_plotting.plot_taylor_diagram(
            target_stdev=conv_ratio * numpy.nanmean(
                t[scalar_evaluation.TARGET_STDEV_KEY].values[j, :]
            ),
            prediction_stdev=conv_ratio * numpy.nanmean(
                t[scalar_evaluation.PREDICTION_STDEV_KEY].values[j, :]
            ),
            correlation=numpy.nanmean(
                t[scalar_evaluation.CORRELATION_KEY].values[j, :]
            ),
            marker_colour=TAYLOR_MARKER_COLOUR,
            figure_object=figure_object
        )

        axes_object.set_ylabel('Stdev of mean predictions', labelpad=40)
        axes_object.set_xlabel('Stdev of targets', labelpad=40)
        figure_object.suptitle('Taylor diagram for {0:s}'.format(
            TARGET_FIELD_TO_VERBOSE_WITH_UNITS[target_field_names[j]]
        ))

        figure_file_name = '{0:s}/taylor_diagram_{1:s}.jpg'.format(
            output_dir_name,
            target_field_names[j].replace('_', '-')
        )

        print('Saving figure to file: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        evaluation_file_name=getattr(
            INPUT_ARG_OBJECT, EVALUATION_FILE_ARG_NAME
        ),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
