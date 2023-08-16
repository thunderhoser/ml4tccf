"""Evaluates trained neural net for intensity estimation."""

import os
import sys
import glob
import argparse
import numpy
import xarray
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import scalar_evaluation
import neural_net_training_intensity as nn_training
import scalar_evaluation_plotting as scalar_eval_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

METRES_PER_SECOND_TO_KT = 3.6 / 1.852

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_PATTERN_ARG_NAME = 'input_prediction_file_pattern'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_PATTERN_HELP_STRING = (
    'Glob pattern for input files.  Each should be produced by '
    '`neural_net_training_intensity.write_prediction_file`, and predictions in '
    'all these files will be evaluated together.'
)
OUTPUT_DIR_HELP_STRING = 'Name of output directory.  Stuff will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_PATTERN_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_PATTERN_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(prediction_file_pattern, output_dir_name):
    """Evaluates trained neural net for intensity estimation.

    This is effectively the main method.

    :param prediction_file_pattern: See documentation at top of file.
    :param output_dir_name: Same.
    :raises: ValueError: if no prediction files could be found.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    prediction_file_names = glob.glob(prediction_file_pattern)
    if len(prediction_file_names) == 0:
        error_string = (
            'Could not find any prediction file with the following pattern: '
            '"{0:s}"'
        ).format(prediction_file_pattern)

        raise ValueError(error_string)

    prediction_file_names.sort()

    target_intensities_m_s01 = numpy.array([], dtype=float)
    predicted_intensities_m_s01 = numpy.array([], dtype=float)

    for this_file_name in prediction_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_prediction_table_xarray = xarray.open_dataset(this_file_name)
        tptx = this_prediction_table_xarray

        target_intensities_m_s01 = numpy.concatenate((
            target_intensities_m_s01,
            tptx[nn_training.TARGET_INTENSITY_KEY].values
        ))
        predicted_intensities_m_s01 = numpy.concatenate((
            predicted_intensities_m_s01,
            tptx[nn_training.PREDICTED_INTENSITY_KEY].values
        ))

    print(SEPARATOR_STRING)

    target_intensities_kt = METRES_PER_SECOND_TO_KT * target_intensities_m_s01
    predicted_intensities_kt = (
        METRES_PER_SECOND_TO_KT * predicted_intensities_m_s01
    )

    (
        mean_predictions_kt, mean_observations_kt, example_counts
    ) = scalar_evaluation._get_reliability_curve_one_variable(
        target_values=target_intensities_kt,
        predicted_values=predicted_intensities_kt,
        is_var_direction=False,
        num_bins=30, min_bin_edge=30., max_bin_edge=180., invert=False
    )

    (
        _, inv_mean_observations_kt, inv_example_counts
    ) = scalar_evaluation._get_reliability_curve_one_variable(
        target_values=target_intensities_kt,
        predicted_values=predicted_intensities_kt,
        is_var_direction=False,
        num_bins=30, min_bin_edge=30., max_bin_edge=180., invert=True
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    scalar_eval_plotting.plot_attributes_diagram(
        figure_object=figure_object,
        axes_object=axes_object,
        mean_predictions=mean_predictions_kt,
        mean_observations=mean_observations_kt,
        mean_value_in_training=46.5832769126608,
        min_value_to_plot=30., max_value_to_plot=180.
    )
    scalar_eval_plotting.plot_inset_histogram(
        figure_object=figure_object,
        bin_centers=mean_predictions_kt,
        bin_counts=example_counts,
        has_predictions=True,
        bar_colour=scalar_eval_plotting.RELIABILITY_LINE_COLOUR
    )
    scalar_eval_plotting.plot_inset_histogram(
        figure_object=figure_object,
        bin_centers=mean_predictions_kt,
        bin_counts=inv_example_counts,
        has_predictions=False,
        bar_colour=scalar_eval_plotting.RELIABILITY_LINE_COLOUR
    )

    reliability_kt2 = scalar_evaluation._get_reliability(
        binned_mean_predictions=mean_predictions_kt,
        binned_mean_observations=mean_observations_kt,
        binned_example_counts=example_counts,
        is_var_direction=False
    )

    mean_absolute_error_kt = numpy.mean(
        numpy.absolute(predicted_intensities_kt - target_intensities_kt)
    )
    mean_signed_error_kt = numpy.mean(
        predicted_intensities_kt - target_intensities_kt
    )
    rmse_kt = numpy.sqrt(
        numpy.mean((predicted_intensities_kt - target_intensities_kt) ** 2)
    )

    title_string = (
        'MAE = {0:.2f} kt; bias = {1:.2f} kt ; RMSE = {2:.2f} kt;\n'
        'reliability = {3:.2f} kt'
    ).format(
        mean_absolute_error_kt, mean_signed_error_kt, rmse_kt, reliability_kt2
    )
    title_string += r'$^{2}$'
    print(title_string)

    axes_object.set_title(title_string)
    figure_file_name = '{0:s}/attributes_diagram.jpg'.format(output_dir_name)

    print('Saving figure to file: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_pattern=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_PATTERN_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
