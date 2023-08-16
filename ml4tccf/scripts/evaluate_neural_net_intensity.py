"""Evaluates trained neural net for intensity estimation."""

import glob
import argparse
import numpy
import xarray
from ml4tccf.machine_learning import \
    neural_net_training_intensity as nn_training

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
METRES_PER_SECOND_TO_KT = 3.6 / 1.852

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
            tptx[nn_training.TARGET_INTENSITIES_KEY].values
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

    mean_absolute_error_kt = numpy.mean(
        numpy.absolute(predicted_intensities_kt - target_intensities_kt)
    )
    mean_signed_error_kt = numpy.mean(
        predicted_intensities_kt - target_intensities_kt
    )
    rmse_kt = numpy.sqrt(
        numpy.mean((predicted_intensities_kt - target_intensities_kt) ** 2)
    )

    print((
        'Mean absolute error = {0:.2f} kt ... mean signed error = {1:.2f} kt '
        '... RMSE = {2:.2f} kt'
    ).format(
        mean_absolute_error_kt, mean_signed_error_kt, rmse_kt
    ))


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_pattern=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_PATTERN_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
