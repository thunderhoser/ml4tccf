"""Converts gridded predictions to scalar predictions."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import prediction_io
import scalar_prediction_io
import misc_utils
import gridded_prediction_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
OUTPUT_DIR_ARG_NAME = 'output_prediction_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file (will be read by `prediction_io.read_file`).'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Converted (scalar) predictions will be written '
    'here by `scalar_prediction_io.write_file`, to an exact path determined by '
    '`prediction_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_file_name, output_dir_name):
    """Converts gridded predictions to scalar predictions.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param output_dir_name: Same.
    """

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    gridded_prediction_table_xarray = prediction_io.read_file(input_file_name)
    gpt = gridded_prediction_table_xarray

    gridded_target_matrix = (
        gpt[gridded_prediction_utils.TARGET_MATRIX_KEY].values
    )
    gridded_prediction_matrix = (
        gpt[gridded_prediction_utils.PREDICTION_MATRIX_KEY].values
    )

    num_examples = gridded_prediction_matrix.shape[0]
    ensemble_size = gridded_prediction_matrix.shape[-1]

    actual_row_offsets = numpy.full(num_examples, numpy.nan)
    actual_column_offsets = numpy.full(num_examples, numpy.nan)
    predicted_row_offset_matrix = numpy.full(
        (num_examples, ensemble_size), numpy.nan
    )
    predicted_column_offset_matrix = numpy.full(
        (num_examples, ensemble_size), numpy.nan
    )

    for i in range(num_examples):
        actual_row_offsets[i], actual_column_offsets[i] = (
            misc_utils.target_matrix_to_centroid(gridded_target_matrix[i, ...])
        )

        for j in range(ensemble_size):
            (
                predicted_row_offset_matrix[i, j],
                predicted_column_offset_matrix[i, j]
            ) = misc_utils.prediction_matrix_to_centroid(
                gridded_prediction_matrix[i, ..., j]
            )

    cyclone_id_string = prediction_io.file_name_to_cyclone_id(input_file_name)
    scalar_target_matrix = numpy.transpose(numpy.vstack((
        actual_row_offsets,
        actual_column_offsets,
        gpt[gridded_prediction_utils.GRID_SPACING_KEY].values,
        gpt[gridded_prediction_utils.ACTUAL_CENTER_LATITUDE_KEY].values
    )))
    scalar_prediction_matrix = numpy.stack(
        (predicted_row_offset_matrix, predicted_column_offset_matrix), axis=-2
    )

    output_file_name = prediction_io.find_file(
        directory_name=output_dir_name, cyclone_id_string=cyclone_id_string,
        raise_error_if_missing=False
    )

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    scalar_prediction_io.write_file(
        netcdf_file_name=output_file_name,
        target_matrix=scalar_target_matrix,
        prediction_matrix=scalar_prediction_matrix,
        cyclone_id_string=cyclone_id_string,
        target_times_unix_sec=
        gpt[gridded_prediction_utils.TARGET_TIME_KEY].values,
        model_file_name=gpt.attrs[gridded_prediction_utils.MODEL_FILE_KEY],
        isotonic_model_file_name=None
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
