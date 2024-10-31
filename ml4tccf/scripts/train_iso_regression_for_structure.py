"""Trains isotonic regression for TC-structure parameters."""

import glob
import argparse
from ml4tccf.io import structure_prediction_io as prediction_io
from ml4tccf.utils import structure_prediction_utils as prediction_utils
from ml4tccf.machine_learning import \
    isotonic_regression_for_structure as isotonic_regression

INPUT_FILE_PATTERN_ARG_NAME = 'input_prediction_file_pattern'
OUTPUT_FILE_ARG_NAME = 'output_model_file_name'

INPUT_FILE_PATTERN_HELP_STRING = (
    'Glob pattern for input files.  Each will be read by '
    '`structure_prediction_io.read_file`, and predictions in all these files '
    'will be used for training.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file (Dill format).  Isotonic-regression models will be '
    'written here by `isotonic_regression_for_structure.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_PATTERN_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_PATTERN_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(prediction_file_pattern, output_file_name):
    """Trains isotonic regression for TC-structure parameters.

    This is effectively the main method.

    :param prediction_file_pattern: See documentation at top of this script.
    :param output_file_name: Same.
    :raises: ValueError: if predictions were already made with isotonic
        regression.
    """

    # TODO(thunderhoser): Implement ValueError.  Add isotonic_model as metadata
    # field for predictions.

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

    prediction_table_xarray = prediction_utils.concat_over_examples(
        prediction_tables_xarray
    )
    target_field_to_model_object = (
        isotonic_regression.train_models(prediction_table_xarray)
    )

    print('Writing isotonic-regression models to: "{0:s}"...'.format(
        output_file_name
    ))
    isotonic_regression.write_file(
        dill_file_name=output_file_name,
        target_field_to_model_object=target_field_to_model_object
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_pattern=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_PATTERN_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
