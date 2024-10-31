"""Applies isotonic regression for TC-structure parameters."""

import glob
import argparse
import numpy
from ml4tccf.io import structure_prediction_io as prediction_io
from ml4tccf.utils import structure_prediction_utils as prediction_utils
from ml4tccf.machine_learning import \
    isotonic_regression_for_structure as isotonic_regression

INPUT_FILE_PATTERN_ARG_NAME = 'input_prediction_file_pattern'
MODEL_FILE_ARG_NAME = 'input_model_file_name'
OUTPUT_PREDICTION_DIR_ARG_NAME = 'output_prediction_dir_name'

INPUT_FILE_PATTERN_HELP_STRING = (
    'Glob pattern for input files.  Each will be read by '
    '`structure_prediction_io.read_file`, and predictions in all these files '
    'will be bias-corrected.'
)
MODEL_FILE_HELP_STRING = (
    'Path to file with set of trained isotonic-regression models.  Will be read'
    ' by `isotonic_regression_for_structure.read_file`.'
)
OUTPUT_PREDICTION_DIR_HELP_STRING = (
    'Path to output directory.  Bias-corrected predictions will be written '
    'here by `structure_prediction_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_PATTERN_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_PATTERN_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_PREDICTION_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_PREDICTION_DIR_HELP_STRING
)


def _run(input_prediction_file_pattern, model_file_name,
         output_prediction_dir_name):
    """Applies isotonic regression for TC-structure parameters.

    This is effectively the main method.

    :param input_prediction_file_pattern: See documentation at top of this
        script.
    :param model_file_name: Same.
    :param output_prediction_dir_name: Same.
    :raises: ValueError: if predictions in `input_prediction_file_pattern` were
        made with isotonic regression.
    """

    # TODO(thunderhoser): Implement ValueError.  Add isotonic_model as metadata
    # field for predictions.

    input_prediction_file_names = glob.glob(input_prediction_file_pattern)
    if len(input_prediction_file_names) == 0:
        error_string = (
            'Could not find any prediction file with the following pattern: '
            '"{0:s}"'
        ).format(input_prediction_file_pattern)

        raise ValueError(error_string)

    input_prediction_file_names.sort()
    num_files = len(input_prediction_file_names)

    print('Reading isotonic-regression models from: "{0:s}"...'.format(
        model_file_name
    ))
    target_field_to_model_object = isotonic_regression.read_file(
        model_file_name
    )

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(
            input_prediction_file_names[i]
        ))
        prediction_table_xarray = prediction_io.read_file(
            input_prediction_file_names[i]
        )
        prediction_table_xarray = isotonic_regression.apply_models(
            prediction_table_xarray=prediction_table_xarray,
            target_field_to_model_object=target_field_to_model_object
        )
        ptx = prediction_table_xarray

        cyclone_id_strings = ptx[prediction_utils.CYCLONE_ID_KEY].values
        assert len(numpy.unique(cyclone_id_strings)) == 1
        cyclone_id_string = cyclone_id_strings[0]

        try:
            cyclone_id_string = cyclone_id_string.decode('utf-8')
        except:
            pass

        output_file_name = prediction_io.find_file(
            directory_name=output_prediction_dir_name,
            cyclone_id_string=cyclone_id_string,
            raise_error_if_missing=False
        )

        print('Writing bias-corrected predictions to: "{0:s}"...'.format(
            output_file_name
        ))

        try:
            baseline_prediction_matrix = (
                ptx[prediction_utils.BASELINE_PREDICTION_KEY].values
            )
        except:
            baseline_prediction_matrix = None

        prediction_io.write_file(
            netcdf_file_name=output_file_name,
            target_matrix=ptx[prediction_utils.TARGET_KEY].values,
            prediction_matrix=ptx[prediction_utils.PREDICTION_KEY].values,
            baseline_prediction_matrix=baseline_prediction_matrix,
            cyclone_id_string=cyclone_id_string,
            target_times_unix_sec=ptx[prediction_utils.TARGET_TIME_KEY].values,
            model_file_name=ptx.attrs[prediction_utils.MODEL_FILE_KEY]
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_file_pattern=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_PATTERN_ARG_NAME
        ),
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        output_prediction_dir_name=getattr(
            INPUT_ARG_OBJECT, OUTPUT_PREDICTION_DIR_ARG_NAME
        )
    )
