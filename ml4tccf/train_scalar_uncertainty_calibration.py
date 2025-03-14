"""Trains uncertainty-calibration models for scalar target vars (x and y)."""

import os
import sys
import glob
import argparse

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import prediction_io
import scalar_prediction_utils as prediction_utils
import scalar_uncertainty_calibration as uncertainty_calib

INPUT_FILE_PATTERN_ARG_NAME = 'input_prediction_file_pattern'
OUTPUT_DIR_ARG_NAME = 'output_model_dir_name'

INPUT_FILE_PATTERN_HELP_STRING = (
    'Glob pattern for input files.  Each will be read by '
    '`prediction_io.read_file`, and predictions in all these files will be '
    'used for training.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Uncertainty-calibration models will be written '
    'here by `uncertainty_calibration.write_file`, to a file name determined '
    'by `uncertainty_calibration.find_file`.'
)

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
    """Trains uncertainty-calibration models for scalar target vars (x and y).

    This is effectively the main method.

    :param prediction_file_pattern: See documentation at top of file.
    :param output_dir_name: Same.
    :raises: ValueError: if predictions were already post-processed with
        uncertainty calibration.
    """

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

    pt = prediction_table_xarray
    if pt.attrs[prediction_utils.UNCERTAINTY_CALIB_MODEL_FILE_KEY] is not None:
        raise ValueError(
            'Predictions used for training uncertainty calibration must be '
            'made with base model only (i.e., must not already include '
            'uncertainty calibration).'
        )

    x_coord_model_object, y_coord_model_object = (
        uncertainty_calib.train_models(prediction_table_xarray)
    )

    output_file_name = uncertainty_calib.find_file(
        model_dir_name=output_dir_name, raise_error_if_missing=False
    )
    print('Writing uncertainty-calibration models to: "{0:s}"...'.format(
        output_file_name
    ))
    uncertainty_calib.write_file(
        dill_file_name=output_file_name,
        x_coord_model_object=x_coord_model_object,
        y_coord_model_object=y_coord_model_object
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_pattern=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_PATTERN_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
