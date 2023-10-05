"""Trains nadir-dependent isotonic-regression models for scalar target vars."""

import glob
import argparse
import numpy
from ml4tccf.io import prediction_io
from ml4tccf.utils import scalar_prediction_utils as prediction_utils
from ml4tccf.machine_learning import \
    scalar_nadir_dependent_iso_regression as isotonic_regression

INPUT_FILE_PATTERN_ARG_NAME = 'input_prediction_file_pattern'
EBTRK_FILE_ARG_NAME = 'input_extended_best_track_file_name'
NADIR_RELATIVE_X_CUTOFFS_ARG_NAME = 'nadir_relative_x_cutoffs_metres'
NADIR_RELATIVE_Y_CUTOFFS_ARG_NAME = 'nadir_relative_y_cutoffs_metres'
MIN_SAMPLE_SIZE_ARG_NAME = 'min_training_sample_size'
OUTPUT_DIR_ARG_NAME = 'output_model_dir_name'

INPUT_FILE_PATTERN_HELP_STRING = (
    'Glob pattern for input files.  Each will be read by '
    '`prediction_io.read_file`, and predictions in all these files will be '
    'used for training.'
)
EBTRK_FILE_HELP_STRING = (
    'Path to file with extended best-track data (contains TC-center locations).'
    '  Will be read by `extended_best_track_io.read_file`.'
)
NADIR_RELATIVE_X_CUTOFFS_HELP_STRING = (
    'Category cutoffs for nadir-relative x-coordinate.  Please leave -inf and '
    '+inf out of this list, as they will be added automatically.'
)
NADIR_RELATIVE_Y_CUTOFFS_HELP_STRING = (
    'Category cutoffs for nadir-relative y-coordinate.  Please leave -inf and '
    '+inf out of this list, as they will be added automatically.'
)
MIN_SAMPLE_SIZE_HELP_STRING = (
    'Minimum sample size (number of examples) for training one model.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Isotonic-regression models will be written here'
    ' by `scalar_nadir_dependent_iso_regression.write_file`, to a file name '
    'determined by `scalar_nadir_dependent_iso_regression.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_PATTERN_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_PATTERN_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EBTRK_FILE_ARG_NAME, type=str, required=True,
    help=EBTRK_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NADIR_RELATIVE_X_CUTOFFS_ARG_NAME, type=float, nargs='+',
    required=True, help=NADIR_RELATIVE_X_CUTOFFS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NADIR_RELATIVE_Y_CUTOFFS_ARG_NAME, type=float, nargs='+',
    required=True, help=NADIR_RELATIVE_Y_CUTOFFS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_SAMPLE_SIZE_ARG_NAME, type=int, required=True,
    help=MIN_SAMPLE_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(prediction_file_pattern, ebtrk_file_name,
         nadir_relative_x_cutoffs_metres, nadir_relative_y_cutoffs_metres,
         min_training_sample_size, output_dir_name):
    """Trains nadir-dependent isotonic-regression models for scalar target vars.

    This is effectively the main method.

    :param prediction_file_pattern: See documentation at top of file.
    :param ebtrk_file_name: Same.
    :param nadir_relative_x_cutoffs_metres: Same.
    :param nadir_relative_y_cutoffs_metres: Same.
    :param min_training_sample_size: Same.
    :param output_dir_name: Same.
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
    if (
            pt.attrs[prediction_utils.ISOTONIC_MODEL_FILE_KEY] is not None or
            pt.attrs[prediction_utils.NADIR_DEP_ISO_MODEL_FILE_KEY] is not None
    ):
        raise ValueError(
            'Predictions used for training isotonic regression must be made'
            ' with base model only (i.e., must not already include isotonic'
            ' regression).'
        )

    x_coord_model_objects, y_coord_model_objects = (
        isotonic_regression.train_models(
            prediction_table_xarray=prediction_table_xarray,
            nadir_relative_x_cutoffs_metres=nadir_relative_x_cutoffs_metres,
            nadir_relative_y_cutoffs_metres=nadir_relative_y_cutoffs_metres,
            ebtrk_file_name=ebtrk_file_name,
            min_training_sample_size=min_training_sample_size
        )
    )

    output_file_name = isotonic_regression.find_file(
        model_dir_name=output_dir_name, raise_error_if_missing=False
    )
    print('Writing isotonic-regression models to: "{0:s}"...'.format(
        output_file_name
    ))
    isotonic_regression.write_file(
        dill_file_name=output_file_name,
        x_coord_model_objects=x_coord_model_objects,
        y_coord_model_objects=y_coord_model_objects,
        nadir_relative_x_cutoffs_metres=nadir_relative_x_cutoffs_metres,
        nadir_relative_y_cutoffs_metres=nadir_relative_y_cutoffs_metres
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_pattern=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_PATTERN_ARG_NAME
        ),
        ebtrk_file_name=getattr(INPUT_ARG_OBJECT, EBTRK_FILE_ARG_NAME),
        nadir_relative_x_cutoffs_metres=numpy.array(
            getattr(INPUT_ARG_OBJECT, NADIR_RELATIVE_X_CUTOFFS_ARG_NAME),
            dtype=float
        ),
        nadir_relative_y_cutoffs_metres=numpy.array(
            getattr(INPUT_ARG_OBJECT, NADIR_RELATIVE_Y_CUTOFFS_ARG_NAME),
            dtype=float
        ),
        min_training_sample_size=getattr(
            INPUT_ARG_OBJECT, MIN_SAMPLE_SIZE_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
