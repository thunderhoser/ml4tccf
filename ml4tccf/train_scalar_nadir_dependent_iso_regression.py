"""Trains nadir-dependent isotonic-regression models for scalar target vars."""

import os
import sys
import glob
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import prediction_io
import scalar_prediction_utils as prediction_utils
import scalar_nadir_dependent_iso_regression as iso_reg_1d_bins
import scalar_nadir_dependent_iso_reg_2d_bins as iso_reg_2d_bins

INPUT_FILE_PATTERN_ARG_NAME = 'input_prediction_file_pattern'
EBTRK_FILE_ARG_NAME = 'input_extended_best_track_file_name'
NADIR_RELATIVE_X_CUTOFFS_ARG_NAME = 'nadir_relative_x_cutoffs_metres'
NADIR_RELATIVE_Y_CUTOFFS_ARG_NAME = 'nadir_relative_y_cutoffs_metres'
USE_2D_BINS_ARG_NAME = 'use_2d_bins'
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
USE_2D_BINS_HELP_STRING = (
    'Boolean flag.  Either way, the nadir-relative coordinate space will be '
    'divided into 2-D bins.  If this flag is 1, there will be a separate pair '
    'of isotonic-regression models (one to correct x-coordinate of TC center, '
    'one to correct y-coord of TC center) for every 2-D bin.  If this flag is '
    '0, there will be one x-coord-correcting isotonic-regression model for '
    'every bin along the nadir-relative x-coord, plus one y-coord-correcting '
    'IR model for every bin along the nadir-relative y-coord.'
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
    '--' + USE_2D_BINS_ARG_NAME, type=int, required=True,
    help=USE_2D_BINS_HELP_STRING
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
         use_2d_bins, min_training_sample_size, output_dir_name):
    """Trains nadir-dependent isotonic-regression models for scalar target vars.

    This is effectively the main method.

    :param prediction_file_pattern: See documentation at top of file.
    :param ebtrk_file_name: Same.
    :param nadir_relative_x_cutoffs_metres: Same.
    :param nadir_relative_y_cutoffs_metres: Same.
    :param use_2d_bins: Same.
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
            pt.attrs[prediction_utils.ISOTONIC_MODEL_FILE_KEY] is not None
    ):
        raise ValueError(
            'Predictions used for training isotonic regression must be made'
            ' with base model only (i.e., must not already include isotonic'
            ' regression).'
        )

    if not use_2d_bins:
        x_coord_model_objects, y_coord_model_objects = (
            iso_reg_1d_bins.train_models(
                prediction_table_xarray=prediction_table_xarray,
                nadir_relative_x_cutoffs_metres=nadir_relative_x_cutoffs_metres,
                nadir_relative_y_cutoffs_metres=nadir_relative_y_cutoffs_metres,
                ebtrk_file_name=ebtrk_file_name,
                min_training_sample_size=min_training_sample_size
            )
        )

        output_file_name = iso_reg_1d_bins.find_file(
            model_dir_name=output_dir_name, raise_error_if_missing=False
        )
        print('Writing isotonic-regression models to: "{0:s}"...'.format(
            output_file_name
        ))
        iso_reg_1d_bins.write_file(
            dill_file_name=output_file_name,
            x_coord_model_objects=x_coord_model_objects,
            y_coord_model_objects=y_coord_model_objects,
            nadir_relative_x_cutoffs_metres=nadir_relative_x_cutoffs_metres,
            nadir_relative_y_cutoffs_metres=nadir_relative_y_cutoffs_metres
        )

        return

    x_coord_model_matrix, y_coord_model_matrix = (
        iso_reg_2d_bins.train_models(
            prediction_table_xarray=prediction_table_xarray,
            nadir_relative_x_cutoffs_metres=nadir_relative_x_cutoffs_metres,
            nadir_relative_y_cutoffs_metres=nadir_relative_y_cutoffs_metres,
            ebtrk_file_name=ebtrk_file_name,
            min_training_sample_size=min_training_sample_size
        )
    )

    output_file_name = iso_reg_2d_bins.find_file(
        model_dir_name=output_dir_name, raise_error_if_missing=False
    )
    print('Writing isotonic-regression models to: "{0:s}"...'.format(
        output_file_name
    ))
    iso_reg_2d_bins.write_file(
        dill_file_name=output_file_name,
        x_coord_model_matrix=x_coord_model_matrix,
        y_coord_model_matrix=y_coord_model_matrix,
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
        use_2d_bins=bool(getattr(INPUT_ARG_OBJECT, USE_2D_BINS_ARG_NAME)),
        min_training_sample_size=getattr(
            INPUT_ARG_OBJECT, MIN_SAMPLE_SIZE_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
