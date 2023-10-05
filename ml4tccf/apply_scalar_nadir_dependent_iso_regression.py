"""Applies nadir-dependent isotonic-regression models to scalar target vars."""

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
import scalar_prediction_utils
import scalar_nadir_dependent_iso_regression as isotonic_regression

INPUT_PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
MODEL_FILE_ARG_NAME = 'input_model_file_name'
EBTRK_FILE_ARG_NAME = 'input_extended_best_track_file_name'
OUTPUT_PREDICTION_FILE_ARG_NAME = 'output_prediction_file_name'

INPUT_PREDICTION_FILE_HELP_STRING = (
    'Path to file containing model predictions before isotonic regression.  '
    'Will be read by `prediction_io.read_file`.'
)
MODEL_FILE_HELP_STRING = (
    'Path to file with set of trained isotonic-regression models.  Will be read'
    ' by `scalar_nadir_dependent_iso_regression.read_file`.'
)
EBTRK_FILE_HELP_STRING = (
    'Path to file with extended best-track data (contains TC-center locations).'
    '  Will be read by `extended_best_track_io.read_file`.'
)
OUTPUT_PREDICTION_FILE_HELP_STRING = (
    'Path to output file, containing model predictions after isotonic '
    'regression.  Will be written by `prediction_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_PREDICTION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EBTRK_FILE_ARG_NAME, type=str, required=True,
    help=EBTRK_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_PREDICTION_FILE_HELP_STRING
)


def _run(input_prediction_file_name, model_file_name, ebtrk_file_name,
         output_prediction_file_name):
    """Applies nadir-dependent isotonic-regression models to scalar target vars.

    This is effectively the main method.

    :param input_prediction_file_name: See documentation at top of file.
    :param model_file_name: Same.
    :param output_prediction_file_name: Same.
    :raises: ValueError: if predictions in `input_prediction_file_name` were
        made with isotonic regression.
    """

    print('Reading original predictions from: "{0:s}"...'.format(
        input_prediction_file_name
    ))
    prediction_table_xarray = prediction_io.read_file(
        input_prediction_file_name
    )

    pt = prediction_table_xarray
    if (
            pt.attrs[scalar_prediction_utils.ISOTONIC_MODEL_FILE_KEY]
            is not None
    ):
        raise ValueError(
            'Input predictions must be made with base model only (i.e., must '
            'not already include isotonic regression).'
        )

    print('Reading isotonic-regression models from: "{0:s}"...'.format(
        model_file_name
    ))
    (
        x_coord_model_objects, y_coord_model_objects,
        nadir_relative_x_cutoffs_metres, nadir_relative_y_cutoffs_metres
    ) = isotonic_regression.read_file(model_file_name)

    prediction_table_xarray = isotonic_regression.apply_models(
        prediction_table_xarray=prediction_table_xarray,
        x_coord_model_objects=x_coord_model_objects,
        y_coord_model_objects=y_coord_model_objects,
        nadir_relative_x_cutoffs_metres=nadir_relative_x_cutoffs_metres,
        nadir_relative_y_cutoffs_metres=nadir_relative_y_cutoffs_metres,
        ebtrk_file_name=ebtrk_file_name
    )
    pt = prediction_table_xarray

    target_matrix = numpy.transpose(numpy.vstack((
        pt[scalar_prediction_utils.ACTUAL_ROW_OFFSET_KEY].values,
        pt[scalar_prediction_utils.ACTUAL_COLUMN_OFFSET_KEY].values,
        pt[scalar_prediction_utils.GRID_SPACING_KEY].values,
        pt[scalar_prediction_utils.ACTUAL_CENTER_LATITUDE_KEY].values
    )))
    prediction_matrix = numpy.stack((
        pt[scalar_prediction_utils.PREDICTED_ROW_OFFSET_KEY].values,
        pt[scalar_prediction_utils.PREDICTED_COLUMN_OFFSET_KEY].values
    ), axis=-2)

    cyclone_id_strings = pt[scalar_prediction_utils.CYCLONE_ID_KEY].values
    assert len(numpy.unique(cyclone_id_strings)) == 1
    cyclone_id_string = cyclone_id_strings[0].decode('utf-8')

    print('Writing new predictions to: "{0:s}"...'.format(
        output_prediction_file_name
    ))
    scalar_prediction_io.write_file(
        netcdf_file_name=output_prediction_file_name,
        target_matrix=target_matrix,
        prediction_matrix=prediction_matrix,
        cyclone_id_string=cyclone_id_string,
        target_times_unix_sec=
        pt[scalar_prediction_utils.TARGET_TIME_KEY].values,
        model_file_name=pt.attrs[scalar_prediction_utils.MODEL_FILE_KEY],
        isotonic_model_file_name=model_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, INPUT_PREDICTION_FILE_ARG_NAME
        ),
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        ebtrk_file_name=getattr(INPUT_ARG_OBJECT, EBTRK_FILE_ARG_NAME),
        output_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, OUTPUT_PREDICTION_FILE_ARG_NAME
        )
    )
