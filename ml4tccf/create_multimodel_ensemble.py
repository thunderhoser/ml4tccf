"""Creates multi-model ensemble."""

import os
import sys
import copy
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import prediction_io
import scalar_prediction_io
import scalar_prediction_utils

TOLERANCE = 1e-6
MODEL_FILE_KEY = scalar_prediction_utils.MODEL_FILE_KEY
ISOTONIC_MODEL_FILE_KEY = scalar_prediction_utils.ISOTONIC_MODEL_FILE_KEY

INPUT_FILES_ARG_NAME = 'input_prediction_file_names'
MAX_ENSEMBLE_SIZE_ARG_NAME = 'max_total_ensemble_size'
OUTPUT_FILE_ARG_NAME = 'output_prediction_file_name'

INPUT_FILES_HELP_STRING = (
    'List of paths to input files.  Each file will be read by '
    '`prediction_io.read_file`, and predictions from all these files will be '
    'concatenated along the final (ensemble-member) axis.'
)
MAX_ENSEMBLE_SIZE_HELP_STRING = (
    'Maximum size of total ensemble, after concatenating predictions from all '
    'input files along the final axis.  In other words, the size of the final '
    'axis may not exceed {0:s}.  If it does, {0:s} predictions will be '
    'randomly selected.'
).format(MAX_ENSEMBLE_SIZE_ARG_NAME)

OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  All predictions will be written here by '
    '`scalar_prediction_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_ENSEMBLE_SIZE_ARG_NAME, type=int, required=True,
    help=MAX_ENSEMBLE_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_file_names, max_ensemble_size, output_file_name):
    """Creates multi-model ensemble.

    This is effectively the main method.

    :param input_file_names: See documentation at top of file.
    :param max_ensemble_size: Same.
    :param output_file_name: Same.
    """

    error_checking.assert_is_geq(max_ensemble_size, 2)

    num_models = len(input_file_names)
    prediction_tables_xarray = [None] * num_models
    model_file_names = [''] * num_models
    isotonic_model_file_names = [''] * num_models

    for i in range(num_models):
        print('Reading data from: "{0:s}"...'.format(input_file_names[i]))
        prediction_tables_xarray[i] = prediction_io.read_file(
            input_file_names[i]
        )

        model_file_names[i] = copy.deepcopy(
            prediction_tables_xarray[i].attrs[MODEL_FILE_KEY]
        )
        prediction_tables_xarray[i].attrs[MODEL_FILE_KEY] = model_file_names[0]

        isotonic_model_file_names[i] = copy.deepcopy(
            prediction_tables_xarray[i].attrs[ISOTONIC_MODEL_FILE_KEY]
        )
        prediction_tables_xarray[i].attrs[ISOTONIC_MODEL_FILE_KEY] = (
            isotonic_model_file_names[0]
        )

    prediction_table_xarray = (
        scalar_prediction_utils.concat_over_ensemble_members(
            prediction_tables_xarray=prediction_tables_xarray,
            use_only_common_examples=True
        )
    )

    pt = prediction_table_xarray
    ensemble_size = (
        pt[scalar_prediction_utils.PREDICTED_ROW_OFFSET_KEY].values.shape[-1]
    )

    if ensemble_size > max_ensemble_size:
        member_indices = numpy.linspace(
            0, ensemble_size - 1, num=ensemble_size, dtype=int
        )
        desired_indices = numpy.random.choice(
            member_indices, size=max_ensemble_size, replace=False
        )
        pt = pt.isel(indexers={
            scalar_prediction_utils.ENSEMBLE_MEMBER_DIM_KEY: desired_indices
        })

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

    target_times_unix_sec = pt[scalar_prediction_utils.TARGET_TIME_KEY].values
    dummy_model_file_name = ' '.join(model_file_names)

    if all([m is None for m in isotonic_model_file_names]):
        dummy_iso_model_file_name = None
    else:
        dummy_iso_model_file_name = ' '.join(isotonic_model_file_names)

    cyclone_id_strings = pt[scalar_prediction_utils.CYCLONE_ID_KEY].values
    assert len(numpy.unique(cyclone_id_strings)) == 1
    cyclone_id_string = cyclone_id_strings[0].decode('utf-8')

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    scalar_prediction_io.write_file(
        netcdf_file_name=output_file_name,
        target_matrix=target_matrix,
        prediction_matrix=prediction_matrix,
        cyclone_id_string=cyclone_id_string,
        target_times_unix_sec=target_times_unix_sec,
        model_file_name=dummy_model_file_name,
        isotonic_model_file_name=dummy_iso_model_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        max_ensemble_size=getattr(INPUT_ARG_OBJECT, MAX_ENSEMBLE_SIZE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
