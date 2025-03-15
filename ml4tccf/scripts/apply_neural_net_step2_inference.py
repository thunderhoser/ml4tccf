"""Step 2 for two-step NN inference: feed example files to NNs."""

import os
import argparse
import warnings
import numpy
import xarray
from ml4tccf.io import prediction_io
from ml4tccf.io import scalar_prediction_io
from ml4tccf.io import gridded_prediction_io
from ml4tccf.utils import satellite_utils
from ml4tccf.machine_learning import neural_net_utils as nn_utils
from ml4tccf.scripts import \
    apply_neural_net_step1_preprocessing as preprocessing

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

LARGE_INTEGER = int(1e10)
NUM_EXAMPLES_PER_BATCH = 10

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
CYCLONE_ID_ARG_NAME = 'cyclone_id_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model (will be read by `nn_utils.read_model`).'
)
EXAMPLE_DIR_HELP_STRING = (
    'Path to directory with pre-processed examples.  Files therein will be '
    'found by `apply_neural_net_step1_preprocessing.find_file` and read by '
    '`apply_neural_net_step1_preprocessing.read_file`.'
)
CYCLONE_ID_HELP_STRING = (
    'Will apply neural net to data from this cyclone.  Cyclone ID must be in '
    'format "yyyyBBnn".'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Results will be written by '
    '`scalar_prediction_io.write_file` or `gridded_prediction_io.write_file`, '
    'to a location in this directory determined by `prediction_io.find_file`.  '
    'If you would rather specify the file path directly, leave this argument '
    'alone.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CYCLONE_ID_ARG_NAME, type=str, required=True,
    help=CYCLONE_ID_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(model_file_name, example_dir_name, cyclone_id_string, output_dir_name):
    """Step 2 for two-step NN inference: feed example files to NNs.

    This is effectively the main method.

    :param model_file_name: See documentation at top of this script.
    :param example_dir_name: Same.
    :param cyclone_id_string: Same.
    :param output_dir_name: Same.
    """

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = nn_utils.read_model(model_file_name)
    model_metafile_name = nn_utils.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = nn_utils.read_metafile(model_metafile_name)

    example_file_name = preprocessing.find_file(
        directory_name=example_dir_name,
        cyclone_id_string=cyclone_id_string,
        raise_error_if_missing=False
    )

    if not os.path.isfile(example_file_name):
        return

    print('Reading pre-processed examples from: "{0:s}"...'.format(
        example_file_name
    ))
    (
        predictor_matrices,
        target_matrix,
        grid_spacings_km,
        cyclone_center_latitudes_deg_n,
        target_times_unix_sec
    ) = preprocessing.read_file(example_file_name)

    num_lag_times_in_file = predictor_matrices[0].shape[-2]
    vod = model_metadata_dict[nn_utils.VALIDATION_OPTIONS_KEY]
    num_lag_times_in_model = len(vod[nn_utils.LAG_TIMES_KEY])
    assert not num_lag_times_in_model > num_lag_times_in_file

    if num_lag_times_in_model < num_lag_times_in_file:
        # TODO(thunderhoser): Subsetting by lag time is HACKY.  I assume that if
        # the file contains N lag times and the model takes K lag times, with
        # K < N, then the last K lag times in the file are the desired ones.
        warning_string = (
            'Subsetting by lag time is HACKY.  I assume that if the file contains '
            'N lag times and the model takes K lag times, with K < N, then the '
            'last K lag times in the file are the desired ones.'
        )
        warnings.warn(warning_string)

        print((
            'Subsetting satellite predictors from {0:d} to {1:d} lag times...'
        ).format(
            num_lag_times_in_file, num_lag_times_in_model
        ))

        predictor_matrices[0] = (
            predictor_matrices[0][..., -num_lag_times_in_model:, :]
        )

    num_grid_rows_in_file = predictor_matrices[0].shape[1]
    num_grid_rows_in_model = vod[nn_utils.NUM_GRID_ROWS_KEY]
    assert not num_grid_rows_in_model > num_grid_rows_in_file

    if num_grid_rows_in_model < num_grid_rows_in_file:
        print((
            'Subsetting satellite predictors from {0:d} to {1:d} pixels per '
            'side...'
        ).format(
            num_grid_rows_in_file, num_grid_rows_in_model
        ))

        num_examples = predictor_matrices[0].shape[0]
        num_wavelengths = predictor_matrices[0].shape[-1]

        coord_dict = {
            satellite_utils.TIME_DIM: numpy.linspace(
                0, num_examples - 1, num_examples, dtype=int
            ),
            satellite_utils.LOW_RES_ROW_DIM: numpy.linspace(
                0, num_grid_rows_in_file - 1, num_grid_rows_in_file, dtype=int
            ),
            satellite_utils.LOW_RES_COLUMN_DIM: numpy.linspace(
                0, num_grid_rows_in_file - 1, num_grid_rows_in_file, dtype=int
            ),
            'lag_time': numpy.linspace(
                0, num_lag_times_in_model - 1, num_lag_times_in_model, dtype=int
            ),
            satellite_utils.LOW_RES_WAVELENGTH_DIM: 1e-6 * numpy.linspace(
                1, num_wavelengths, num_wavelengths, dtype=float
            )
        }

        these_dims = (
            satellite_utils.TIME_DIM,
            satellite_utils.LOW_RES_ROW_DIM,
            satellite_utils.LOW_RES_COLUMN_DIM,
            'lag_time',
            satellite_utils.LOW_RES_WAVELENGTH_DIM
        )

        main_data_dict = {
            satellite_utils.BRIGHTNESS_TEMPERATURE_KEY: (
                these_dims, predictor_matrices[0]
            )
        }

        satellite_table_xarray = xarray.Dataset(
            data_vars=main_data_dict, coords=coord_dict
        )
        satellite_table_xarray = satellite_utils.subset_grid(
            satellite_table_xarray=satellite_table_xarray,
            num_rows_to_keep=num_grid_rows_in_model,
            num_columns_to_keep=num_grid_rows_in_model,
            for_high_res=False
        )

        predictor_matrices[0] = satellite_table_xarray[
            satellite_utils.BRIGHTNESS_TEMPERATURE_KEY
        ].values

    print(SEPARATOR_STRING)
    prediction_matrix = nn_utils.apply_model(
        model_object=model_object, predictor_matrices=predictor_matrices,
        num_examples_per_batch=NUM_EXAMPLES_PER_BATCH, verbose=True
    )
    print(SEPARATOR_STRING)
    del predictor_matrices

    output_file_name = prediction_io.find_file(
        directory_name=output_dir_name,
        cyclone_id_string=cyclone_id_string,
        raise_error_if_missing=False
    )

    print('Writing results to: "{0:s}"...'.format(output_file_name))

    if vod[nn_utils.SEMANTIC_SEG_FLAG_KEY]:
        gridded_prediction_io.write_file(
            netcdf_file_name=output_file_name,
            target_matrix=target_matrix,
            prediction_matrix=prediction_matrix,
            grid_spacings_km=grid_spacings_km,
            cyclone_center_latitudes_deg_n=cyclone_center_latitudes_deg_n,
            cyclone_id_string=cyclone_id_string,
            target_times_unix_sec=target_times_unix_sec,
            model_file_name=model_file_name
        )
    else:
        scalar_prediction_io.write_file(
            netcdf_file_name=output_file_name,
            target_matrix=target_matrix,
            prediction_matrix=prediction_matrix,
            cyclone_id_string=cyclone_id_string,
            target_times_unix_sec=target_times_unix_sec,
            model_file_name=model_file_name,
            isotonic_model_file_name=None,
            uncertainty_calib_model_file_name=None
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        cyclone_id_string=getattr(INPUT_ARG_OBJECT, CYCLONE_ID_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
