"""Applies (does inference with) NN trained to predict TC-structure params."""

import os
import argparse
from ml4tccf.io import structure_prediction_io
from ml4tccf.machine_learning import neural_net_utils as nn_utils
from ml4tccf.machine_learning import \
    neural_net_training_structure as nn_training

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

LARGE_INTEGER = int(1e10)
NUM_EXAMPLES_PER_BATCH = 10

MODEL_FILE_ARG_NAME = 'input_model_file_name'
SATELLITE_DIR_ARG_NAME = 'input_satellite_dir_name'
A_DECK_FILE_ARG_NAME = 'input_a_deck_file_name'
TARGET_FILE_ARG_NAME = 'input_target_file_name'
CYCLONE_ID_ARG_NAME = 'cyclone_id_string'
SYNOPTIC_TIMES_ONLY_ARG_NAME = 'synoptic_times_only'
DISABLE_GPUS_ARG_NAME = 'disable_gpus'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model (will be read by `nn_utils.read_model`).'
)
SATELLITE_DIR_HELP_STRING = (
    'Path to directory with satellite (image) predictors.  Files therein will '
    'be found by `satellite_io.find_file` and read by `satellite_io.read_file`.'
)
A_DECK_FILE_HELP_STRING = (
    'Path to file with A-deck (scalar) predictors.  This file will be read by '
    '`a_deck_io.read_file`.'
)
TARGET_FILE_HELP_STRING = (
    'Path to file with target fields.  This file will be read by '
    '`extended_best_track_io.read_file`.'
)
CYCLONE_ID_HELP_STRING = (
    'Will apply neural net to data from this cyclone.  Cyclone ID must be in '
    'format "yyyyBBnn".'
)
SYNOPTIC_TIMES_ONLY_HELP_STRING = (
    'Boolean flag.  If 1, only synoptic times can be target times.  If 0, any '
    '10-min time step can be a target time.'
)
DISABLE_GPUS_HELP_STRING = (
    'Boolean flag.  If 1, will disable GPUs and use only CPUs.  This argument '
    'is HIGHLY RECOMMENDED in any environment, besides Hera (or some machine '
    'where every job runs on a different node), where this script could be '
    'running multiple times at once.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Results will be written here by '
    '`structure_prediction_io.write_file`, to an exact location determined by '
    '`structure_prediction_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SATELLITE_DIR_ARG_NAME, type=str, required=True,
    help=SATELLITE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + A_DECK_FILE_ARG_NAME, type=str, required=True,
    help=A_DECK_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_FILE_ARG_NAME, type=str, required=True,
    help=TARGET_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CYCLONE_ID_ARG_NAME, type=str, required=True,
    help=CYCLONE_ID_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SYNOPTIC_TIMES_ONLY_ARG_NAME, type=int, required=False,
    default=1, help=SYNOPTIC_TIMES_ONLY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DISABLE_GPUS_ARG_NAME, type=int, required=False,
    default=0, help=DISABLE_GPUS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(model_file_name, satellite_dir_name, a_deck_file_name,
         target_file_name, cyclone_id_string, synoptic_times_only,
         disable_gpus, output_dir_name):
    """Applies (does inference with) NN trained to predict TC-structure params.

    :param model_file_name: See documentation at top of this script.
    :param satellite_dir_name: Same.
    :param a_deck_file_name: Same.
    :param target_file_name: Same.
    :param cyclone_id_string: Same.
    :param synoptic_times_only: Same.
    :param disable_gpus: Same.
    :param output_dir_name: Same.
    """

    if disable_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = nn_utils.read_model(model_file_name)
    model_metafile_name = nn_utils.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = nn_utils.read_metafile(model_metafile_name)
    vod = model_metadata_dict[nn_utils.VALIDATION_OPTIONS_KEY]

    vod[nn_training.SATELLITE_DIRECTORY_KEY] = satellite_dir_name
    vod[nn_training.A_DECK_FILE_KEY] = a_deck_file_name
    vod[nn_training.TARGET_FILE_KEY] = target_file_name
    vod[nn_training.SYNOPTIC_TIMES_ONLY_KEY] = synoptic_times_only
    validation_option_dict = vod

    data_dict = nn_training.create_data(
        option_dict=validation_option_dict,
        cyclone_id_string=cyclone_id_string,
        num_target_times=LARGE_INTEGER
    )

    if data_dict is None:
        return

    predictor_matrices = data_dict[nn_training.PREDICTOR_MATRICES_KEY]
    target_matrix = data_dict[nn_training.TARGET_MATRIX_KEY]
    target_times_unix_sec = data_dict[nn_training.TARGET_TIMES_KEY]

    print(SEPARATOR_STRING)
    prediction_matrix = nn_training.apply_model(
        model_object=model_object,
        predictor_matrices=predictor_matrices,
        num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
        verbose=True
    )
    print(SEPARATOR_STRING)

    if validation_option_dict[nn_training.DO_RESIDUAL_PREDICTION_KEY]:
        resid_baseline_prediction_matrix = predictor_matrices[-1]
    else:
        resid_baseline_prediction_matrix = None

    del predictor_matrices

    output_file_name = structure_prediction_io.find_file(
        directory_name=output_dir_name,
        cyclone_id_string=cyclone_id_string,
        raise_error_if_missing=False
    )

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    structure_prediction_io.write_file(
        netcdf_file_name=output_file_name,
        target_matrix=target_matrix,
        prediction_matrix=prediction_matrix,
        baseline_prediction_matrix=resid_baseline_prediction_matrix,
        cyclone_id_string=cyclone_id_string,
        target_times_unix_sec=target_times_unix_sec,
        model_file_name=model_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        satellite_dir_name=getattr(INPUT_ARG_OBJECT, SATELLITE_DIR_ARG_NAME),
        a_deck_file_name=getattr(INPUT_ARG_OBJECT, A_DECK_FILE_ARG_NAME),
        target_file_name=getattr(INPUT_ARG_OBJECT, TARGET_FILE_ARG_NAME),
        cyclone_id_string=getattr(INPUT_ARG_OBJECT, CYCLONE_ID_ARG_NAME),
        synoptic_times_only=bool(
            getattr(INPUT_ARG_OBJECT, SYNOPTIC_TIMES_ONLY_ARG_NAME)
        ),
        disable_gpus=bool(
            getattr(INPUT_ARG_OBJECT, DISABLE_GPUS_ARG_NAME)
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
