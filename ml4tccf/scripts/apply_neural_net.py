"""Applies trained neural net -- inference time!"""

import os
import argparse
import numpy
from gewittergefahr.gg_utils import error_checking
from ml4tccf.io import prediction_io
from ml4tccf.io import scalar_prediction_io
from ml4tccf.io import gridded_prediction_io
from ml4tccf.machine_learning import neural_net_utils as nn_utils
from ml4tccf.machine_learning import \
    neural_net_training_cira_ir as nn_training_cira_ir
from ml4tccf.machine_learning import \
    neural_net_training_simple as nn_training_simple
from ml4tccf.machine_learning import \
    neural_net_training_fancy as nn_training_fancy

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

LARGE_INTEGER = int(1e10)
NUM_EXAMPLES_PER_BATCH = 10

MODEL_FILE_ARG_NAME = 'input_model_file_name'
SATELLITE_DIR_ARG_NAME = 'input_satellite_dir_name'
A_DECK_FILE_ARG_NAME = 'input_a_deck_file_name'
CYCLONE_ID_ARG_NAME = 'cyclone_id_string'
VALID_DATE_ARG_NAME = 'valid_date_string'
NUM_TRANSLATIONS_ARG_NAME = 'data_aug_num_translations'
RANDOM_SEED_ARG_NAME = 'random_seed'
REMOVE_TROPICAL_SYSTEMS_ARG_NAME = 'remove_tropical_systems'
SYNOPTIC_TIMES_ONLY_ARG_NAME = 'synoptic_times_only'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model (will be read by `nn_utils.read_model`).'
)
SATELLITE_DIR_HELP_STRING = (
    'Name of directory with satellite (predictor) data.  Files therein will be '
    'found by `satellite_io.find_file` and read by `satellite_io.read_file`.'
)
A_DECK_FILE_HELP_STRING = (
    'Name of file with ATCF (A-deck) scalars.  This file will be read by '
    '`a_deck_io.read_file`.'
)
CYCLONE_ID_HELP_STRING = (
    'Will apply neural net to data from this cyclone.  Cyclone ID must be in '
    'format "yyyyBBnn".'
)
VALID_DATE_HELP_STRING = (
    'Valid date (format "yyyymmdd").  Will apply neural net to all valid times '
    'on this date.  If you want to apply the NN to all valid times for the '
    'cyclone, leave this argument alone.'
)
NUM_TRANSLATIONS_HELP_STRING = (
    'Number of translations for each cyclone snapshot (one snapshot = one '
    'cyclone at one target time).  Total number of data samples will be '
    'num_snapshots * {0:s}.'
).format(
    NUM_TRANSLATIONS_ARG_NAME
)
RANDOM_SEED_HELP_STRING = (
    'Random seed.  This will determine, among other things, the exact '
    'translations used in data augmentation.  For example, suppose you want to '
    'ensure that for a given cyclone object, two models see the same random '
    'translations.  Then you would set this seed to be equal for the two '
    'models.'
)
REMOVE_TROPICAL_SYSTEMS_HELP_STRING = (
    'Boolean flag.  If 1, the NN will be applied only to non-tropical systems, '
    'regardless of how it was trained.'
)
SYNOPTIC_TIMES_ONLY_HELP_STRING = (
    'Boolean flag.  If 1, only synoptic times can be target times.  If 0, any '
    '10-min time step can be a target time.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Results will be written by '
    '`scalar_prediction_io.write_file` or `gridded_prediction_io.write_file`, '
    'to a location in this directory determined by `prediction_io.find_file`.  '
    'If you would rather specify the file path directly, leave this argument '
    'alone.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Results will be written here by '
    '`scalar_prediction_io.write_file` or `gridded_prediction_io.write_file`.  '
    'If you would rather just specify the output directory and not the total '
    'path, leave this argument alone.'
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
    '--' + CYCLONE_ID_ARG_NAME, type=str, required=True,
    help=CYCLONE_ID_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + VALID_DATE_ARG_NAME, type=str, required=False, default='',
    help=VALID_DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TRANSLATIONS_ARG_NAME, type=int, required=True,
    help=NUM_TRANSLATIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + RANDOM_SEED_ARG_NAME, type=int, required=False, default=-1,
    help=RANDOM_SEED_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + REMOVE_TROPICAL_SYSTEMS_ARG_NAME, type=int, required=False,
    default=0, help=REMOVE_TROPICAL_SYSTEMS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SYNOPTIC_TIMES_ONLY_ARG_NAME, type=int, required=False,
    default=1, help=SYNOPTIC_TIMES_ONLY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=False, default='',
    help=OUTPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=False, default='',
    help=OUTPUT_FILE_HELP_STRING
)


def _run(model_file_name, satellite_dir_name, a_deck_file_name,
         cyclone_id_string, valid_date_string, data_aug_num_translations,
         random_seed, remove_tropical_systems, synoptic_times_only,
         output_dir_name, output_file_name):
    """Applies trained neural net -- inference time!

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param satellite_dir_name: Same.
    :param a_deck_file_name: Same.
    :param cyclone_id_string: Same.
    :param valid_date_string: Same.
    :param data_aug_num_translations: Same.
    :param random_seed: Same.
    :param remove_tropical_systems: Same.
    :param synoptic_times_only: Same.
    :param output_dir_name: Same.
    :param output_file_name: Same.
    """

    if random_seed != -1:
        numpy.random.seed(random_seed)
    if valid_date_string == '':
        valid_date_string = None

    if output_dir_name == '':
        output_dir_name = None
    else:
        output_file_name = None

    if output_file_name == '':
        output_file_name = None
    else:
        output_dir_name = None

    assert not (output_dir_name is None and output_file_name is None)

    error_checking.assert_is_geq(data_aug_num_translations, 1)

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = nn_utils.read_model(model_file_name)
    model_metafile_name = nn_utils.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = nn_utils.read_metafile(model_metafile_name)

    validation_option_dict = (
        model_metadata_dict[nn_utils.VALIDATION_OPTIONS_KEY]
    )
    validation_option_dict[nn_utils.SATELLITE_DIRECTORY_KEY] = (
        satellite_dir_name
    )
    validation_option_dict[nn_utils.A_DECK_FILE_KEY] = a_deck_file_name
    validation_option_dict[nn_utils.DATA_AUG_NUM_TRANS_KEY] = (
        data_aug_num_translations
    )

    if remove_tropical_systems:
        validation_option_dict[nn_utils.REMOVE_TROPICAL_KEY] = True
        validation_option_dict[nn_utils.REMOVE_NONTROPICAL_KEY] = False

    validation_option_dict[nn_utils.SYNOPTIC_TIMES_ONLY_KEY] = (
        synoptic_times_only
    )

    data_type_string = model_metadata_dict[nn_utils.DATA_TYPE_KEY]

    if data_type_string == nn_utils.CIRA_IR_DATA_TYPE_STRING:
        assert valid_date_string is None

        data_dict = nn_training_cira_ir.create_data(
            option_dict=validation_option_dict,
            cyclone_id_string=cyclone_id_string,
            num_target_times=LARGE_INTEGER
        )
    elif data_type_string == nn_utils.RG_SIMPLE_DATA_TYPE_STRING:
        data_dict = nn_training_simple.create_data(
            option_dict=validation_option_dict,
            cyclone_id_string=cyclone_id_string,
            valid_date_string=valid_date_string,
            num_target_times=LARGE_INTEGER
        )
    else:
        assert valid_date_string is None

        data_dict = nn_training_fancy.create_data(
            option_dict=validation_option_dict,
            cyclone_id_string=cyclone_id_string,
            num_target_times=LARGE_INTEGER
        )

    if data_dict is None:
        return

    predictor_matrices = data_dict[nn_utils.PREDICTOR_MATRICES_KEY]
    target_matrix = data_dict[nn_utils.TARGET_MATRIX_KEY]
    grid_spacings_km = data_dict[nn_utils.GRID_SPACINGS_KEY]
    cyclone_center_latitudes_deg_n = data_dict[nn_utils.CENTER_LATITUDES_KEY]
    target_times_unix_sec = data_dict[nn_utils.TARGET_TIMES_KEY]

    if len(target_matrix.shape) == 4:
        target_matrix = target_matrix[..., 0]

    print(SEPARATOR_STRING)

    prediction_matrix = nn_utils.apply_model(
        model_object=model_object, predictor_matrices=predictor_matrices,
        num_examples_per_batch=NUM_EXAMPLES_PER_BATCH, verbose=True
    )

    print(SEPARATOR_STRING)
    del predictor_matrices

    if output_file_name is None:
        output_file_name = prediction_io.find_file(
            directory_name=output_dir_name,
            cyclone_id_string=cyclone_id_string,
            raise_error_if_missing=False
        )

    print('Writing results to: "{0:s}"...'.format(output_file_name))

    if validation_option_dict[nn_utils.SEMANTIC_SEG_FLAG_KEY]:
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
            isotonic_model_file_name=None
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        satellite_dir_name=getattr(INPUT_ARG_OBJECT, SATELLITE_DIR_ARG_NAME),
        a_deck_file_name=getattr(INPUT_ARG_OBJECT, A_DECK_FILE_ARG_NAME),
        cyclone_id_string=getattr(INPUT_ARG_OBJECT, CYCLONE_ID_ARG_NAME),
        valid_date_string=getattr(INPUT_ARG_OBJECT, VALID_DATE_ARG_NAME),
        data_aug_num_translations=getattr(
            INPUT_ARG_OBJECT, NUM_TRANSLATIONS_ARG_NAME
        ),
        random_seed=getattr(INPUT_ARG_OBJECT, RANDOM_SEED_ARG_NAME),
        remove_tropical_systems=bool(
            getattr(INPUT_ARG_OBJECT, REMOVE_TROPICAL_SYSTEMS_ARG_NAME)
        ),
        synoptic_times_only=bool(
            getattr(INPUT_ARG_OBJECT, SYNOPTIC_TIMES_ONLY_ARG_NAME)
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )