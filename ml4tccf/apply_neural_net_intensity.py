"""Applies trained neural net for intensity estimation -- inference time!"""

import os
import sys
import argparse

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import prediction_io
import neural_net_utils as nn_utils
import neural_net_training_intensity as nn_training

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

LARGE_INTEGER = int(1e10)
NUM_EXAMPLES_PER_BATCH = 32

MODEL_FILE_ARG_NAME = 'input_model_file_name'
SATELLITE_DIR_ARG_NAME = 'input_satellite_dir_name'
EBTRK_FILE_ARG_NAME = 'input_extended_best_track_file_name'
CENTER_FIXING_MODEL_FILE_ARG_NAME = 'input_center_fixing_model_file_name'
CYCLONE_ID_ARG_NAME = 'cyclone_id_string'
NUM_TRANSLATIONS_ARG_NAME = 'data_aug_num_translations'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model (will be read by `nn_utils.read_model`).'
)
SATELLITE_DIR_HELP_STRING = (
    'Name of directory with satellite (predictor) data.  Files therein will be '
    'found by `satellite_io.find_file` and read by `satellite_io.read_file`.'
)
EBTRK_FILE_HELP_STRING = (
    'Path to EBTRK (extended best-track) file.  True intensities will be read '
    'from here by `extended_best_track_io.read_file`.'
)
CENTER_FIXING_MODEL_FILE_HELP_STRING = (
    'Path to auxiliary neural net that does center-fixing (will be read by '
    '`neural_net_utils.read_model`).  If the intensity-estimation NN does not '
    'use an auxiliary center-fixing NN, leave this argument alone.'
)
CYCLONE_ID_HELP_STRING = (
    'Will apply neural net to data from this cyclone.  Cyclone ID must be in '
    'format "yyyyBBnn".'
)
NUM_TRANSLATIONS_HELP_STRING = (
    'Number of translations for each TC object.  Total number of data samples '
    'will be num_objects * {0:s}.  If the intensity-estimation NN was not '
    'trained with data augmentation, make this 0.'
).format(NUM_TRANSLATIONS_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Results will be written by '
    '`scalar_prediction_io.write_file` or `gridded_prediction_io.write_file`, '
    'to a location in this directory determined by `prediction_io.find_file`.'
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
    '--' + EBTRK_FILE_ARG_NAME, type=str, required=True,
    help=EBTRK_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CENTER_FIXING_MODEL_FILE_ARG_NAME, type=str, required=False,
    default='', help=CENTER_FIXING_MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CYCLONE_ID_ARG_NAME, type=str, required=True,
    help=CYCLONE_ID_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TRANSLATIONS_ARG_NAME, type=int, required=True,
    help=NUM_TRANSLATIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(model_file_name, satellite_dir_name, ebtrk_file_name,
         center_fixing_model_file_name, cyclone_id_string,
         data_aug_num_translations, output_dir_name):
    """Applies trained neural net for intensity estimation -- inference time!

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param satellite_dir_name: Same.
    :param ebtrk_file_name: Same.
    :param center_fixing_model_file_name: Same.
    :param cyclone_id_string: Same.
    :param data_aug_num_translations: Same.
    :param output_dir_name: Same.
    """

    error_checking.assert_is_geq(data_aug_num_translations, 0)
    if data_aug_num_translations == 0:
        center_fixing_model_file_name = ''

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
    validation_option_dict[nn_utils.DATA_AUG_NUM_TRANS_KEY] = (
        data_aug_num_translations
    )
    validation_option_dict[nn_utils.SYNOPTIC_TIMES_ONLY_KEY] = True

    if center_fixing_model_file_name == '':
        center_fixing_model_object = None
    else:
        print('Reading auxiliary center-fixing NN from: "{0:s}"...'.format(
            center_fixing_model_file_name
        ))
        center_fixing_model_object = nn_utils.read_model(
            center_fixing_model_file_name
        )

    if data_aug_num_translations == 0:
        data_dict = nn_training.create_data_no_trans(
            option_dict=validation_option_dict,
            ebtrk_file_name=ebtrk_file_name,
            cyclone_id_string=cyclone_id_string,
            num_target_times=LARGE_INTEGER
        )
    else:
        data_dict = nn_training.create_data(
            option_dict=validation_option_dict,
            ebtrk_file_name=ebtrk_file_name,
            center_fixing_model_object=center_fixing_model_object,
            cyclone_id_string=cyclone_id_string,
            num_target_times=LARGE_INTEGER
        )

    if data_dict is None:
        return

    print(SEPARATOR_STRING)

    predictor_matrices = data_dict[nn_training.PREDICTOR_MATRICES_KEY]
    target_intensities_m_s01 = data_dict[nn_training.TARGET_INTENSITIES_KEY]
    target_times_unix_sec = data_dict[nn_training.TARGET_TIMES_KEY]

    predicted_intensities_m_s01 = nn_training.apply_model(
        model_object=model_object, predictor_matrices=predictor_matrices,
        num_examples_per_batch=NUM_EXAMPLES_PER_BATCH, verbose=True
    )
    print(SEPARATOR_STRING)

    output_file_name = prediction_io.find_file(
        directory_name=output_dir_name, cyclone_id_string=cyclone_id_string,
        raise_error_if_missing=False
    )

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    nn_training.write_prediction_file(
        netcdf_file_name=output_file_name,
        target_intensities_m_s01=target_intensities_m_s01,
        predicted_intensities_m_s01=predicted_intensities_m_s01,
        cyclone_id_string=cyclone_id_string,
        target_times_unix_sec=target_times_unix_sec,
        model_file_name=model_file_name
    )

if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        satellite_dir_name=getattr(INPUT_ARG_OBJECT, SATELLITE_DIR_ARG_NAME),
        ebtrk_file_name=getattr(INPUT_ARG_OBJECT, EBTRK_FILE_ARG_NAME),
        center_fixing_model_file_name=getattr(
            INPUT_ARG_OBJECT, CENTER_FIXING_MODEL_FILE_ARG_NAME
        ),
        cyclone_id_string=getattr(INPUT_ARG_OBJECT, CYCLONE_ID_ARG_NAME),
        data_aug_num_translations=getattr(
            INPUT_ARG_OBJECT, NUM_TRANSLATIONS_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
