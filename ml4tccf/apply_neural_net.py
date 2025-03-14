"""Applies trained neural net -- inference time!"""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import prediction_io
import scalar_prediction_io
import gridded_prediction_io
import neural_net_utils as nn_utils
import neural_net_training_cira_ir as nn_training_cira_ir
import neural_net_training_simple as nn_training_simple
import neural_net_training_fancy as nn_training_fancy

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

LARGE_INTEGER = int(1e10)
NUM_EXAMPLES_PER_BATCH = 10

MODEL_FILE_ARG_NAME = 'input_model_file_name'
SATELLITE_DIR_ARG_NAME = 'input_satellite_dir_name'
A_DECK_FILE_ARG_NAME = 'input_a_deck_file_name'
CYCLONE_ID_ARG_NAME = 'cyclone_id_string'
VALID_DATE_ARG_NAME = 'valid_date_string'
SHORT_TRACK_DIR_ARG_NAME = 'input_short_track_dir_name'
SHORT_TRACK_MAX_LEAD_ARG_NAME = 'short_track_max_lead_minutes'
SHORT_TRACK_DIFF_CENTERS_ARG_NAME = 'short_track_center_each_lag_diffly'
NUM_TRANSLATIONS_ARG_NAME = 'data_aug_num_translations'
MEAN_TRANSLATION_DIST_ARG_NAME = 'data_aug_mean_translation_low_res_px'
STDEV_TRANSLATION_DIST_ARG_NAME = 'data_aug_stdev_translation_low_res_px'
DATA_AUG_UNIFORM_DIST_ARG_NAME = 'data_aug_uniform_dist_flag'
MEAN_TRANS_DIST_WITHIN_ARG_NAME = 'data_aug_within_mean_trans_px'
STDEV_TRANS_DIST_WITHIN_ARG_NAME = 'data_aug_within_stdev_trans_px'
DATA_AUG_WITHIN_UNIFORM_DIST_ARG_NAME = 'data_aug_within_uniform_dist_flag'
RANDOM_SEED_ARG_NAME = 'random_seed'
REMOVE_TROPICAL_SYSTEMS_ARG_NAME = 'remove_tropical_systems'
SYNOPTIC_TIMES_ONLY_ARG_NAME = 'synoptic_times_only'
DISABLE_GPUS_ARG_NAME = 'disable_gpus'
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
SHORT_TRACK_DIR_HELP_STRING = (
    'Path to directory with short-track data (files therein will be found by '
    '`short_track_io.find_file` and read by `short_track_io.read_file`).  To '
    'use the same setting as during training, leave this argument alone.  To '
    'omit short track in the first guess and use random translations only, '
    'make this argument the empty string "".'
)
SHORT_TRACK_MAX_LEAD_HELP_STRING = (
    '[used only if `{0:s}` is specified] Max lead time for short-track '
    'forecasts (any longer-lead forecast will not be used in first guess).  To '
    'use the same setting as during training, leave this argument alone.'
).format(
    SHORT_TRACK_DIR_ARG_NAME
)
SHORT_TRACK_DIFF_CENTERS_HELP_STRING = (
    '[used only if `{0:s}` is specified] Boolean flag.  If 1 (0), for a given '
    'data sample, the first guess will involve a different (the same) lat/long '
    'center for each lag time.  To use the same setting as during training, '
    'leave this argument alone.'
).format(
    SHORT_TRACK_DIR_ARG_NAME
)
NUM_TRANSLATIONS_HELP_STRING = (
    'Number of translations for each TC object.  Total number of data samples '
    'will be num_tc_objects * {0:s}.'
).format(
    NUM_TRANSLATIONS_ARG_NAME
)
MEAN_TRANSLATION_DIST_HELP_STRING = (
    'Mean whole-track translation distance (units of IR pixels, or low-res '
    'pixels).  To use the same setting as during training, leave this argument '
    'alone.'
)
STDEV_TRANSLATION_DIST_HELP_STRING = (
    'Standard deviation of whole-track translation distance (units of IR '
    'pixels, or low-res pixels).  To use the same setting as during training, '
    'leave this argument alone.'
)
DATA_AUG_UNIFORM_DIST_HELP_STRING = (
    'Boolean flag.  If 1, whole-track translation distances will actually be '
    'drawn from a uniform distribution, with min of 0 pixels and max of '
    '{0:s} + 3 * {1:s}.  If 0, whole-track translation distances will actually '
    'be drawn from Gaussian.  To use the same setting as during training, '
    'leave this argument alone.'
).format(
    MEAN_TRANSLATION_DIST_ARG_NAME, STDEV_TRANSLATION_DIST_ARG_NAME
)
MEAN_TRANS_DIST_WITHIN_HELP_STRING = (
    'Mean within-track translation distance (units of IR pixels, or low-res '
    'pixels).  To use the same setting as during training, leave this argument '
    'alone.'
)
STDEV_TRANS_DIST_WITHIN_HELP_STRING = (
    'Standard deviation of within-track translation distance (units of IR '
    'pixels, or low-res pixels).  To use the same setting as during training, '
    'leave this argument alone.'
)
DATA_AUG_WITHIN_UNIFORM_DIST_HELP_STRING = (
    'Boolean flag.  If 1, within-track translation distances will actually be '
    'drawn from a uniform distribution, with min of 0 pixels and max of '
    '{0:s} + 3 * {1:s}.  If 0, within-track translation distances will '
    'actually be drawn from Gaussian.  To use the same setting as during '
    'training, leave this argument alone.'
).format(
    MEAN_TRANS_DIST_WITHIN_ARG_NAME, STDEV_TRANS_DIST_WITHIN_ARG_NAME
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
DISABLE_GPUS_HELP_STRING = (
    'Boolean flag.  If 1, will disable GPUs and use only CPUs.  This argument '
    'is HIGHLY RECOMMENDED in any environment, besides Hera (or some machine '
    'where every job runs on a different node), where this script could be '
    'running multiple times at once.'
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
    '--' + SHORT_TRACK_DIR_ARG_NAME, type=str, required=False, default='same',
    help=SHORT_TRACK_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SHORT_TRACK_MAX_LEAD_ARG_NAME, type=int, required=False, default=-1,
    help=SHORT_TRACK_MAX_LEAD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SHORT_TRACK_DIFF_CENTERS_ARG_NAME, type=int, required=False,
    default=-2, help=SHORT_TRACK_DIFF_CENTERS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TRANSLATIONS_ARG_NAME, type=int, required=True,
    help=NUM_TRANSLATIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MEAN_TRANSLATION_DIST_ARG_NAME, type=float, required=False,
    default=-1., help=MEAN_TRANSLATION_DIST_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + STDEV_TRANSLATION_DIST_ARG_NAME, type=float, required=False,
    default=-1., help=STDEV_TRANSLATION_DIST_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DATA_AUG_UNIFORM_DIST_ARG_NAME, type=int, required=False,
    default=-1, help=DATA_AUG_UNIFORM_DIST_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MEAN_TRANS_DIST_WITHIN_ARG_NAME, type=float, required=False,
    default=-1., help=MEAN_TRANS_DIST_WITHIN_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + STDEV_TRANS_DIST_WITHIN_ARG_NAME, type=float, required=False,
    default=-1., help=STDEV_TRANS_DIST_WITHIN_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DATA_AUG_WITHIN_UNIFORM_DIST_ARG_NAME, type=int, required=False,
    default=-1, help=DATA_AUG_WITHIN_UNIFORM_DIST_HELP_STRING
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
    '--' + DISABLE_GPUS_ARG_NAME, type=int, required=False,
    default=0, help=DISABLE_GPUS_HELP_STRING
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
         cyclone_id_string, valid_date_string,
         short_track_dir_name, short_track_max_lead_minutes,
         short_track_center_each_lag_diffly, data_aug_num_translations,
         data_aug_mean_translation_low_res_px,
         data_aug_stdev_translation_low_res_px, data_aug_uniform_dist_flag,
         data_aug_within_mean_trans_px, data_aug_within_stdev_trans_px,
         data_aug_within_uniform_dist_flag,
         random_seed, remove_tropical_systems, synoptic_times_only,
         disable_gpus, output_dir_name, output_file_name):
    """Applies trained neural net -- inference time!

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param satellite_dir_name: Same.
    :param a_deck_file_name: Same.
    :param cyclone_id_string: Same.
    :param valid_date_string: Same.
    :param short_track_dir_name: Same.
    :param short_track_max_lead_minutes: Same.
    :param short_track_center_each_lag_diffly: Same.
    :param data_aug_num_translations: Same.
    :param data_aug_mean_translation_low_res_px: Same.
    :param data_aug_stdev_translation_low_res_px: Same.
    :param data_aug_uniform_dist_flag: Same.
    :param data_aug_within_mean_trans_px: Same.
    :param data_aug_within_stdev_trans_px: Same.
    :param data_aug_within_uniform_dist_flag: Same.
    :param random_seed: Same.
    :param remove_tropical_systems: Same.
    :param synoptic_times_only: Same.
    :param disable_gpus: Same.
    :param output_dir_name: Same.
    :param output_file_name: Same.
    """

    if disable_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if random_seed != -1:
        numpy.random.seed(random_seed)
    if valid_date_string == '':
        valid_date_string = None

    if output_dir_name == '':
        output_dir_name = None
    if output_dir_name is not None or output_file_name == '':
        output_file_name = None

    assert not (output_dir_name is None and output_file_name is None)

    if short_track_dir_name == 'same':
        short_track_dir_name = None
    if short_track_max_lead_minutes <= 0:
        short_track_max_lead_minutes = None
    if short_track_center_each_lag_diffly < 0:
        short_track_center_each_lag_diffly = None
    else:
        short_track_center_each_lag_diffly = bool(
            short_track_center_each_lag_diffly
        )

    if data_aug_mean_translation_low_res_px < 0:
        data_aug_mean_translation_low_res_px = None
    if data_aug_stdev_translation_low_res_px < 0:
        data_aug_stdev_translation_low_res_px = None
    if data_aug_within_mean_trans_px < 0:
        data_aug_within_mean_trans_px = None
    if data_aug_within_stdev_trans_px < 0:
        data_aug_within_stdev_trans_px = None

    if data_aug_uniform_dist_flag < 0:
        data_aug_uniform_dist_flag = None
    else:
        data_aug_uniform_dist_flag = bool(data_aug_uniform_dist_flag)

    if data_aug_within_uniform_dist_flag < 0:
        data_aug_within_uniform_dist_flag = None
    else:
        data_aug_within_uniform_dist_flag = bool(
            data_aug_within_uniform_dist_flag
        )

    error_checking.assert_is_geq(data_aug_num_translations, 1)

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = nn_utils.read_model(model_file_name)
    model_metafile_name = nn_utils.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = nn_utils.read_metafile(model_metafile_name)
    vod = model_metadata_dict[nn_utils.VALIDATION_OPTIONS_KEY]

    vod[nn_utils.SATELLITE_DIRECTORY_KEY] = satellite_dir_name
    vod[nn_utils.A_DECK_FILE_KEY] = a_deck_file_name
    vod[nn_utils.DATA_AUG_NUM_TRANS_KEY] = data_aug_num_translations

    if short_track_dir_name is not None:
        vod[nn_training_simple.SHORT_TRACK_DIR_KEY] = short_track_dir_name
    if short_track_max_lead_minutes is not None:
        vod[nn_training_simple.SHORT_TRACK_MAX_LEAD_KEY] = (
            short_track_max_lead_minutes
        )
    if short_track_center_each_lag_diffly is not None:
        vod[nn_training_simple.SHORT_TRACK_DIFF_CENTERS_KEY] = (
            short_track_center_each_lag_diffly
        )
    if data_aug_mean_translation_low_res_px is not None:
        vod[nn_utils.DATA_AUG_MEAN_TRANS_KEY] = (
            data_aug_mean_translation_low_res_px
        )
    if data_aug_stdev_translation_low_res_px is not None:
        vod[nn_utils.DATA_AUG_STDEV_TRANS_KEY] = (
            data_aug_stdev_translation_low_res_px
        )
    if data_aug_within_mean_trans_px is not None:
        vod[nn_training_simple.DATA_AUG_WITHIN_MEAN_TRANS_KEY] = (
            data_aug_within_mean_trans_px
        )
    if data_aug_within_stdev_trans_px is not None:
        vod[nn_training_simple.DATA_AUG_WITHIN_STDEV_TRANS_KEY] = (
            data_aug_within_stdev_trans_px
        )
    if data_aug_uniform_dist_flag is not None:
        vod[nn_training_simple.DATA_AUG_UNIFORM_DIST_KEY] = (
            data_aug_uniform_dist_flag
        )
    if data_aug_within_uniform_dist_flag is not None:
        vod[nn_training_simple.DATA_AUG_WITHIN_UNIFORM_DIST_KEY] = (
            data_aug_within_uniform_dist_flag
        )

    if remove_tropical_systems:
        vod[nn_utils.REMOVE_TROPICAL_KEY] = True
        vod[nn_utils.REMOVE_NONTROPICAL_KEY] = False

    vod[nn_utils.SYNOPTIC_TIMES_ONLY_KEY] = synoptic_times_only
    validation_option_dict = vod

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
            isotonic_model_file_name=None,
            uncertainty_calib_model_file_name=None
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        satellite_dir_name=getattr(INPUT_ARG_OBJECT, SATELLITE_DIR_ARG_NAME),
        a_deck_file_name=getattr(INPUT_ARG_OBJECT, A_DECK_FILE_ARG_NAME),
        cyclone_id_string=getattr(INPUT_ARG_OBJECT, CYCLONE_ID_ARG_NAME),
        valid_date_string=getattr(INPUT_ARG_OBJECT, VALID_DATE_ARG_NAME),
        short_track_dir_name=getattr(
            INPUT_ARG_OBJECT, SHORT_TRACK_DIR_ARG_NAME
        ),
        short_track_max_lead_minutes=getattr(
            INPUT_ARG_OBJECT, SHORT_TRACK_MAX_LEAD_ARG_NAME
        ),
        short_track_center_each_lag_diffly=getattr(
            INPUT_ARG_OBJECT, SHORT_TRACK_DIFF_CENTERS_ARG_NAME
        ),
        data_aug_num_translations=getattr(
            INPUT_ARG_OBJECT, NUM_TRANSLATIONS_ARG_NAME
        ),
        data_aug_mean_translation_low_res_px=getattr(
            INPUT_ARG_OBJECT, MEAN_TRANSLATION_DIST_ARG_NAME
        ),
        data_aug_stdev_translation_low_res_px=getattr(
            INPUT_ARG_OBJECT, STDEV_TRANSLATION_DIST_ARG_NAME
        ),
        data_aug_uniform_dist_flag=getattr(
            INPUT_ARG_OBJECT, DATA_AUG_UNIFORM_DIST_ARG_NAME
        ),
        data_aug_within_mean_trans_px=getattr(
            INPUT_ARG_OBJECT, MEAN_TRANS_DIST_WITHIN_ARG_NAME
        ),
        data_aug_within_stdev_trans_px=getattr(
            INPUT_ARG_OBJECT, STDEV_TRANS_DIST_WITHIN_ARG_NAME
        ),
        data_aug_within_uniform_dist_flag=getattr(
            INPUT_ARG_OBJECT, DATA_AUG_WITHIN_UNIFORM_DIST_ARG_NAME
        ),
        random_seed=getattr(INPUT_ARG_OBJECT, RANDOM_SEED_ARG_NAME),
        remove_tropical_systems=bool(
            getattr(INPUT_ARG_OBJECT, REMOVE_TROPICAL_SYSTEMS_ARG_NAME)
        ),
        synoptic_times_only=bool(
            getattr(INPUT_ARG_OBJECT, SYNOPTIC_TIMES_ONLY_ARG_NAME)
        ),
        disable_gpus=bool(
            getattr(INPUT_ARG_OBJECT, DISABLE_GPUS_ARG_NAME)
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
