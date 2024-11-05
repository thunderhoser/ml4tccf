"""Real-time version of apply_neural_net.py.

There are two differences between the real-time (here) and development
(apply_neural_net.py) version of this script:

[1] The development version has an input argument called
    `data_aug_num_translations`, which specifies the number of random
    translations to apply to every TC sample.  This is because in development
    mode every image center already coincides with the best-track TC center,
    which is our "ground truth" -- so unless you create a random offset between
    the image center and best-track TC center, there is nothing for the neural
    net to do.  But in real time, every image center is a first guess coming
    from, e.g., an operational forecast.  This best is not the "true" best-track
    center, so there is already something for the neural net to correct; thus,
    there is no need to randomly translate the image, creating a further offset
    between the image center and best-track TC center.
[2] The development version applies the neural net only to synoptic times
    (0000, 0600, 1200, 1800 UTC), since these are the times with the best
    "ground truth".  But in real time, we want the ability to produce neural-net
    estimates whenever.

UPDATE: I added back the "data augmentation" input args, since there are some
situations where we want to add random translations to the first-guess center,
even in real time.
"""

import os
import argparse
import numpy
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
DISABLE_GPUS_ARG_NAME = 'disable_gpus'
NUM_TRANSLATIONS_ARG_NAME = 'data_aug_num_translations'
MEAN_TRANSLATION_DIST_ARG_NAME = 'data_aug_mean_translation_low_res_px'
STDEV_TRANSLATION_DIST_ARG_NAME = 'data_aug_stdev_translation_low_res_px'
NUM_TRANSLATIONS_PER_STEP_ARG_NAME = 'data_aug_num_translations_per_step'
RANDOM_SEED_ARG_NAME = 'random_seed'
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
DISABLE_GPUS_HELP_STRING = (
    'Boolean flag.  If 1, will disable GPUs and use only CPUs.  This argument '
    'is HIGHLY RECOMMENDED in any environment, besides Hera (or some machine '
    'where every job runs on a different node), where this script could be '
    'running multiple times at once.'
)
NUM_TRANSLATIONS_HELP_STRING = (
    'Number of random translations to apply to each TC snapshot (one snapshot '
    '= one TC at one time).  Total number of TC samples will be num_snapshots '
    '* {0:s}.  WARNING: Use this argument only if you want to add random '
    'translations to the first-guess TC center.  If you want to just leave the '
    'first-guess TC center alone and use this as the image center (which is '
    'the default behaviour for the real-time version of GeoCenter), then make '
    'this argument 0.'
).format(
    NUM_TRANSLATIONS_ARG_NAME
)
MEAN_TRANSLATION_DIST_HELP_STRING = (
    '[used only if {0:s} > 0] Mean translation distance (units of IR pixels).  '
    'If you want to keep the same mean translation distance used in training, '
    'leave this argument alone.'
).format(
    NUM_TRANSLATIONS_ARG_NAME
)
STDEV_TRANSLATION_DIST_HELP_STRING = (
    '[used only if {0:s} > 0] Standard deviation of translation distance '
    '(units of IR pixels).  If you want to keep the same stdev translation '
    'distance used in training, leave this argument alone.'
).format(
    NUM_TRANSLATIONS_ARG_NAME
)
NUM_TRANSLATIONS_PER_STEP_HELP_STRING = (
    '[used only if {0:s} > 0] Number of random translations per step.  This '
    'argument is useful if you are doing a large number of random translations '
    '(>~ 20).  For example, if you are doing 100 random translations, the '
    'predictor matrix must be 100 times larger than usual -- which will cause '
    'an out-of-memory error on most systems.  You may instead wish to do the '
    'random translations in 5 steps, for example.  In this case, each step '
    'will consist of 100/5 = 20 random translations, followed by applying the '
    'NN to the predictor matrix, then promptly deleting the predictor matrix '
    'so that it doesn''t cause an out-of-memory error.  If you would rather do '
    'all translations at once, leave this argument alone.'
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
    '--' + DISABLE_GPUS_ARG_NAME, type=int, required=False,
    default=0, help=DISABLE_GPUS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TRANSLATIONS_ARG_NAME, type=int, required=False, default=-1,
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
    '--' + NUM_TRANSLATIONS_PER_STEP_ARG_NAME, type=int, required=False,
    default=-1, help=NUM_TRANSLATIONS_PER_STEP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + RANDOM_SEED_ARG_NAME, type=int, required=False, default=6695,
    help=RANDOM_SEED_HELP_STRING
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
         cyclone_id_string, valid_date_string, disable_gpus,
         data_aug_num_translations, data_aug_mean_translation_low_res_px,
         data_aug_stdev_translation_low_res_px,
         data_aug_num_translations_per_step, random_seed,
         output_dir_name, output_file_name):
    """Real-time version of apply_neural_net.py.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param satellite_dir_name: Same.
    :param a_deck_file_name: Same.
    :param cyclone_id_string: Same.
    :param valid_date_string: Same.
    :param disable_gpus: Same.
    :param data_aug_num_translations: Same.
    :param data_aug_mean_translation_low_res_px: Same.
    :param data_aug_stdev_translation_low_res_px: Same.
    :param data_aug_num_translations_per_step: Same.
    :param random_seed: Same.
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
    else:
        output_file_name = None

    if output_file_name == '':
        output_file_name = None
    else:
        output_dir_name = None

    assert not (output_dir_name is None and output_file_name is None)

    if data_aug_num_translations < 0:
        data_aug_num_translations = None
    if data_aug_mean_translation_low_res_px < 0:
        data_aug_mean_translation_low_res_px = None
    if data_aug_stdev_translation_low_res_px < 0:
        data_aug_stdev_translation_low_res_px = None
    if data_aug_num_translations_per_step <= 0:
        data_aug_num_translations_per_step = None

    if data_aug_num_translations is None:
        data_aug_num_translations_per_step = None
    else:
        if data_aug_num_translations_per_step is None:
            data_aug_num_translations_per_step = data_aug_num_translations + 0

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
    validation_option_dict[nn_utils.REMOVE_TROPICAL_KEY] = False
    validation_option_dict[nn_utils.REMOVE_NONTROPICAL_KEY] = False
    validation_option_dict[nn_utils.SYNOPTIC_TIMES_ONLY_KEY] = False

    if data_aug_num_translations is None:

        # First-guess TC center will be left alone.
        validation_option_dict[nn_utils.DATA_AUG_NUM_TRANS_KEY] = 1
        validation_option_dict[nn_utils.DATA_AUG_MEAN_TRANS_KEY] = 1e-6
        validation_option_dict[nn_utils.DATA_AUG_STDEV_TRANS_KEY] = 1e-6
    else:

        # First-guess TC center will be perturbed.
        validation_option_dict[nn_utils.DATA_AUG_NUM_TRANS_KEY] = (
            data_aug_num_translations
        )

        if data_aug_mean_translation_low_res_px is not None:
            validation_option_dict[nn_utils.DATA_AUG_MEAN_TRANS_KEY] = (
                data_aug_mean_translation_low_res_px
            )
        if data_aug_stdev_translation_low_res_px is not None:
            validation_option_dict[nn_utils.DATA_AUG_STDEV_TRANS_KEY] = (
                data_aug_stdev_translation_low_res_px
            )

    validation_option_dict[nn_utils.SATELLITE_DIRECTORY_KEY] = (
        satellite_dir_name
    )
    validation_option_dict[nn_utils.A_DECK_FILE_KEY] = a_deck_file_name

    data_type_string = model_metadata_dict[nn_utils.DATA_TYPE_KEY]
    target_matrix = numpy.array([], dtype=float)
    prediction_matrix = numpy.array([], dtype=float)
    grid_spacings_km = numpy.array([], dtype=float)
    cyclone_center_latitudes_deg_n = numpy.array([], dtype=float)
    target_times_unix_sec = numpy.array([], dtype=int)

    while True:
        if data_aug_num_translations is not None:
            num_translations_done = target_matrix.shape[0]
            this_num_translations = min([
                data_aug_num_translations - num_translations_done,
                data_aug_num_translations_per_step
            ])

            validation_option_dict[nn_utils.DATA_AUG_NUM_TRANS_KEY] = (
                this_num_translations
            )
            print('this_num_translations = {0:d}'.format(this_num_translations))

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

        these_predictor_matrices = data_dict[nn_utils.PREDICTOR_MATRICES_KEY]
        this_target_matrix = data_dict[nn_utils.TARGET_MATRIX_KEY]
        these_grid_spacings_km = data_dict[nn_utils.GRID_SPACINGS_KEY]
        these_center_latitudes_deg_n = data_dict[nn_utils.CENTER_LATITUDES_KEY]
        these_target_times_unix_sec = data_dict[nn_utils.TARGET_TIMES_KEY]

        if len(this_target_matrix.shape) == 4:
            this_target_matrix = this_target_matrix[..., 0]

        print(SEPARATOR_STRING)
        this_prediction_matrix = nn_utils.apply_model(
            model_object=model_object,
            predictor_matrices=these_predictor_matrices,
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            verbose=True
        )
        print(SEPARATOR_STRING)

        del these_predictor_matrices

        if target_matrix.size == 0:
            target_matrix = this_target_matrix + 0.
            prediction_matrix = this_prediction_matrix + 0.
            grid_spacings_km = these_grid_spacings_km + 0.
            cyclone_center_latitudes_deg_n = these_center_latitudes_deg_n + 0.
            target_times_unix_sec = these_target_times_unix_sec + 0
        else:
            target_matrix = numpy.concatenate(
                [target_matrix, this_target_matrix], axis=0
            )
            prediction_matrix = numpy.concatenate(
                [prediction_matrix, this_prediction_matrix], axis=0
            )
            grid_spacings_km = numpy.concatenate(
                [grid_spacings_km, these_grid_spacings_km], axis=0
            )
            cyclone_center_latitudes_deg_n = numpy.concatenate(
                [cyclone_center_latitudes_deg_n, these_center_latitudes_deg_n],
                axis=0
            )
            target_times_unix_sec = numpy.concatenate(
                [target_times_unix_sec, these_target_times_unix_sec], axis=0
            )

        print('Shape of target_matrix = {0:s}'.format(str(target_matrix.shape)))

        if data_aug_num_translations is None:
            break

        num_translations_done = target_matrix.shape[0]
        if num_translations_done >= data_aug_num_translations:
            break

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
        disable_gpus=bool(
            getattr(INPUT_ARG_OBJECT, DISABLE_GPUS_ARG_NAME)
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
        data_aug_num_translations_per_step=getattr(
            INPUT_ARG_OBJECT, NUM_TRANSLATIONS_PER_STEP_ARG_NAME
        ),
        random_seed=getattr(INPUT_ARG_OBJECT, RANDOM_SEED_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
