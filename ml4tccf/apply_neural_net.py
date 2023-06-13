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
NUM_EXAMPLES_PER_BATCH = 32

MODEL_FILE_ARG_NAME = 'input_model_file_name'
SATELLITE_DIR_ARG_NAME = 'input_satellite_dir_name'
CYCLONE_ID_ARG_NAME = 'cyclone_id_string'
NUM_BNN_ITERATIONS_ARG_NAME = 'num_bnn_iterations'
MAX_ENSEMBLE_SIZE_ARG_NAME = 'max_ensemble_size'
NUM_TRANSLATIONS_ARG_NAME = 'data_aug_num_translations'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model (will be read by `nn_utils.read_model`).'
)
SATELLITE_DIR_HELP_STRING = (
    'Name of directory with satellite (predictor) data.  Files therein will be '
    'found by `satellite_io.find_file` and read by `satellite_io.read_file`.'
)
CYCLONE_ID_HELP_STRING = (
    'Will apply neural net to data from this cyclone.  Cyclone ID must be in '
    'format "yyyyBBnn".'
)
NUM_BNN_ITERATIONS_HELP_STRING = (
    '[used only if the model is a Bayesian neural net (BNN)] Number of times '
    'to run the BNN.'
)
MAX_ENSEMBLE_SIZE_HELP_STRING = (
    'Max ensemble size.  If final ensemble size ends up being larger, it will '
    'be randomly subset to equal {0:s}.'
).format(MAX_ENSEMBLE_SIZE_ARG_NAME)

NUM_TRANSLATIONS_HELP_STRING = (
    'Number of translations for each cyclone snapshot (one snapshot = one '
    'cyclone at one target time).  Total number of data samples will be '
    'num_snapshots * {0:s}.'
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
    '--' + CYCLONE_ID_ARG_NAME, type=str, required=True,
    help=CYCLONE_ID_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BNN_ITERATIONS_ARG_NAME, type=int, required=False, default=1,
    help=NUM_BNN_ITERATIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_ENSEMBLE_SIZE_ARG_NAME, type=int, required=False,
    default=LARGE_INTEGER, help=MAX_ENSEMBLE_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TRANSLATIONS_ARG_NAME, type=int, required=True,
    help=NUM_TRANSLATIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(model_file_name, satellite_dir_name, cyclone_id_string,
         num_bnn_iterations, max_ensemble_size, data_aug_num_translations,
         output_dir_name):
    """Applies trained neural net -- inference time!

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param satellite_dir_name: Same.
    :param cyclone_id_string: Same.
    :param num_bnn_iterations: Same.
    :param max_ensemble_size: Same.
    :param data_aug_num_translations: Same.
    :param output_dir_name: Same.
    """

    error_checking.assert_is_geq(data_aug_num_translations, 1)

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = nn_utils.read_model(model_file_name)
    model_metafile_name = nn_utils.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = nn_utils.read_metafile(model_metafile_name)

    if not model_metadata_dict[nn_utils.IS_MODEL_BNN_KEY]:
        num_bnn_iterations = 1

    error_checking.assert_is_geq(num_bnn_iterations, 1)

    validation_option_dict = (
        model_metadata_dict[nn_utils.VALIDATION_OPTIONS_KEY]
    )
    validation_option_dict[nn_utils.SATELLITE_DIRECTORY_KEY] = (
        satellite_dir_name
    )
    validation_option_dict[nn_utils.DATA_AUG_NUM_TRANS_KEY] = (
        data_aug_num_translations
    )

    # TODO(thunderhoser): I might want to make this an input arg to the script.
    validation_option_dict[nn_utils.SYNOPTIC_TIMES_ONLY_KEY] = True

    # TODO(thunderhoser): With many target times and many translations, I might
    # get out-of-memory errors.  In this case, I will write a new version of
    # this script, which calls `create_data` once without data augmentation and
    # then augments each example many times.

    data_type_string = model_metadata_dict[nn_utils.DATA_TYPE_KEY]

    if data_type_string == nn_utils.CIRA_IR_DATA_TYPE_STRING:
        data_dict = nn_training_cira_ir.create_data(
            option_dict=validation_option_dict,
            cyclone_id_string=cyclone_id_string,
            num_target_times=LARGE_INTEGER
        )
    elif data_type_string == nn_utils.RG_SIMPLE_DATA_TYPE_STRING:
        data_dict = nn_training_simple.create_data(
            option_dict=validation_option_dict,
            cyclone_id_string=cyclone_id_string,
            num_target_times=LARGE_INTEGER
        )
    else:
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

    prediction_matrix = None
    inner_ensemble_size = -1
    print(SEPARATOR_STRING)

    for k in range(num_bnn_iterations):
        this_prediction_matrix = nn_utils.apply_model(
            model_object=model_object, predictor_matrices=predictor_matrices,
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH, verbose=True
        )

        if prediction_matrix is None:
            inner_ensemble_size = this_prediction_matrix.shape[-1]
            total_ensemble_size = inner_ensemble_size * num_bnn_iterations
            dimensions = (
                this_prediction_matrix.shape[:-1] + (total_ensemble_size,)
            )
            prediction_matrix = numpy.full(dimensions, numpy.nan)

        first_index = k * inner_ensemble_size
        last_index = first_index + inner_ensemble_size
        prediction_matrix[..., first_index:last_index] = this_prediction_matrix

        print(SEPARATOR_STRING)

    del predictor_matrices

    total_ensemble_size = prediction_matrix.shape[-1]
    if total_ensemble_size > max_ensemble_size:
        ensemble_indices = numpy.linspace(
            0, total_ensemble_size - 1, num=total_ensemble_size, dtype=int
        )
        ensemble_indices = numpy.random.choice(
            ensemble_indices, size=max_ensemble_size, replace=False
        )
        prediction_matrix = prediction_matrix[..., ensemble_indices]

    if validation_option_dict[nn_utils.SEMANTIC_SEG_FLAG_KEY]:
        output_file_name = prediction_io.find_file(
            directory_name=output_dir_name, cyclone_id_string=cyclone_id_string,
            raise_error_if_missing=False
        )

        print('Writing results to: "{0:s}"...'.format(output_file_name))
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
        output_file_name = prediction_io.find_file(
            directory_name=output_dir_name, cyclone_id_string=cyclone_id_string,
            raise_error_if_missing=False
        )

        print('Writing results to: "{0:s}"...'.format(output_file_name))
        scalar_prediction_io.write_file(
            netcdf_file_name=output_file_name,
            target_matrix=target_matrix,
            prediction_matrix=prediction_matrix,
            cyclone_id_string=cyclone_id_string,
            target_times_unix_sec=target_times_unix_sec,
            model_file_name=model_file_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        satellite_dir_name=getattr(INPUT_ARG_OBJECT, SATELLITE_DIR_ARG_NAME),
        cyclone_id_string=getattr(INPUT_ARG_OBJECT, CYCLONE_ID_ARG_NAME),
        num_bnn_iterations=getattr(
            INPUT_ARG_OBJECT, NUM_BNN_ITERATIONS_ARG_NAME
        ),
        max_ensemble_size=getattr(INPUT_ARG_OBJECT, MAX_ENSEMBLE_SIZE_ARG_NAME),
        data_aug_num_translations=getattr(
            INPUT_ARG_OBJECT, NUM_TRANSLATIONS_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
