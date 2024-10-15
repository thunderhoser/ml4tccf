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
"""

import os
import argparse
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
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=False, default='',
    help=OUTPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=False, default='',
    help=OUTPUT_FILE_HELP_STRING
)


def _run(model_file_name, satellite_dir_name, a_deck_file_name,
         cyclone_id_string, valid_date_string, disable_gpus,
         output_dir_name, output_file_name):
    """Real-time version of apply_neural_net.py.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param satellite_dir_name: Same.
    :param a_deck_file_name: Same.
    :param cyclone_id_string: Same.
    :param valid_date_string: Same.
    :param disable_gpus: Same.
    :param output_dir_name: Same.
    :param output_file_name: Same.
    """

    if disable_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
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

    validation_option_dict[nn_utils.DATA_AUG_NUM_TRANS_KEY] = 1
    validation_option_dict[nn_utils.DATA_AUG_MEAN_TRANS_KEY] = 1e-6
    validation_option_dict[nn_utils.DATA_AUG_STDEV_TRANS_KEY] = 1e-6

    validation_option_dict[nn_utils.SATELLITE_DIRECTORY_KEY] = (
        satellite_dir_name
    )
    validation_option_dict[nn_utils.A_DECK_FILE_KEY] = a_deck_file_name

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
        print('Writing results to: "{0:s}"...'.format(output_file_name))
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
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
