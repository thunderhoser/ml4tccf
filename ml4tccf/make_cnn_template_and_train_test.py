"""Makes CNN template and trains the model -- TEST ONLY."""

import os
import sys
import numpy
import tensorflow

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import architecture_utils
import neural_net
import cnn_architecture
import custom_losses

# from tensorflow.config.experimental import list_physical_devices, set_memory_growth
# gpus = list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tensorflow.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tensorflow.config.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)

OPTION_DICT = {
    cnn_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([600, 600, 7], dtype=int),
    cnn_architecture.INCLUDE_HIGH_RES_KEY: True,
    cnn_architecture.NUM_CONV_LAYERS_KEY: numpy.full(9, 2, dtype=int),
    cnn_architecture.NUM_CHANNELS_KEY: numpy.array([
        2, 2, 4, 4, 16, 16,
        20, 20, 24, 24, 28, 28, 32, 32, 36, 36,
        40, 40
    ], dtype=int),
    cnn_architecture.CONV_DROPOUT_RATES_KEY: numpy.full(18, 0.),
    cnn_architecture.NUM_NEURONS_KEY:
        numpy.array([1024, 128, 100, 100], dtype=int),
    cnn_architecture.DENSE_DROPOUT_RATES_KEY: numpy.array([0.5, 0.5, 0.5, 0]),
    cnn_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    cnn_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    cnn_architecture.L2_WEIGHT_KEY: 1e-6,
    cnn_architecture.USE_BATCH_NORM_KEY: True,
    cnn_architecture.ENSEMBLE_SIZE_KEY: 50,
    cnn_architecture.LOSS_FUNCTION_KEY: custom_losses.crps_kilometres()
}

OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/'
    'cnn_template_test'
)

SATELLITE_DIRECTORY_KEY = 'satellite_dir_name'
YEARS_KEY = 'years'
LAG_TIMES_KEY = 'lag_times_minutes'
HIGH_RES_WAVELENGTHS_KEY = 'high_res_wavelengths_microns'
LOW_RES_WAVELENGTHS_KEY = 'low_res_wavelengths_microns'
BATCH_SIZE_KEY = 'num_examples_per_batch'
MAX_EXAMPLES_PER_CYCLONE_KEY = 'max_examples_per_cyclone'
NUM_GRID_ROWS_KEY = 'num_rows_low_res'
NUM_GRID_COLUMNS_KEY = 'num_columns_low_res'
DATA_AUG_NUM_TRANS_KEY = 'data_aug_num_translations'
DATA_AUG_MEAN_TRANS_KEY = 'data_aug_mean_translation_low_res_px'
DATA_AUG_STDEV_TRANS_KEY = 'data_aug_stdev_translation_low_res_px'
LAG_TIME_TOLERANCE_KEY = 'lag_time_tolerance_sec'
MAX_MISSING_LAG_TIMES_KEY = 'max_num_missing_lag_times'
MAX_INTERP_GAP_KEY = 'max_interp_gap_sec'
SENTINEL_VALUE_KEY = 'sentinel_value'

TRAINING_OPTION_DICT = {
    SATELLITE_DIRECTORY_KEY: '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_project/satellite_data/processed',
    YEARS_KEY: numpy.array([2021], dtype=int),
    LAG_TIMES_KEY: numpy.array([0, 30, 60], dtype=int),
    HIGH_RES_WAVELENGTHS_KEY: numpy.array([0.64]),
    LOW_RES_WAVELENGTHS_KEY: numpy.array([3.9, 6.95, 8.5, 10.35, 11.2, 12.3, 13.3]),
    BATCH_SIZE_KEY: 4,
    MAX_EXAMPLES_PER_CYCLONE_KEY: 2,
    NUM_GRID_ROWS_KEY: 626,
    NUM_GRID_COLUMNS_KEY: 626,
    DATA_AUG_NUM_TRANS_KEY: 8,
    DATA_AUG_MEAN_TRANS_KEY: 15.,
    DATA_AUG_STDEV_TRANS_KEY: 7.5,
    LAG_TIME_TOLERANCE_KEY: 900,
    MAX_MISSING_LAG_TIMES_KEY: 1,
    MAX_INTERP_GAP_KEY: 3600,
    SENTINEL_VALUE_KEY: -10.
}

VALIDATION_OPTION_DICT = {
    SATELLITE_DIRECTORY_KEY: '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_project/satellite_data/processed',
    YEARS_KEY: numpy.array([2021], dtype=int),
    LAG_TIME_TOLERANCE_KEY: 900,
    MAX_MISSING_LAG_TIMES_KEY: 1,
    MAX_INTERP_GAP_KEY: 3600,
}


def _run():
    """Makes CNN template and trains the model -- TEST ONLY.

    This is effectively the main method.
    """

    model_object = cnn_architecture.create_model(OPTION_DICT)
    output_file_name = '{0:s}/model.h5'.format(OUTPUT_DIR_NAME)

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    print('Writing model to: "{0:s}"...'.format(output_file_name))
    model_object.save(
        filepath=output_file_name, overwrite=True, include_optimizer=True
    )

    neural_net.train_model(
        model_object=model_object,
        output_dir_name=OUTPUT_DIR_NAME, num_epochs=10,
        num_training_batches_per_epoch=3,
        training_option_dict=TRAINING_OPTION_DICT,
        num_validation_batches_per_epoch=3,
        validation_option_dict=VALIDATION_OPTION_DICT,
        loss_function_string='custom_losses.crps_kilometres()',
        do_early_stopping=True, plateau_lr_multiplier=0.6,
        bnn_architecture_dict=None
    )


if __name__ == '__main__':
    _run()
