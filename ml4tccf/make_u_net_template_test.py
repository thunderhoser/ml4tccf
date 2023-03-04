"""Makes U-net template -- TEST ONLY."""

import os
import sys
import copy
import numpy
import keras

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import architecture_utils
import u_net_architecture
import custom_losses_gridded
import neural_net

LOSS_FUNCTION_STRING = (
    'custom_losses_gridded.fractions_skill_score('
    'half_window_size_px=10, use_as_loss_function=True, function_name="loss"'
    ')'
)
OPTIMIZER_FUNCTION_STRING = 'keras.optimizers.Adam()'

DEFAULT_OPTION_DICT = {
    u_net_architecture.INPUT_DIMENSIONS_LOW_RES_KEY:
        numpy.array([600, 600, 1], dtype=int),
    u_net_architecture.INPUT_DIMENSIONS_HIGH_RES_KEY:
        numpy.array([5000, 5000, 3], dtype=int),
    u_net_architecture.INCLUDE_HIGH_RES_KEY: False,
    u_net_architecture.CONV_LAYER_COUNTS_KEY: numpy.full(9, 2, dtype=int),
    u_net_architecture.OUTPUT_CHANNEL_COUNTS_KEY: numpy.array(
        [8, 16, 24, 32, 40, 48, 56, 64, 72], dtype=int
    ),
    u_net_architecture.CONV_DROPOUT_RATES_KEY: [numpy.full(2, 0.)] * 9,
    u_net_architecture.UPCONV_DROPOUT_RATES_KEY: numpy.array([
        0, 0, 0, 0, 0, 0.3, 0.3, 0.3
    ]),
    u_net_architecture.SKIP_DROPOUT_RATES_KEY: [numpy.full(2, 0.)] * 8,
    u_net_architecture.INCLUDE_PENULTIMATE_KEY: False,
    u_net_architecture.PENULTIMATE_DROPOUT_RATE_KEY: 0.,
    u_net_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    u_net_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    u_net_architecture.L2_WEIGHT_KEY: 1e-6,
    u_net_architecture.USE_BATCH_NORM_KEY: True,
    u_net_architecture.ENSEMBLE_SIZE_KEY: 5,
    u_net_architecture.LOSS_FUNCTION_KEY:
        custom_losses_gridded.fractions_skill_score(
            half_window_size_px=10, use_as_loss_function=True,
            function_name='loss'
        ),
    u_net_architecture.OPTIMIZER_FUNCTION_KEY: keras.optimizers.Adam()
}

OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/'
    'u_net_template_test'
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
SEMANTIC_SEG_FLAG_KEY = 'semantic_segmentation_flag'
TARGET_SMOOOTHER_STDEV_KEY = 'target_smoother_stdev_km'

TRAINING_OPTION_DICT = {
    SATELLITE_DIRECTORY_KEY: '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_project/satellite_data/processed',
    YEARS_KEY: numpy.array([2017], dtype=int),
    LAG_TIMES_KEY: numpy.array([0], dtype=int),
    HIGH_RES_WAVELENGTHS_KEY: numpy.array([]),
    LOW_RES_WAVELENGTHS_KEY: numpy.array([3.9, 6.185, 6.95, 7.34, 8.5, 9.61, 10.35, 11.2, 12.3, 13.3]),
    BATCH_SIZE_KEY: 8,
    MAX_EXAMPLES_PER_CYCLONE_KEY: 2,
    NUM_GRID_ROWS_KEY: 600,
    NUM_GRID_COLUMNS_KEY: 600,
    DATA_AUG_NUM_TRANS_KEY: 4,
    DATA_AUG_MEAN_TRANS_KEY: 15.,
    DATA_AUG_STDEV_TRANS_KEY: 7.5,
    LAG_TIME_TOLERANCE_KEY: 900,
    MAX_MISSING_LAG_TIMES_KEY: 1,
    MAX_INTERP_GAP_KEY: 3600,
    SENTINEL_VALUE_KEY: -10.,
    SEMANTIC_SEG_FLAG_KEY: True,
    TARGET_SMOOOTHER_STDEV_KEY: 1e-6
}

VALIDATION_OPTION_DICT = {
    SATELLITE_DIRECTORY_KEY: '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_project/satellite_data/processed',
    YEARS_KEY: numpy.array([2020], dtype=int),
    LAG_TIME_TOLERANCE_KEY: 900,
    MAX_MISSING_LAG_TIMES_KEY: 1,
    MAX_INTERP_GAP_KEY: 3600,
}


def _run():
    """Makes U-net template -- TEST ONLY.

    This is effectively the main method.
    """

    model_object = u_net_architecture.create_model(DEFAULT_OPTION_DICT)
    output_file_name = '{0:s}/model.h5'.format(OUTPUT_DIR_NAME)

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    print('Writing model to: "{0:s}"...'.format(output_file_name))
    model_object.save(
        filepath=output_file_name, overwrite=True, include_optimizer=True
    )

    option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)
    option_dict[u_net_architecture.LOSS_FUNCTION_KEY] = (
        LOSS_FUNCTION_STRING
    )
    option_dict[u_net_architecture.OPTIMIZER_FUNCTION_KEY] = (
        'keras.optimizers.Adam()'
    )

    neural_net.train_model(
        model_object=model_object,
        output_dir_name=OUTPUT_DIR_NAME, num_epochs=10,
        num_training_batches_per_epoch=32,
        training_option_dict=TRAINING_OPTION_DICT,
        num_validation_batches_per_epoch=16,
        validation_option_dict=VALIDATION_OPTION_DICT,
        loss_function_string=LOSS_FUNCTION_STRING,
        optimizer_function_string=OPTIMIZER_FUNCTION_STRING,
        plateau_patience_epochs=10,
        plateau_learning_rate_multiplier=0.6,
        early_stopping_patience_epochs=50,
        architecture_dict=option_dict, is_model_bnn=False
    )


if __name__ == '__main__':
    _run()
