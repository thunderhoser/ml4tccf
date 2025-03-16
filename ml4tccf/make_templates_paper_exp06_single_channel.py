"""Makes templates for Paper Experiment 6 (one predictor channel)."""

import os
import sys
import copy
import numpy
import keras
import tensorflow

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import architecture_utils
import neural_net_utils
import temporal_cnn_architecture as tcnn_architecture
import custom_losses_scalar

OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/'
    'paper_experiment06_single_channel/templates'
)

ENSEMBLE_SIZE = 50

OPTIMIZER_FUNCTION = keras.optimizers.AdamW()
OPTIMIZER_STRING = 'keras.optimizers.AdamW()'
LOSS_FUNCTION_STRING = 'custom_losses_scalar.coord_avg_crps_kilometres'

DEFAULT_OPTION_DICT = {
    # tcnn_architecture.INPUT_DIMENSIONS_LOW_RES_KEY:
    #     numpy.array([580, 900, 21], dtype=int),
    # tcnn_architecture.INPUT_DIMENSIONS_SCALAR_KEY:
    #     numpy.array([NUM_SCALAR_PREDICTORS], dtype=int),
    tcnn_architecture.INCLUDE_HIGH_RES_KEY: False,
    tcnn_architecture.INCLUDE_SCALAR_DATA_KEY: True,
    tcnn_architecture.START_WITH_POOLING_KEY: False,
    # tcnn_architecture.NUM_CONV_LAYERS_KEY: numpy.full(
    #     NUM_CONV_BLOCKS, 1, dtype=int
    # ),
    # tcnn_architecture.NUM_CHANNELS_KEY: numpy.array(
    #     [16, 16, 24, 24, 32, 32, 40, 40, 48, 48, 56, 56, 64, 64], dtype=int
    # ),
    # tcnn_architecture.CONV_DROPOUT_RATES_KEY: numpy.full(NUM_CONV_BLOCKS, 0.),
    # tcnn_architecture.NUM_NEURONS_KEY:
    #     numpy.array([1024, 128, 50, 50], dtype=int),
    tcnn_architecture.DENSE_DROPOUT_RATES_KEY: numpy.array([0.5, 0.5, 0.5, 0]),
    tcnn_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    tcnn_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    tcnn_architecture.L2_WEIGHT_KEY: 1e-6,
    tcnn_architecture.USE_BATCH_NORM_KEY: True,
    tcnn_architecture.ENSEMBLE_SIZE_KEY: ENSEMBLE_SIZE,
    tcnn_architecture.LOSS_FUNCTION_KEY:
        custom_losses_scalar.coord_avg_crps_kilometres,
    tcnn_architecture.OPTIMIZER_FUNCTION_KEY: OPTIMIZER_FUNCTION
}

MAIN_POOLING_FACTOR = 2
INCLUDE_ATCF = True
NUM_GRID_ROWS = 400
NUM_LAG_TIMES = 5


def _run():
    """Makes templates for Paper Experiment 6 (one predictor channel).

    This is effectively the main method.
    """

    option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)
    input_dimensions = numpy.array(
        [NUM_GRID_ROWS, NUM_GRID_ROWS, NUM_LAG_TIMES, 1], dtype=int
    )

    if NUM_GRID_ROWS == 300:
        if MAIN_POOLING_FACTOR == 2:
            pooling_size_by_conv_block_px = numpy.full(
                7, 2, dtype=int
            )
            num_channels_multipliers = numpy.array([
                1, 2, 3, 4, 5, 6, 7
            ])
            num_pixels_coarsest = 16
        else:
            pooling_size_by_conv_block_px = numpy.array(
                [2, 3, 3, 3, 2], dtype=int
            )
            num_channels_multipliers = numpy.array([
                1, 2, 3 + 2. / 3, 5 + 1. / 3, 7
            ])
            num_pixels_coarsest = 25

    elif NUM_GRID_ROWS == 400:
        if MAIN_POOLING_FACTOR == 2:
            pooling_size_by_conv_block_px = numpy.full(
                7, 2, dtype=int
            )
            num_channels_multipliers = numpy.array([
                1, 2, 3, 4, 5, 6, 7
            ])
            num_pixels_coarsest = 36
        else:
            pooling_size_by_conv_block_px = numpy.array(
                [2, 3, 3, 3, 2], dtype=int
            )
            num_channels_multipliers = numpy.array([
                1, 2, 3 + 2. / 3, 5 + 1. / 3, 7
            ])
            num_pixels_coarsest = 49

    elif NUM_GRID_ROWS == 500:
        if MAIN_POOLING_FACTOR == 2:
            pooling_size_by_conv_block_px = numpy.full(
                7, 2, dtype=int
            )
            num_channels_multipliers = numpy.array([
                1, 2, 3, 4, 5, 6, 7
            ])
            num_pixels_coarsest = 49
        else:
            pooling_size_by_conv_block_px = numpy.array(
                [2, 3, 3, 3, 2, 2], dtype=int
            )
            num_channels_multipliers = numpy.array([
                1, 2, 3 + 1. / 3, 4 + 2. / 3, 6, 7
            ])
            num_pixels_coarsest = 16

    else:
        if MAIN_POOLING_FACTOR == 2:
            pooling_size_by_conv_block_px = numpy.full(
                8, 2, dtype=int
            )
            num_channels_multipliers = numpy.array([
                1, 2, 3, 4, 5, 6, 7, 8
            ])
            num_pixels_coarsest = 16
        else:
            pooling_size_by_conv_block_px = numpy.array(
                [2, 3, 3, 3, 2, 2], dtype=int
            )
            num_channels_multipliers = numpy.array([
                1, 2, 3 + 1. / 3, 4 + 2. / 3, 6, 7
            ])
            num_pixels_coarsest = 25

    num_conv_blocks = len(pooling_size_by_conv_block_px)
    num_channels_by_conv_layer = numpy.round(
        10 * num_channels_multipliers
    ).astype(int)

    num_dense_layers = len(
        option_dict[tcnn_architecture.DENSE_DROPOUT_RATES_KEY]
    )

    if INCLUDE_ATCF:
        num_scalar_predictors = 9
    else:
        num_scalar_predictors = 0

    (
        dense_neuron_counts
    ) = architecture_utils.get_dense_layer_dimensions(
        num_input_units=(
            num_scalar_predictors +
            num_pixels_coarsest * num_channels_by_conv_layer[-1]
        ),
        num_classes=2,
        num_dense_layers=num_dense_layers,
        for_classification=False
    )[1]

    dense_neuron_counts[-1] = 2 * ENSEMBLE_SIZE
    dense_neuron_counts[-2] = max([
        dense_neuron_counts[-1], dense_neuron_counts[-2]
    ])

    option_dict.update({
        tcnn_architecture.INPUT_DIMENSIONS_LOW_RES_KEY:
            input_dimensions,
        tcnn_architecture.NUM_CONV_LAYERS_KEY:
            numpy.full(num_conv_blocks, 1, dtype=int),
        tcnn_architecture.POOLING_SIZE_KEY:
            pooling_size_by_conv_block_px,
        tcnn_architecture.NUM_CHANNELS_KEY:
            num_channels_by_conv_layer,
        tcnn_architecture.CONV_DROPOUT_RATES_KEY:
            numpy.full(num_conv_blocks, 0.),
        tcnn_architecture.NUM_NEURONS_KEY: dense_neuron_counts
    })

    if INCLUDE_ATCF:
        option_dict.update({
            tcnn_architecture.INPUT_DIMENSIONS_SCALAR_KEY:
                numpy.array([num_scalar_predictors], dtype=int),
            tcnn_architecture.INCLUDE_SCALAR_DATA_KEY: True
        })
    else:
        option_dict.update({
            tcnn_architecture.INPUT_DIMENSIONS_SCALAR_KEY: None,
            tcnn_architecture.INCLUDE_SCALAR_DATA_KEY: False
        })

    model_object = tcnn_architecture.create_model(option_dict)

    output_file_name = '{0:s}/model.keras'.format(OUTPUT_DIR_NAME)
    file_system_utils.mkdir_recursive_if_necessary(
        file_name=output_file_name
    )

    print('Writing model to: "{0:s}"...'.format(output_file_name))
    model_object.save(
        filepath=output_file_name, overwrite=True,
        include_optimizer=True
    )

    metafile_name = neural_net_utils.find_metafile(
        model_dir_name=os.path.split(output_file_name)[0],
        raise_error_if_missing=False
    )

    option_dict[tcnn_architecture.LOSS_FUNCTION_KEY] = LOSS_FUNCTION_STRING
    option_dict[tcnn_architecture.OPTIMIZER_FUNCTION_KEY] = OPTIMIZER_STRING

    neural_net_utils.write_metafile(
        pickle_file_name=metafile_name,
        num_epochs=100,
        num_training_batches_per_epoch=32,
        training_option_dict={
            neural_net_utils.SEMANTIC_SEG_FLAG_KEY: False
        },
        num_validation_batches_per_epoch=16,
        validation_option_dict={
            neural_net_utils.SEMANTIC_SEG_FLAG_KEY: False
        },
        loss_function_string=LOSS_FUNCTION_STRING,
        optimizer_function_string=OPTIMIZER_STRING,
        plateau_patience_epochs=10,
        plateau_learning_rate_multiplier=0.6,
        early_stopping_patience_epochs=50,
        cnn_architecture_dict=None,
        structure_cnn_architecture_dict=None,
        u_net_architecture_dict=None,
        temporal_convnext_architecture_dict=None,
        temporal_cnn_architecture_dict=option_dict,
        data_type_string=neural_net_utils.RG_SIMPLE_DATA_TYPE_STRING,
        train_with_shuffled_data=True
    )


if __name__ == '__main__':
    _run()
