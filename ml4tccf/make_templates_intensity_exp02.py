"""Makes CNN templates for Intensity Experiment 2."""

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
import neural_net_utils
import temporal_cnn_architecture as tcnn_architecture
import custom_losses_scalar
import custom_metrics_structure

OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/'
    'intensity_experiment01/templates'
)

ENSEMBLE_SIZE = 2
NUM_DENSE_LAYERS = 4
NUM_SCALAR_PREDICTORS = 13

LOSS_FUNCTION = custom_losses_scalar.dwmse_for_structure_params(
    channel_weights=numpy.array([1.]),
    function_name='loss_dwmse'
)
LOSS_FUNCTION_STRING = (
    'custom_losses_scalar.dwmse_for_structure_params('
        'channel_weights=numpy.array([1.]), '
        'function_name="loss_dwmse"'
    ')'
)

OPTIMIZER_FUNCTION = keras.optimizers.Nadam()
OPTIMIZER_FUNCTION_STRING = 'keras.optimizers.Nadam()'

DEFAULT_OPTION_DICT = {
    # tcnn_architecture.INPUT_DIMENSIONS_LOW_RES_KEY:
    #     numpy.array([580, 900, 21], dtype=int),
    # tcnn_architecture.INPUT_DIMENSIONS_SCALAR_KEY:
    #     numpy.array([NUM_SCALAR_PREDICTORS], dtype=int),
    tcnn_architecture.INCLUDE_HIGH_RES_KEY: False,
    # tcnn_architecture.INCLUDE_SCALAR_DATA_KEY: True,
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
    # tcnn_architecture.DENSE_DROPOUT_RATES_KEY: numpy.array([0.5, 0.5, 0.5, 0]),
    tcnn_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    tcnn_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    # tcnn_architecture.L2_WEIGHT_KEY: 1e-6,
    tcnn_architecture.USE_BATCH_NORM_KEY: True,
    tcnn_architecture.ENSEMBLE_SIZE_KEY: ENSEMBLE_SIZE,
    tcnn_architecture.LOSS_FUNCTION_KEY: LOSS_FUNCTION,
    tcnn_architecture.OPTIMIZER_FUNCTION_KEY: OPTIMIZER_FUNCTION,
    tcnn_architecture.INTENSITY_INDEX_KEY: 0,
    tcnn_architecture.R34_INDEX_KEY: 1,
    tcnn_architecture.R50_INDEX_KEY: 2,
    tcnn_architecture.R64_INDEX_KEY: 3,
    tcnn_architecture.RMW_INDEX_KEY: 4,
    tcnn_architecture.USE_PHYSICAL_CONSTRAINTS_KEY: True,
    tcnn_architecture.DO_RESIDUAL_PREDICTION_KEY: False,
    tcnn_architecture.PREDICT_INTENSITY_ONLY_KEY: True
}

DROPOUT_RATES_AXIS1 = numpy.array([0.1, 0.2, 0.3, 0.4])
L2_WEIGHTS_AXIS2 = numpy.logspace(-9, -5, num=5, dtype=float)
USE_A_DECK_FLAGS_AXIS3 = numpy.array([0, 1], dtype=bool)


def _run():
    """Makes CNN templates for Intensity Experiment 2.

    This is effectively the main method.
    """

    for i in range(len(DROPOUT_RATES_AXIS1)):
        for j in range(len(L2_WEIGHTS_AXIS2)):
            for k in range(len(USE_A_DECK_FLAGS_AXIS3)):
                option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)
                input_dimensions = numpy.array([400, 400, 7, 3], dtype=int)

                pooling_size_by_conv_block_px = numpy.full(7, 2, dtype=int)
                num_channels_multipliers = numpy.array([1, 2, 3, 4, 5, 6, 7])
                num_pixels_coarsest = 36

                num_conv_blocks = len(pooling_size_by_conv_block_px)
                num_channels_by_conv_layer = numpy.round(
                    10 * num_channels_multipliers
                ).astype(int)

                num_flattened_neurons = (
                    int(USE_A_DECK_FLAGS_AXIS3[k]) * NUM_SCALAR_PREDICTORS +
                    num_pixels_coarsest * num_channels_by_conv_layer[-1]
                )

                dense_neuron_counts = (
                    architecture_utils.get_dense_layer_dimensions(
                        num_input_units=num_flattened_neurons,
                        num_classes=1,
                        num_dense_layers=NUM_DENSE_LAYERS,
                        for_classification=False
                    )[1]
                )

                dense_neuron_counts[-1] = ENSEMBLE_SIZE
                dense_neuron_counts[-2] = max([
                    dense_neuron_counts[-1], dense_neuron_counts[-2]
                ])
                
                dense_dropout_rates = numpy.concatenate([
                    numpy.full(3, DROPOUT_RATES_AXIS1[i]),
                    numpy.array([0.])
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
                    tcnn_architecture.NUM_NEURONS_KEY: dense_neuron_counts,
                    tcnn_architecture.DENSE_DROPOUT_RATES_KEY:
                        dense_dropout_rates,
                    tcnn_architecture.L2_WEIGHT_KEY: L2_WEIGHTS_AXIS2[j],
                    tcnn_architecture.INCLUDE_SCALAR_DATA_KEY:
                        USE_A_DECK_FLAGS_AXIS3[k],
                    tcnn_architecture.INPUT_DIMENSIONS_SCALAR_KEY: (
                        numpy.array([NUM_SCALAR_PREDICTORS], dtype=int)
                        if USE_A_DECK_FLAGS_AXIS3[k]
                        else None
                    )
                })

                model_object = tcnn_architecture.create_model_for_intensity(
                    option_dict
                )

                output_file_name = (
                    '{0:s}/dropout-rate={1:.1f}_l2-weight={2:.10f}_'
                    'use-a-decks={3:d}/model.h5'
                ).format(
                    OUTPUT_DIR_NAME,
                    DROPOUT_RATES_AXIS1[i],
                    L2_WEIGHTS_AXIS2[j],
                    int(USE_A_DECK_FLAGS_AXIS3[k])
                )

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

                option_dict[tcnn_architecture.LOSS_FUNCTION_KEY] = (
                    LOSS_FUNCTION_STRING
                )
                option_dict[tcnn_architecture.OPTIMIZER_FUNCTION_KEY] = (
                    OPTIMIZER_FUNCTION_STRING
                )

                neural_net_utils.write_metafile(
                    pickle_file_name=metafile_name,
                    num_epochs=100,
                    num_training_batches_per_epoch=32,
                    training_option_dict={
                        neural_net_utils.SEMANTIC_SEG_FLAG_KEY: False,
                        neural_net_utils.A_DECK_FILE_KEY: ''
                    },
                    num_validation_batches_per_epoch=16,
                    validation_option_dict={
                        neural_net_utils.SEMANTIC_SEG_FLAG_KEY: False,
                        neural_net_utils.A_DECK_FILE_KEY: ''
                    },
                    loss_function_string=LOSS_FUNCTION_STRING,
                    optimizer_function_string=OPTIMIZER_FUNCTION_STRING,
                    plateau_patience_epochs=10,
                    plateau_learning_rate_multiplier=0.6,
                    early_stopping_patience_epochs=50,
                    architecture_dict=option_dict,
                    is_model_bnn=False,
                    data_type_string=neural_net_utils.RG_SIMPLE_DATA_TYPE_STRING,
                    train_with_shuffled_data=True
                )


if __name__ == '__main__':
    _run()
