"""Makes templates for Experiment 12."""

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
import accum_grad_optimizer

OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/'
    'experiment12_regularization/templates'
)

NUM_SCALAR_PREDICTORS = 5
# NUM_CONV_BLOCKS = 7
ENSEMBLE_SIZE = 50
OPTIMIZER_STRING = 'keras.optimizers.Adam()'
LOSS_FUNCTION_STRING = 'custom_losses_scalar.coord_avg_crps_kilometres'

DEFAULT_OPTION_DICT = {
    # tcnn_architecture.INPUT_DIMENSIONS_LOW_RES_KEY:
    #     numpy.array([580, 900, 21], dtype=int),
    tcnn_architecture.INPUT_DIMENSIONS_SCALAR_KEY:
        numpy.array([NUM_SCALAR_PREDICTORS], dtype=int),
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
    # tcnn_architecture.DENSE_DROPOUT_RATES_KEY: numpy.array([0.5, 0.5, 0.5, 0]),
    tcnn_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    tcnn_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    # tcnn_architecture.L2_WEIGHT_KEY: 1e-6,
    tcnn_architecture.USE_BATCH_NORM_KEY: True,
    tcnn_architecture.ENSEMBLE_SIZE_KEY: ENSEMBLE_SIZE,
    tcnn_architecture.LOSS_FUNCTION_KEY:
        custom_losses_scalar.coord_avg_crps_kilometres,
    tcnn_architecture.OPTIMIZER_FUNCTION_KEY: keras.optimizers.Adam()
}

DENSE_LAYER_COUNTS_AXIS1 = numpy.array([3, 4, 5, 6], dtype=int)
DENSE_LAYER_DROPOUT_RATES_AXIS2 = numpy.array([0.125, 0.250, 0.375, 0.500])
L2_WEIGHTS_AXIS3 = numpy.power(10., numpy.array([-7.0, -6.5, -6.0, -5.5, -5.0]))


def _run():
    """Makes templates for Experiment 12.

    This is effectively the main method.
    """

    for i in range(len(DENSE_LAYER_COUNTS_AXIS1)):
        for j in range(len(DENSE_LAYER_DROPOUT_RATES_AXIS2)):
            for k in range(len(L2_WEIGHTS_AXIS3)):
                option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)
                input_dimensions = numpy.array([300, 300, 6, 3], dtype=int)

                pooling_size_by_conv_block_px = numpy.full(
                    7, 2, dtype=int
                )
                num_pixels_coarsest = 16

                num_conv_blocks = len(pooling_size_by_conv_block_px)
                num_channels_by_conv_layer = numpy.linspace(
                    1, num_conv_blocks, num=num_conv_blocks, dtype=int
                ) * 10

                num_dense_layers = DENSE_LAYER_COUNTS_AXIS1[i]

                dense_neuron_counts = (
                    architecture_utils.get_dense_layer_dimensions(
                        num_input_units=(
                            NUM_SCALAR_PREDICTORS +
                            num_pixels_coarsest * num_channels_by_conv_layer[-1]
                        ),
                        num_classes=2,
                        num_dense_layers=num_dense_layers,
                        for_classification=False
                    )[1]
                )

                dense_neuron_counts[-1] = 2 * ENSEMBLE_SIZE
                dense_neuron_counts[-2] = max([
                    dense_neuron_counts[-1], dense_neuron_counts[-2]
                ])

                dense_layer_dropout_rates = numpy.full(
                    num_dense_layers, DENSE_LAYER_DROPOUT_RATES_AXIS2[j]
                )
                dense_layer_dropout_rates[-1] = 0.

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
                        dense_layer_dropout_rates,
                    tcnn_architecture.L2_WEIGHT_KEY: L2_WEIGHTS_AXIS3[k]
                })

                model_object = tcnn_architecture.create_model(option_dict)

                output_file_name = (
                    '{0:s}/num-dense-layers={1:d}_dense-dropout-rate={2:.3f}_'
                    'l2-weight={3:.10f}/model.h5'
                ).format(
                    OUTPUT_DIR_NAME,
                    DENSE_LAYER_COUNTS_AXIS1[i],
                    DENSE_LAYER_DROPOUT_RATES_AXIS2[j],
                    L2_WEIGHTS_AXIS3[k]
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
                    OPTIMIZER_STRING
                )

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
                    architecture_dict=option_dict,
                    is_model_bnn=False,
                    data_type_string=neural_net_utils.RG_SIMPLE_DATA_TYPE_STRING,
                    train_with_shuffled_data=True
                )


if __name__ == '__main__':
    _run()
