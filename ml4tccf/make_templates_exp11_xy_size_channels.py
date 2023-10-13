"""Makes templates for Experiment 11."""

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
    'experiment11_xy_size_channels/templates'
)

NUM_SCALAR_PREDICTORS = 5
# NUM_CONV_BLOCKS = 7
ENSEMBLE_SIZE = 50
NUM_DENSE_LAYERS = 4
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
    tcnn_architecture.DENSE_DROPOUT_RATES_KEY: numpy.array([0.5, 0.5, 0.5, 0]),
    tcnn_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    tcnn_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    tcnn_architecture.L2_WEIGHT_KEY: 1e-6,
    tcnn_architecture.USE_BATCH_NORM_KEY: True,
    tcnn_architecture.ENSEMBLE_SIZE_KEY: ENSEMBLE_SIZE,
    tcnn_architecture.LOSS_FUNCTION_KEY:
        custom_losses_scalar.coord_avg_crps_kilometres,
    tcnn_architecture.OPTIMIZER_FUNCTION_KEY: keras.optimizers.Adam()
}

GRID_ROW_COUNTS_AXIS1 = numpy.array([300, 300, 400, 500, 590], dtype=int)
GRID_ROW_COUNTS_AXIS1_DUMMY = ['300-pool-by-2', '300', '400', '500', '590']
MAIN_POOLING_SIZES_PX_AXIS1 = numpy.array([2, 3, 3, 3, 3], dtype=int)
LAG_TIME_COUNTS_AXIS2 = numpy.array([3, 4, 5, 6, 7], dtype=int)
FIRST_LAYER_FILTER_COUNTS_AXIS3 = numpy.array([10, 15, 20, 25], dtype=int)


def _run():
    """Makes templates for Experiment 11.

    This is effectively the main method.
    """

    for i in range(len(GRID_ROW_COUNTS_AXIS1)):
        for j in range(len(LAG_TIME_COUNTS_AXIS2)):
            for k in range(len(FIRST_LAYER_FILTER_COUNTS_AXIS3)):
                option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)
                input_dimensions = numpy.array([
                    GRID_ROW_COUNTS_AXIS1[i], GRID_ROW_COUNTS_AXIS1[i],
                    LAG_TIME_COUNTS_AXIS2[j], 3
                ], dtype=int)

                if GRID_ROW_COUNTS_AXIS1[i] == 300:
                    if MAIN_POOLING_SIZES_PX_AXIS1[i] == 2:
                        pooling_size_by_conv_block_px = numpy.full(
                            7, 2, dtype=int
                        )
                        num_pixels_coarsest = 16
                    else:
                        pooling_size_by_conv_block_px = numpy.array(
                            [2, 3, 3, 3, 2], dtype=int
                        )
                        num_pixels_coarsest = 25
                elif GRID_ROW_COUNTS_AXIS1[i] == 400:
                    pooling_size_by_conv_block_px = numpy.array(
                        [2, 3, 3, 3, 2], dtype=int
                    )
                    num_pixels_coarsest = 49
                else:
                    pooling_size_by_conv_block_px = numpy.array(
                        [2, 3, 3, 3, 2, 2], dtype=int
                    )

                    if GRID_ROW_COUNTS_AXIS1[i] == 500:
                        num_pixels_coarsest = 16
                    else:
                        num_pixels_coarsest = 25

                num_conv_blocks = len(pooling_size_by_conv_block_px)
                num_channels_by_conv_layer = numpy.linspace(
                    1, num_conv_blocks, num=num_conv_blocks, dtype=int
                ) * FIRST_LAYER_FILTER_COUNTS_AXIS3[k]

                (
                    dense_neuron_counts
                ) = architecture_utils.get_dense_layer_dimensions(
                    num_input_units=(
                        NUM_SCALAR_PREDICTORS +
                        num_pixels_coarsest * num_channels_by_conv_layer[-1]
                    ),
                    num_classes=2,
                    num_dense_layers=NUM_DENSE_LAYERS,
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

                model_object = tcnn_architecture.create_model(option_dict)

                output_file_name = (
                    '{0:s}/num-grid-rows={1:s}_num-lag-times={2:d}_'
                    'num-first-layer-filters={3:02d}/model.h5'
                ).format(
                    OUTPUT_DIR_NAME,
                    GRID_ROW_COUNTS_AXIS1_DUMMY[i],
                    LAG_TIME_COUNTS_AXIS2[j],
                    FIRST_LAYER_FILTER_COUNTS_AXIS3[k]
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
