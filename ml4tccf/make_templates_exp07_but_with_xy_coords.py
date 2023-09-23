"""Makes templates for Experiment 7 with xy-coords as predictors."""

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
import cnn_architecture
import custom_losses_scalar
import accum_grad_optimizer

OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/'
    'experiment07_but_with_xy_coords/templates'
)

NUM_SCALAR_PREDICTORS = 9
NUM_CONV_BLOCKS = 6
ENSEMBLE_SIZE = 50
NUM_DENSE_LAYERS = 4
LOSS_FUNCTION_STRING = 'custom_losses_scalar.coord_avg_crps_kilometres'

DEFAULT_OPTION_DICT = {
    # cnn_architecture.INPUT_DIMENSIONS_LOW_RES_KEY:
    #     numpy.array([580, 900, 21], dtype=int),
    cnn_architecture.INPUT_DIMENSIONS_SCALAR_KEY:
        numpy.array([NUM_SCALAR_PREDICTORS], dtype=int),
    cnn_architecture.INCLUDE_HIGH_RES_KEY: False,
    cnn_architecture.INCLUDE_SCALAR_DATA_KEY: True,
    cnn_architecture.START_WITH_POOLING_KEY: True,
    cnn_architecture.NUM_CONV_LAYERS_KEY: numpy.full(
        NUM_CONV_BLOCKS, 1, dtype=int
    ),
    # cnn_architecture.NUM_CHANNELS_KEY: numpy.array(
    #     [16, 16, 24, 24, 32, 32, 40, 40, 48, 48, 56, 56, 64, 64], dtype=int
    # ),
    cnn_architecture.CONV_DROPOUT_RATES_KEY: numpy.full(NUM_CONV_BLOCKS, 0.),
    # cnn_architecture.NUM_NEURONS_KEY:
    #     numpy.array([1024, 128, 50, 50], dtype=int),
    cnn_architecture.DENSE_DROPOUT_RATES_KEY: numpy.array([0.5, 0.5, 0.5, 0]),
    cnn_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    cnn_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    cnn_architecture.L2_WEIGHT_KEY: 1e-6,
    cnn_architecture.USE_BATCH_NORM_KEY: True,
    cnn_architecture.ENSEMBLE_SIZE_KEY: ENSEMBLE_SIZE,
    cnn_architecture.LOSS_FUNCTION_KEY:
        custom_losses_scalar.coord_avg_crps_kilometres,
    # cnn_architecture.OPTIMIZER_FUNCTION_KEY:
    #     accum_grad_optimizer.convert_to_accumulate_gradient_optimizer(
    #         orig_optimizer=keras.optimizers.Adam(), update_params_frequency=5,
    #         accumulate_sum_or_mean=True
    #     )
}

FIRST_LAYER_FILTER_COUNTS = numpy.array([16, 24, 32, 40], dtype=int)
USE_XY_FLAGS = numpy.array([0, 1], dtype=bool)
BATCHES_PER_UPDATE_COUNTS = numpy.array([1, 2, 3], dtype=int)


def _run():
    """Makes templates for Experiment 7 with xy-coords as predictors.

    This is effectively the main method.
    """

    for i in range(len(FIRST_LAYER_FILTER_COUNTS)):
        for j in range(len(USE_XY_FLAGS)):
            for k in range(len(BATCHES_PER_UPDATE_COUNTS)):
                option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)

                num_input_channels = 3 * (3 + 2 * int(USE_XY_FLAGS[j]))
                input_dimensions = numpy.array(
                    [580, 900, num_input_channels], dtype=int
                )

                num_channels_by_conv_layer = numpy.linspace(
                    1, NUM_CONV_BLOCKS, num=NUM_CONV_BLOCKS, dtype=int
                ) * FIRST_LAYER_FILTER_COUNTS[i]

                (
                    dense_neuron_counts
                ) = architecture_utils.get_dense_layer_dimensions(
                    num_input_units=
                    NUM_SCALAR_PREDICTORS + 28 * num_channels_by_conv_layer[-1],
                    num_classes=2,
                    num_dense_layers=NUM_DENSE_LAYERS,
                    for_classification=False
                )[1]

                dense_neuron_counts[-1] = 2 * ENSEMBLE_SIZE
                dense_neuron_counts[-2] = max([
                    dense_neuron_counts[-1], dense_neuron_counts[-2]
                ])

                if BATCHES_PER_UPDATE_COUNTS[k] > 1:
                    optimizer_function = (
                        accum_grad_optimizer.convert_to_accumulate_gradient_optimizer(
                            orig_optimizer=keras.optimizers.Adam(),
                            update_params_frequency=BATCHES_PER_UPDATE_COUNTS[k],
                            accumulate_sum_or_mean=False
                        )
                    )

                    optimizer_string = (
                        'accum_grad_optimizer.convert_to_accumulate_gradient_optimizer('
                        'orig_optimizer=keras.optimizers.Adam(), '
                        'update_params_frequency={0:d}, '
                        'accumulate_sum_or_mean=False)'.format(
                            BATCHES_PER_UPDATE_COUNTS[k]
                        )
                    )
                else:
                    optimizer_function = keras.optimizers.Adam()
                    optimizer_string = 'keras.optimizers.Adam()'

                option_dict.update({
                    cnn_architecture.INPUT_DIMENSIONS_LOW_RES_KEY: input_dimensions,
                    cnn_architecture.NUM_CHANNELS_KEY: num_channels_by_conv_layer,
                    cnn_architecture.NUM_NEURONS_KEY: dense_neuron_counts,
                    cnn_architecture.OPTIMIZER_FUNCTION_KEY: optimizer_function
                })
                model_object = cnn_architecture.create_model(option_dict)

                output_file_name = (
                    '{0:s}/num-first-layer-filters={1:02d}_use-xy-coords={2:d}_'
                    'num-batches-per-update={3:d}/model.h5'
                ).format(
                    OUTPUT_DIR_NAME,
                    FIRST_LAYER_FILTER_COUNTS[i],
                    int(USE_XY_FLAGS[j]),
                    BATCHES_PER_UPDATE_COUNTS[k]
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

                option_dict[cnn_architecture.LOSS_FUNCTION_KEY] = (
                    LOSS_FUNCTION_STRING
                )
                option_dict[cnn_architecture.OPTIMIZER_FUNCTION_KEY] = (
                    optimizer_string
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
                    optimizer_function_string=optimizer_string,
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
