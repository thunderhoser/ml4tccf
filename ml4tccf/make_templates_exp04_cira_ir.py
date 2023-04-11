"""Makes templates for Experiment 4 (CIRA IR data)."""

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
import neural_net
import cnn_architecture
import custom_losses_scalar
import accum_grad_optimizer

OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/'
    'experiment04_cira_ir/templates'
)

NUM_CONV_BLOCKS = 7
ENSEMBLE_SIZE = 100
NUM_DENSE_LAYERS = 4
LOSS_FUNCTION_STRING = 'custom_losses_scalar.coord_avg_crps_kilometres'

DEFAULT_OPTION_DICT = {
    cnn_architecture.INPUT_DIMENSIONS_LOW_RES_KEY:
        numpy.array([290, 450, 5], dtype=int),
    cnn_architecture.INCLUDE_HIGH_RES_KEY: False,
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

FIRST_LAYER_CHANNEL_COUNTS = numpy.array([8, 12, 16, 20, 24], dtype=int)
BATCHES_PER_UPDATE_COUNTS = numpy.array([1, 2, 3, 4, 5], dtype=int)


def _run():
    """Makes templates for Experiment 4 (CIRA IR data).

    This is effectively the main method.
    """

    num_channel_counts = len(FIRST_LAYER_CHANNEL_COUNTS)
    num_batch_counts = len(BATCHES_PER_UPDATE_COUNTS)

    for i in range(num_channel_counts):
        for j in range(num_batch_counts):
            option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)

            num_channels_by_conv_layer = (
                FIRST_LAYER_CHANNEL_COUNTS[i] *
                numpy.linspace(
                    1, NUM_CONV_BLOCKS, num=NUM_CONV_BLOCKS, dtype=int
                )
            )

            dense_neuron_counts = (
                architecture_utils.get_dense_layer_dimensions(
                    num_input_units=
                    4 * 7 * num_channels_by_conv_layer[-1],
                    num_classes=2,
                    num_dense_layers=NUM_DENSE_LAYERS, for_classification=False
                )[1]
            )

            dense_neuron_counts[-1] = 2 * ENSEMBLE_SIZE
            dense_neuron_counts[-2] = max([
                dense_neuron_counts[-1], dense_neuron_counts[-2]
            ])

            if BATCHES_PER_UPDATE_COUNTS[j] > 1:
                optimizer_function = (
                    accum_grad_optimizer.convert_to_accumulate_gradient_optimizer(
                        orig_optimizer=keras.optimizers.Adam(),
                        update_params_frequency=BATCHES_PER_UPDATE_COUNTS[j],
                        accumulate_sum_or_mean=False
                    )
                )

                optimizer_string = (
                    'accum_grad_optimizer.convert_to_accumulate_gradient_optimizer('
                    'orig_optimizer=keras.optimizers.Adam(), '
                    'update_params_frequency={0:d}, '
                    'accumulate_sum_or_mean=False)'.format(
                        BATCHES_PER_UPDATE_COUNTS[j]
                    )
                )
            else:
                optimizer_function = keras.optimizers.Adam()
                optimizer_string = 'keras.optimizers.Adam()'

            option_dict.update({
                cnn_architecture.NUM_CHANNELS_KEY: num_channels_by_conv_layer,
                cnn_architecture.NUM_NEURONS_KEY: dense_neuron_counts,
                cnn_architecture.OPTIMIZER_FUNCTION_KEY: optimizer_function
            })

            model_object = cnn_architecture.create_model(option_dict)
            output_file_name = (
                '{0:s}/num-first-layer-channels={1:02d}_'
                'num-batches-per-update={2:02d}/model.h5'
            ).format(
                OUTPUT_DIR_NAME, FIRST_LAYER_CHANNEL_COUNTS[i],
                BATCHES_PER_UPDATE_COUNTS[j]
            )

            file_system_utils.mkdir_recursive_if_necessary(
                file_name=output_file_name
            )

            print('Writing model to: "{0:s}"...'.format(output_file_name))
            model_object.save(
                filepath=output_file_name,
                overwrite=True, include_optimizer=True
            )

            metafile_name = neural_net.find_metafile(
                model_dir_name=os.path.split(output_file_name)[0],
                raise_error_if_missing=False
            )

            option_dict[cnn_architecture.LOSS_FUNCTION_KEY] = (
                LOSS_FUNCTION_STRING
            )
            option_dict[cnn_architecture.OPTIMIZER_FUNCTION_KEY] = (
                optimizer_string
            )

            neural_net._write_metafile(
                pickle_file_name=metafile_name,
                num_epochs=100,
                num_training_batches_per_epoch=32,
                training_option_dict=None,
                num_validation_batches_per_epoch=16,
                validation_option_dict=None,
                loss_function_string=LOSS_FUNCTION_STRING,
                optimizer_function_string=optimizer_string,
                plateau_patience_epochs=10,
                plateau_learning_rate_multiplier=0.6,
                early_stopping_patience_epochs=50,
                architecture_dict=option_dict,
                is_model_bnn=False,
                use_cira_ir_data=True
            )


if __name__ == '__main__':
    _run()
