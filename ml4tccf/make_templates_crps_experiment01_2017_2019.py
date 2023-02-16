"""Makes templates for first experiment with CRPS, training on 2017-19."""

import os
import sys
import copy
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import architecture_utils
import neural_net
import cnn_architecture
import custom_losses

OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/'
    'crps_experiment01_2017-2019/templates'
)

NUM_CONV_BLOCKS = 8
# ENSEMBLE_SIZE = 25
ENSEMBLE_SIZE = 5

DEFAULT_OPTION_DICT = {
    cnn_architecture.INPUT_DIMENSIONS_LOW_RES_KEY:
        numpy.array([600, 600, 1], dtype=int),
    cnn_architecture.INCLUDE_HIGH_RES_KEY: False,
    # cnn_architecture.NUM_CONV_LAYERS_KEY: numpy.full(7, 2, dtype=int),
    # cnn_architecture.NUM_CHANNELS_KEY: numpy.array(
    #     [16, 16, 24, 24, 32, 32, 40, 40, 48, 48, 56, 56, 64, 64], dtype=int
    # ),
    # cnn_architecture.CONV_DROPOUT_RATES_KEY: numpy.full(14, 0.),
    # cnn_architecture.NUM_NEURONS_KEY:
    #     numpy.array([1024, 128, 50, 50], dtype=int),
    # cnn_architecture.DENSE_DROPOUT_RATES_KEY: numpy.array([0.25, 0.25, 0.25, 0]),
    cnn_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    cnn_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    # cnn_architecture.L2_WEIGHT_KEY: 1e-7,
    cnn_architecture.USE_BATCH_NORM_KEY: True,
    cnn_architecture.ENSEMBLE_SIZE_KEY: ENSEMBLE_SIZE,
    cnn_architecture.LOSS_FUNCTION_KEY:
        custom_losses.discretized_mean_sq_dist_kilometres2
}

DENSE_LAYER_DROPOUT_RATES = numpy.array([0.2, 0.3, 0.4, 0.5])
DENSE_LAYER_COUNTS = numpy.array([2, 3, 4], dtype=int)
CONV_LAYER_L2_WEIGHTS = numpy.logspace(-7, -5, num=5)
CONV_LAYER_BY_BLOCK_COUNTS = numpy.array([1, 2], dtype=int)


def _run():
    """Makes templates for first experiment with CRPS, training on 2017-19.

    This is effectively the main method.
    """

    num_dropout_rates = len(DENSE_LAYER_DROPOUT_RATES)
    num_dense_layer_counts = len(DENSE_LAYER_COUNTS)
    num_l2_weights = len(CONV_LAYER_L2_WEIGHTS)
    num_conv_layer_counts = len(CONV_LAYER_BY_BLOCK_COUNTS)

    for i in range(num_dropout_rates):
        for j in range(num_dense_layer_counts):
            for k in range(num_l2_weights):
                for m in range(num_conv_layer_counts):
                    option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)

                    dense_dropout_rates = numpy.full(
                        DENSE_LAYER_COUNTS[j], DENSE_LAYER_DROPOUT_RATES[i]
                    )
                    dense_dropout_rates[-1] = 0.

                    num_conv_layers_by_block = numpy.full(
                        NUM_CONV_BLOCKS, CONV_LAYER_BY_BLOCK_COUNTS[m],
                        dtype=int
                    )

                    num_channels_by_conv_layer = numpy.array(
                        [8, 16, 24, 32, 40, 48, 56, 64], dtype=int
                    )
                    num_channels_by_conv_layer = numpy.ravel(numpy.repeat(
                        numpy.expand_dims(num_channels_by_conv_layer, axis=1),
                        repeats=CONV_LAYER_BY_BLOCK_COUNTS[m],
                        axis=1
                    ))

                    conv_dropout_rates = numpy.full(
                        len(num_channels_by_conv_layer), 0.
                    )

                    dense_neuron_counts = (
                        architecture_utils.get_dense_layer_dimensions(
                            num_input_units=
                            4 * 4 * num_channels_by_conv_layer[-1],
                            num_classes=2,
                            num_dense_layers=DENSE_LAYER_COUNTS[j],
                            for_classification=False
                        )[1]
                    )

                    dense_neuron_counts[-1] = 2 * ENSEMBLE_SIZE
                    dense_neuron_counts[-2] = max([
                        dense_neuron_counts[-1], dense_neuron_counts[-2]
                    ])

                    option_dict.update({
                        cnn_architecture.NUM_CONV_LAYERS_KEY:
                            num_conv_layers_by_block,
                        cnn_architecture.NUM_CHANNELS_KEY:
                            num_channels_by_conv_layer,
                        cnn_architecture.CONV_DROPOUT_RATES_KEY:
                            conv_dropout_rates,
                        cnn_architecture.NUM_NEURONS_KEY: dense_neuron_counts,
                        cnn_architecture.DENSE_DROPOUT_RATES_KEY:
                            dense_dropout_rates,
                        cnn_architecture.L2_WEIGHT_KEY: CONV_LAYER_L2_WEIGHTS[k]
                    })

                    model_object = cnn_architecture.create_model(option_dict)
                    output_file_name = (
                        '{0:s}/num-dense-layers={1:d}_'
                        'dense-dropout-rate={2:.1f}_'
                        'num-conv-layers-per-block={3:d}_l2-weight={4:.10f}/'
                        'model.h5'
                    ).format(
                        OUTPUT_DIR_NAME,
                        DENSE_LAYER_COUNTS[j],
                        DENSE_LAYER_DROPOUT_RATES[i],
                        CONV_LAYER_BY_BLOCK_COUNTS[m],
                        CONV_LAYER_L2_WEIGHTS[k]
                    )

                    file_system_utils.mkdir_recursive_if_necessary(
                        file_name=output_file_name
                    )

                    print('Writing model to: "{0:s}"...'.format(
                        output_file_name
                    ))
                    model_object.save(
                        filepath=output_file_name,
                        overwrite=True, include_optimizer=True
                    )

                    metafile_name = neural_net.find_metafile(
                        model_dir_name=os.path.split(output_file_name)[0],
                        raise_error_if_missing=False
                    )

                    neural_net._write_metafile(
                        pickle_file_name=metafile_name,
                        num_epochs=100,
                        num_training_batches_per_epoch=32,
                        training_option_dict=None,
                        num_validation_batches_per_epoch=16,
                        validation_option_dict=None,
                        loss_function_string=
                        'custom_losses.discretized_mean_sq_dist_kilometres2',
                        plateau_patience_epochs=10,
                        plateau_learning_rate_multiplier=0.6,
                        early_stopping_patience_epochs=50,
                        bnn_architecture_dict=None
                    )


if __name__ == '__main__':
    _run()
