"""Makes CNN template for fastfast version of Structure Experiment 1."""

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

OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/'
    'structure_experiment01_fastfast/template'
)

NUM_SCALAR_PREDICTORS = 9
ENSEMBLE_SIZE = 50

OPTIMIZER_FUNCTION = keras.optimizers.Nadam(
    gradient_accumulation_steps=20, clipnorm=1.0
)
OPTIMIZER_FUNCTION_STRING = (
    'keras.optimizers.Nadam('
    'gradient_accumulation_steps=20, clipnorm=1.0'
    ')'
)

# Intensity (kt), R34 (km), R50 (km), R64 (km), RMW (km)
CHANNEL_WEIGHTS = numpy.array(
    [0.22710066, 0.01812499, 0.10875182, 0.5450069, 0.10101562]
)
LOSS_FUNCTION = custom_losses_scalar.constrained_dwcrps_for_structure_params(
    channel_weights=CHANNEL_WEIGHTS,
    intensity_index=0, r34_index=1, r50_index=2, r64_index=3, rmw_index=4,
    function_name='loss_constrained_dwcrps'
)
LOSS_FUNCTION_STRING = (
    'custom_losses_scalar.constrained_dwcrps_for_structure_params('
    'channel_weights=numpy.array([0.22710066, 0.01812499, 0.10875182, 0.5450069, 0.10101562]), '
    'intensity_index=0, r34_index=1, r50_index=2, r64_index=3, rmw_index=4, '
    'function_name="loss_constrained_dwcrps"'
    ')'
)

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
    tcnn_architecture.LOSS_FUNCTION_KEY: LOSS_FUNCTION,
    tcnn_architecture.OPTIMIZER_FUNCTION_KEY: OPTIMIZER_FUNCTION,
    tcnn_architecture.INTENSITY_INDEX_KEY: 0,
    tcnn_architecture.R34_INDEX_KEY: 1,
    tcnn_architecture.R50_INDEX_KEY: 2,
    tcnn_architecture.R64_INDEX_KEY: 3,
    tcnn_architecture.RMW_INDEX_KEY: 4,
    tcnn_architecture.USE_PHYSICAL_CONSTRAINTS_KEY: False
}


def _run():
    """Makes CNN template for fastfast version of Structure Experiment 1.

    This is effectively the main method.
    """

    option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)
    input_dimensions = numpy.array([800, 800, 7, 3], dtype=int)

    pooling_size_by_conv_block_px = numpy.full(
        8, 2, dtype=int
    )
    num_channels_multipliers = numpy.array([
        1, 2, 3, 4, 5, 6, 7, 8
    ])
    num_pixels_coarsest = 36

    num_conv_blocks = len(pooling_size_by_conv_block_px)
    num_channels_by_conv_layer = numpy.round(
        10 * num_channels_multipliers
    ).astype(int)

    num_dense_layers = len(
        option_dict[tcnn_architecture.DENSE_DROPOUT_RATES_KEY]
    )

    dense_neuron_counts = (
        architecture_utils.get_dense_layer_dimensions(
            num_input_units=(
                NUM_SCALAR_PREDICTORS +
                num_pixels_coarsest * num_channels_by_conv_layer[-1]
            ),
            num_classes=5,
            num_dense_layers=num_dense_layers,
            for_classification=False
        )[1]
    )

    dense_neuron_counts[-1] = 5 * ENSEMBLE_SIZE
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

    model_object = tcnn_architecture.create_model_for_structure(option_dict)

    output_file_name = '{0:s}/model.h5'.format(OUTPUT_DIR_NAME)
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

    neural_net_utils.read_model(output_file_name)


if __name__ == '__main__':
    _run()
