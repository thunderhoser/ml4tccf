"""Makes Bayesian CNN template -- TEST ONLY."""

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
import cnn_architecture_bayesian as bcnn_architecture
import custom_losses

DEFAULT_OPTION_DICT = {
    bcnn_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([1250, 1250, 7], dtype=int),
    bcnn_architecture.INCLUDE_HIGH_RES_KEY: True,
    bcnn_architecture.NUM_CONV_LAYERS_KEY: numpy.full(11, 2, dtype=int),
    bcnn_architecture.NUM_CHANNELS_KEY: numpy.array([
        2, 2, 4, 4, 16, 16, 32, 32, 64, 64, 96, 96, 128, 128, 184, 184,
        256, 256, 384, 384, 512, 512
    ], dtype=int),
    bcnn_architecture.CONV_DROPOUT_RATES_KEY: numpy.full(22, 0.),
    bcnn_architecture.NUM_NEURONS_KEY:
        numpy.array([1024, 128, 50, 50], dtype=int),
    bcnn_architecture.DENSE_DROPOUT_RATES_KEY: numpy.array([0.5, 0.5, 0.5, 0]),
    bcnn_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    bcnn_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    bcnn_architecture.L2_WEIGHT_KEY: 1e-6,
    bcnn_architecture.USE_BATCH_NORM_KEY: True,
    bcnn_architecture.ENSEMBLE_SIZE_KEY: 50,
    bcnn_architecture.LOSS_FUNCTION_KEY: custom_losses.crps_kilometres()
}

OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/'
    'bcnn_template_test'
)


def _run():
    """Makes Bayesian CNN template -- TEST ONLY.

    This is effectively the main method.
    """

    option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)
    option_dict[bcnn_architecture.KL_SCALING_FACTOR_KEY] = 1e-5
    option_dict[bcnn_architecture.CONV_LAYER_TYPES_KEY] = (
        [bcnn_architecture.POINT_ESTIMATE_TYPE_STRING] * 18 +
        [
            bcnn_architecture.FLIPOUT_TYPE_STRING,
            bcnn_architecture.REPARAMETERIZATION_TYPE_STRING
        ] * 2
    )
    option_dict[bcnn_architecture.DENSE_LAYER_TYPES_KEY] = [
        bcnn_architecture.POINT_ESTIMATE_TYPE_STRING,
        bcnn_architecture.FLIPOUT_TYPE_STRING,
        bcnn_architecture.REPARAMETERIZATION_TYPE_STRING,
        bcnn_architecture.FLIPOUT_TYPE_STRING
    ]

    model_object = bcnn_architecture.create_model(option_dict)
    output_file_name = '{0:s}/model.h5'.format(OUTPUT_DIR_NAME)

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    print('Writing model to: "{0:s}"...'.format(output_file_name))
    model_object.save(
        filepath=output_file_name, overwrite=True, include_optimizer=True
    )


if __name__ == '__main__':
    _run()
