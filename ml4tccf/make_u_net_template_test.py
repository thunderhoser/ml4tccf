"""Makes U-net template -- TEST ONLY."""

import os
import sys
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import architecture_utils
import u_net_architecture
import custom_losses_gridded

LOSS_FUNCTION_STRING = (
    'custom_losses_gridded.fractions_skill_score('
    'half_window_size_px=0., use_as_loss_function=True, function_name="loss"'
    ')'
)
OPTIMIZER_FUNCTION_STRING = 'keras.optimizers.Adam()'

DEFAULT_OPTION_DICT = {
    u_net_architecture.INPUT_DIMENSIONS_LOW_RES_KEY:
        numpy.array([1250, 1250, 21], dtype=int),
    u_net_architecture.INPUT_DIMENSIONS_HIGH_RES_KEY:
        numpy.array([5000, 5000, 3], dtype=int),
    u_net_architecture.INCLUDE_HIGH_RES_KEY: True,
    u_net_architecture.CONV_LAYER_COUNTS_KEY: numpy.full(11, 2, dtype=int),
    u_net_architecture.OUTPUT_CHANNEL_COUNTS_KEY: numpy.array(
        [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176 ], dtype=int
    ),
    u_net_architecture.CONV_DROPOUT_RATES_KEY: [numpy.full(2, 0.)] * 11,
    u_net_architecture.UPCONV_DROPOUT_RATES_KEY: numpy.array([
        0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5
    ]),
    u_net_architecture.SKIP_DROPOUT_RATES_KEY: [numpy.full(2, 0.)] * 10,
    u_net_architecture.INCLUDE_PENULTIMATE_KEY: False,
    u_net_architecture.PENULTIMATE_DROPOUT_RATE_KEY: 0.,
    u_net_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    u_net_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    u_net_architecture.L2_WEIGHT_KEY: 1e-6,
    u_net_architecture.USE_BATCH_NORM_KEY: True,
    u_net_architecture.ENSEMBLE_SIZE_KEY: 50,
    u_net_architecture.LOSS_FUNCTION_KEY:
        custom_losses_gridded.fractions_skill_score(
            half_window_size_px=0., use_as_loss_function=True,
            function_name='loss'
        ),
    u_net_architecture.OPTIMIZER_FUNCTION_KEY: keras.optimizers.Adam()
}

OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/'
    'u_net_template_test'
)


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


if __name__ == '__main__':
    _run()
