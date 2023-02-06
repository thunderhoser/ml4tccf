"""Makes CNN template -- TEST ONLY."""

import os
import sys
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import architecture_utils
import cnn_architecture

OPTION_DICT = {
    cnn_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([1250, 1250, 7], dtype=int),
    cnn_architecture.INCLUDE_HIGH_RES_KEY: True,
    cnn_architecture.NUM_CONV_LAYERS_KEY: numpy.full(11, 2, dtype=int),
    cnn_architecture.NUM_CHANNELS_KEY: numpy.array([
        2, 2, 4, 4, 16, 16, 32, 32, 64, 64, 96, 96, 128, 128, 184, 184,
        256, 256, 384, 384, 512, 512
    ], dtype=int),
    cnn_architecture.CONV_DROPOUT_RATES_KEY: numpy.full(22, 0.),
    cnn_architecture.NUM_NEURONS_KEY:
        numpy.array([1024, 128, 16, 2], dtype=int),
    cnn_architecture.DENSE_DROPOUT_RATES_KEY: numpy.array([0.5, 0.5, 0.5, 0]),
    cnn_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    cnn_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    cnn_architecture.L2_WEIGHT_KEY: 1e-6,
    cnn_architecture.USE_BATCH_NORM_KEY: True
}

OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/'
    'cnn_template_test'
)


def _run():
    """Makes CNN template -- TEST ONLY.

    This is effectively the main method.
    """

    model_object = cnn_architecture.create_model(OPTION_DICT)
    output_file_name = '{0:s}/model.h5'.format(OUTPUT_DIR_NAME)

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    print('Writing model to: "{0:s}"...'.format(output_file_name))
    model_object.save(
        filepath=output_file_name, overwrite=True, include_optimizer=True
    )


if __name__ == '__main__':
    _run()
