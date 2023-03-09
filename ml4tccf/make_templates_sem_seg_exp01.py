"""Makes templates for first experiment with semantic segmentation."""

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
import u_net_architecture
import custom_losses_gridded
import neural_net

# TODO(thunderhoser): Still need to fuck with batch size, num channels, and
# regularization.
OPTIMIZER_FUNCTION_STRING = 'keras.optimizers.Adam()'

DEFAULT_OPTION_DICT = {
    # u_net_architecture.INPUT_DIMENSIONS_LOW_RES_KEY:
    #     numpy.array([600, 600, 1], dtype=int),
    u_net_architecture.INPUT_DIMENSIONS_HIGH_RES_KEY:
        numpy.array([5000, 5000, 3], dtype=int),
    u_net_architecture.INCLUDE_HIGH_RES_KEY: False,
    u_net_architecture.CONV_LAYER_COUNTS_KEY: numpy.full(9, 2, dtype=int),
    u_net_architecture.OUTPUT_CHANNEL_COUNTS_KEY: numpy.array(
        [16, 24, 32, 40, 48, 56, 64, 72, 80], dtype=int
    ),
    u_net_architecture.CONV_DROPOUT_RATES_KEY: [numpy.full(2, 0.)] * 9,
    u_net_architecture.UPCONV_DROPOUT_RATES_KEY: numpy.array([
        0, 0, 0, 0, 0, 0.25, 0.25, 0.25
    ]),
    u_net_architecture.SKIP_DROPOUT_RATES_KEY: [numpy.full(2, 0.)] * 8,
    u_net_architecture.INCLUDE_PENULTIMATE_KEY: True,
    u_net_architecture.PENULTIMATE_DROPOUT_RATE_KEY: 0.25,
    u_net_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    u_net_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    u_net_architecture.L2_WEIGHT_KEY: 1e-7,
    u_net_architecture.USE_BATCH_NORM_KEY: True,
    u_net_architecture.ENSEMBLE_SIZE_KEY: 1,
    # u_net_architecture.LOSS_FUNCTION_KEY:
    #     custom_losses_gridded.fractions_skill_score(
    #         half_window_size_px=10, use_as_loss_function=True,
    #         function_name='loss'
    #     ),
    u_net_architecture.OPTIMIZER_FUNCTION_KEY: keras.optimizers.Adam()
}

OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/'
    'semantic_segmentation_experiment01'
)

LAG_TIME_COUNTS = numpy.array([1, 2, 4, 6, 8, 10, 12], dtype=int)
LOSS_FUNCTION_SCORES = ['fss', 'gerrity', 'brier']
LOSS_FUNCTION_NEIGH_SIZES_PX = numpy.array([3, 5, 7, 9], dtype=int)


def _run():
    """Makes templates for first experiment with semantic segmentation.

    This is effectively the main method.
    """

    num_lag_time_counts = len(LAG_TIME_COUNTS)
    num_loss_function_scores = len(LOSS_FUNCTION_SCORES)
    num_loss_function_neighs = len(LOSS_FUNCTION_NEIGH_SIZES_PX)

    for i in range(num_lag_time_counts):
        for j in range(num_loss_function_scores):
            for k in range(num_loss_function_neighs):
                option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)

                option_dict[
                    u_net_architecture.INPUT_DIMENSIONS_LOW_RES_KEY
                ] = numpy.array([600, 600, LAG_TIME_COUNTS[i]], dtype=int)

                if LOSS_FUNCTION_SCORES[j] == 'fss':
                    loss_function = custom_losses_gridded.fractions_skill_score(
                        half_window_size_px=LOSS_FUNCTION_NEIGH_SIZES_PX[k],
                        use_as_loss_function=True, function_name='loss'
                    )
                    loss_function_string = (
                        'custom_losses_gridded.fractions_skill_score('
                        'half_window_size_px={0:d}, use_as_loss_function=True, '
                        'function_name="loss")'.format(
                            LOSS_FUNCTION_NEIGH_SIZES_PX[k]
                        )
                    )
                elif LOSS_FUNCTION_SCORES[j] == 'gerrity':
                    loss_function = custom_losses_gridded.gerrity_score(
                        half_window_size_px=LOSS_FUNCTION_NEIGH_SIZES_PX[k],
                        use_as_loss_function=True, function_name='loss'
                    )
                    loss_function_string = (
                        'custom_losses_gridded.gerrity_score('
                        'half_window_size_px={0:d}, use_as_loss_function=True, '
                        'function_name="loss")'.format(
                            LOSS_FUNCTION_NEIGH_SIZES_PX[k]
                        )
                    )
                else:
                    loss_function = custom_losses_gridded.brier_score(
                        half_window_size_px=LOSS_FUNCTION_NEIGH_SIZES_PX[k],
                        function_name='loss'
                    )
                    loss_function_string = (
                        'custom_losses_gridded.brier_score('
                        'half_window_size_px={0:d}, '
                        'function_name="loss")'.format(
                            LOSS_FUNCTION_NEIGH_SIZES_PX[k]
                        )
                    )

                option_dict[
                    u_net_architecture.LOSS_FUNCTION_KEY
                ] = loss_function

                model_object = u_net_architecture.create_model(option_dict)
                output_file_name = (
                    '{0:s}/num-lag-times={1:02d}_loss={2:s}{3:d}/model.h5'
                ).format(
                    OUTPUT_DIR_NAME, LAG_TIME_COUNTS[i],
                    LOSS_FUNCTION_SCORES[j], LOSS_FUNCTION_NEIGH_SIZES_PX[k]
                )

                file_system_utils.mkdir_recursive_if_necessary(
                    file_name=output_file_name
                )
                print('Writing model to: "{0:s}"...'.format(output_file_name))
                model_object.save(
                    filepath=output_file_name, overwrite=True,
                    include_optimizer=True
                )

                option_dict[u_net_architecture.LOSS_FUNCTION_KEY] = (
                    loss_function_string
                )
                option_dict[u_net_architecture.OPTIMIZER_FUNCTION_KEY] = (
                    OPTIMIZER_FUNCTION_STRING
                )

                metafile_name = neural_net.find_metafile(
                    model_dir_name=os.path.split(output_file_name)[0],
                    raise_error_if_missing=False
                )

                neural_net._write_metafile(
                    pickle_file_name=metafile_name,
                    num_epochs=1000,
                    num_training_batches_per_epoch=32,
                    training_option_dict=None,
                    num_validation_batches_per_epoch=16,
                    validation_option_dict=None,
                    loss_function_string=loss_function_string,
                    optimizer_function_string=OPTIMIZER_FUNCTION_STRING,
                    plateau_patience_epochs=10,
                    plateau_learning_rate_multiplier=0.6,
                    early_stopping_patience_epochs=50,
                    architecture_dict=option_dict,
                    is_model_bnn=False
                )


if __name__ == '__main__':
    _run()
