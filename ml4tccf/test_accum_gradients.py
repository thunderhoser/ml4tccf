"""Tests optimizer with 'accumulated gradients' over many batches."""

import os
import sys
import copy
import numpy
import tensorflow
from tensorflow.python.keras import backend as K
import tensorflow.keras as tf_keras

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import architecture_utils
import neural_net
import cnn_architecture
import custom_losses

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


def convert_to_accumulate_gradient_optimizer(orig_optimizer, update_params_frequency, accumulate_sum_or_mean=True):
    if update_params_frequency < 1:
        raise ValueError('update_params_frequency must be >= 1')
    print('update_params_frequency: %s' % update_params_frequency)
    print('accumulate_sum_or_mean: %s' % accumulate_sum_or_mean)
    orig_get_gradients = orig_optimizer.get_gradients
    orig_get_updates = orig_optimizer.get_updates
    accumulated_iterations = K.variable(0, dtype='int64', name='accumulated_iterations')
    orig_optimizer.accumulated_iterations = accumulated_iterations

    def updated_get_gradients(self, loss, params):
        return self.accumulate_gradient_accumulators

    def updated_get_updates(self, loss, params):
        self.accumulate_gradient_accumulators = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        updates_accumulated_iterations = K.update_add(accumulated_iterations, 1)
        new_grads = orig_get_gradients(loss, params)
        if not accumulate_sum_or_mean:
            new_grads = [g / K.cast(update_params_frequency, K.dtype(g)) for g in new_grads]
        self.updated_grads = [K.update_add(p, g) for p, g in zip(self.accumulate_gradient_accumulators, new_grads)]
        def update_function():
            with tensorflow.control_dependencies(orig_get_updates(loss, params)):
                reset_grads = [K.update(p, K.zeros(K.int_shape(p), dtype=K.dtype(p))) for p in self.accumulate_gradient_accumulators]
            return tensorflow.group(*(reset_grads + [updates_accumulated_iterations]))
        def just_store_function():
            return tensorflow.group(*[updates_accumulated_iterations])

        update_switch = K.equal((updates_accumulated_iterations) % update_params_frequency, 0)

        with tensorflow.control_dependencies(self.updated_grads):
            self.updates = [K.switch(update_switch, update_function, just_store_function)]
            return self.updates

    orig_optimizer.get_gradients = updated_get_gradients.__get__(orig_optimizer, type(orig_optimizer))
    orig_optimizer.get_updates = updated_get_updates.__get__(orig_optimizer, type(orig_optimizer))
    return orig_optimizer


if __name__ == '__main__':
    # opt = AdamAccumulate(lr=0.001, decay=1e-5, accum_iters=5)
    print(tf_keras.optimizers.Adam())
    opt = convert_to_accumulate_gradient_optimizer(orig_optimizer=tf_keras.optimizers.Adam(), update_params_frequency=5, accumulate_sum_or_mean=True)
    # opt = Adam()
    DEFAULT_OPTION_DICT[cnn_architecture.OPTIMIZER_FUNCTION_KEY] = opt

    i = 0
    j = 0
    k = 0
    m = 0

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

    model_object = cnn_architecture.create_model(
        option_dict=option_dict
    )

    training_option_dict = {
        neural_net.SATELLITE_DIRECTORY_KEY: '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_project/satellite_data/processed/normalized_params_from_2017-2019',
        neural_net.YEARS_KEY: numpy.array([2017, 2018, 2019], dtype=int),
        neural_net.LAG_TIMES_KEY: numpy.array([0], dtype=int),
        neural_net.HIGH_RES_WAVELENGTHS_KEY: numpy.array([]),
        neural_net.LOW_RES_WAVELENGTHS_KEY: numpy.array([11.2]),
        neural_net.BATCH_SIZE_KEY: 8,
        neural_net.MAX_EXAMPLES_PER_CYCLONE_KEY: 2,
        neural_net.NUM_GRID_ROWS_KEY: 600,
        neural_net.NUM_GRID_COLUMNS_KEY: 600,
        neural_net.DATA_AUG_NUM_TRANS_KEY: 8,
        neural_net.DATA_AUG_MEAN_TRANS_KEY: 15.,
        neural_net.DATA_AUG_STDEV_TRANS_KEY: 7.5,
        neural_net.LAG_TIME_TOLERANCE_KEY: 900,
        neural_net.MAX_MISSING_LAG_TIMES_KEY: 1,
        neural_net.MAX_INTERP_GAP_KEY: 3600,
        neural_net.SENTINEL_VALUE_KEY: -10.
    }

    validation_option_dict = {
        neural_net.SATELLITE_DIRECTORY_KEY: '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_project/satellite_data/processed/normalized_params_from_2017-2019',
        neural_net.YEARS_KEY: numpy.array([2020], dtype=int),
        neural_net.LAG_TIME_TOLERANCE_KEY: 900,
        neural_net.MAX_MISSING_LAG_TIMES_KEY: 1,
        neural_net.MAX_INTERP_GAP_KEY: 3600
    }

    neural_net.train_model(
        model_object=model_object, output_dir_name='/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/test_accum_gradients',
        num_epochs=1000,
        num_training_batches_per_epoch=64,
        training_option_dict=training_option_dict,
        num_validation_batches_per_epoch=32,
        validation_option_dict=validation_option_dict,
        loss_function_string='custom_losses.discretized_mean_sq_dist_kilometres2',
        optimizer_function_string='convert_to_accumulate_gradient_optimizer(orig_optimizer=keras.optimizers.Adam(), update_params_frequency=5, accumulate_sum_or_mean=True)',
        plateau_patience_epochs=10,
        plateau_learning_rate_multiplier=0.6,
        early_stopping_patience_epochs=50,
        architecture_dict=option_dict,
        is_model_bnn=False
    )