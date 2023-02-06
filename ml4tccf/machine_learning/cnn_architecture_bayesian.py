"""Methods for creating Bayesian CNN architecture."""

import numpy
import keras
import keras.layers
import tensorflow_probability as tf_prob
from tensorflow_probability.python.distributions import \
    kullback_leibler as kl_lib
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import architecture_utils
from ml4tccf.machine_learning import cnn_architecture

POINT_ESTIMATE_TYPE_STRING = 'point_estimate'
FLIPOUT_TYPE_STRING = 'flipout'
REPARAMETERIZATION_TYPE_STRING = 'reparameterization'
VALID_CONV_LAYER_TYPE_STRINGS = [
    POINT_ESTIMATE_TYPE_STRING, FLIPOUT_TYPE_STRING,
    REPARAMETERIZATION_TYPE_STRING
]

INPUT_DIMENSIONS_KEY = cnn_architecture.INPUT_DIMENSIONS_KEY
INCLUDE_HIGH_RES_KEY = cnn_architecture.INCLUDE_HIGH_RES_KEY
NUM_CONV_LAYERS_KEY = cnn_architecture.NUM_CONV_LAYERS_KEY
NUM_CHANNELS_KEY = cnn_architecture.NUM_CHANNELS_KEY
CONV_DROPOUT_RATES_KEY = cnn_architecture.CONV_DROPOUT_RATES_KEY
NUM_NEURONS_KEY = cnn_architecture.NUM_NEURONS_KEY
DENSE_DROPOUT_RATES_KEY = cnn_architecture.DENSE_DROPOUT_RATES_KEY
INNER_ACTIV_FUNCTION_KEY = cnn_architecture.INNER_ACTIV_FUNCTION_KEY
INNER_ACTIV_FUNCTION_ALPHA_KEY = cnn_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY
L2_WEIGHT_KEY = cnn_architecture.L2_WEIGHT_KEY
USE_BATCH_NORM_KEY = cnn_architecture.USE_BATCH_NORM_KEY
ENSEMBLE_SIZE_KEY = cnn_architecture.ENSEMBLE_SIZE_KEY
LOSS_FUNCTION_KEY = cnn_architecture.LOSS_FUNCTION_KEY

KL_SCALING_FACTOR_KEY = 'kl_divergence_scaling_factor'
CONV_LAYER_TYPES_KEY = 'conv_layer_type_strings'
DENSE_LAYER_TYPES_KEY = 'dense_layer_type_strings'


def _check_layer_type(layer_type_string):
    """Ensures that convolutional-layer type is valid.

    :param layer_type_string: Layer type (must be in list
        VALID_CONV_LAYER_TYPE_STRINGS).
    :raises ValueError: if
        `layer_type_string not in VALID_CONV_LAYER_TYPE_STRINGS`.
    """

    error_checking.assert_is_string(layer_type_string)

    if layer_type_string not in VALID_CONV_LAYER_TYPE_STRINGS:
        error_string = (
            'Valid conv-layer types (listed below) do not include "{0:s}":'
            '\n{1:s}'
        ).format(layer_type_string, str(VALID_CONV_LAYER_TYPE_STRINGS))

        raise ValueError(error_string)


def _check_input_args(option_dict):
    """Error-checks input arguments.

    C = number of convolutional layers
    D = number of dense layers

    :param option_dict: Dictionary with keys listed in the documentation for
        `cnn_architecture.check_input_args`, plus the following additional keys.
    option_dict["kl_divergence_scaling_factor"]: Scaling factor for KL loss
        (Kullback-Leibler divergence) in Bayesian layers.
    option_dict["conv_layer_type_strings"]: length-C list with types of
        convolutional layers.  Each list item must be accepted by
        `_check_layer_type`.
    option_dict["dense_layer_type_strings"]: length-D list with types of dense
        layers.  Each list item must be accepted by `_check_layer_type`.

    :return: option_dict: Same as input but maybe with default values added.
    """

    option_dict = cnn_architecture.check_input_args(option_dict)

    error_checking.assert_is_greater(option_dict[KL_SCALING_FACTOR_KEY], 0.)
    error_checking.assert_is_less_than(option_dict[KL_SCALING_FACTOR_KEY], 1.)

    num_conv_layers = numpy.sum(option_dict[NUM_CONV_LAYERS_KEY])
    num_dense_layers = len(option_dict[NUM_NEURONS_KEY])

    error_checking.assert_is_string_list(option_dict[CONV_LAYER_TYPES_KEY])
    error_checking.assert_is_numpy_array(
        numpy.array(option_dict[CONV_LAYER_TYPES_KEY]),
        exact_dimensions=numpy.array([num_conv_layers], dtype=int)
    )

    for t in option_dict[CONV_LAYER_TYPES_KEY]:
        _check_layer_type(t)

    error_checking.assert_is_string_list(option_dict[DENSE_LAYER_TYPES_KEY])
    error_checking.assert_is_numpy_array(
        numpy.array(option_dict[DENSE_LAYER_TYPES_KEY]),
        exact_dimensions=numpy.array([num_dense_layers], dtype=int)
    )

    for t in option_dict[DENSE_LAYER_TYPES_KEY]:
        _check_layer_type(t)

    return option_dict


def _get_2d_conv_layer(
        previous_layer_object, layer_type_string, num_filters,
        weight_regularizer, kl_divergence_scaling_factor):
    """Creates conv layer for 2 spatial dimensions.

    :param previous_layer_object: Previous layer (instance of
        `keras.layers.Layer` or similar).
    :param layer_type_string: See documentation for `_check_layer_type`.
    :param num_filters: Number of filters, i.e., number of output channels.
    :param weight_regularizer: Instance of `keras.regularizers.l1_l2`.
    :param kl_divergence_scaling_factor: Scaling factor for Kullback-Leibler
        divergence.
    :return: layer_object: New conv layer (instance of `keras.layers.Conv1D` or
        similar).
    """

    _check_layer_type(layer_type_string)

    if layer_type_string == POINT_ESTIMATE_TYPE_STRING:
        return architecture_utils.get_2d_conv_layer(
            num_kernel_rows=3, num_kernel_columns=3,
            num_rows_per_stride=1, num_columns_per_stride=1,
            num_filters=num_filters,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=weight_regularizer
        )(previous_layer_object)

    if layer_type_string == FLIPOUT_TYPE_STRING:
        return tf_prob.layers.Convolution2DFlipout(
            filters=num_filters, kernel_size=(3, 3), strides=(1, 1),
            padding='same', data_format='channels_last', dilation_rate=(1, 1),
            activation=None,
            kernel_divergence_fn=(
                lambda q, p, ignore:
                kl_divergence_scaling_factor * kl_lib.kl_divergence(q, p)
            ),
            bias_divergence_fn=(
                lambda q, p, ignore:
                kl_divergence_scaling_factor * kl_lib.kl_divergence(q, p)
            )
        )(previous_layer_object)

    return tf_prob.layers.Convolution2DReparameterization(
        filters=num_filters, kernel_size=(3, 3), strides=(1, 1),
        padding='same', data_format='channels_last', dilation_rate=(1, 1),
        activation=None,
        kernel_divergence_fn=(
            lambda q, p, ignore:
            kl_divergence_scaling_factor * kl_lib.kl_divergence(q, p)
        ),
        bias_divergence_fn=(
            lambda q, p, ignore:
            kl_divergence_scaling_factor * kl_lib.kl_divergence(q, p)
        )
    )(previous_layer_object)


def _get_dense_layer(
        previous_layer_object, layer_type_string, num_units,
        kl_divergence_scaling_factor):
    """Creates dense layer.

    :param previous_layer_object: Previous layer (instance of
        `keras.layers.Layer` or similar).
    :param layer_type_string: See documentation for `_check_layer_type`.
    :param num_units: Number of units, i.e., output neurons.
    :param kl_divergence_scaling_factor: Scaling factor for Kullback-Leibler
        divergence.
    :return: layer_object: New dense layer (instance of `keras.layers.Dense` or
        similar).
    """

    _check_layer_type(layer_type_string)

    if layer_type_string == POINT_ESTIMATE_TYPE_STRING:
        return architecture_utils.get_dense_layer(
            num_output_units=num_units
        )(previous_layer_object)

    if layer_type_string == FLIPOUT_TYPE_STRING:
        return tf_prob.layers.DenseFlipout(
            units=num_units, activation=None,
            kernel_divergence_fn=(
                lambda q, p, ignore:
                kl_divergence_scaling_factor * kl_lib.kl_divergence(q, p)
            ),
            bias_divergence_fn=(
                lambda q, p, ignore:
                kl_divergence_scaling_factor * kl_lib.kl_divergence(q, p)
            )
        )(previous_layer_object)

    return tf_prob.layers.DenseReparameterization(
        units=num_units, activation=None,
        kernel_divergence_fn=(
            lambda q, p, ignore:
            kl_divergence_scaling_factor * kl_lib.kl_divergence(q, p)
        ),
        bias_divergence_fn=(
            lambda q, p, ignore:
            kl_divergence_scaling_factor * kl_lib.kl_divergence(q, p)
        )
    )(previous_layer_object)


def create_model(option_dict):
    """Creates Bayesian CNN.

    :param option_dict: See documentation for `_check_input_args`.
    :return: model_object: Untrained (but compiled) instance of
        `keras.models.Model`.
    """

    option_dict = _check_input_args(option_dict)

    input_dimensions_low_res = option_dict[INPUT_DIMENSIONS_KEY]
    include_high_res_data = option_dict[INCLUDE_HIGH_RES_KEY]
    num_conv_layers_by_block = option_dict[NUM_CONV_LAYERS_KEY]
    num_channels_by_conv_layer = option_dict[NUM_CHANNELS_KEY]
    dropout_rate_by_conv_layer = option_dict[CONV_DROPOUT_RATES_KEY]
    num_neurons_by_dense_layer = option_dict[NUM_NEURONS_KEY]
    dropout_rate_by_dense_layer = option_dict[DENSE_DROPOUT_RATES_KEY]
    inner_activ_function_name = option_dict[INNER_ACTIV_FUNCTION_KEY]
    inner_activ_function_alpha = option_dict[INNER_ACTIV_FUNCTION_ALPHA_KEY]
    l2_weight = option_dict[L2_WEIGHT_KEY]
    use_batch_normalization = option_dict[USE_BATCH_NORM_KEY]
    loss_function = option_dict[LOSS_FUNCTION_KEY]
    ensemble_size = option_dict[ENSEMBLE_SIZE_KEY]

    kl_divergence_scaling_factor = option_dict[KL_SCALING_FACTOR_KEY]
    conv_layer_type_strings = option_dict[CONV_LAYER_TYPES_KEY]
    dense_layer_type_strings = option_dict[DENSE_LAYER_TYPES_KEY]

    input_layer_object_low_res = keras.layers.Input(
        shape=tuple(input_dimensions_low_res.tolist())
    )

    if include_high_res_data:
        these_dim = (
            4 * input_dimensions_low_res[0], 4 * input_dimensions_low_res[1], 1
        )
        input_layer_object_high_res = keras.layers.Input(shape=these_dim)
    else:
        input_layer_object_high_res = None

    l2_function = architecture_utils.get_weight_regularizer(l2_weight=l2_weight)
    layer_index = -1
    layer_object = None

    if include_high_res_data:
        for block_index in range(2):
            for _ in range(num_conv_layers_by_block[block_index]):
                layer_index += 1
                k = layer_index

                layer_object = _get_2d_conv_layer(
                    previous_layer_object=(
                        input_layer_object_high_res if k == 0 else layer_object
                    ),
                    layer_type_string=conv_layer_type_strings[k],
                    num_filters=num_channels_by_conv_layer[k],
                    weight_regularizer=l2_function,
                    kl_divergence_scaling_factor=kl_divergence_scaling_factor
                )

                layer_object = architecture_utils.get_activation_layer(
                    activation_function_string=inner_activ_function_name,
                    alpha_for_relu=inner_activ_function_alpha,
                    alpha_for_elu=inner_activ_function_alpha
                )(layer_object)

                if dropout_rate_by_conv_layer[k] > 0:
                    layer_object = architecture_utils.get_dropout_layer(
                        dropout_fraction=dropout_rate_by_conv_layer[k]
                    )(layer_object)

                if use_batch_normalization:
                    layer_object = architecture_utils.get_batch_norm_layer()(
                        layer_object
                    )

            layer_object = architecture_utils.get_2d_pooling_layer(
                num_rows_in_window=2, num_columns_in_window=2,
                num_rows_per_stride=2, num_columns_per_stride=2,
                pooling_type_string=architecture_utils.MAX_POOLING_STRING
            )(layer_object)

        layer_object = keras.layers.Concatenate(axis=-1)([
            layer_object, input_layer_object_low_res
        ])

    num_conv_blocks = len(num_conv_layers_by_block)

    for block_index in range(2, num_conv_blocks):
        for _ in range(num_conv_layers_by_block[block_index]):
            layer_index += 1
            k = layer_index

            layer_object = _get_2d_conv_layer(
                previous_layer_object=(
                    layer_object if include_high_res_data
                    else input_layer_object_low_res
                ),
                layer_type_string=conv_layer_type_strings[k],
                num_filters=num_channels_by_conv_layer[k],
                weight_regularizer=l2_function,
                kl_divergence_scaling_factor=kl_divergence_scaling_factor
            )

            layer_object = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha
            )(layer_object)

            if dropout_rate_by_conv_layer[k] > 0:
                layer_object = architecture_utils.get_dropout_layer(
                    dropout_fraction=dropout_rate_by_conv_layer[k]
                )(layer_object)

            if use_batch_normalization:
                layer_object = architecture_utils.get_batch_norm_layer()(
                    layer_object
                )

        if block_index != num_conv_blocks - 1:
            layer_object = architecture_utils.get_2d_pooling_layer(
                num_rows_in_window=2, num_columns_in_window=2,
                num_rows_per_stride=2, num_columns_per_stride=2,
                pooling_type_string=architecture_utils.MAX_POOLING_STRING
            )(layer_object)

    layer_object = architecture_utils.get_flattening_layer()(layer_object)
    num_dense_layers = len(num_neurons_by_dense_layer)

    for i in range(num_dense_layers):
        layer_object = _get_dense_layer(
            previous_layer_object=layer_object,
            layer_type_string=dense_layer_type_strings[i],
            num_units=num_neurons_by_dense_layer[i],
            kl_divergence_scaling_factor=kl_divergence_scaling_factor
        )

        if i == num_dense_layers - 1:
            break

        layer_object = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha
        )(layer_object)

        if dropout_rate_by_dense_layer[i] > 0:
            layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=dropout_rate_by_dense_layer[i]
            )(layer_object)

        if use_batch_normalization:
            layer_object = architecture_utils.get_batch_norm_layer()(
                layer_object
            )

    if ensemble_size > 1:
        num_target_vars = float(num_neurons_by_dense_layer[-1]) / ensemble_size
        assert numpy.isclose(
            num_target_vars, numpy.round(num_target_vars), atol=1e-6
        )

        num_target_vars = int(numpy.round(num_target_vars))
        layer_object = keras.layers.Reshape(
            target_shape=(num_target_vars, ensemble_size)
        )(layer_object)

    if include_high_res_data:
        input_layer_objects = [
            input_layer_object_high_res, input_layer_object_low_res
        ]
    else:
        input_layer_objects = input_layer_object_low_res

    model_object = keras.models.Model(
        inputs=input_layer_objects, outputs=layer_object
    )
    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adam(), metrics=[]
    )
    model_object.summary()

    return model_object
