"""Methods for creating CNN architecture."""

import os
import sys
import numpy
import keras
import keras.layers

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import architecture_utils

INPUT_DIMENSIONS_LOW_RES_KEY = 'input_dimensions_low_res'
INPUT_DIMENSIONS_HIGH_RES_KEY = 'input_dimensions_high_res'
INCLUDE_HIGH_RES_KEY = 'include_high_res_data'
NUM_CONV_LAYERS_KEY = 'num_conv_layers_by_block'
NUM_CHANNELS_KEY = 'num_channels_by_conv_layer'
CONV_DROPOUT_RATES_KEY = 'dropout_rate_by_conv_layer'
NUM_NEURONS_KEY = 'num_neurons_by_dense_layer'
DENSE_DROPOUT_RATES_KEY = 'dropout_rate_by_dense_layer'
INNER_ACTIV_FUNCTION_KEY = 'inner_activ_function_name'
INNER_ACTIV_FUNCTION_ALPHA_KEY = 'inner_activ_function_alpha'
L2_WEIGHT_KEY = 'l2_weight'
USE_BATCH_NORM_KEY = 'use_batch_normalization'
ENSEMBLE_SIZE_KEY = 'ensemble_size'
LOSS_FUNCTION_KEY = 'loss_function'

DEFAULT_OPTION_DICT = {
    INCLUDE_HIGH_RES_KEY: True,
    NUM_CONV_LAYERS_KEY: numpy.full(11, 2, dtype=int),
    NUM_CHANNELS_KEY: numpy.array([
        2, 2, 4, 4, 16, 16, 32, 32, 64, 64, 96, 96, 128, 128, 184, 184,
        256, 256, 384, 384, 512, 512
    ], dtype=int),
    CONV_DROPOUT_RATES_KEY: numpy.full(22, 0.),
    NUM_NEURONS_KEY: numpy.array([1024, 128, 16, 2], dtype=int),
    DENSE_DROPOUT_RATES_KEY: numpy.array([0.5, 0.5, 0.5, 0]),
    INNER_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    USE_BATCH_NORM_KEY: True
}


def check_input_args(option_dict):
    """Error-checks input arguments.

    B = number of convolutional blocks
    C = number of convolutional layers
    D = number of dense layers

    :param option_dict: Dictionary with the following keys.
    option_dict["input_dimensions_low_res"]:
        numpy array with input dimensions for low-resolution satellite data:
        [num_grid_rows, num_grid_columns, num_lag_times * num_wavelengths].
    option_dict["input_dimensions_high_res"]:
        numpy array with input dimensions for high-resolution satellite data:
        [num_grid_rows, num_grid_columns, num_lag_times * num_wavelengths].
        If you are not including high-res data, make this None.
    option_dict["include_high_res_data"]: Boolean flag.  If True, will create
        architecture that includes high-resolution satellite data (1/4 the grid
        spacing of low-resolution data) as input.
    option_dict["num_conv_layers_by_block"]: length-B numpy array with number of
        conv layers for each block.
    option_dict["num_channels_by_conv_layer"]: length-C numpy array with number
        of channels for each conv layer.
    option_dict["dropout_rate_by_conv_layer"]: length-C numpy array with dropout
        rate for each conv layer.  Use number <= 0 to indicate no-dropout.
    option_dict["num_neurons_by_dense_layer"]: length-D numpy array with number
        of output neurons for each dense layer.
    option_dict["dropout_rate_by_dense_layer"]: length-D numpy array with
        dropout rate for each dense layer.  Use number <= 0 to indicate
        no-dropout.
    option_dict["inner_activ_function_name"]: Name of activation function for
        all non-output layers.  Must be accepted by
        `architecture_utils.check_activation_function`.
    option_dict["inner_activ_function_alpha"]: Alpha (slope parameter) for
        activation function for all non-output layers.  Applies only to ReLU and
        eLU.
    option_dict["l2_weight"]: Strength of L2 regularization (for conv layers
        only).
    option_dict["use_batch_normalization"]: Boolean flag.  If True, will use
        batch normalization after each non-output layer.
    option_dict["ensemble_size"]: Number of ensemble members.
    option_dict["loss_function"]: Loss function.

    :return: option_dict: Same as input but maybe with default values added.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    error_checking.assert_is_numpy_array(
        option_dict[INPUT_DIMENSIONS_LOW_RES_KEY],
        exact_dimensions=numpy.array([3], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[INPUT_DIMENSIONS_LOW_RES_KEY]
    )
    error_checking.assert_is_greater_numpy_array(
        option_dict[INPUT_DIMENSIONS_LOW_RES_KEY], 0
    )

    error_checking.assert_is_boolean(option_dict[INCLUDE_HIGH_RES_KEY])

    if option_dict[INCLUDE_HIGH_RES_KEY]:
        error_checking.assert_is_numpy_array(
            option_dict[INPUT_DIMENSIONS_HIGH_RES_KEY],
            exact_dimensions=numpy.array([3], dtype=int)
        )
        error_checking.assert_is_integer_numpy_array(
            option_dict[INPUT_DIMENSIONS_HIGH_RES_KEY]
        )
        error_checking.assert_is_greater_numpy_array(
            option_dict[INPUT_DIMENSIONS_HIGH_RES_KEY], 0
        )

        error_checking.assert_equals(
            option_dict[INPUT_DIMENSIONS_HIGH_RES_KEY][0],
            4 * option_dict[INPUT_DIMENSIONS_LOW_RES_KEY][0]
        )
        error_checking.assert_equals(
            option_dict[INPUT_DIMENSIONS_HIGH_RES_KEY][1],
            4 * option_dict[INPUT_DIMENSIONS_LOW_RES_KEY][1]
        )

    error_checking.assert_is_numpy_array(
        option_dict[NUM_CONV_LAYERS_KEY], num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[NUM_CONV_LAYERS_KEY]
    )
    error_checking.assert_is_geq_numpy_array(
        option_dict[NUM_CONV_LAYERS_KEY], 1
    )

    num_conv_layers = numpy.sum(option_dict[NUM_CONV_LAYERS_KEY])

    error_checking.assert_is_numpy_array(
        option_dict[NUM_CHANNELS_KEY],
        exact_dimensions=numpy.array([num_conv_layers], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(option_dict[NUM_CHANNELS_KEY])
    error_checking.assert_is_geq_numpy_array(option_dict[NUM_CHANNELS_KEY], 1)

    error_checking.assert_is_numpy_array(
        option_dict[CONV_DROPOUT_RATES_KEY],
        exact_dimensions=numpy.array([num_conv_layers], dtype=int)
    )
    error_checking.assert_is_leq_numpy_array(
        option_dict[CONV_DROPOUT_RATES_KEY], 1.
    )

    error_checking.assert_is_numpy_array(
        option_dict[NUM_NEURONS_KEY], num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(option_dict[NUM_NEURONS_KEY])
    error_checking.assert_is_geq_numpy_array(option_dict[NUM_NEURONS_KEY], 2)

    num_dense_layers = len(option_dict[NUM_NEURONS_KEY])
    error_checking.assert_is_numpy_array(
        option_dict[DENSE_DROPOUT_RATES_KEY],
        exact_dimensions=numpy.array([num_dense_layers], dtype=int)
    )
    error_checking.assert_is_leq_numpy_array(
        option_dict[DENSE_DROPOUT_RATES_KEY], 1.
    )

    error_checking.assert_is_geq(option_dict[L2_WEIGHT_KEY], 0.)
    error_checking.assert_is_boolean(option_dict[USE_BATCH_NORM_KEY])
    error_checking.assert_is_integer(option_dict[ENSEMBLE_SIZE_KEY])
    error_checking.assert_is_geq(option_dict[ENSEMBLE_SIZE_KEY], 1)

    return option_dict


def create_model(option_dict):
    """Creates CNN.

    :param option_dict: See documentation for `check_input_args`.
    :return: model_object: Untrained (but compiled) instance of
        `keras.models.Model`.
    """

    option_dict = check_input_args(option_dict)

    input_dimensions_low_res = option_dict[INPUT_DIMENSIONS_LOW_RES_KEY]
    input_dimensions_high_res = option_dict[INPUT_DIMENSIONS_HIGH_RES_KEY]
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

    input_layer_object_low_res = keras.layers.Input(
        shape=tuple(input_dimensions_low_res.tolist())
    )

    if include_high_res_data:
        input_layer_object_high_res = keras.layers.Input(
            shape=tuple(input_dimensions_high_res.tolist())
        )
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

                if k == 0:
                    layer_object = architecture_utils.get_2d_conv_layer(
                        num_kernel_rows=3, num_kernel_columns=3,
                        num_rows_per_stride=1, num_columns_per_stride=1,
                        num_filters=num_channels_by_conv_layer[k],
                        padding_type_string=
                        architecture_utils.YES_PADDING_STRING,
                        weight_regularizer=l2_function
                    )(input_layer_object_high_res)
                else:
                    layer_object = architecture_utils.get_2d_conv_layer(
                        num_kernel_rows=3, num_kernel_columns=3,
                        num_rows_per_stride=1, num_columns_per_stride=1,
                        num_filters=num_channels_by_conv_layer[k],
                        padding_type_string=
                        architecture_utils.YES_PADDING_STRING,
                        weight_regularizer=l2_function
                    )(layer_object)

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

            if include_high_res_data:
                layer_object = architecture_utils.get_2d_conv_layer(
                    num_kernel_rows=3, num_kernel_columns=3,
                    num_rows_per_stride=1, num_columns_per_stride=1,
                    num_filters=num_channels_by_conv_layer[k],
                    padding_type_string=
                    architecture_utils.YES_PADDING_STRING,
                    weight_regularizer=l2_function
                )(layer_object)
            else:
                layer_object = architecture_utils.get_2d_conv_layer(
                    num_kernel_rows=3, num_kernel_columns=3,
                    num_rows_per_stride=1, num_columns_per_stride=1,
                    num_filters=num_channels_by_conv_layer[k],
                    padding_type_string=
                    architecture_utils.YES_PADDING_STRING,
                    weight_regularizer=l2_function
                )(input_layer_object_low_res)

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
        layer_object = architecture_utils.get_dense_layer(
            num_output_units=num_neurons_by_dense_layer[i]
        )(layer_object)

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
