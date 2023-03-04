"""Methods for creating U-net architecture."""

import numpy
import keras
from keras import backend as K
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import architecture_utils
from ml4tccf.machine_learning import neural_net

INPUT_DIMENSIONS_LOW_RES_KEY = 'input_dimensions_low_res'
INPUT_DIMENSIONS_HIGH_RES_KEY = 'input_dimensions_high_res'
INCLUDE_HIGH_RES_KEY = 'include_high_res_data'
CONV_LAYER_COUNTS_KEY = 'num_conv_layers_by_level'
OUTPUT_CHANNEL_COUNTS_KEY = 'num_output_channels_by_level'
CONV_DROPOUT_RATES_KEY = 'conv_dropout_rates_by_level'
UPCONV_DROPOUT_RATES_KEY = 'upconv_dropout_rate_by_level'
SKIP_DROPOUT_RATES_KEY = 'skip_dropout_rates_by_level'
INCLUDE_PENULTIMATE_KEY = 'include_penultimate_conv'
PENULTIMATE_DROPOUT_RATE_KEY = 'penultimate_conv_dropout_rate'
INNER_ACTIV_FUNCTION_KEY = 'inner_activ_function_name'
INNER_ACTIV_FUNCTION_ALPHA_KEY = 'inner_activ_function_alpha'
L2_WEIGHT_KEY = 'l2_weight'
USE_BATCH_NORM_KEY = 'use_batch_normalization'
ENSEMBLE_SIZE_KEY = 'ensemble_size'
LOSS_FUNCTION_KEY = 'loss_function'
OPTIMIZER_FUNCTION_KEY = 'optimizer_function'

DEFAULT_OPTION_DICT = {
    INCLUDE_HIGH_RES_KEY: False,
    CONV_LAYER_COUNTS_KEY: numpy.full(8, 2, dtype=int),
    OUTPUT_CHANNEL_COUNTS_KEY:
        numpy.array([16, 24, 32, 48, 64, 96, 128, 192], dtype=int),
    CONV_DROPOUT_RATES_KEY: [numpy.full(2, 0.)] * 8,
    UPCONV_DROPOUT_RATES_KEY: numpy.full(7, 0.),
    SKIP_DROPOUT_RATES_KEY: [numpy.full(2, 0.)] * 7,
    INCLUDE_PENULTIMATE_KEY: True,
    PENULTIMATE_DROPOUT_RATE_KEY: 0.,
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
    option_dict["num_conv_layers_by_level"]: length-(L + 1) numpy array with
        number of conv layers at each level.
    option_dict["num_output_channels_by_level"]: length-(L + 1) numpy array with
        number of output channels at each level.
    option_dict["conv_dropout_rates_by_level"]: length-(L + 1) list, where each
        list item is a 1-D numpy array of dropout rates.  The [k]th list item
        should be an array with length = number of conv layers at the [k]th
        level.  Use values <= 0 to omit dropout.
    option_dict["upconv_dropout_rate_by_level"]: length-L numpy array of dropout
        rates for upconv layers.
    option_dict["skip_dropout_rates_by_level"]: length-L list, where each list
        item is a 1-D numpy array of dropout rates.  The [k]th list item
        should be an array with length = number of conv layers at the [k]th
        level.  Use values <= 0 to omit dropout.
    option_dict["include_penultimate_conv"]: Boolean flag.  If True, will put in
        extra conv layer (with 3 x 3 filter) before final pixelwise conv.
    option_dict["penultimate_conv_dropout_rate"]: Dropout rate for penultimate
        conv layer.
    option_dict["inner_activ_function_name"]: Name of activation function for
        all inner (non-output) layers.  Must be accepted by
        `architecture_utils.check_activation_function`.
    option_dict["inner_activ_function_alpha"]: Alpha (slope parameter) for
        activation function for all inner layers.  Applies only to ReLU and eLU.
    option_dict["l2_weight"]: Strength of L2 regularization (for conv layers
        only).
    option_dict["use_batch_normalization"]: Boolean flag.  If True, will use
        batch normalization after each non-output layer.
    option_dict["ensemble_size"]: Number of ensemble members.
    option_dict["loss_function"]: Loss function.
    option_dict["optimizer_function"]: Optimizer function.

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
        option_dict[CONV_LAYER_COUNTS_KEY], num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[CONV_LAYER_COUNTS_KEY]
    )
    error_checking.assert_is_greater_numpy_array(
        option_dict[CONV_LAYER_COUNTS_KEY], 0
    )

    num_levels = len(option_dict[CONV_LAYER_COUNTS_KEY]) - 1
    expected_dim = numpy.array([num_levels + 1], dtype=int)

    error_checking.assert_is_numpy_array(
        option_dict[OUTPUT_CHANNEL_COUNTS_KEY], exact_dimensions=expected_dim
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[OUTPUT_CHANNEL_COUNTS_KEY]
    )
    error_checking.assert_is_geq_numpy_array(
        option_dict[OUTPUT_CHANNEL_COUNTS_KEY], 2
    )

    assert len(option_dict[CONV_DROPOUT_RATES_KEY]) == num_levels + 1

    for k in range(num_levels + 1):
        these_dim = numpy.array(
            [option_dict[CONV_LAYER_COUNTS_KEY][k]], dtype=int
        )
        error_checking.assert_is_numpy_array(
            option_dict[CONV_DROPOUT_RATES_KEY][k], exact_dimensions=these_dim
        )
        error_checking.assert_is_leq_numpy_array(
            option_dict[CONV_DROPOUT_RATES_KEY][k], 1., allow_nan=True
        )

    expected_dim = numpy.array([num_levels], dtype=int)

    error_checking.assert_is_numpy_array(
        option_dict[UPCONV_DROPOUT_RATES_KEY], exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        option_dict[UPCONV_DROPOUT_RATES_KEY], 1., allow_nan=True
    )

    assert len(option_dict[SKIP_DROPOUT_RATES_KEY]) == num_levels

    for k in range(num_levels):
        these_dim = numpy.array(
            [option_dict[CONV_LAYER_COUNTS_KEY][k]], dtype=int
        )
        error_checking.assert_is_numpy_array(
            option_dict[SKIP_DROPOUT_RATES_KEY][k], exact_dimensions=these_dim
        )
        error_checking.assert_is_leq_numpy_array(
            option_dict[SKIP_DROPOUT_RATES_KEY][k], 1., allow_nan=True
        )

    error_checking.assert_is_boolean(option_dict[INCLUDE_PENULTIMATE_KEY])
    error_checking.assert_is_leq(
        option_dict[PENULTIMATE_DROPOUT_RATE_KEY], 1., allow_nan=True
    )

    error_checking.assert_is_geq(option_dict[L2_WEIGHT_KEY], 0.)
    error_checking.assert_is_boolean(option_dict[USE_BATCH_NORM_KEY])
    error_checking.assert_is_integer(option_dict[ENSEMBLE_SIZE_KEY])
    error_checking.assert_is_geq(option_dict[ENSEMBLE_SIZE_KEY], 1)

    return option_dict


def make_norm_sum_function():
    """Creates function that normalizes the sum of probs in every grid to be 1.

    :return: norm_sum_function: Function handle (see below).
    """

    def norm_sum_function(orig_prediction_tensor):
        """Normalizes sum of probabilities in every grid to be 1.0.

        E = number of examples
        M = number of rows in grid
        N = number of columns in grid
        S = ensemble size

        :param orig_prediction_tensor: E-by-M-by-N-by-S tensor of probabilities.
        :return: new_prediction_tensor: Same but with normalized sums.
        """

        return (
            orig_prediction_tensor /
            K.sum(orig_prediction_tensor, axis=(1, 2), keepdims=True)
        )

    return norm_sum_function


def create_model(option_dict):
    """Creates U-net.

    :param option_dict: See documentation for `check_input_args`.
    :return: model_object: Untrained (but compiled) instance of
        `keras.models.Model`.
    """

    option_dict = check_input_args(option_dict)

    input_dimensions_low_res = option_dict[INPUT_DIMENSIONS_LOW_RES_KEY]
    include_high_res_data = option_dict[INCLUDE_HIGH_RES_KEY]
    num_conv_layers_by_level = option_dict[CONV_LAYER_COUNTS_KEY]
    num_output_channels_by_level = option_dict[OUTPUT_CHANNEL_COUNTS_KEY]
    conv_dropout_rates_by_level = option_dict[CONV_DROPOUT_RATES_KEY]
    upconv_dropout_rate_by_level = option_dict[UPCONV_DROPOUT_RATES_KEY]
    skip_dropout_rates_by_level = option_dict[SKIP_DROPOUT_RATES_KEY]
    # include_penultimate_conv = option_dict[INCLUDE_PENULTIMATE_KEY]
    # penultimate_conv_dropout_rate = option_dict[PENULTIMATE_DROPOUT_RATE_KEY]
    inner_activ_function_name = option_dict[INNER_ACTIV_FUNCTION_KEY]
    inner_activ_function_alpha = option_dict[INNER_ACTIV_FUNCTION_ALPHA_KEY]
    l2_weight = option_dict[L2_WEIGHT_KEY]
    use_batch_normalization = option_dict[USE_BATCH_NORM_KEY]
    loss_function = option_dict[LOSS_FUNCTION_KEY]
    optimizer_function = option_dict[OPTIMIZER_FUNCTION_KEY]
    ensemble_size = option_dict[ENSEMBLE_SIZE_KEY]

    input_layer_object_low_res = keras.layers.Input(
        shape=tuple(input_dimensions_low_res.tolist())
    )

    if include_high_res_data:
        input_dimensions_high_res = option_dict[INPUT_DIMENSIONS_HIGH_RES_KEY]

        input_layer_object_high_res = keras.layers.Input(
            shape=tuple(input_dimensions_high_res.tolist())
        )
    else:
        input_layer_object_high_res = None

    l2_function = architecture_utils.get_weight_regularizer(l2_weight=l2_weight)
    layer_object = None

    if include_high_res_data:
        for level_index in range(2):
            for i in range(num_conv_layers_by_level[level_index]):
                if i == 0 and level_index == 0:
                    layer_object = architecture_utils.get_2d_conv_layer(
                        num_kernel_rows=3, num_kernel_columns=3,
                        num_rows_per_stride=1, num_columns_per_stride=1,
                        num_filters=num_output_channels_by_level[level_index],
                        padding_type_string=
                        architecture_utils.YES_PADDING_STRING,
                        weight_regularizer=l2_function
                    )(input_layer_object_high_res)
                else:
                    layer_object = architecture_utils.get_2d_conv_layer(
                        num_kernel_rows=3, num_kernel_columns=3,
                        num_rows_per_stride=1, num_columns_per_stride=1,
                        num_filters=num_output_channels_by_level[level_index],
                        padding_type_string=
                        architecture_utils.YES_PADDING_STRING,
                        weight_regularizer=l2_function
                    )(layer_object)

                layer_object = architecture_utils.get_activation_layer(
                    activation_function_string=inner_activ_function_name,
                    alpha_for_relu=inner_activ_function_alpha,
                    alpha_for_elu=inner_activ_function_alpha
                )(layer_object)

                if conv_dropout_rates_by_level[level_index][i] > 0:
                    layer_object = architecture_utils.get_dropout_layer(
                        dropout_fraction=
                        conv_dropout_rates_by_level[level_index][i]
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

        num_conv_layers_by_level = num_conv_layers_by_level[2:]
        num_output_channels_by_level = num_output_channels_by_level[2:]
        conv_dropout_rates_by_level = conv_dropout_rates_by_level[2:]
        upconv_dropout_rate_by_level = upconv_dropout_rate_by_level[2:]
        skip_dropout_rates_by_level = skip_dropout_rates_by_level[2:]
    else:
        layer_object = input_layer_object_low_res

    num_levels = len(num_conv_layers_by_level)
    conv_layer_by_level = [None] * (num_levels + 1)
    pooling_layer_by_level = [None] * num_levels

    for i in range(num_levels + 1):
        for j in range(num_conv_layers_by_level[i]):
            if j == 0:
                if i == 0:
                    this_input_layer_object = layer_object
                else:
                    this_input_layer_object = pooling_layer_by_level[i - 1]
            else:
                this_input_layer_object = conv_layer_by_level[i]

            conv_layer_by_level[i] = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=num_output_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=l2_function
            )(this_input_layer_object)

            conv_layer_by_level[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha
            )(conv_layer_by_level[i])

            if conv_dropout_rates_by_level[i][j] > 0:
                conv_layer_by_level[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=conv_dropout_rates_by_level[i][j]
                )(conv_layer_by_level[i])

            if use_batch_normalization:
                conv_layer_by_level[i] = (
                    architecture_utils.get_batch_norm_layer()(
                        conv_layer_by_level[i]
                    )
                )

        if i == num_levels:
            break

        pooling_layer_by_level[i] = architecture_utils.get_2d_pooling_layer(
            num_rows_in_window=2, num_columns_in_window=2,
            num_rows_per_stride=2, num_columns_per_stride=2,
            pooling_type_string=architecture_utils.MAX_POOLING_STRING
        )(conv_layer_by_level[i])

    upconv_layer_by_level = [None] * num_levels
    skip_layer_by_level = [None] * num_levels
    merged_layer_by_level = [None] * num_levels

    try:
        this_layer_object = keras.layers.UpSampling2D(
            size=(2, 2), interpolation='bilinear'
        )(conv_layer_by_level[num_levels])
    except:
        this_layer_object = keras.layers.UpSampling2D(
            size=(2, 2)
        )(conv_layer_by_level[num_levels])

    i = num_levels - 1
    upconv_layer_by_level[i] = architecture_utils.get_2d_conv_layer(
        num_kernel_rows=2, num_kernel_columns=2,
        num_rows_per_stride=1, num_columns_per_stride=1,
        num_filters=num_output_channels_by_level[i],
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=l2_function
    )(this_layer_object)

    if upconv_dropout_rate_by_level[i] > 0:
        upconv_layer_by_level[i] = architecture_utils.get_dropout_layer(
            dropout_fraction=upconv_dropout_rate_by_level[i]
        )(upconv_layer_by_level[i])

    num_upconv_rows = upconv_layer_by_level[i].get_shape()[1]
    num_desired_rows = conv_layer_by_level[i].get_shape()[1]
    num_padding_rows = num_desired_rows - num_upconv_rows

    num_upconv_columns = upconv_layer_by_level[i].get_shape()[2]
    num_desired_columns = conv_layer_by_level[i].get_shape()[2]
    num_padding_columns = num_desired_columns - num_upconv_columns

    if num_padding_rows + num_padding_columns > 0:
        padding_arg = ((0, num_padding_rows), (0, num_padding_columns))

        upconv_layer_by_level[i] = keras.layers.ZeroPadding2D(
            padding=padding_arg
        )(upconv_layer_by_level[i])

    merged_layer_by_level[i] = keras.layers.Concatenate(axis=-1)(
        [conv_layer_by_level[i], upconv_layer_by_level[i]]
    )

    level_indices = numpy.linspace(
        0, num_levels - 1, num=num_levels, dtype=int
    )[::-1]

    for i in level_indices:
        for j in range(num_conv_layers_by_level[i]):
            if j == 0:
                this_input_layer_object = merged_layer_by_level[i]
            else:
                this_input_layer_object = skip_layer_by_level[i]

            skip_layer_by_level[i] = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=num_output_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=l2_function
            )(this_input_layer_object)

            skip_layer_by_level[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha
            )(skip_layer_by_level[i])

            this_dropout_rate = skip_dropout_rates_by_level[i][j]

            if this_dropout_rate > 0:
                skip_layer_by_level[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=this_dropout_rate
                )(skip_layer_by_level[i])

            if use_batch_normalization:
                skip_layer_by_level[i] = (
                    architecture_utils.get_batch_norm_layer()(
                        skip_layer_by_level[i]
                    )
                )

        # TODO(thunderhoser): Is this right?
        if i == 0:
            break

        try:
            this_layer_object = keras.layers.UpSampling2D(
                size=(2, 2), interpolation='bilinear'
            )(skip_layer_by_level[i])
        except:
            this_layer_object = keras.layers.UpSampling2D(
                size=(2, 2)
            )(skip_layer_by_level[i])

        upconv_layer_by_level[i - 1] = architecture_utils.get_2d_conv_layer(
            num_kernel_rows=2, num_kernel_columns=2,
            num_rows_per_stride=1, num_columns_per_stride=1,
            num_filters=num_output_channels_by_level[i - 1],
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=l2_function
        )(this_layer_object)

        if upconv_dropout_rate_by_level[i - 1] > 0:
            upconv_layer_by_level[i - 1] = architecture_utils.get_dropout_layer(
                dropout_fraction=upconv_dropout_rate_by_level[i - 1]
            )(upconv_layer_by_level[i - 1])

        num_upconv_rows = upconv_layer_by_level[i - 1].get_shape()[1]
        num_desired_rows = conv_layer_by_level[i - 1].get_shape()[1]
        num_padding_rows = num_desired_rows - num_upconv_rows

        num_upconv_columns = upconv_layer_by_level[i - 1].get_shape()[2]
        num_desired_columns = conv_layer_by_level[i - 1].get_shape()[2]
        num_padding_columns = num_desired_columns - num_upconv_columns

        if num_padding_rows + num_padding_columns > 0:
            padding_arg = ((0, num_padding_rows), (0, num_padding_columns))

            upconv_layer_by_level[i - 1] = keras.layers.ZeroPadding2D(
                padding=padding_arg
            )(upconv_layer_by_level[i - 1])

        merged_layer_by_level[i - 1] = keras.layers.Concatenate(axis=-1)(
            [conv_layer_by_level[i - 1], upconv_layer_by_level[i - 1]]
        )

    skip_layer_by_level[0] = architecture_utils.get_2d_conv_layer(
        num_kernel_rows=1, num_kernel_columns=1,
        num_rows_per_stride=1, num_columns_per_stride=1,
        num_filters=ensemble_size,
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=l2_function
    )(skip_layer_by_level[0])

    output_layer_object = keras.layers.Lambda(
        make_norm_sum_function(), name='normalize_sums'
    )(skip_layer_by_level[0])

    if include_high_res_data:
        input_layer_objects = [
            input_layer_object_high_res, input_layer_object_low_res
        ]
    else:
        input_layer_objects = input_layer_object_low_res

    model_object = keras.models.Model(
        inputs=input_layer_objects, outputs=output_layer_object
    )
    model_object.compile(
        loss=loss_function, optimizer=optimizer_function,
        metrics=neural_net.METRIC_FUNCTION_LIST_GRIDDED
    )
    model_object.summary()

    return model_object
