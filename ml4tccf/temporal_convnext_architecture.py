"""Methods for creating temporal-CNN architecture.

This is a CNN with TimeDistributed layers to handle inputs (satellite images) at
different lag times.
"""

import os
import sys
import numpy
import keras
import keras.layers as layers
import tensorflow

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import architecture_utils
import neural_net_utils

try:
    THIS_LAYER = layers.Input(shape=(3, 4, 5))
except:
    import tensorflow.keras as keras
    import tensorflow.keras.layers as layers


INPUT_DIMENSIONS_LOW_RES_KEY = 'input_dimensions_low_res'
INPUT_DIMENSIONS_HIGH_RES_KEY = 'input_dimensions_high_res'
INPUT_DIMENSIONS_SCALAR_KEY = 'input_dimensions_scalar'
INCLUDE_HIGH_RES_KEY = 'include_high_res_data'
INCLUDE_SCALAR_DATA_KEY = 'include_scalar_data'
NUM_CONV_LAYERS_KEY = 'num_conv_layers_by_block'
USE_RESIDUAL_BLOCKS_KEY = 'use_residual_blocks'
USE_CONVNEXT_BLOCKS_KEY = 'use_convnext_blocks'
POOLING_SIZE_KEY = 'pooling_size_by_block_px'
NUM_CHANNELS_KEY = 'num_channels_by_conv_block'
CONV_DROPOUT_RATES_KEY = 'dropout_rate_by_conv_block'

FC_MODULE_NUM_CONV_LAYERS_KEY = 'forecast_module_num_conv_layers'
FC_MODULE_DROPOUT_RATES_KEY = 'forecast_module_dropout_rates'
FC_MODULE_USE_3D_CONV = 'forecast_module_use_3d_conv'

NUM_NEURONS_KEY = 'num_neurons_by_dense_layer'
DENSE_DROPOUT_RATES_KEY = 'dropout_rate_by_dense_layer'
INNER_ACTIV_FUNCTION_KEY = 'inner_activ_function_name'
INNER_ACTIV_FUNCTION_ALPHA_KEY = 'inner_activ_function_alpha'
L2_WEIGHT_KEY = 'l2_weight'
USE_BATCH_NORM_KEY = 'use_batch_normalization'
ENSEMBLE_SIZE_KEY = 'ensemble_size'
LOSS_FUNCTION_KEY = 'loss_function'
OPTIMIZER_FUNCTION_KEY = 'optimizer_function'
START_WITH_POOLING_KEY = 'start_with_pooling_layer'

DEFAULT_OPTION_DICT = {
    INCLUDE_HIGH_RES_KEY: True,
    INCLUDE_SCALAR_DATA_KEY: False,
    NUM_CONV_LAYERS_KEY: numpy.full(11, 2, dtype=int),
    USE_RESIDUAL_BLOCKS_KEY: False,
    USE_CONVNEXT_BLOCKS_KEY: False,
    POOLING_SIZE_KEY: numpy.full(11, 2, dtype=int),
    NUM_CHANNELS_KEY: numpy.array(
        [2, 4, 16, 32, 64, 96, 128, 184, 256, 384, 512], dtype=int
    ),
    CONV_DROPOUT_RATES_KEY: numpy.full(11, 0.),
    FC_MODULE_NUM_CONV_LAYERS_KEY: 1,
    FC_MODULE_DROPOUT_RATES_KEY: numpy.array([0.]),
    FC_MODULE_USE_3D_CONV: True,
    NUM_NEURONS_KEY: numpy.array([1024, 128, 16, 2], dtype=int),
    DENSE_DROPOUT_RATES_KEY: numpy.array([0.5, 0.5, 0.5, 0]),
    INNER_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    USE_BATCH_NORM_KEY: True,
    START_WITH_POOLING_KEY: False
}

LARGE_INTEGER = int(1e12)

EPSILON_FOR_LAYER_NORM = 1e-6
EXPANSION_FACTOR_FOR_CONVNEXT = 4
INIT_VALUE_FOR_LAYER_SCALE = 1e-6


# @keras.saving.register_keras_serializable()
class LayerScale(keras.layers.Layer):
    """Layer-scale module.

    Scavenged from: https://github.com/danielabdi-noaa/HRRRemulator/blob/
                    master/tfmodel/convnext.py
    """

    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, _):
        self.gamma = self.add_weight(
            shape=(self.projection_dim,),
            initializer=keras.initializers.Constant(self.init_values),
            trainable=True,
            name="gamma",
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
            }
        )
        return config


# @keras.saving.register_keras_serializable()
class StochasticDepth(keras.layers.Layer):
    def __init__(self, survival_prob=0.9, **kwargs):
        super().__init__(**kwargs)
        self.survival_prob = survival_prob

    def call(self, inputs, training=None):
        if not training:
            return inputs[0] + inputs[1]

        batch_size = tensorflow.shape(inputs[0])[0]
        random_tensor = self.survival_prob + tensorflow.random.uniform(
            [batch_size, 1, 1, 1]
        )
        binary_tensor = tensorflow.floor(random_tensor)
        output = inputs[0] + binary_tensor * inputs[1] / self.survival_prob
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "survival_prob": self.survival_prob
            }
        )
        return config


def __dimension_to_int(dimension_object):
    """Converts `tensorflow.Dimension` object to integer.

    :param dimension_object: `tensorflow.Dimension` object.
    :return: dimension_int: Integer.
    """

    try:
        dimension = dimension_object.value
    except AttributeError:
        dimension = dimension_object

    try:
        return int(dimension)
    except TypeError:
        return LARGE_INTEGER


def __get_2d_convnext_block(
        input_layer_object, num_conv_layers, filter_size_px, num_filters,
        do_time_distributed_conv, regularizer_object,
        do_activation, dropout_rate, basic_layer_name):
    """Creates ConvNext block for data with 2 spatial dimensions.

    L = number of conv layers

    :param input_layer_object: See documentation for `_get_2d_conv_block`.
    :param num_conv_layers: Same.
    :param filter_size_px: Same.
    :param num_filters: Same.
    :param do_time_distributed_conv: Same.
    :param regularizer_object: Same.
    :param do_activation: Same.
    :param dropout_rate: Same.
    :param basic_layer_name: Same.
    :return: output_layer_object: Same.
    """

    # TODO(thunderhoser): HACK.
    if filter_size_px == 3:
        actual_filter_size_px = 7
    else:
        actual_filter_size_px = filter_size_px + 0

    current_layer_object = None

    for i in range(num_conv_layers):
        if i == 0:
            this_input_layer_object = input_layer_object
        else:
            this_input_layer_object = current_layer_object

        this_name = '{0:s}_conv{1:d}'.format(basic_layer_name, i)
        current_layer_object = architecture_utils.get_2d_depthwise_conv_layer(
            num_kernel_rows=actual_filter_size_px,
            num_kernel_columns=actual_filter_size_px,
            num_rows_per_stride=1,
            num_columns_per_stride=1,
            num_filters=num_filters,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )

        if do_time_distributed_conv:
            current_layer_object = keras.layers.TimeDistributed(
                current_layer_object, name=this_name
            )(this_input_layer_object)
        else:
            current_layer_object = current_layer_object(this_input_layer_object)

        this_name = '{0:s}_lyrnorm{1:d}'.format(basic_layer_name, i)
        current_layer_object = keras.layers.LayerNormalization(
            epsilon=EPSILON_FOR_LAYER_NORM, name=this_name
        )(
            current_layer_object
        )

        this_name = '{0:s}_dense{1:d}a'.format(basic_layer_name, i)
        dense_layer_object = architecture_utils.get_dense_layer(
            num_output_units=EXPANSION_FACTOR_FOR_CONVNEXT * num_filters,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )
        current_layer_object = dense_layer_object(current_layer_object)

        if do_activation:
            this_name = '{0:s}_gelu{1:d}'.format(basic_layer_name, i)
            current_layer_object = keras.layers.Activation(
                'gelu', name=this_name
            )(current_layer_object)

        this_name = '{0:s}_dense{1:d}b'.format(basic_layer_name, i)
        dense_layer_object = architecture_utils.get_dense_layer(
            num_output_units=num_filters,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )
        current_layer_object = dense_layer_object(current_layer_object)

        this_name = '{0:s}_lyrscale{1:d}'.format(basic_layer_name, i)
        current_layer_object = LayerScale(
            INIT_VALUE_FOR_LAYER_SCALE, num_filters, name=this_name
        )(current_layer_object)

        if i != num_conv_layers - 1:
            continue

        if input_layer_object.shape[-1] == num_filters:
            new_layer_object = input_layer_object
        else:
            this_name = '{0:s}_preresidual_conv'.format(basic_layer_name)
            new_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=1,
                num_kernel_columns=1,
                num_rows_per_stride=1,
                num_columns_per_stride=1,
                num_filters=num_filters,
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object,
                layer_name=this_name
            )

            if do_time_distributed_conv:
                new_layer_object = keras.layers.TimeDistributed(
                    new_layer_object, name=this_name
                )(input_layer_object)
            else:
                new_layer_object = new_layer_object(input_layer_object)

        this_name = '{0:s}_residual'.format(basic_layer_name)

        if dropout_rate > 0:
            current_layer_object = StochasticDepth(
                survival_prob=1. - dropout_rate, name=this_name
            )([new_layer_object, current_layer_object])
        else:
            current_layer_object = keras.layers.Add(name=this_name)([
                new_layer_object, current_layer_object
            ])

    return current_layer_object


def __get_3d_convnext_block(
        input_layer_object, num_time_steps, num_conv_layers, filter_size_px,
        regularizer_object, do_activation, dropout_rate, basic_layer_name):
    """Creates ConvNext block for data with 3 spatial dimensions.

    :param input_layer_object: See documentation for `_get_3d_conv_block`.
    :param num_time_steps: Same.
    :param num_conv_layers: Same.
    :param filter_size_px: Same.
    :param regularizer_object: Same.
    :param do_activation: Same.
    :param dropout_rate: Same.
    :param basic_layer_name: Same.
    :return: output_layer_object: Same.
    """

    current_layer_object = None
    num_filters = __dimension_to_int(input_layer_object.shape[-1])

    for i in range(num_conv_layers):
        this_name = '{0:s}_conv{1:d}'.format(basic_layer_name, i)

        if i == 0:
            current_layer_object = architecture_utils.get_3d_conv_layer(
                num_kernel_rows=filter_size_px,
                num_kernel_columns=filter_size_px,
                num_kernel_heights=num_time_steps,
                num_rows_per_stride=1,
                num_columns_per_stride=1,
                num_heights_per_stride=1,
                num_filters=num_filters,
                padding_type_string=architecture_utils.NO_PADDING_STRING,
                weight_regularizer=regularizer_object,
                layer_name=this_name
            )
            current_layer_object = current_layer_object(input_layer_object)

            new_dims = (
                current_layer_object.shape[1:3] +
                (current_layer_object.shape[-1],)
            )

            this_name = '{0:s}_remove-time-dim'.format(basic_layer_name)
            current_layer_object = keras.layers.Reshape(
                target_shape=new_dims, name=this_name
            )(current_layer_object)
        else:
            dwconv_layer_object = (
                architecture_utils.get_2d_depthwise_conv_layer(
                    num_kernel_rows=filter_size_px,
                    num_kernel_columns=filter_size_px,
                    num_rows_per_stride=1,
                    num_columns_per_stride=1,
                    num_filters=num_filters,
                    padding_type_string=architecture_utils.YES_PADDING_STRING,
                    weight_regularizer=regularizer_object,
                    layer_name=this_name
                )
            )
            current_layer_object = dwconv_layer_object(current_layer_object)

        this_name = '{0:s}_lyrnorm{1:d}'.format(basic_layer_name, i)
        current_layer_object = keras.layers.LayerNormalization(
            epsilon=EPSILON_FOR_LAYER_NORM, name=this_name
        )(
            current_layer_object
        )

        this_name = '{0:s}_dense{1:d}a'.format(basic_layer_name, i)
        dense_layer_object = architecture_utils.get_dense_layer(
            num_output_units=EXPANSION_FACTOR_FOR_CONVNEXT * num_filters,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )
        current_layer_object = dense_layer_object(current_layer_object)

        if do_activation:
            this_name = '{0:s}_gelu{1:d}'.format(basic_layer_name, i)
            current_layer_object = keras.layers.Activation(
                'gelu', name=this_name
            )(current_layer_object)

        this_name = '{0:s}_dense{1:d}b'.format(basic_layer_name, i)
        dense_layer_object = architecture_utils.get_dense_layer(
            num_output_units=num_filters,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )
        current_layer_object = dense_layer_object(current_layer_object)

        this_name = '{0:s}_lyrscale{1:d}'.format(basic_layer_name, i)
        current_layer_object = LayerScale(
            INIT_VALUE_FOR_LAYER_SCALE, num_filters, name=this_name
        )(current_layer_object)

        if i != num_conv_layers - 1:
            continue

        this_name = '{0:s}_preresidual_avg'.format(basic_layer_name)
        this_layer_object = architecture_utils.get_3d_pooling_layer(
            num_rows_in_window=1,
            num_columns_in_window=1,
            num_heights_in_window=num_time_steps,
            num_rows_per_stride=1,
            num_columns_per_stride=1,
            num_heights_per_stride=num_time_steps,
            pooling_type_string=architecture_utils.MEAN_POOLING_STRING,
            layer_name=this_name
        )(input_layer_object)

        new_dims = (
            this_layer_object.shape[1:3] +
            (this_layer_object.shape[-1],)
        )

        this_name = '{0:s}_preresidual_squeeze'.format(basic_layer_name)
        this_layer_object = keras.layers.Reshape(
            target_shape=new_dims, name=this_name
        )(this_layer_object)

        this_name = '{0:s}_residual'.format(basic_layer_name)

        if dropout_rate > 0:
            current_layer_object = StochasticDepth(
                survival_prob=1. - dropout_rate, name=this_name
            )([this_layer_object, current_layer_object])
        else:
            current_layer_object = keras.layers.Add(name=this_name)([
                this_layer_object, current_layer_object
            ])

    return current_layer_object


def _get_2d_conv_block(
        input_layer_object, do_residual, do_convnext,
        num_conv_layers, filter_size_px, num_filters, do_time_distributed_conv,
        regularizer_object,
        activation_function_name, activation_function_alpha,
        dropout_rates, use_batch_norm, basic_layer_name):
    """Creates convolutional block for data with 2 spatial dimensions.

    L = number of conv layers

    :param input_layer_object: Input layer to block.
    :param do_residual: Boolean flag.  If True, this will be a residual block.
    :param do_convnext: Boolean flag.  If True, this will be a ConvNext block.
        If `do_residual == do_convnext == False`, this will be a basic
        convolutional block.
    :param num_conv_layers: Number of conv layers in block.
    :param filter_size_px: Filter size for conv layers.  The same filter size
        will be used in both dimensions, and the same filter size will be used
        for every conv layer.
    :param num_filters: Number of filters -- same for every conv layer.
    :param do_time_distributed_conv: Boolean flag.  If True (False), will do
        time-distributed (basic) convolution.
    :param regularizer_object: Regularizer for conv layers (instance of
        `keras.regularizers.l1_l2` or similar).
    :param activation_function_name: Name of activation function -- same for
        every conv layer.  Must be accepted by
        `architecture_utils.check_activation_function`.
    :param activation_function_alpha: Alpha (slope parameter) for activation
        function -- same for every conv layer.  Applies only to ReLU and eLU.
    :param dropout_rates: Dropout rates for conv layers.  This can be a scalar
        (applied to every conv layer) or length-L numpy array.
    :param use_batch_norm: Boolean flag.  If True, will use batch normalization.
    :param basic_layer_name: Basic layer name.  Each layer name will be made
        unique by adding a suffix.
    :return: output_layer_object: Output layer from block.
    """

    # Check input args.
    if do_residual:
        num_conv_layers = max([num_conv_layers, 2])

    try:
        _ = len(dropout_rates)
    except:
        dropout_rates = numpy.full(num_conv_layers, dropout_rates)

    if len(dropout_rates) < num_conv_layers:
        dropout_rates = numpy.concatenate([
            dropout_rates, dropout_rates[[-1]]
        ])

    assert len(dropout_rates) == num_conv_layers

    # Handle ConvNext block.
    if do_convnext:
        current_layer_object = __get_2d_convnext_block(
            input_layer_object=input_layer_object,
            num_conv_layers=1,
            filter_size_px=filter_size_px,
            num_filters=num_filters,
            do_time_distributed_conv=do_time_distributed_conv,
            regularizer_object=regularizer_object,
            do_activation=activation_function_name is not None,
            dropout_rate=dropout_rates[0],
            basic_layer_name=(
                basic_layer_name if num_conv_layers == 1
                else basic_layer_name + '_0'
            )
        )

        for i in range(num_conv_layers):
            if i == 0:
                continue

            current_layer_object = __get_2d_convnext_block(
                input_layer_object=input_layer_object,
                num_conv_layers=1,
                filter_size_px=filter_size_px,
                num_filters=num_filters,
                do_time_distributed_conv=do_time_distributed_conv,
                regularizer_object=regularizer_object,
                do_activation=activation_function_name is not None,
                dropout_rate=dropout_rates[0],
                basic_layer_name='{0:s}_{1:d}'.format(basic_layer_name, i)
            )

        return current_layer_object

    # Create residual block or basic conv block.
    current_layer_object = None

    for i in range(num_conv_layers):
        if i == 0:
            this_input_layer_object = input_layer_object
        else:
            this_input_layer_object = current_layer_object

        this_name = '{0:s}_conv{1:d}'.format(basic_layer_name, i)
        current_layer_object = architecture_utils.get_2d_conv_layer(
            num_kernel_rows=filter_size_px,
            num_kernel_columns=filter_size_px,
            num_rows_per_stride=1,
            num_columns_per_stride=1,
            num_filters=num_filters,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )

        if do_time_distributed_conv:
            current_layer_object = keras.layers.TimeDistributed(
                current_layer_object, name=this_name
            )(this_input_layer_object)
        else:
            current_layer_object = current_layer_object(this_input_layer_object)

        if i == num_conv_layers - 1 and do_residual:
            if __dimension_to_int(input_layer_object.shape[-1]) == num_filters:
                new_layer_object = input_layer_object
            else:
                this_name = '{0:s}_preresidual_conv'.format(basic_layer_name)
                new_layer_object = architecture_utils.get_2d_conv_layer(
                    num_kernel_rows=filter_size_px,
                    num_kernel_columns=filter_size_px,
                    num_rows_per_stride=1,
                    num_columns_per_stride=1,
                    num_filters=num_filters,
                    padding_type_string=architecture_utils.YES_PADDING_STRING,
                    weight_regularizer=regularizer_object,
                    layer_name=this_name
                )

                if do_time_distributed_conv:
                    new_layer_object = keras.layers.TimeDistributed(
                        new_layer_object, name=this_name
                    )(input_layer_object)
                else:
                    new_layer_object = new_layer_object(input_layer_object)

            this_name = '{0:s}_residual'.format(basic_layer_name)
            current_layer_object = keras.layers.Add(name=this_name)([
                current_layer_object, new_layer_object
            ])

        if activation_function_name is not None:
            this_name = '{0:s}_activ{1:d}'.format(basic_layer_name, i)
            current_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=activation_function_name,
                alpha_for_relu=activation_function_alpha,
                alpha_for_elu=activation_function_alpha,
                layer_name=this_name
            )(current_layer_object)

        if dropout_rates[i] > 0:
            this_name = '{0:s}_dropout{1:d}'.format(basic_layer_name, i)
            current_layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=dropout_rates[i], layer_name=this_name
            )(current_layer_object)

        if use_batch_norm:
            this_name = '{0:s}_bn{1:d}'.format(basic_layer_name, i)
            current_layer_object = architecture_utils.get_batch_norm_layer(
                layer_name=this_name
            )(current_layer_object)

    return current_layer_object


def _get_3d_conv_block(
        input_layer_object, do_residual, do_convnext,
        num_conv_layers, filter_size_px, regularizer_object,
        activation_function_name, activation_function_alpha,
        dropout_rates, use_batch_norm, basic_layer_name):
    """Creates convolutional block for data with 3 spatial dimensions.

    :param input_layer_object: Input layer to block (with 3 spatial dims).
    :param do_residual: See documentation for `_get_2d_conv_block`.
    :param do_convnext: Same.
    :param num_conv_layers: Same.
    :param filter_size_px: Same.
    :param regularizer_object: Same.
    :param activation_function_name: Same.
    :param activation_function_alpha: Same.
    :param dropout_rates: Same.
    :param use_batch_norm: Same.
    :param basic_layer_name: Same.
    :return: output_layer_object: Output layer from block (with 2 spatial dims).
    """

    # Check input args.
    if do_residual:
        num_conv_layers = max([num_conv_layers, 2])

    try:
        _ = len(dropout_rates)
    except:
        dropout_rates = numpy.full(num_conv_layers, dropout_rates)

    if len(dropout_rates) < num_conv_layers:
        dropout_rates = numpy.concatenate([
            dropout_rates, dropout_rates[[-1]]
        ])

    assert len(dropout_rates) == num_conv_layers

    # Handle ConvNext block.
    num_time_steps = __dimension_to_int(input_layer_object.shape[-2])

    if do_convnext:
        current_layer_object = __get_3d_convnext_block(
            input_layer_object=input_layer_object,
            num_conv_layers=1,
            filter_size_px=filter_size_px,
            num_time_steps=num_time_steps,
            regularizer_object=regularizer_object,
            do_activation=activation_function_name is not None,
            dropout_rate=dropout_rates[0],
            basic_layer_name=(
                basic_layer_name if num_conv_layers == 1
                else basic_layer_name + '_0'
            )
        )

        for i in range(num_conv_layers):
            if i == 0:
                continue

            current_layer_object = __get_3d_convnext_block(
                input_layer_object=input_layer_object,
                num_conv_layers=1,
                filter_size_px=filter_size_px,
                num_time_steps=num_time_steps,
                regularizer_object=regularizer_object,
                do_activation=activation_function_name is not None,
                dropout_rate=dropout_rates[0],
                basic_layer_name='{0:s}_{1:d}'.format(basic_layer_name, i)
            )

        return current_layer_object

    # Create residual block or basic conv block.
    current_layer_object = None
    num_filters = __dimension_to_int(input_layer_object.shape[-1])

    for i in range(num_conv_layers):
        this_name = '{0:s}_conv{1:d}'.format(basic_layer_name, i)

        if i == 0:
            current_layer_object = architecture_utils.get_3d_conv_layer(
                num_kernel_rows=filter_size_px,
                num_kernel_columns=filter_size_px,
                num_kernel_heights=num_time_steps,
                num_rows_per_stride=1,
                num_columns_per_stride=1,
                num_heights_per_stride=1,
                num_filters=num_filters,
                padding_type_string=architecture_utils.NO_PADDING_STRING,
                weight_regularizer=regularizer_object,
                layer_name=this_name
            )(input_layer_object)

            new_dims = (
                __dimension_to_int(current_layer_object.shape[1]),
                __dimension_to_int(current_layer_object.shape[2]),
                __dimension_to_int(current_layer_object.shape[-1])
            )

            this_name = '{0:s}_remove-time-dim'.format(basic_layer_name)
            current_layer_object = keras.layers.Reshape(
                target_shape=new_dims, name=this_name
            )(current_layer_object)
        else:
            current_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=filter_size_px,
                num_kernel_columns=filter_size_px,
                num_rows_per_stride=1,
                num_columns_per_stride=1,
                num_filters=num_filters,
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object
            )(current_layer_object)

        if i == num_conv_layers - 1 and do_residual:
            this_name = '{0:s}_preresidual_avg'.format(basic_layer_name)
            this_layer_object = architecture_utils.get_3d_pooling_layer(
                num_rows_in_window=1,
                num_columns_in_window=1,
                num_heights_in_window=num_time_steps,
                num_rows_per_stride=1,
                num_columns_per_stride=1,
                num_heights_per_stride=num_time_steps,
                pooling_type_string=architecture_utils.MEAN_POOLING_STRING,
                layer_name=this_name
            )(input_layer_object)

            new_dims = (
                __dimension_to_int(this_layer_object.shape[1]),
                __dimension_to_int(this_layer_object.shape[2]),
                __dimension_to_int(this_layer_object.shape[-1])
            )

            this_name = '{0:s}_preresidual_squeeze'.format(basic_layer_name)
            this_layer_object = keras.layers.Reshape(
                target_shape=new_dims, name=this_name
            )(this_layer_object)

            this_name = '{0:s}_residual'.format(basic_layer_name)
            current_layer_object = keras.layers.Add(name=this_name)([
                current_layer_object, this_layer_object
            ])

        this_name = '{0:s}_activ{1:d}'.format(basic_layer_name, i)
        current_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=activation_function_name,
            alpha_for_relu=activation_function_alpha,
            alpha_for_elu=activation_function_alpha,
            layer_name=this_name
        )(current_layer_object)

        if dropout_rates[i] > 0:
            this_name = '{0:s}_dropout{1:d}'.format(basic_layer_name, i)
            current_layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=dropout_rates[i], layer_name=this_name
            )(current_layer_object)

        if use_batch_norm:
            this_name = '{0:s}_bn{1:d}'.format(basic_layer_name, i)
            current_layer_object = architecture_utils.get_batch_norm_layer(
                layer_name=this_name
            )(current_layer_object)

    return current_layer_object


def check_input_args(option_dict):
    """Error-checks input arguments.

    B = number of convolutional blocks
    F = number of convolutional layers in forecasting module
    D = number of dense layers

    :param option_dict: Dictionary with the following keys.
    option_dict["input_dimensions_low_res"]:
        numpy array with input dimensions for low-resolution satellite data:
        [num_grid_rows, num_grid_columns, num_lag_times, num_wavelengths].
    option_dict["input_dimensions_high_res"]:
        numpy array with input dimensions for high-resolution satellite data:
        [num_grid_rows, num_grid_columns, num_lag_times, num_wavelengths].
        If you are not including high-res data, make this None.
    option_dict["input_dimensions_scalar"]:
        numpy array with input dimensions for scalar data: [num_fields].
        If you are not including scalar data, make this None.
    option_dict["include_high_res_data"]: Boolean flag.  If True, will create
        architecture that includes high-resolution satellite data (1/4 the grid
        spacing of low-resolution data) as input.
    option_dict["include_scalar_data"]: Boolean flag.  If True, will create
        architecture that includes scalar data as input.
    option_dict["num_conv_layers_by_block"]: length-B numpy array with number of
        conv layers for each block.
    option_dict["do_convnext"]: Boolean flag.  If True (False), will use
        ConvNeXt (basic convolutional blocks).
    option_dict["pooling_size_by_block_px"]: length-B numpy array with size of
        max-pooling window for each block.  For example, if you want 2-by-2
        pooling in the [j]th block, make pooling_size_by_block_px[j] = 2.
    option_dict["num_channels_by_conv_block"]: length-B numpy array with number
        of channels for each conv block.
    option_dict["dropout_rate_by_conv_block"]: length-B numpy array with dropout
        rate for each conv block.  Use number <= 0 to indicate no-dropout.
    option_dict["forecast_module_num_conv_layers"]: F in the above definitions.
    option_dict["forecast_module_dropout_rates"]: length-F numpy array with
        dropout rate for each conv layer in forecasting module.  Use number
        <= 0 to indicate no-dropout.
    option_dict["forecast_module_use_3d_conv"]: Boolean flag.  Determines
        whether forecasting module will use 2-D or 3-D convolution.
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
    option_dict["optimizer_function"]: Optimizer function.
    option_dict["start_with_pooling_layer"]: Boolean flag.

    :return: option_dict: Same as input but maybe with default values added.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    error_checking.assert_is_numpy_array(
        option_dict[INPUT_DIMENSIONS_LOW_RES_KEY],
        exact_dimensions=numpy.array([4], dtype=int)
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
            exact_dimensions=numpy.array([4], dtype=int)
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

    error_checking.assert_is_boolean(option_dict[INCLUDE_SCALAR_DATA_KEY])
    if option_dict[INCLUDE_SCALAR_DATA_KEY]:
        error_checking.assert_is_numpy_array(
            option_dict[INPUT_DIMENSIONS_SCALAR_KEY],
            exact_dimensions=numpy.array([1], dtype=int)
        )
        error_checking.assert_is_integer_numpy_array(
            option_dict[INPUT_DIMENSIONS_SCALAR_KEY]
        )
        error_checking.assert_is_greater_numpy_array(
            option_dict[INPUT_DIMENSIONS_SCALAR_KEY], 0
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
    error_checking.assert_is_boolean(option_dict[USE_RESIDUAL_BLOCKS_KEY])
    error_checking.assert_is_boolean(option_dict[USE_CONVNEXT_BLOCKS_KEY])

    num_conv_blocks = len(option_dict[NUM_CONV_LAYERS_KEY])

    error_checking.assert_is_numpy_array(
        option_dict[POOLING_SIZE_KEY],
        exact_dimensions=numpy.array([num_conv_blocks], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[POOLING_SIZE_KEY]
    )
    error_checking.assert_is_geq_numpy_array(
        option_dict[POOLING_SIZE_KEY], 2
    )

    error_checking.assert_is_numpy_array(
        option_dict[NUM_CHANNELS_KEY],
        exact_dimensions=numpy.array([num_conv_blocks], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(option_dict[NUM_CHANNELS_KEY])
    error_checking.assert_is_geq_numpy_array(option_dict[NUM_CHANNELS_KEY], 1)

    error_checking.assert_is_numpy_array(
        option_dict[CONV_DROPOUT_RATES_KEY],
        exact_dimensions=numpy.array([num_conv_blocks], dtype=int)
    )
    error_checking.assert_is_leq_numpy_array(
        option_dict[CONV_DROPOUT_RATES_KEY], 1.
    )

    fc_module_num_conv_layers = option_dict[FC_MODULE_NUM_CONV_LAYERS_KEY]
    error_checking.assert_is_integer(fc_module_num_conv_layers)
    error_checking.assert_is_greater(fc_module_num_conv_layers, 0)

    expected_dim = numpy.array([fc_module_num_conv_layers], dtype=int)

    fc_module_dropout_rates = option_dict[FC_MODULE_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        fc_module_dropout_rates, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        fc_module_dropout_rates, 1., allow_nan=True
    )

    error_checking.assert_is_boolean(option_dict[FC_MODULE_USE_3D_CONV])

    error_checking.assert_is_numpy_array(
        option_dict[NUM_NEURONS_KEY], num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(option_dict[NUM_NEURONS_KEY])
    # error_checking.assert_is_geq_numpy_array(option_dict[NUM_NEURONS_KEY], 2)

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
    error_checking.assert_is_boolean(option_dict[START_WITH_POOLING_KEY])

    return option_dict


def create_model(option_dict):
    """Creates CNN.

    :param option_dict: See documentation for `check_input_args`.
    :return: model_object: Untrained (but compiled) instance of
        `keras.models.Model`.
    """

    option_dict = check_input_args(option_dict)

    input_dimensions_low_res = option_dict[INPUT_DIMENSIONS_LOW_RES_KEY]
    include_high_res_data = option_dict[INCLUDE_HIGH_RES_KEY]
    include_scalar_data = option_dict[INCLUDE_SCALAR_DATA_KEY]
    num_conv_layers_by_block = option_dict[NUM_CONV_LAYERS_KEY]
    use_residual_blocks = option_dict[USE_RESIDUAL_BLOCKS_KEY]
    use_convnext_blocks = option_dict[USE_CONVNEXT_BLOCKS_KEY]
    pooling_size_by_block_px = option_dict[POOLING_SIZE_KEY]
    num_channels_by_conv_block = option_dict[NUM_CHANNELS_KEY]
    dropout_rate_by_conv_block = option_dict[CONV_DROPOUT_RATES_KEY]
    forecast_module_num_conv_layers = option_dict[FC_MODULE_NUM_CONV_LAYERS_KEY]
    forecast_module_dropout_rates = option_dict[FC_MODULE_DROPOUT_RATES_KEY]
    forecast_module_use_3d_conv = option_dict[FC_MODULE_USE_3D_CONV]
    num_neurons_by_dense_layer = option_dict[NUM_NEURONS_KEY]
    dropout_rate_by_dense_layer = option_dict[DENSE_DROPOUT_RATES_KEY]
    inner_activ_function_name = option_dict[INNER_ACTIV_FUNCTION_KEY]
    inner_activ_function_alpha = option_dict[INNER_ACTIV_FUNCTION_ALPHA_KEY]
    l2_weight = option_dict[L2_WEIGHT_KEY]
    use_batch_normalization = option_dict[USE_BATCH_NORM_KEY]
    loss_function = option_dict[LOSS_FUNCTION_KEY]
    optimizer_function = option_dict[OPTIMIZER_FUNCTION_KEY]
    ensemble_size = option_dict[ENSEMBLE_SIZE_KEY]
    start_with_pooling_layer = option_dict[START_WITH_POOLING_KEY]

    input_layer_object_low_res = layers.Input(
        shape=tuple(input_dimensions_low_res.tolist())
    )
    layer_object_low_res = layers.Permute(
        dims=(3, 1, 2, 4), name='low_res_put-time-first'
    )(input_layer_object_low_res)

    num_lag_times = input_dimensions_low_res[2]

    if start_with_pooling_layer:
        layer_object_low_res = (
            architecture_utils.get_2d_pooling_layer(
                num_rows_in_window=2, num_columns_in_window=2,
                num_rows_per_stride=2, num_columns_per_stride=2,
                pooling_type_string=architecture_utils.MAX_POOLING_STRING
            )(input_layer_object_low_res)
        )

    if include_high_res_data:
        input_dimensions_high_res = option_dict[INPUT_DIMENSIONS_HIGH_RES_KEY]
        input_layer_object_high_res = layers.Input(
            shape=tuple(input_dimensions_high_res.tolist())
        )
        layer_object_high_res = layers.Permute(
            dims=(3, 1, 2, 4), name='high_res_put-time-first'
        )(input_layer_object_high_res)

        if start_with_pooling_layer:
            layer_object_high_res = (
                architecture_utils.get_2d_pooling_layer(
                    num_rows_in_window=2, num_columns_in_window=2,
                    num_rows_per_stride=2, num_columns_per_stride=2,
                    pooling_type_string=architecture_utils.MAX_POOLING_STRING
                )(input_layer_object_high_res)
            )
    else:
        input_layer_object_high_res = None
        layer_object_high_res = None

    if include_scalar_data:
        input_dimensions_scalar = option_dict[INPUT_DIMENSIONS_SCALAR_KEY]
        input_layer_object_scalar = layers.Input(
            shape=tuple(input_dimensions_scalar.tolist())
        )
    else:
        input_layer_object_scalar = None

    l2_function = architecture_utils.get_weight_regularizer(l2_weight=l2_weight)
    layer_object = None

    num_conv_blocks = len(num_conv_layers_by_block)
    conv_layer_objects = [None] * num_conv_blocks
    forecast_module_layer_objects = [None] * num_conv_blocks
    pooling_layer_objects = [None] * (num_conv_blocks - 1)

    if include_high_res_data:
        for i in range(2):
            conv_layer_objects[i] = _get_2d_conv_block(
                input_layer_object=(
                    layer_object_high_res if layer_object is None
                    else layer_object
                ),
                do_residual=use_residual_blocks,
                do_convnext=use_convnext_blocks,
                num_conv_layers=num_conv_layers_by_block[i],
                filter_size_px=3,
                num_filters=num_channels_by_conv_block[i],
                do_time_distributed_conv=True,
                regularizer_object=l2_function,
                activation_function_name=inner_activ_function_name,
                activation_function_alpha=inner_activ_function_alpha,
                dropout_rates=dropout_rate_by_conv_block[i],
                use_batch_norm=use_batch_normalization,
                basic_layer_name='encoder_level{0:d}'.format(i)
            )

            this_name = 'fcst_level{0:d}_put-time-last'.format(i)
            forecast_module_layer_objects[i] = layers.Permute(
                dims=(2, 3, 1, 4), name=this_name
            )(conv_layer_objects[i])

            if num_lag_times == 1:
                orig_dims = forecast_module_layer_objects[i].shape
                orig_dims = numpy.array(
                    [__dimension_to_int(d) for d in orig_dims], dtype=int
                )
                new_dims = tuple(orig_dims[1:-2].tolist()) + (orig_dims[-1],)

                this_name = 'fcst_level{0:d}_remove-time-dim'.format(i)
                forecast_module_layer_objects[i] = keras.layers.Reshape(
                    target_shape=new_dims, name=this_name
                )(forecast_module_layer_objects[i])

            elif forecast_module_use_3d_conv:
                forecast_module_layer_objects[i] = _get_3d_conv_block(
                    input_layer_object=forecast_module_layer_objects[i],
                    do_residual=use_residual_blocks,
                    do_convnext=use_convnext_blocks,
                    num_conv_layers=forecast_module_num_conv_layers,
                    filter_size_px=1,
                    regularizer_object=l2_function,
                    activation_function_name=inner_activ_function_name,
                    activation_function_alpha=inner_activ_function_alpha,
                    dropout_rates=forecast_module_dropout_rates,
                    use_batch_norm=use_batch_normalization,
                    basic_layer_name='fcst_level{0:d}'.format(i)
                )
            else:
                orig_dims = forecast_module_layer_objects[i].shape
                orig_dims = numpy.array(
                    [__dimension_to_int(d) for d in orig_dims], dtype=int
                )
                new_dims = (
                    tuple(orig_dims[1:-2].tolist()) +
                    (orig_dims[-2] * orig_dims[-1],)
                )

                this_name = 'fcst_level{0:d}_remove-time-dim'.format(i)
                forecast_module_layer_objects[i] = keras.layers.Reshape(
                    target_shape=new_dims, name=this_name
                )(forecast_module_layer_objects[i])

                forecast_module_layer_objects[i] = _get_2d_conv_block(
                    input_layer_object=forecast_module_layer_objects[i],
                    do_residual=use_residual_blocks,
                    do_convnext=use_convnext_blocks,
                    num_conv_layers=forecast_module_num_conv_layers,
                    filter_size_px=1,
                    num_filters=num_channels_by_conv_block[i],
                    do_time_distributed_conv=False,
                    regularizer_object=l2_function,
                    activation_function_name=inner_activ_function_name,
                    activation_function_alpha=inner_activ_function_alpha,
                    dropout_rates=forecast_module_dropout_rates,
                    use_batch_norm=use_batch_normalization,
                    basic_layer_name='fcst_level{0:d}'.format(i)
                )

            this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
                num_rows_in_window=pooling_size_by_block_px[i],
                num_columns_in_window=pooling_size_by_block_px[i],
                num_rows_per_stride=pooling_size_by_block_px[i],
                num_columns_per_stride=pooling_size_by_block_px[i],
                pooling_type_string=architecture_utils.MAX_POOLING_STRING
            )
            pooling_layer_objects[i] = layers.TimeDistributed(
                this_pooling_layer_object
            )(conv_layer_objects[i])

        layer_object = layers.Concatenate(axis=-1)([
            pooling_layer_objects[i], layer_object_low_res
        ])
    else:
        layer_object = layer_object_low_res

    start_index = 2 if include_high_res_data else 0

    for i in range(start_index, num_conv_blocks):
        conv_layer_objects[i] = _get_2d_conv_block(
            input_layer_object=(
                layer_object if i == start_index
                else pooling_layer_objects[i - 1]
            ),
            do_residual=use_residual_blocks,
            do_convnext=use_convnext_blocks,
            num_conv_layers=num_conv_layers_by_block[i],
            filter_size_px=3,
            num_filters=num_channels_by_conv_block[i],
            do_time_distributed_conv=True,
            regularizer_object=l2_function,
            activation_function_name=inner_activ_function_name,
            activation_function_alpha=inner_activ_function_alpha,
            dropout_rates=dropout_rate_by_conv_block[i],
            use_batch_norm=use_batch_normalization,
            basic_layer_name='encoder_level{0:d}'.format(i)
        )

        this_name = 'fcst_level{0:d}_put-time-last'.format(i)
        forecast_module_layer_objects[i] = layers.Permute(
            dims=(2, 3, 1, 4), name=this_name
        )(conv_layer_objects[i])

        if num_lag_times == 1:
            orig_dims = forecast_module_layer_objects[i].shape
            orig_dims = numpy.array(
                [__dimension_to_int(d) for d in orig_dims], dtype=int
            )
            new_dims = tuple(orig_dims[1:-2].tolist()) + (orig_dims[-1],)

            this_name = 'fcst_level{0:d}_remove-time-dim'.format(i)
            forecast_module_layer_objects[i] = keras.layers.Reshape(
                target_shape=new_dims, name=this_name
            )(forecast_module_layer_objects[i])

        elif forecast_module_use_3d_conv:
            forecast_module_layer_objects[i] = _get_3d_conv_block(
                input_layer_object=forecast_module_layer_objects[i],
                do_residual=use_residual_blocks,
                do_convnext=use_convnext_blocks,
                num_conv_layers=forecast_module_num_conv_layers,
                filter_size_px=1,
                regularizer_object=l2_function,
                activation_function_name=inner_activ_function_name,
                activation_function_alpha=inner_activ_function_alpha,
                dropout_rates=forecast_module_dropout_rates,
                use_batch_norm=use_batch_normalization,
                basic_layer_name='fcst_level{0:d}'.format(i)
            )
        else:
            orig_dims = forecast_module_layer_objects[i].shape
            orig_dims = numpy.array(
                [__dimension_to_int(d) for d in orig_dims], dtype=int
            )
            new_dims = (
                tuple(orig_dims[1:-2].tolist()) +
                (orig_dims[-2] * orig_dims[-1],)
            )

            this_name = 'fcst_level{0:d}_remove-time-dim'.format(i)
            forecast_module_layer_objects[i] = keras.layers.Reshape(
                target_shape=new_dims, name=this_name
            )(forecast_module_layer_objects[i])

            forecast_module_layer_objects[i] = _get_2d_conv_block(
                input_layer_object=forecast_module_layer_objects[i],
                do_residual=use_residual_blocks,
                do_convnext=use_convnext_blocks,
                num_conv_layers=forecast_module_num_conv_layers,
                filter_size_px=1,
                num_filters=num_channels_by_conv_block[i],
                do_time_distributed_conv=False,
                regularizer_object=l2_function,
                activation_function_name=inner_activ_function_name,
                activation_function_alpha=inner_activ_function_alpha,
                dropout_rates=forecast_module_dropout_rates,
                use_batch_norm=use_batch_normalization,
                basic_layer_name='fcst_level{0:d}'.format(i)
            )

        if i == num_conv_blocks - 1:
            continue

        this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
            num_rows_in_window=pooling_size_by_block_px[i],
            num_columns_in_window=pooling_size_by_block_px[i],
            num_rows_per_stride=pooling_size_by_block_px[i],
            num_columns_per_stride=pooling_size_by_block_px[i],
            pooling_type_string=architecture_utils.MAX_POOLING_STRING
        )
        pooling_layer_objects[i] = layers.TimeDistributed(
            this_pooling_layer_object
        )(conv_layer_objects[i])

    layer_object = architecture_utils.get_flattening_layer()(
        conv_layer_objects[-1]
    )
    if include_scalar_data:
        layer_object = layers.Concatenate(axis=-1)([
            layer_object, input_layer_object_scalar
        ])

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

    num_target_vars = float(num_neurons_by_dense_layer[-1]) / ensemble_size
    assert numpy.isclose(
        num_target_vars, numpy.round(num_target_vars), atol=1e-6
    )

    num_target_vars = int(numpy.round(num_target_vars))
    layer_object = layers.Reshape(
        target_shape=(num_target_vars, ensemble_size)
    )(layer_object)

    if include_high_res_data:
        input_layer_objects = [
            input_layer_object_high_res, input_layer_object_low_res
        ]
        if include_scalar_data:
            input_layer_objects.append(input_layer_object_scalar)
    else:
        if include_scalar_data:
            input_layer_objects = [
                input_layer_object_low_res, input_layer_object_scalar
            ]
        else:
            input_layer_objects = input_layer_object_low_res

    model_object = keras.models.Model(
        inputs=input_layer_objects, outputs=layer_object
    )
    model_object.compile(
        loss=loss_function, optimizer=optimizer_function,
        metrics=neural_net_utils.METRIC_FUNCTION_LIST_SCALAR
    )
    model_object.summary()

    return model_object
