"""Temporal-CNN architecture to predict TC-structure parameters."""

import os
import sys
import numpy
import keras
import keras.layers as layers
from tensorflow.keras import backend as K

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import architecture_utils
import temporal_cnn_architecture as basic_arch

try:
    input_layer_object_low_res = layers.Input(shape=(3, 4, 5))
except:
    import tensorflow.keras as keras
    import tensorflow.keras.layers as layers

NUM_STRUCTURE_PARAMETERS = 5

INPUT_DIMENSIONS_LOW_RES_KEY = basic_arch.INPUT_DIMENSIONS_LOW_RES_KEY
INPUT_DIMENSIONS_HIGH_RES_KEY = basic_arch.INPUT_DIMENSIONS_HIGH_RES_KEY
INPUT_DIMENSIONS_SCALAR_KEY = basic_arch.INPUT_DIMENSIONS_SCALAR_KEY
INCLUDE_HIGH_RES_KEY = basic_arch.INCLUDE_HIGH_RES_KEY
INCLUDE_SCALAR_DATA_KEY = basic_arch.INCLUDE_SCALAR_DATA_KEY
NUM_CONV_LAYERS_KEY = basic_arch.NUM_CONV_LAYERS_KEY
POOLING_SIZE_KEY = basic_arch.POOLING_SIZE_KEY
NUM_CHANNELS_KEY = basic_arch.NUM_CHANNELS_KEY
CONV_DROPOUT_RATES_KEY = basic_arch.CONV_DROPOUT_RATES_KEY

FC_MODULE_NUM_CONV_LAYERS_KEY = basic_arch.FC_MODULE_NUM_CONV_LAYERS_KEY
FC_MODULE_DROPOUT_RATES_KEY = basic_arch.FC_MODULE_DROPOUT_RATES_KEY
FC_MODULE_USE_3D_CONV = basic_arch.FC_MODULE_USE_3D_CONV

NUM_NEURONS_KEY = basic_arch.NUM_NEURONS_KEY
DENSE_DROPOUT_RATES_KEY = basic_arch.DENSE_DROPOUT_RATES_KEY
INNER_ACTIV_FUNCTION_KEY = basic_arch.INNER_ACTIV_FUNCTION_KEY
INNER_ACTIV_FUNCTION_ALPHA_KEY = basic_arch.INNER_ACTIV_FUNCTION_ALPHA_KEY
L2_WEIGHT_KEY = basic_arch.L2_WEIGHT_KEY
USE_BATCH_NORM_KEY = basic_arch.USE_BATCH_NORM_KEY
ENSEMBLE_SIZE_KEY = basic_arch.ENSEMBLE_SIZE_KEY
LOSS_FUNCTION_KEY = basic_arch.LOSS_FUNCTION_KEY
OPTIMIZER_FUNCTION_KEY = basic_arch.OPTIMIZER_FUNCTION_KEY
START_WITH_POOLING_KEY = basic_arch.START_WITH_POOLING_KEY

INTENSITY_INDEX_KEY = 'intensity_index'
R34_INDEX_KEY = 'r34_index'
R50_INDEX_KEY = 'r50_index'
R64_INDEX_KEY = 'r64_index'
RMW_INDEX_KEY = 'rmw_index'
USE_PHYSICAL_CONSTRAINTS_KEY = 'use_physical_constraints'
DO_RESIDUAL_PREDICTION_KEY = 'do_residual_prediction'
PREDICT_INTENSITY_ONLY_KEY = 'predict_intensity_only'
METRIC_FUNCTIONS_KEY = 'metric_functions'

DEFAULT_OPTION_DICT = {
    INCLUDE_HIGH_RES_KEY: False,
    NUM_CONV_LAYERS_KEY: numpy.full(11, 2, dtype=int),
    POOLING_SIZE_KEY: numpy.full(10, 2, dtype=int),
    NUM_CHANNELS_KEY: numpy.array([
        2, 2, 4, 4, 16, 16, 32, 32, 64, 64, 96, 96, 128, 128, 184, 184,
        256, 256, 384, 384, 512, 512
    ], dtype=int),
    CONV_DROPOUT_RATES_KEY: numpy.full(22, 0.),
    FC_MODULE_NUM_CONV_LAYERS_KEY: 1,
    FC_MODULE_DROPOUT_RATES_KEY: numpy.array([0.]),
    FC_MODULE_USE_3D_CONV: True,
    NUM_NEURONS_KEY: numpy.array([1024, 128, 16, 2], dtype=int),
    DENSE_DROPOUT_RATES_KEY: numpy.array([0.5, 0.5, 0.5, 0]),
    INNER_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    USE_BATCH_NORM_KEY: True,
    START_WITH_POOLING_KEY: False,
    INTENSITY_INDEX_KEY: 0,
    R34_INDEX_KEY: 1,
    R50_INDEX_KEY: 2,
    R64_INDEX_KEY: 3,
    RMW_INDEX_KEY: 4,
    PREDICT_INTENSITY_ONLY_KEY: False
}


class PhysicalConstraintLayer(layers.Layer):
    def __init__(self, intensity_index, r34_index, r50_index, r64_index,
                 rmw_index, **kwargs):
        super(PhysicalConstraintLayer, self).__init__(**kwargs)

        self.intensity_index = intensity_index
        self.r34_index = r34_index
        self.r50_index = r50_index
        self.r64_index = r64_index
        self.rmw_index = rmw_index

    def call(self, inputs):
        new_intensity_tensor = inputs[..., self.intensity_index]
        new_r34_tensor = inputs[..., self.r34_index]
        new_r50_tensor = inputs[..., self.r50_index]
        new_r64_tensor = inputs[..., self.r64_index]
        new_rmw_tensor = inputs[..., self.rmw_index]

        new_r50_tensor = new_r50_tensor + new_r64_tensor
        new_r34_tensor = new_r34_tensor + new_r50_tensor

        # new_r34_tensor = tensorflow.where(
        #     inputs[..., self.intensity_index] < 34.,
        #     tensorflow.zeros_like(new_r34_tensor),
        #     new_r34_tensor
        # )
        # new_r50_tensor = tensorflow.where(
        #     inputs[..., self.intensity_index] < 50.,
        #     tensorflow.zeros_like(new_r50_tensor),
        #     new_r50_tensor
        # )
        # new_r64_tensor = tensorflow.where(
        #     inputs[..., self.intensity_index] < 64.,
        #     tensorflow.zeros_like(new_r64_tensor),
        #     new_r64_tensor
        # )

        new_tensors = [
            K.expand_dims(new_intensity_tensor, axis=-1),
            K.expand_dims(new_r34_tensor, axis=-1),
            K.expand_dims(new_r50_tensor, axis=-1),
            K.expand_dims(new_r64_tensor, axis=-1),
            K.expand_dims(new_rmw_tensor, axis=-1)
        ]

        new_indices = numpy.array([
            self.intensity_index,
            self.r34_index,
            self.r50_index,
            self.r64_index,
            self.rmw_index
        ], dtype=int)

        new_tensors = [new_tensors[i] for i in numpy.argsort(new_indices)]
        new_inputs = K.concatenate(new_tensors, axis=-1)

        return new_inputs


def check_input_args(option_dict):
    """Error-checks input arguments.

    :param option_dict: Dictionary with the following keys.
    option_dict["input_dimensions_low_res"]: See documentation for
        `temporal_cnn_architecture.check_input_args`.
    option_dict["input_dimensions_high_res"]: Same.
    option_dict["input_dimensions_scalar"]: Same.
    option_dict["include_high_res_data"]: Same.
    option_dict["include_scalar_data"]: Same.
    option_dict["num_conv_layers_by_block"]: Same.
    option_dict["pooling_size_by_block_px"]: Same.
    option_dict["num_channels_by_conv_layer"]: Same.
    option_dict["dropout_rate_by_conv_layer"]: Same.
    option_dict["forecast_module_num_conv_layers"]: Same.
    option_dict["forecast_module_dropout_rates"]: Same.
    option_dict["forecast_module_use_3d_conv"]: Same.
    option_dict["num_neurons_by_dense_layer"]: Same.
    option_dict["dropout_rate_by_dense_layer"]: Same.
    option_dict["inner_activ_function_name"]: Same.
    option_dict["inner_activ_function_alpha"]: Same.
    option_dict["l2_weight"]: Same.
    option_dict["use_batch_normalization"]: Same.
    option_dict["ensemble_size"]: Same.
    option_dict["loss_function"]: Same.
    option_dict["optimizer_function"]: Same.
    option_dict["start_with_pooling_layer"]: Same.
    option_dict["intensity_index"]: Channel index for TC intensity.
    option_dict["r34_index"]: Channel index for radius of 34-kt wind.
    option_dict["r50_index"]: Channel index for radius of 50-kt wind.
    option_dict["r64_index"]: Channel index for radius of 64-kt wind.
    option_dict["rmw_index"]: Channel index for radius of max wind.
    option_dict["use_physical_constraints"]: Boolean flag.  If True, will use
        physical constraints to make predictions of TC-structure parameters
        consistent.
    option_dict["do_residual_prediction"]: Boolean flag.  If True, will do
        residual prediction.
    option_dict["predict_intensity_only"]: Boolean flag.  If True, will predict
        intensity only and not the other structure parameters.
    option_dict["metric_functions"]: 1-D list of metric functions.

    :return: option_dict: Same as input but maybe with default values added.
    """

    option_dict = basic_arch.check_input_args(option_dict)

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    error_checking.assert_is_list(option_dict[METRIC_FUNCTIONS_KEY])

    error_checking.assert_is_boolean(option_dict[PREDICT_INTENSITY_ONLY_KEY])
    if option_dict[PREDICT_INTENSITY_ONLY_KEY]:
        option_dict[USE_PHYSICAL_CONSTRAINTS_KEY] = False
        option_dict[DO_RESIDUAL_PREDICTION_KEY] = False
        return option_dict

    error_checking.assert_is_integer(option_dict[INTENSITY_INDEX_KEY])
    error_checking.assert_is_geq(option_dict[INTENSITY_INDEX_KEY], 0)
    error_checking.assert_is_integer(option_dict[R34_INDEX_KEY])
    error_checking.assert_is_geq(option_dict[R34_INDEX_KEY], 0)
    error_checking.assert_is_integer(option_dict[R50_INDEX_KEY])
    error_checking.assert_is_geq(option_dict[R50_INDEX_KEY], 0)
    error_checking.assert_is_integer(option_dict[R64_INDEX_KEY])
    error_checking.assert_is_geq(option_dict[R64_INDEX_KEY], 0)
    error_checking.assert_is_integer(option_dict[RMW_INDEX_KEY])
    error_checking.assert_is_geq(option_dict[RMW_INDEX_KEY], 0)
    error_checking.assert_is_boolean(option_dict[USE_PHYSICAL_CONSTRAINTS_KEY])
    error_checking.assert_is_boolean(option_dict[DO_RESIDUAL_PREDICTION_KEY])

    all_indices = numpy.array([
        option_dict[INTENSITY_INDEX_KEY], option_dict[R34_INDEX_KEY],
        option_dict[R50_INDEX_KEY], option_dict[R64_INDEX_KEY],
        option_dict[RMW_INDEX_KEY]
    ], dtype=int)

    error_checking.assert_equals(
        len(numpy.unique(all_indices)),
        len(all_indices)
    )

    return option_dict


def create_model(option_dict):
    """Creates CNN for TC-structure parameters.

    :param option_dict: See documentation for `check_input_args`.
    :return: model_object: Untrained (but compiled) instance of
        `keras.models.Model`.
    """

    option_dict = check_input_args(option_dict)

    input_dimensions_low_res = option_dict[INPUT_DIMENSIONS_LOW_RES_KEY]
    include_high_res_data = option_dict[INCLUDE_HIGH_RES_KEY]
    include_scalar_data = option_dict[INCLUDE_SCALAR_DATA_KEY]
    num_conv_layers_by_block = option_dict[NUM_CONV_LAYERS_KEY]
    pooling_size_by_block_px = option_dict[POOLING_SIZE_KEY]
    num_channels_by_conv_layer = option_dict[NUM_CHANNELS_KEY]
    dropout_rate_by_conv_layer = option_dict[CONV_DROPOUT_RATES_KEY]
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

    intensity_index = option_dict[INTENSITY_INDEX_KEY]
    r34_index = option_dict[R34_INDEX_KEY]
    r50_index = option_dict[R50_INDEX_KEY]
    r64_index = option_dict[R64_INDEX_KEY]
    rmw_index = option_dict[RMW_INDEX_KEY]
    use_physical_constraints = option_dict[USE_PHYSICAL_CONSTRAINTS_KEY]
    do_residual_prediction = option_dict[DO_RESIDUAL_PREDICTION_KEY]
    metric_functions = option_dict[METRIC_FUNCTIONS_KEY]

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

    num_target_vars = float(num_neurons_by_dense_layer[-1]) / ensemble_size
    assert numpy.isclose(
        num_target_vars, numpy.round(num_target_vars), atol=1e-6
    )
    num_target_vars = int(numpy.round(num_target_vars))

    if do_residual_prediction:
        input_layer_object_resid_baseline = layers.Input(
            shape=(num_target_vars,)
        )
    else:
        input_layer_object_resid_baseline = None

    if include_scalar_data:
        input_dimensions_scalar = option_dict[INPUT_DIMENSIONS_SCALAR_KEY]
        input_layer_object_scalar = layers.Input(
            shape=tuple(input_dimensions_scalar.tolist())
        )
    else:
        input_layer_object_scalar = None

    l2_function = architecture_utils.get_weight_regularizer(l2_weight=l2_weight)
    layer_index = -1
    layer_object = None

    if include_high_res_data:
        for block_index in range(2):
            for _ in range(num_conv_layers_by_block[block_index]):
                layer_index += 1
                k = layer_index

                this_conv_layer_object = architecture_utils.get_2d_conv_layer(
                    num_kernel_rows=3, num_kernel_columns=3,
                    num_rows_per_stride=1, num_columns_per_stride=1,
                    num_filters=num_channels_by_conv_layer[k],
                    padding_type_string=
                    architecture_utils.YES_PADDING_STRING,
                    weight_regularizer=l2_function
                )
                if k == 0:
                    layer_object = layers.TimeDistributed(
                        this_conv_layer_object
                    )(layer_object_high_res)
                else:
                    layer_object = layers.TimeDistributed(
                        this_conv_layer_object
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

            this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
                num_rows_in_window=pooling_size_by_block_px[block_index],
                num_columns_in_window=pooling_size_by_block_px[block_index],
                num_rows_per_stride=pooling_size_by_block_px[block_index],
                num_columns_per_stride=pooling_size_by_block_px[block_index],
                pooling_type_string=architecture_utils.MAX_POOLING_STRING
            )
            layer_object = layers.TimeDistributed(
                this_pooling_layer_object
            )(layer_object)

        layer_object = layers.Concatenate(axis=-1)([
            layer_object, layer_object_low_res
        ])
    else:
        layer_object = layer_object_low_res

    num_conv_blocks = len(num_conv_layers_by_block)
    start_index = 2 if include_high_res_data else 0

    for block_index in range(start_index, num_conv_blocks):
        for _ in range(num_conv_layers_by_block[block_index]):
            layer_index += 1
            k = layer_index

            this_conv_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=num_channels_by_conv_layer[k],
                padding_type_string=
                architecture_utils.YES_PADDING_STRING,
                weight_regularizer=l2_function
            )
            layer_object = layers.TimeDistributed(
                this_conv_layer_object
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

        if block_index != num_conv_blocks - 1:
            this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
                num_rows_in_window=pooling_size_by_block_px[block_index],
                num_columns_in_window=pooling_size_by_block_px[block_index],
                num_rows_per_stride=pooling_size_by_block_px[block_index],
                num_columns_per_stride=pooling_size_by_block_px[block_index],
                pooling_type_string=architecture_utils.MAX_POOLING_STRING
            )
            layer_object = layers.TimeDistributed(
                this_pooling_layer_object
            )(layer_object)

    forecast_module_layer_object = layers.Permute(
        dims=(2, 3, 1, 4), name='fcst_module_put-time-last'
    )(layer_object)

    if not forecast_module_use_3d_conv:
        orig_dims = forecast_module_layer_object.get_shape()
        new_dims = orig_dims[1:-2] + [orig_dims[-2] * orig_dims[-1]]

        forecast_module_layer_object = layers.Reshape(
            target_shape=new_dims, name='fcst_module_remove-time-dim'
        )(forecast_module_layer_object)

    for j in range(forecast_module_num_conv_layers):
        if forecast_module_use_3d_conv:
            if j == 0:
                forecast_module_layer_object = (
                    architecture_utils.get_3d_conv_layer(
                        num_kernel_rows=1, num_kernel_columns=1,
                        num_kernel_heights=num_lag_times,
                        num_rows_per_stride=1, num_columns_per_stride=1,
                        num_heights_per_stride=1,
                        num_filters=num_channels_by_conv_layer[-1],
                        padding_type_string=
                        architecture_utils.NO_PADDING_STRING,
                        weight_regularizer=l2_function
                    )(forecast_module_layer_object)
                )

                new_dims = (
                    forecast_module_layer_object.shape[1:-2] +
                    (forecast_module_layer_object.shape[-1],)
                )
                forecast_module_layer_object = layers.Reshape(
                    target_shape=new_dims, name='fcst_module_remove-time-dim'
                )(forecast_module_layer_object)
            else:
                forecast_module_layer_object = (
                    architecture_utils.get_2d_conv_layer(
                        num_kernel_rows=3, num_kernel_columns=3,
                        num_rows_per_stride=1, num_columns_per_stride=1,
                        num_filters=num_channels_by_conv_layer[-1],
                        padding_type_string=
                        architecture_utils.YES_PADDING_STRING,
                        weight_regularizer=l2_function
                    )(forecast_module_layer_object)
                )
        else:
            forecast_module_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=num_channels_by_conv_layer[-1],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=l2_function
            )(forecast_module_layer_object)

        forecast_module_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha
        )(forecast_module_layer_object)

        if forecast_module_dropout_rates[j] > 0:
            forecast_module_layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=forecast_module_dropout_rates[j]
            )(forecast_module_layer_object)

        if use_batch_normalization:
            forecast_module_layer_object = (
                architecture_utils.get_batch_norm_layer()(
                    forecast_module_layer_object
                )
            )

    layer_object = architecture_utils.get_flattening_layer()(
        forecast_module_layer_object
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
            # layer_object = architecture_utils.get_activation_layer(
            #     activation_function_string=
            #     architecture_utils.RELU_FUNCTION_STRING,
            #     alpha_for_relu=0.,
            #     alpha_for_elu=0.
            # )(layer_object)

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

    layer_object = layers.Reshape(
        target_shape=(num_target_vars, ensemble_size)
    )(layer_object)

    if do_residual_prediction:
        new_dims = (num_target_vars, 1)
        layer_object_resid_baseline = keras.layers.Reshape(
            target_shape=new_dims,
            name='reshape_predn_baseline'
        )(input_layer_object_resid_baseline)

        layer_object = keras.layers.Add(name='output_add_baseline')([
            layer_object, layer_object_resid_baseline
        ])

    layer_object = architecture_utils.get_activation_layer(
        activation_function_string=architecture_utils.RELU_FUNCTION_STRING,
        alpha_for_relu=0.,
        alpha_for_elu=0.
    )(layer_object)

    if use_physical_constraints:
        layer_object = layers.Permute(
            dims=(2, 1), name='output_channels_last'
        )(layer_object)

        layer_object = PhysicalConstraintLayer(
            intensity_index=intensity_index,
            r34_index=r34_index,
            r50_index=r50_index,
            r64_index=r64_index,
            rmw_index=rmw_index,
            name='output_phys_constraints'
        )(layer_object)

        layer_object = layers.Permute(
            dims=(2, 1), name='output_channels_first'
        )(layer_object)

    input_layer_objects = []
    if include_high_res_data:
        input_layer_objects.append(input_layer_object_high_res)

    input_layer_objects.append(input_layer_object_low_res)
    if include_scalar_data:
        input_layer_objects.append(input_layer_object_scalar)
    if do_residual_prediction:
        input_layer_objects.append(input_layer_object_resid_baseline)

    model_object = keras.models.Model(
        inputs=input_layer_objects, outputs=layer_object
    )
    model_object.compile(
        loss=loss_function, optimizer=optimizer_function,
        metrics=metric_functions
    )
    model_object.summary()

    return model_object


def create_model_for_intensity(option_dict):
    """Creates CNN for TC intensity only.

    :param option_dict: See documentation for `check_input_args`.
    :return: model_object: Untrained (but compiled) instance of
        `keras.models.Model`.
    """

    option_dict = check_input_args(option_dict)

    input_dimensions_low_res = option_dict[INPUT_DIMENSIONS_LOW_RES_KEY]
    include_high_res_data = option_dict[INCLUDE_HIGH_RES_KEY]
    include_scalar_data = option_dict[INCLUDE_SCALAR_DATA_KEY]
    num_conv_layers_by_block = option_dict[NUM_CONV_LAYERS_KEY]
    pooling_size_by_block_px = option_dict[POOLING_SIZE_KEY]
    num_channels_by_conv_layer = option_dict[NUM_CHANNELS_KEY]
    dropout_rate_by_conv_layer = option_dict[CONV_DROPOUT_RATES_KEY]
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
    metric_functions = option_dict[METRIC_FUNCTIONS_KEY]

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

    num_target_vars = float(num_neurons_by_dense_layer[-1]) / ensemble_size
    assert numpy.isclose(
        num_target_vars, numpy.round(num_target_vars), atol=1e-6
    )
    num_target_vars = int(numpy.round(num_target_vars))

    if include_scalar_data:
        input_dimensions_scalar = option_dict[INPUT_DIMENSIONS_SCALAR_KEY]
        input_layer_object_scalar = layers.Input(
            shape=tuple(input_dimensions_scalar.tolist())
        )
    else:
        input_layer_object_scalar = None

    l2_function = architecture_utils.get_weight_regularizer(l2_weight=l2_weight)
    layer_index = -1
    layer_object = None

    if include_high_res_data:
        for block_index in range(2):
            for _ in range(num_conv_layers_by_block[block_index]):
                layer_index += 1
                k = layer_index

                this_conv_layer_object = architecture_utils.get_2d_conv_layer(
                    num_kernel_rows=3, num_kernel_columns=3,
                    num_rows_per_stride=1, num_columns_per_stride=1,
                    num_filters=num_channels_by_conv_layer[k],
                    padding_type_string=
                    architecture_utils.YES_PADDING_STRING,
                    weight_regularizer=l2_function
                )
                if k == 0:
                    layer_object = layers.TimeDistributed(
                        this_conv_layer_object
                    )(layer_object_high_res)
                else:
                    layer_object = layers.TimeDistributed(
                        this_conv_layer_object
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

            this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
                num_rows_in_window=pooling_size_by_block_px[block_index],
                num_columns_in_window=pooling_size_by_block_px[block_index],
                num_rows_per_stride=pooling_size_by_block_px[block_index],
                num_columns_per_stride=pooling_size_by_block_px[block_index],
                pooling_type_string=architecture_utils.MAX_POOLING_STRING
            )
            layer_object = layers.TimeDistributed(
                this_pooling_layer_object
            )(layer_object)

        layer_object = layers.Concatenate(axis=-1)([
            layer_object, layer_object_low_res
        ])
    else:
        layer_object = layer_object_low_res

    num_conv_blocks = len(num_conv_layers_by_block)
    start_index = 2 if include_high_res_data else 0

    for block_index in range(start_index, num_conv_blocks):
        for _ in range(num_conv_layers_by_block[block_index]):
            layer_index += 1
            k = layer_index

            this_conv_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=num_channels_by_conv_layer[k],
                padding_type_string=
                architecture_utils.YES_PADDING_STRING,
                weight_regularizer=l2_function
            )
            layer_object = layers.TimeDistributed(
                this_conv_layer_object
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

        if block_index != num_conv_blocks - 1:
            this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
                num_rows_in_window=pooling_size_by_block_px[block_index],
                num_columns_in_window=pooling_size_by_block_px[block_index],
                num_rows_per_stride=pooling_size_by_block_px[block_index],
                num_columns_per_stride=pooling_size_by_block_px[block_index],
                pooling_type_string=architecture_utils.MAX_POOLING_STRING
            )
            layer_object = layers.TimeDistributed(
                this_pooling_layer_object
            )(layer_object)

    forecast_module_layer_object = layers.Permute(
        dims=(2, 3, 1, 4), name='fcst_module_put-time-last'
    )(layer_object)

    if not forecast_module_use_3d_conv:
        orig_dims = forecast_module_layer_object.get_shape()
        new_dims = orig_dims[1:-2] + [orig_dims[-2] * orig_dims[-1]]

        forecast_module_layer_object = layers.Reshape(
            target_shape=new_dims, name='fcst_module_remove-time-dim'
        )(forecast_module_layer_object)

    for j in range(forecast_module_num_conv_layers):
        if forecast_module_use_3d_conv:
            if j == 0:
                forecast_module_layer_object = (
                    architecture_utils.get_3d_conv_layer(
                        num_kernel_rows=1, num_kernel_columns=1,
                        num_kernel_heights=num_lag_times,
                        num_rows_per_stride=1, num_columns_per_stride=1,
                        num_heights_per_stride=1,
                        num_filters=num_channels_by_conv_layer[-1],
                        padding_type_string=
                        architecture_utils.NO_PADDING_STRING,
                        weight_regularizer=l2_function
                    )(forecast_module_layer_object)
                )

                new_dims = (
                    forecast_module_layer_object.shape[1:-2] +
                    (forecast_module_layer_object.shape[-1],)
                )
                forecast_module_layer_object = layers.Reshape(
                    target_shape=new_dims, name='fcst_module_remove-time-dim'
                )(forecast_module_layer_object)
            else:
                forecast_module_layer_object = (
                    architecture_utils.get_2d_conv_layer(
                        num_kernel_rows=3, num_kernel_columns=3,
                        num_rows_per_stride=1, num_columns_per_stride=1,
                        num_filters=num_channels_by_conv_layer[-1],
                        padding_type_string=
                        architecture_utils.YES_PADDING_STRING,
                        weight_regularizer=l2_function
                    )(forecast_module_layer_object)
                )
        else:
            forecast_module_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=num_channels_by_conv_layer[-1],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=l2_function
            )(forecast_module_layer_object)

        forecast_module_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha
        )(forecast_module_layer_object)

        if forecast_module_dropout_rates[j] > 0:
            forecast_module_layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=forecast_module_dropout_rates[j]
            )(forecast_module_layer_object)

        if use_batch_normalization:
            forecast_module_layer_object = (
                architecture_utils.get_batch_norm_layer()(
                    forecast_module_layer_object
                )
            )

    layer_object = architecture_utils.get_flattening_layer()(
        forecast_module_layer_object
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
            layer_object = architecture_utils.get_activation_layer(
                activation_function_string=
                architecture_utils.RELU_FUNCTION_STRING,
                alpha_for_relu=0.,
                alpha_for_elu=0.
            )(layer_object)

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

    layer_object = layers.Reshape(
        target_shape=(num_target_vars, ensemble_size)
    )(layer_object)

    input_layer_objects = []
    if include_high_res_data:
        input_layer_objects.append(input_layer_object_high_res)

    input_layer_objects.append(input_layer_object_low_res)
    if include_scalar_data:
        input_layer_objects.append(input_layer_object_scalar)

    model_object = keras.models.Model(
        inputs=input_layer_objects, outputs=layer_object
    )
    model_object.compile(
        loss=loss_function, optimizer=optimizer_function,
        metrics=metric_functions
    )
    model_object.summary()

    return model_object
