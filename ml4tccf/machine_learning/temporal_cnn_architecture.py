"""Methods for creating temporal-CNN architecture.

This is a CNN with TimeDistributed layers to handle inputs (satellite images) at
different lag times.
"""

import numpy
import keras
import tensorflow
import keras.layers as layers
from tensorflow.keras import backend as K
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import architecture_utils
from ml4tccf.machine_learning import neural_net_utils
from ml4tccf.machine_learning import custom_metrics_structure

try:
    input_layer_object_low_res = layers.Input(shape=(3, 4, 5))
except:
    import tensorflow.keras as keras
    import tensorflow.keras.layers as layers

NUM_STRUCTURE_PARAMETERS = 5
KERNEL_INIT_NAME_FOR_STRUCTURE = 'he_uniform'
BIAS_INIT_NAME_FOR_STRUCTURE = 'zeros'

INPUT_DIMENSIONS_LOW_RES_KEY = 'input_dimensions_low_res'
INPUT_DIMENSIONS_HIGH_RES_KEY = 'input_dimensions_high_res'
INPUT_DIMENSIONS_SCALAR_KEY = 'input_dimensions_scalar'
INCLUDE_HIGH_RES_KEY = 'include_high_res_data'
INCLUDE_SCALAR_DATA_KEY = 'include_scalar_data'
NUM_CONV_LAYERS_KEY = 'num_conv_layers_by_block'
POOLING_SIZE_KEY = 'pooling_size_by_block_px'
NUM_CHANNELS_KEY = 'num_channels_by_conv_layer'
CONV_DROPOUT_RATES_KEY = 'dropout_rate_by_conv_layer'

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

INTENSITY_INDEX_KEY = 'intensity_index'
R34_INDEX_KEY = 'r34_index'
R50_INDEX_KEY = 'r50_index'
R64_INDEX_KEY = 'r64_index'
RMW_INDEX_KEY = 'rmw_index'
USE_PHYSICAL_CONSTRAINTS_KEY = 'use_physical_constraints'
DO_RESIDUAL_PREDICTION_KEY = 'do_residual_prediction'
PREDICT_INTENSITY_ONLY_KEY = 'predict_intensity_only'

DEFAULT_OPTION_DICT = {
    INCLUDE_HIGH_RES_KEY: True,
    INCLUDE_SCALAR_DATA_KEY: False,
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
    INTENSITY_INDEX_KEY: None,
    R34_INDEX_KEY: None,
    R50_INDEX_KEY: None,
    R64_INDEX_KEY: None,
    RMW_INDEX_KEY: None,
    USE_PHYSICAL_CONSTRAINTS_KEY: True,
    DO_RESIDUAL_PREDICTION_KEY: False,
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
        new_r50_tensor = (
            inputs[..., self.r50_index] + inputs[..., self.r64_index]
        )
        new_r34_tensor = (
            inputs[..., self.r34_index] + inputs[..., self.r50_index]
        )
        new_r64_tensor = inputs[..., self.r64_index]
        new_rmw_tensor = inputs[..., self.rmw_index]

        new_r34_tensor = tensorflow.where(
            inputs[..., self.intensity_index] < 34.,
            tensorflow.zeros_like(new_r34_tensor),
            new_r34_tensor
        )
        new_r50_tensor = tensorflow.where(
            inputs[..., self.intensity_index] < 50.,
            tensorflow.zeros_like(new_r50_tensor),
            new_r50_tensor
        )
        new_r64_tensor = tensorflow.where(
            inputs[..., self.intensity_index] < 64.,
            tensorflow.zeros_like(new_r64_tensor),
            new_r64_tensor
        )

        new_rmw_tensor = tensorflow.where(
            inputs[..., self.intensity_index] >= 34.,
            tensorflow.minimum(new_rmw_tensor, new_r34_tensor),
            new_rmw_tensor
        )
        new_rmw_tensor = tensorflow.where(
            inputs[..., self.intensity_index] >= 50.,
            tensorflow.minimum(new_rmw_tensor, new_r50_tensor),
            new_rmw_tensor
        )
        new_rmw_tensor = tensorflow.where(
            inputs[..., self.intensity_index] >= 64.,
            tensorflow.minimum(new_rmw_tensor, new_r64_tensor),
            new_rmw_tensor
        )

        # new_inputs = tensorflow.tensor_scatter_nd_update(
        #     inputs,
        #     tensorflow.expand_dims(self.r34_index, axis=-1),
        #     new_r34_tensor
        # )
        # new_inputs = tensorflow.tensor_scatter_nd_update(
        #     new_inputs,
        #     tensorflow.expand_dims(self.r50_index, axis=-1),
        #     new_r50_tensor
        # )
        # new_inputs = tensorflow.tensor_scatter_nd_update(
        #     new_inputs,
        #     tensorflow.expand_dims(self.r64_index, axis=-1),
        #     new_r64_tensor
        # )
        # new_inputs = tensorflow.tensor_scatter_nd_update(
        #     new_inputs,
        #     tensorflow.expand_dims(self.rmw_index, axis=-1),
        #     new_rmw_tensor
        # )

        new_tensors = [
            K.expand_dims(inputs[..., self.intensity_index], axis=-1),
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

        sort_indices = numpy.argsort(new_indices)
        new_tensors = [new_tensors[i] for i in numpy.argsort(new_indices)]
        new_inputs = K.concatenate(new_tensors, axis=-1)

        return new_inputs


def check_input_args(option_dict):
    """Error-checks input arguments.

    B = number of convolutional blocks
    C = number of convolutional layers
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
    option_dict["pooling_size_by_block_px"]: length-B numpy array with size of
        max-pooling window for each block.  For example, if you want 2-by-2
        pooling in the [j]th block, make pooling_size_by_block_px[j] = 2.
    option_dict["num_channels_by_conv_layer"]: length-C numpy array with number
        of channels for each conv layer.
    option_dict["dropout_rate_by_conv_layer"]: length-C numpy array with dropout
        rate for each conv layer.  Use number <= 0 to indicate no-dropout.
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
    option_dict["intensity_index"]: Channel index for TC intensity.  If you are
        predicting TC center and not TC-structure parameters, make this None.
    option_dict["r34_index"]: Channel index for radius of 34-kt wind.  If you
        are predicting TC center and not TC-structure parameters, make this
        None.
    option_dict["r50_index"]: Channel index for radius of 50-kt wind.  If you
        are predicting TC center and not TC-structure parameters, make this
        None.
    option_dict["r64_index"]: Channel index for radius of 64-kt wind.  If you
        are predicting TC center and not TC-structure parameters, make this
        None.
    option_dict["rmw_index"]: Channel index for radius of max wind.  If you
        are predicting TC center and not TC-structure parameters, make this
        None.
    option_dict["use_physical_constraints"]: Boolean flag.  If True, will use
        physical constraints to make predictions of TC-structure parameters
        consistent.  If you are predicting TC center and not TC-structure
        parameters, make this False.
    option_dict["do_residual_prediction"]: Boolean flag.  If True, will do
        residual prediction for TC-structure parameters.  If you are predicting
        TC center and not TC-structure parameters, make this False.

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

    num_conv_layers = numpy.sum(option_dict[NUM_CONV_LAYERS_KEY])

    error_checking.assert_is_numpy_array(
        option_dict[POOLING_SIZE_KEY],
        exact_dimensions=numpy.array([num_conv_layers], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[POOLING_SIZE_KEY]
    )
    error_checking.assert_is_geq_numpy_array(
        option_dict[POOLING_SIZE_KEY], 2
    )

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

    error_checking.assert_is_boolean(option_dict[PREDICT_INTENSITY_ONLY_KEY])
    if option_dict[PREDICT_INTENSITY_ONLY_KEY]:
        option_dict[USE_PHYSICAL_CONSTRAINTS_KEY] = False
        option_dict[DO_RESIDUAL_PREDICTION_KEY] = False
        return option_dict

    predict_structure_params = (
        option_dict[INTENSITY_INDEX_KEY] is not None
        or option_dict[R34_INDEX_KEY] is not None
        or option_dict[R50_INDEX_KEY] is not None
        or option_dict[R64_INDEX_KEY] is not None
        or option_dict[RMW_INDEX_KEY] is not None
    )

    if not predict_structure_params:
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
                        weight_regularizer=l2_function,
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


def create_model_for_structure(option_dict):
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
                    weight_regularizer=l2_function,
                    kernel_init_name=KERNEL_INIT_NAME_FOR_STRUCTURE,
                    bias_init_name=BIAS_INIT_NAME_FOR_STRUCTURE
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
                weight_regularizer=l2_function,
                kernel_init_name=KERNEL_INIT_NAME_FOR_STRUCTURE,
                bias_init_name=BIAS_INIT_NAME_FOR_STRUCTURE
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
                        weight_regularizer=l2_function,
                        kernel_init_name=KERNEL_INIT_NAME_FOR_STRUCTURE,
                        bias_init_name=BIAS_INIT_NAME_FOR_STRUCTURE
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
                        weight_regularizer=l2_function,
                        kernel_init_name=KERNEL_INIT_NAME_FOR_STRUCTURE,
                        bias_init_name=BIAS_INIT_NAME_FOR_STRUCTURE
                    )(forecast_module_layer_object)
                )
        else:
            forecast_module_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=num_channels_by_conv_layer[-1],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=l2_function,
                kernel_init_name=KERNEL_INIT_NAME_FOR_STRUCTURE,
                bias_init_name=BIAS_INIT_NAME_FOR_STRUCTURE
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
            num_output_units=num_neurons_by_dense_layer[i],
            kernel_init_name=KERNEL_INIT_NAME_FOR_STRUCTURE,
            bias_init_name=BIAS_INIT_NAME_FOR_STRUCTURE
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

    if use_physical_constraints:
        layer_object = architecture_utils.get_activation_layer(
            activation_function_string=architecture_utils.RELU_FUNCTION_STRING,
            alpha_for_relu=0.,
            alpha_for_elu=0.
        )(layer_object)

        layer_object = layers.Permute(
            dims=(2, 1), name='output_channels_last'
        )(layer_object)

        r64_layer_object = layers.Lambda(
            lambda x: x[..., r64_index:(r64_index + 1)],
            name='output_slice_r64',
            output_shape=(ensemble_size, 1)
        )(layer_object)
        r50_layer_object = layers.Lambda(
            lambda x: x[..., r50_index:(r50_index + 1)],
            name='output_slice_r50',
            output_shape=(ensemble_size, 1)
        )(layer_object)
        r34_layer_object = layers.Lambda(
            lambda x: x[..., r34_index:(r34_index + 1)],
            name='output_slice_r34',
            output_shape=(ensemble_size, 1)
        )(layer_object)
        intensity_layer_object = layers.Lambda(
            lambda x: x[..., intensity_index:(intensity_index + 1)],
            name='output_slice_intensity',
            output_shape=(ensemble_size, 1)
        )(layer_object)
        rmw_layer_object = layers.Lambda(
            lambda x: x[..., rmw_index:(rmw_index + 1)],
            name='output_slice_rmw',
            output_shape=(ensemble_size, 1)
        )(layer_object)

        r50_layer_object = layers.Add(name='output_add_r50_to_r64')(
            [r50_layer_object, r64_layer_object]
        )
        r34_layer_object = layers.Add(name='output_add_r34_to_r50')(
            [r34_layer_object, r50_layer_object]
        )

        assert (
            intensity_index == 0 and r34_index == 1 and r50_index == 2
            and r64_index == 3 and rmw_index == 4
        )

        layer_object = layers.Concatenate(axis=-1)([
            intensity_layer_object,
            r34_layer_object,
            r50_layer_object,
            r64_layer_object,
            rmw_layer_object
        ])

        # layer_object = PhysicalConstraintLayer(
        #     intensity_index=intensity_index,
        #     r34_index=r34_index,
        #     r50_index=r50_index,
        #     r64_index=r64_index,
        #     rmw_index=rmw_index,
        #     name='output_phys_constraints'
        # )(layer_object)

        layer_object = layers.Permute(
            dims=(2, 1), name='output_channels_first'
        )(layer_object)
    else:
        layer_object = architecture_utils.get_activation_layer(
            activation_function_string=architecture_utils.RELU_FUNCTION_STRING,
            alpha_for_relu=0.,
            alpha_for_elu=0.
        )(layer_object)

    input_layer_objects = []
    if include_high_res_data:
        input_layer_objects.append(input_layer_object_high_res)

    input_layer_objects.append(input_layer_object_low_res)
    if include_scalar_data:
        input_layer_objects.append(input_layer_object_scalar)
    if do_residual_prediction:
        input_layer_objects.append(input_layer_object_resid_baseline)

    metric_functions = [
        custom_metrics_structure.mean_squared_error(
            channel_index=intensity_index,
            function_name='mean_sq_error_intensity_kt2'
        ),
        custom_metrics_structure.mean_squared_error(
            channel_index=r34_index, function_name='mean_sq_error_r34_km2'
        ),
        custom_metrics_structure.mean_squared_error(
            channel_index=r50_index, function_name='mean_sq_error_r50_km2'
        ),
        custom_metrics_structure.mean_squared_error(
            channel_index=r64_index, function_name='mean_sq_error_r64_km2'
        ),
        custom_metrics_structure.mean_squared_error(
            channel_index=rmw_index, function_name='mean_sq_error_rmw_km2'
        ),
        custom_metrics_structure.min_target(
            channel_index=intensity_index,
            function_name='min_target_intensity_kt2'
        ),
        custom_metrics_structure.min_target(
            channel_index=r34_index, function_name='min_target_r34_km2'
        ),
        custom_metrics_structure.min_target(
            channel_index=r50_index, function_name='min_target_r50_km2'
        ),
        custom_metrics_structure.min_target(
            channel_index=r64_index, function_name='min_target_r64_km2'
        ),
        custom_metrics_structure.min_target(
            channel_index=rmw_index, function_name='min_target_rmw_km2'
        ),
        custom_metrics_structure.max_target(
            channel_index=intensity_index,
            function_name='max_target_intensity_kt2'
        ),
        custom_metrics_structure.max_target(
            channel_index=r34_index, function_name='max_target_r34_km2'
        ),
        custom_metrics_structure.max_target(
            channel_index=r50_index, function_name='max_target_r50_km2'
        ),
        custom_metrics_structure.max_target(
            channel_index=r64_index, function_name='max_target_r64_km2'
        ),
        custom_metrics_structure.max_target(
            channel_index=rmw_index, function_name='max_target_rmw_km2'
        ),
        custom_metrics_structure.min_prediction(
            channel_index=intensity_index,
            function_name='min_ens_member_pred_intensity_kt2',
            take_ensemble_mean=False
        ),
        custom_metrics_structure.min_prediction(
            channel_index=r34_index,
            function_name='min_ens_member_pred_r34_km2',
            take_ensemble_mean=False
        ),
        custom_metrics_structure.min_prediction(
            channel_index=r50_index,
            function_name='min_ens_member_pred_r50_km2',
            take_ensemble_mean=False
        ),
        custom_metrics_structure.min_prediction(
            channel_index=r64_index,
            function_name='min_ens_member_pred_r64_km2',
            take_ensemble_mean=False
        ),
        custom_metrics_structure.min_prediction(
            channel_index=rmw_index,
            function_name='min_ens_member_pred_rmw_km2',
            take_ensemble_mean=False
        ),
        custom_metrics_structure.max_prediction(
            channel_index=intensity_index,
            function_name='max_ens_member_pred_intensity_kt2',
            take_ensemble_mean=False
        ),
        custom_metrics_structure.max_prediction(
            channel_index=r34_index,
            function_name='max_ens_member_pred_r34_km2',
            take_ensemble_mean=False
        ),
        custom_metrics_structure.max_prediction(
            channel_index=r50_index,
            function_name='max_ens_member_pred_r50_km2',
            take_ensemble_mean=False
        ),
        custom_metrics_structure.max_prediction(
            channel_index=r64_index,
            function_name='max_ens_member_pred_r64_km2',
            take_ensemble_mean=False
        ),
        custom_metrics_structure.max_prediction(
            channel_index=rmw_index,
            function_name='max_ens_member_pred_rmw_km2',
            take_ensemble_mean=False
        ),
        custom_metrics_structure.min_prediction(
            channel_index=intensity_index,
            function_name='min_ens_mean_pred_intensity_kt2',
            take_ensemble_mean=True
        ),
        custom_metrics_structure.min_prediction(
            channel_index=r34_index,
            function_name='min_ens_mean_pred_r34_km2',
            take_ensemble_mean=True
        ),
        custom_metrics_structure.min_prediction(
            channel_index=r50_index,
            function_name='min_ens_mean_pred_r50_km2',
            take_ensemble_mean=True
        ),
        custom_metrics_structure.min_prediction(
            channel_index=r64_index,
            function_name='min_ens_mean_pred_r64_km2',
            take_ensemble_mean=True
        ),
        custom_metrics_structure.min_prediction(
            channel_index=rmw_index,
            function_name='min_ens_mean_pred_rmw_km2',
            take_ensemble_mean=True
        ),
        custom_metrics_structure.max_prediction(
            channel_index=intensity_index,
            function_name='max_ens_mean_pred_intensity_kt2',
            take_ensemble_mean=True
        ),
        custom_metrics_structure.max_prediction(
            channel_index=r34_index,
            function_name='max_ens_mean_pred_r34_km2',
            take_ensemble_mean=True
        ),
        custom_metrics_structure.max_prediction(
            channel_index=r50_index,
            function_name='max_ens_mean_pred_r50_km2',
            take_ensemble_mean=True
        ),
        custom_metrics_structure.max_prediction(
            channel_index=r64_index,
            function_name='max_ens_mean_pred_r64_km2',
            take_ensemble_mean=True
        ),
        custom_metrics_structure.max_prediction(
            channel_index=rmw_index,
            function_name='max_ens_mean_pred_rmw_km2',
            take_ensemble_mean=True
        )
    ]

    if not use_physical_constraints:
        metric_functions += [
            custom_metrics_structure.mean_squared_error_with_constraints(
                channel_index=r34_index,
                intensity_index=intensity_index,
                r34_index=r34_index,
                r50_index=r50_index,
                r64_index=r64_index,
                rmw_index=rmw_index,
                function_name='mse_constrained_r34_km2'
            ),
            custom_metrics_structure.mean_squared_error_with_constraints(
                channel_index=r50_index,
                intensity_index=intensity_index,
                r34_index=r34_index,
                r50_index=r50_index,
                r64_index=r64_index,
                rmw_index=rmw_index,
                function_name='mse_constrained_r50_km2'
            ),
            custom_metrics_structure.min_prediction_with_constraints(
                channel_index=r34_index,
                intensity_index=intensity_index,
                r34_index=r34_index,
                r50_index=r50_index,
                r64_index=r64_index,
                rmw_index=rmw_index,
                function_name='min_member_pred_constrained_r34_km2',
                take_ensemble_mean=False
            ),
            custom_metrics_structure.min_prediction_with_constraints(
                channel_index=r50_index,
                intensity_index=intensity_index,
                r34_index=r34_index,
                r50_index=r50_index,
                r64_index=r64_index,
                rmw_index=rmw_index,
                function_name='min_member_pred_constrained_r50_km2',
                take_ensemble_mean=False
            ),
            custom_metrics_structure.max_prediction_with_constraints(
                channel_index=r34_index,
                intensity_index=intensity_index,
                r34_index=r34_index,
                r50_index=r50_index,
                r64_index=r64_index,
                rmw_index=rmw_index,
                function_name='max_member_pred_constrained_r34_km2',
                take_ensemble_mean=False
            ),
            custom_metrics_structure.max_prediction_with_constraints(
                channel_index=r50_index,
                intensity_index=intensity_index,
                r34_index=r34_index,
                r50_index=r50_index,
                r64_index=r64_index,
                rmw_index=rmw_index,
                function_name='max_member_pred_constrained_r50_km2',
                take_ensemble_mean=False
            ),
            custom_metrics_structure.min_prediction_with_constraints(
                channel_index=r34_index,
                intensity_index=intensity_index,
                r34_index=r34_index,
                r50_index=r50_index,
                r64_index=r64_index,
                rmw_index=rmw_index,
                function_name='min_mean_pred_constrained_r34_km2',
                take_ensemble_mean=True
            ),
            custom_metrics_structure.min_prediction_with_constraints(
                channel_index=r50_index,
                intensity_index=intensity_index,
                r34_index=r34_index,
                r50_index=r50_index,
                r64_index=r64_index,
                rmw_index=rmw_index,
                function_name='min_mean_pred_constrained_r50_km2',
                take_ensemble_mean=True
            ),
            custom_metrics_structure.max_prediction_with_constraints(
                channel_index=r34_index,
                intensity_index=intensity_index,
                r34_index=r34_index,
                r50_index=r50_index,
                r64_index=r64_index,
                rmw_index=rmw_index,
                function_name='max_mean_pred_constrained_r34_km2',
                take_ensemble_mean=True
            ),
            custom_metrics_structure.max_prediction_with_constraints(
                channel_index=r50_index,
                intensity_index=intensity_index,
                r34_index=r34_index,
                r50_index=r50_index,
                r64_index=r64_index,
                rmw_index=rmw_index,
                function_name='max_mean_pred_constrained_r50_km2',
                take_ensemble_mean=True
            )
        ]

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
                    weight_regularizer=l2_function,
                    kernel_init_name=KERNEL_INIT_NAME_FOR_STRUCTURE,
                    bias_init_name=BIAS_INIT_NAME_FOR_STRUCTURE
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
                weight_regularizer=l2_function,
                kernel_init_name=KERNEL_INIT_NAME_FOR_STRUCTURE,
                bias_init_name=BIAS_INIT_NAME_FOR_STRUCTURE
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
                        weight_regularizer=l2_function,
                        kernel_init_name=KERNEL_INIT_NAME_FOR_STRUCTURE,
                        bias_init_name=BIAS_INIT_NAME_FOR_STRUCTURE
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
                        weight_regularizer=l2_function,
                        kernel_init_name=KERNEL_INIT_NAME_FOR_STRUCTURE,
                        bias_init_name=BIAS_INIT_NAME_FOR_STRUCTURE
                    )(forecast_module_layer_object)
                )
        else:
            forecast_module_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=num_channels_by_conv_layer[-1],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=l2_function,
                kernel_init_name=KERNEL_INIT_NAME_FOR_STRUCTURE,
                bias_init_name=BIAS_INIT_NAME_FOR_STRUCTURE
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
            num_output_units=num_neurons_by_dense_layer[i],
            kernel_init_name=KERNEL_INIT_NAME_FOR_STRUCTURE,
            bias_init_name=BIAS_INIT_NAME_FOR_STRUCTURE
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

    metric_functions = [
        custom_metrics_structure.mean_squared_error(
            channel_index=0,
            function_name='mean_sq_error_intensity_kt2'
        ),
        custom_metrics_structure.min_prediction(
            channel_index=0,
            function_name='min_ens_member_pred_intensity_kt',
            take_ensemble_mean=False
        ),
        custom_metrics_structure.min_prediction(
            channel_index=0,
            function_name='min_pred_intensity_kt',
            take_ensemble_mean=True
        ),
        custom_metrics_structure.max_prediction(
            channel_index=0,
            function_name='max_ens_member_pred_intensity_kt',
            take_ensemble_mean=False
        ),
        custom_metrics_structure.max_prediction(
            channel_index=0,
            function_name='max_pred_intensity_kt',
            take_ensemble_mean=True
        )
    ]

    model_object = keras.models.Model(
        inputs=input_layer_objects, outputs=layer_object
    )
    model_object.compile(
        loss=loss_function, optimizer=optimizer_function,
        metrics=metric_functions
    )
    model_object.summary()

    return model_object
