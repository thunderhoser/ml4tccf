"""Custom losses for predicting TC-structure parameters."""

import numpy
from tensorflow.keras import backend as K
from gewittergefahr.gg_utils import error_checking


def dwcrps_for_structure_params(channel_weights, function_name,
                                test_mode=False):
    """Creates DWCRPS loss function for TC-structure parameters.

    DWCRPS = dual-weighted continuous ranked probability score

    E = number of examples
    C = number of channels = number of target variables
    S = ensemble size = number of ensemble members

    :param channel_weights: length-C numpy array of weights.
    :param function_name: Name of function.
    :param test_mode: Leave this alone.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_numpy_array(channel_weights, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(test_mode)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (DWCRPS).

        :param target_tensor: E-by-C numpy array of correct values.
        :param prediction_tensor: E-by-C-by-S numpy array of predicted values.
        :return: dwcrps: DWCRPS (scalar float).
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

        # Compute dual weights (E-by-C-by-S tensor).
        relevant_target_tensor = K.expand_dims(
            target_tensor[:, :len(channel_weights)], axis=-1
        )
        relevant_prediction_tensor = prediction_tensor
        dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(relevant_prediction_tensor)
        )

        # Turn channel weights into E-by-C tensor.
        channel_weight_tensor = K.cast(
            K.constant(channel_weights), relevant_target_tensor.dtype
        )
        channel_weight_tensor = K.expand_dims(channel_weight_tensor, axis=0)

        # Compute mean absolute errors (E-by-C tensor).
        absolute_error_tensor = K.abs(
            relevant_prediction_tensor - relevant_target_tensor
        )
        mean_prediction_error_tensor = K.mean(
            dual_weight_tensor * absolute_error_tensor, axis=-1
        )

        # Compute mean absolute pairwise differences (E-by-C tensor).
        mean_prediction_diff_tensor = K.map_fn(
            fn=lambda p: K.mean(
                K.maximum(
                    K.abs(K.expand_dims(p, axis=-1)),
                    K.abs(K.expand_dims(p, axis=-2))
                ) *
                K.abs(
                    K.expand_dims(p, axis=-1) -
                    K.expand_dims(p, axis=-2)
                ),
                axis=(-2, -1)
            ),
            elems=prediction_tensor
        )

        # Compute DWCRPS.
        error_tensor = channel_weight_tensor * (
            mean_prediction_error_tensor -
            0.5 * mean_prediction_diff_tensor
        )

        return K.mean(error_tensor)

    loss.__name__ = function_name
    return loss


def apply_physical_constraints(
        target_tensor, prediction_tensor, intensity_index, r34_index, r50_index,
        r64_index, rmw_index):
    """Applies physical constraints for TC-structure parameters.

    E = number of examples
    C = number of channels = number of target variables
    S = ensemble size = number of ensemble members

    :param target_tensor: E-by-C numpy array of correct values.
    :param prediction_tensor: E-by-C-by-S numpy array of predicted values.
    :param intensity_index: Array index for TC-intensity prediction.
    :param r34_index: Array index for radius-of-34-kt-wind prediction.
    :param r50_index: Array index for radius-of-50-kt-wind prediction.
    :param r64_index: Array index for radius-of-64-kt-wind prediction.
    :param rmw_index: Array index of radius-of-max-wind prediction.
    :return: target_tensor: Same as input but with physically consistent values.
    :return: prediction_tensor: Same as input but with physically consistent
        values.
    """

    error_checking.assert_is_integer(intensity_index)
    error_checking.assert_is_geq(intensity_index, 0)
    error_checking.assert_is_integer(r34_index)
    error_checking.assert_is_geq(r34_index, 0)
    error_checking.assert_is_integer(r50_index)
    error_checking.assert_is_geq(r50_index, 0)
    error_checking.assert_is_integer(r64_index)
    error_checking.assert_is_geq(r64_index, 0)
    error_checking.assert_is_integer(rmw_index)
    error_checking.assert_is_geq(rmw_index, 0)

    all_indices = numpy.array(
        [intensity_index, r34_index, r50_index, r64_index, rmw_index],
        dtype=int
    )
    error_checking.assert_equals(
        len(all_indices),
        len(numpy.unique(all_indices))
    )

    predicted_intensity_tensor = K.maximum(
        prediction_tensor[..., intensity_index, :], 0.
    )
    predicted_r34_tensor = K.maximum(
        prediction_tensor[..., r34_index, :], 0.
    )
    predicted_r50_tensor = K.maximum(
        prediction_tensor[..., r50_index, :], 0.
    )
    predicted_r64_tensor = K.maximum(
        prediction_tensor[..., r64_index, :], 0.
    )
    predicted_rmw_tensor = K.maximum(
        prediction_tensor[..., rmw_index, :], 0.
    )

    predicted_r50_tensor = K.maximum(
        predicted_r50_tensor, predicted_r64_tensor
    )
    predicted_r34_tensor = K.maximum(
        predicted_r34_tensor, predicted_r50_tensor
    )

    # predicted_r50_tensor = predicted_r50_tensor + predicted_r64_tensor
    # predicted_r34_tensor = predicted_r34_tensor + predicted_r50_tensor
    # predicted_r34_tensor = tensorflow.where(
    #     predicted_intensity_tensor < 34.,
    #     tensorflow.zeros_like(predicted_r34_tensor),
    #     predicted_r34_tensor
    # )
    # predicted_r50_tensor = tensorflow.where(
    #     predicted_intensity_tensor < 50.,
    #     tensorflow.zeros_like(predicted_r50_tensor),
    #     predicted_r50_tensor
    # )
    # predicted_r64_tensor = tensorflow.where(
    #     predicted_intensity_tensor < 64.,
    #     tensorflow.zeros_like(predicted_r64_tensor),
    #     predicted_r64_tensor
    # )

    new_prediction_tensors = [
        K.expand_dims(predicted_intensity_tensor, axis=-2),
        K.expand_dims(predicted_r34_tensor, axis=-2),
        K.expand_dims(predicted_r50_tensor, axis=-2),
        K.expand_dims(predicted_r64_tensor, axis=-2),
        K.expand_dims(predicted_rmw_tensor, axis=-2)
    ]

    new_indices = numpy.array([
        intensity_index, r34_index, r50_index, r64_index, rmw_index
    ], dtype=int)

    new_prediction_tensors = [
        new_prediction_tensors[i] for i in numpy.argsort(new_indices)
    ]
    prediction_tensor = K.concatenate(new_prediction_tensors, axis=-2)

    return target_tensor, prediction_tensor


def constrained_dwcrps_for_structure_params(
        channel_weights, intensity_index, r34_index, r50_index, r64_index,
        rmw_index, function_name, test_mode=False):
    """Same as `dwcrps_for_structure_params` but with physical constraints.

    :param channel_weights: See documentation for `dwcrps_for_structure_params`.
    :param intensity_index: Array index for TC-intensity prediction.
    :param r34_index: Array index for radius-of-34-kt-wind prediction.
    :param r50_index: Array index for radius-of-50-kt-wind prediction.
    :param r64_index: Array index for radius-of-64-kt-wind prediction.
    :param rmw_index: Array index of radius-of-max-wind prediction.
    :param function_name: Name of function (string).
    :param test_mode: Leave this alone.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_numpy_array(channel_weights, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(test_mode)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (DWCRPS).

        :param target_tensor: E-by-C numpy array of correct values.
        :param prediction_tensor: E-by-C-by-S numpy array of predicted values.
        :return: dwcrps: DWCRPS (scalar float).
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)
        target_tensor, prediction_tensor = apply_physical_constraints(
            target_tensor=target_tensor,
            prediction_tensor=prediction_tensor,
            intensity_index=intensity_index,
            r34_index=r34_index,
            r50_index=r50_index,
            r64_index=r64_index,
            rmw_index=rmw_index
        )

        # Compute dual weights (E-by-C-by-S tensor).
        relevant_target_tensor = K.expand_dims(
            target_tensor[:, :len(channel_weights)], axis=-1
        )
        relevant_prediction_tensor = prediction_tensor
        dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(relevant_prediction_tensor)
        )

        # Turn channel weights into E-by-C tensor.
        channel_weight_tensor = K.cast(
            K.constant(channel_weights), relevant_target_tensor.dtype
        )
        channel_weight_tensor = K.expand_dims(channel_weight_tensor, axis=0)

        # Compute mean absolute errors (E-by-C tensor).
        absolute_error_tensor = K.abs(
            relevant_prediction_tensor - relevant_target_tensor
        )
        mean_prediction_error_tensor = K.mean(
            dual_weight_tensor * absolute_error_tensor, axis=-1
        )

        # Compute mean absolute pairwise differences (E-by-C tensor).
        mean_prediction_diff_tensor = K.map_fn(
            fn=lambda p: K.mean(
                K.maximum(
                    K.abs(K.expand_dims(p, axis=-1)),
                    K.abs(K.expand_dims(p, axis=-2))
                ) *
                K.abs(
                    K.expand_dims(p, axis=-1) -
                    K.expand_dims(p, axis=-2)
                ),
                axis=(-2, -1)
            ),
            elems=prediction_tensor
        )

        # Compute DWCRPS.
        error_tensor = channel_weight_tensor * (
            mean_prediction_error_tensor -
            0.5 * mean_prediction_diff_tensor
        )

        return K.mean(error_tensor)

    loss.__name__ = function_name
    return loss


def dwmse_for_structure_params(channel_weights, function_name, test_mode=False):
    """DWMSE for TC-structure parameters, with*out* physical constraints.

    :param channel_weights: See documentation for `dwcrps_for_structure_params`.
    :param function_name: Name of function (string).
    :param test_mode: Leave this alone.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_numpy_array(channel_weights, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(test_mode)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (DWMSE).

        :param target_tensor: E-by-C numpy array of correct values.
        :param prediction_tensor: E-by-C-by-S numpy array of predicted values.
        :return: dwmse: DWMSE (scalar float).
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

        # Compute dual weights (E-by-C tensor).
        relevant_target_tensor = target_tensor[:, :len(channel_weights)]
        relevant_prediction_tensor = K.mean(prediction_tensor, axis=-1)
        dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(relevant_prediction_tensor)
        )

        # Turn channel weights into E-by-C tensor.
        channel_weight_tensor = K.cast(
            K.constant(channel_weights), relevant_target_tensor.dtype
        )
        channel_weight_tensor = K.expand_dims(channel_weight_tensor, axis=0)

        return K.mean(
            channel_weight_tensor * dual_weight_tensor *
            (relevant_target_tensor - relevant_prediction_tensor) ** 2
        )

    loss.__name__ = function_name
    return loss


def constrained_dwmse_for_structure_params(
        channel_weights, intensity_index, r34_index, r50_index, r64_index,
        rmw_index, function_name, test_mode=False):
    """DWMSE for TC-structure parameters, with physical constraints.

    :param channel_weights: See documentation for `dwcrps_for_structure_params`.
    :param intensity_index: Array index for TC-intensity prediction.
    :param r34_index: Array index for radius-of-34-kt-wind prediction.
    :param r50_index: Array index for radius-of-50-kt-wind prediction.
    :param r64_index: Array index for radius-of-64-kt-wind prediction.
    :param rmw_index: Array index of radius-of-max-wind prediction.
    :param function_name: Name of function (string).
    :param test_mode: Leave this alone.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_numpy_array(channel_weights, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(test_mode)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (DWMSE).

        :param target_tensor: E-by-C numpy array of correct values.
        :param prediction_tensor: E-by-C-by-S numpy array of predicted values.
        :return: dwmse: DWMSE (scalar float).
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)
        target_tensor, prediction_tensor = apply_physical_constraints(
            target_tensor=target_tensor,
            prediction_tensor=prediction_tensor,
            intensity_index=intensity_index,
            r34_index=r34_index,
            r50_index=r50_index,
            r64_index=r64_index,
            rmw_index=rmw_index
        )

        # Compute dual weights (E-by-C tensor).
        relevant_target_tensor = target_tensor[:, :len(channel_weights)]
        relevant_prediction_tensor = K.mean(prediction_tensor, axis=-1)
        dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(relevant_prediction_tensor)
        )

        # Turn channel weights into E-by-C tensor.
        channel_weight_tensor = K.cast(
            K.constant(channel_weights), relevant_target_tensor.dtype
        )
        channel_weight_tensor = K.expand_dims(channel_weight_tensor, axis=0)

        return K.mean(
            channel_weight_tensor * dual_weight_tensor *
            (relevant_target_tensor - relevant_prediction_tensor) ** 2
        )

    loss.__name__ = function_name
    return loss


def dwcrpss_for_structure_params(channel_weights, function_name,
                                 test_mode=False):
    """Creates DWCRPSS loss function for TC-structure parameters.

    DWCRPSS = dual-weighted continuous ranked probability *skill* score

    DWCRPSS compares the actual model's DWCRPS to that of the baseline model.

    E = number of examples
    C = number of channels = number of target variables
    S = ensemble size = number of ensemble members

    :param channel_weights: length-C numpy array of weights.
    :param function_name: Name of function.
    :param test_mode: Leave this alone.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_numpy_array(channel_weights, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(test_mode)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (DWCRPSS).

        :param target_tensor: E-by-C numpy array of correct values.
        :param prediction_tensor: E-by-C-by-S numpy array of predicted values.
        :return: dwcrpss: DWCRPSS (scalar float).
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

        # Compute dual weights (E-by-C-by-S tensor).
        relevant_target_tensor = K.expand_dims(
            target_tensor[:, :len(channel_weights)], axis=-1
        )
        relevant_prediction_tensor = prediction_tensor
        dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(relevant_prediction_tensor)
        )

        # Turn channel weights into E-by-C tensor.
        channel_weight_tensor = K.cast(
            K.constant(channel_weights), relevant_target_tensor.dtype
        )
        channel_weight_tensor = K.expand_dims(channel_weight_tensor, axis=0)

        # Compute mean absolute errors (E-by-C tensor).
        absolute_error_tensor = K.abs(
            relevant_prediction_tensor - relevant_target_tensor
        )
        mean_prediction_error_tensor = K.mean(
            dual_weight_tensor * absolute_error_tensor, axis=-1
        )

        # Compute mean absolute pairwise differences (E-by-C tensor).
        mean_prediction_diff_tensor = K.map_fn(
            fn=lambda p: K.mean(
                K.maximum(
                    K.abs(K.expand_dims(p, axis=-1)),
                    K.abs(K.expand_dims(p, axis=-2))
                ) *
                K.abs(
                    K.expand_dims(p, axis=-1) -
                    K.expand_dims(p, axis=-2)
                ),
                axis=(-2, -1)
            ),
            elems=prediction_tensor
        )

        # Compute DWCRPS of actual model.
        error_tensor = channel_weight_tensor * (
            mean_prediction_error_tensor -
            0.5 * mean_prediction_diff_tensor
        )
        actual_dwcrps = K.mean(error_tensor)

        # Create dual-weight tensor for baseline.
        relevant_baseline_prediction_tensor = K.expand_dims(
            target_tensor[:, len(channel_weights):], axis=-1
        )
        dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(relevant_baseline_prediction_tensor)
        )

        # Compute mean absolute errors for baseline (E-by-C tensor).
        absolute_error_tensor = K.abs(
            relevant_baseline_prediction_tensor - relevant_target_tensor
        )
        mean_prediction_error_tensor = K.mean(
            dual_weight_tensor * absolute_error_tensor, axis=-1
        )
        error_tensor = channel_weight_tensor * mean_prediction_error_tensor

        # Compute DWCRPS of baseline model.
        baseline_dwcrps = K.mean(error_tensor)

        # Return negative skill score.
        return (actual_dwcrps - baseline_dwcrps) / baseline_dwcrps

    loss.__name__ = function_name
    return loss
