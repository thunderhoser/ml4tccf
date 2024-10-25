"""Custom metrics for TC-structure parameters."""

import os
import sys
from tensorflow.keras import backend as K

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import custom_losses_scalar as custom_losses


def mean_squared_error(channel_index, function_name, test_mode=False):
    """Computes mean squared error (MSE) for one channel (target variable).

    :param channel_index: Array index for the desired channel.
    :param function_name: Name of function.
    :param test_mode: Leave this alone.
    :return: metric_function: Function defined below.
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(test_mode)

    def metric(target_tensor, prediction_tensor):
        """Computes MSE.

        E = number of examples
        C = number of channels = number of target variables
        S = ensemble size = number of ensemble members

        :param target_tensor: E-by-C numpy array of correct values.
        :param prediction_tensor: E-by-C-by-S numpy array of predicted values.
        :return: mean_squared_error: MSE (scalar float).
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)
        relevant_target_tensor = target_tensor[:, channel_index]
        relevant_prediction_tensor = K.mean(
            prediction_tensor[:, channel_index, :], axis=-1
        )

        return K.mean(
            (relevant_target_tensor - relevant_prediction_tensor) ** 2
        )

    metric.__name__ = function_name
    return metric


def min_target(channel_index, function_name, test_mode=False):
    """Computes minimum target value for one channel (target variable).

    :param channel_index: See documentation for `mean_squared_error`.
    :param function_name: Same.
    :param test_mode: Same.
    :return: metric_function: Function defined below.
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(test_mode)

    def metric(target_tensor, prediction_tensor):
        """Computes minimum target value.

        :param target_tensor: See documentation for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: min_target: Minimum target value. (scalar float).
        """

        return K.min(target_tensor[:, channel_index])

    metric.__name__ = function_name
    return metric


def max_target(channel_index, function_name, test_mode=False):
    """Computes maximum target value for one channel (target variable).

    :param channel_index: See documentation for `mean_squared_error`.
    :param function_name: Same.
    :param test_mode: Same.
    :return: metric_function: Function defined below.
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(test_mode)

    def metric(target_tensor, prediction_tensor):
        """Computes maximum target value.

        :param target_tensor: See documentation for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: max_target: Maximum target value. (scalar float).
        """

        return K.max(target_tensor[:, channel_index])

    metric.__name__ = function_name
    return metric


def min_prediction(channel_index, function_name, take_ensemble_mean,
                   test_mode=False):
    """Computes minimum predicted value for one channel (target variable).

    :param channel_index: Array index for the desired channel.
    :param function_name: Name of function (string).
    :param take_ensemble_mean: Boolean flag.  If True, will take minimum of
        ensemble-mean predictions.  If False, will take minimum of ensemble-
        member predictions.
    :param test_mode: Leave this alone.
    :return: metric_function: Function defined below.
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(take_ensemble_mean)
    error_checking.assert_is_boolean(test_mode)

    def metric(target_tensor, prediction_tensor):
        """Computes minimum predicted value.

        :param target_tensor: See documentation for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: min_prediction: Minimum predicted value. (scalar float).
        """

        if take_ensemble_mean:
            relevant_prediction_tensor = K.mean(
                prediction_tensor[:, channel_index, :], axis=-1
            )
        else:
            relevant_prediction_tensor = prediction_tensor[:, channel_index, :]

        return K.min(relevant_prediction_tensor)

    metric.__name__ = function_name
    return metric


def max_prediction(channel_index, function_name, take_ensemble_mean,
                   test_mode=False):
    """Computes maximum predicted value for one channel (target variable).

    :param channel_index: Array index for the desired channel.
    :param function_name: Name of function (string).
    :param take_ensemble_mean: Boolean flag.  If True, will take maximum of
        ensemble-mean predictions.  If False, will take maximum of ensemble-
        member predictions.
    :param test_mode: Leave this alone.
    :return: metric_function: Function defined below.
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(take_ensemble_mean)
    error_checking.assert_is_boolean(test_mode)

    def metric(target_tensor, prediction_tensor):
        """Computes maximum predicted value.

        :param target_tensor: See documentation for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: max_prediction: Maximum predicted value. (scalar float).
        """

        if take_ensemble_mean:
            relevant_prediction_tensor = K.mean(
                prediction_tensor[:, channel_index, :], axis=-1
            )
        else:
            relevant_prediction_tensor = prediction_tensor[:, channel_index, :]

        return K.max(relevant_prediction_tensor)

    metric.__name__ = function_name
    return metric


def mean_squared_error_with_constraints(
        channel_index,
        intensity_index, r34_index, r50_index, r64_index, rmw_index,
        function_name, test_mode=False):
    """Same as `mean_squared_error` but after applying physical constraints.

    :param channel_index: Array index for the desired channel.
    :param intensity_index: Array index for TC-intensity prediction.
    :param r34_index: Array index for radius-of-34-kt-wind prediction.
    :param r50_index: Array index for radius-of-50-kt-wind prediction.
    :param r64_index: Array index for radius-of-64-kt-wind prediction.
    :param rmw_index: Array index of radius-of-max-wind prediction.
    :param function_name: Name of function (string).
    :param test_mode: Leave this alone.
    :return: metric_function: Function defined below.
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(test_mode)

    def metric(target_tensor, prediction_tensor):
        """Computes MSE.

        E = number of examples
        C = number of channels = number of target variables
        S = ensemble size = number of ensemble members

        :param target_tensor: E-by-C numpy array of correct values.
        :param prediction_tensor: E-by-C-by-S numpy array of predicted values.
        :return: mean_squared_error: MSE (scalar float).
        """

        target_tensor, prediction_tensor = (
            custom_losses.apply_physical_constraints(
                target_tensor=target_tensor,
                prediction_tensor=prediction_tensor,
                intensity_index=intensity_index,
                r34_index=r34_index,
                r50_index=r50_index,
                r64_index=r64_index,
                rmw_index=rmw_index
            )
        )

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)
        relevant_target_tensor = target_tensor[:, channel_index]
        relevant_prediction_tensor = K.mean(
            prediction_tensor[:, channel_index, :], axis=-1
        )

        return K.mean(
            (relevant_target_tensor - relevant_prediction_tensor) ** 2
        )

    metric.__name__ = function_name
    return metric


def min_target_with_constraints(
        channel_index,
        intensity_index, r34_index, r50_index, r64_index, rmw_index,
        function_name, test_mode=False):
    """Same as `min_target` but after applying physical constraints.

    :param channel_index: Array index for the desired channel.
    :param intensity_index: Array index for TC-intensity prediction.
    :param r34_index: Array index for radius-of-34-kt-wind prediction.
    :param r50_index: Array index for radius-of-50-kt-wind prediction.
    :param r64_index: Array index for radius-of-64-kt-wind prediction.
    :param rmw_index: Array index of radius-of-max-wind prediction.
    :param function_name: Name of function (string).
    :param test_mode: Leave this alone.
    :return: metric_function: Function defined below.
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(test_mode)

    def metric(target_tensor, prediction_tensor):
        """Computes minimum target value.

        :param target_tensor: See documentation for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: min_target: Minimum target value. (scalar float).
        """

        target_tensor, prediction_tensor = (
            custom_losses.apply_physical_constraints(
                target_tensor=target_tensor,
                prediction_tensor=prediction_tensor,
                intensity_index=intensity_index,
                r34_index=r34_index,
                r50_index=r50_index,
                r64_index=r64_index,
                rmw_index=rmw_index
            )
        )

        return K.min(target_tensor[:, channel_index])

    metric.__name__ = function_name
    return metric


def max_target_with_constraints(
        channel_index,
        intensity_index, r34_index, r50_index, r64_index, rmw_index,
        function_name, test_mode=False):
    """Same as `max_target` but after applying physical constraints.

    :param channel_index: Array index for the desired channel.
    :param intensity_index: Array index for TC-intensity prediction.
    :param r34_index: Array index for radius-of-34-kt-wind prediction.
    :param r50_index: Array index for radius-of-50-kt-wind prediction.
    :param r64_index: Array index for radius-of-64-kt-wind prediction.
    :param rmw_index: Array index of radius-of-max-wind prediction.
    :param function_name: Name of function (string).
    :param test_mode: Leave this alone.
    :return: metric_function: Function defined below.
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(test_mode)

    def metric(target_tensor, prediction_tensor):
        """Computes maximum target value.

        :param target_tensor: See documentation for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: max_target: Maximum target value. (scalar float).
        """

        target_tensor, prediction_tensor = (
            custom_losses.apply_physical_constraints(
                target_tensor=target_tensor,
                prediction_tensor=prediction_tensor,
                intensity_index=intensity_index,
                r34_index=r34_index,
                r50_index=r50_index,
                r64_index=r64_index,
                rmw_index=rmw_index
            )
        )

        return K.max(target_tensor[:, channel_index])

    metric.__name__ = function_name
    return metric


def min_prediction_with_constraints(
        channel_index,
        intensity_index, r34_index, r50_index, r64_index, rmw_index,
        take_ensemble_mean, function_name, test_mode=False):
    """Same as `min_prediction` but after applying physical constraints.

    :param channel_index: Array index for the desired channel.
    :param intensity_index: Array index for TC-intensity prediction.
    :param r34_index: Array index for radius-of-34-kt-wind prediction.
    :param r50_index: Array index for radius-of-50-kt-wind prediction.
    :param r64_index: Array index for radius-of-64-kt-wind prediction.
    :param rmw_index: Array index of radius-of-max-wind prediction.
    :param take_ensemble_mean: Boolean flag.  If True, will take minimum of
        ensemble-mean predictions.  If False, will take minimum of ensemble-
        member predictions.
    :param function_name: Name of function (string).
    :param test_mode: Leave this alone.
    :return: metric_function: Function defined below.
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(take_ensemble_mean)
    error_checking.assert_is_boolean(test_mode)

    def metric(target_tensor, prediction_tensor):
        """Computes minimum predicted value.

        :param target_tensor: See documentation for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: min_prediction: Minimum predicted value. (scalar float).
        """

        target_tensor, prediction_tensor = (
            custom_losses.apply_physical_constraints(
                target_tensor=target_tensor,
                prediction_tensor=prediction_tensor,
                intensity_index=intensity_index,
                r34_index=r34_index,
                r50_index=r50_index,
                r64_index=r64_index,
                rmw_index=rmw_index
            )
        )

        if take_ensemble_mean:
            relevant_prediction_tensor = K.mean(
                prediction_tensor[:, channel_index, :], axis=-1
            )
        else:
            relevant_prediction_tensor = prediction_tensor[:, channel_index, :]

        return K.min(relevant_prediction_tensor)

    metric.__name__ = function_name
    return metric


def max_prediction_with_constraints(
        channel_index,
        intensity_index, r34_index, r50_index, r64_index, rmw_index,
        take_ensemble_mean, function_name, test_mode=False):
    """Same as `max_prediction` but after applying physical constraints.

    :param channel_index: Array index for the desired channel.
    :param intensity_index: Array index for TC-intensity prediction.
    :param r34_index: Array index for radius-of-34-kt-wind prediction.
    :param r50_index: Array index for radius-of-50-kt-wind prediction.
    :param r64_index: Array index for radius-of-64-kt-wind prediction.
    :param rmw_index: Array index of radius-of-max-wind prediction.
    :param take_ensemble_mean: Boolean flag.  If True, will take maximum of
        ensemble-mean predictions.  If False, will take maximum of ensemble-
        member predictions.
    :param function_name: Name of function (string).
    :param test_mode: Leave this alone.
    :return: metric_function: Function defined below.
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(take_ensemble_mean)
    error_checking.assert_is_boolean(test_mode)

    def metric(target_tensor, prediction_tensor):
        """Computes maximum predicted value.

        :param target_tensor: See documentation for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: max_prediction: Maximum predicted value. (scalar float).
        """

        target_tensor, prediction_tensor = (
            custom_losses.apply_physical_constraints(
                target_tensor=target_tensor,
                prediction_tensor=prediction_tensor,
                intensity_index=intensity_index,
                r34_index=r34_index,
                r50_index=r50_index,
                r64_index=r64_index,
                rmw_index=rmw_index
            )
        )

        if take_ensemble_mean:
            relevant_prediction_tensor = K.mean(
                prediction_tensor[:, channel_index, :], axis=-1
            )
        else:
            relevant_prediction_tensor = prediction_tensor[:, channel_index, :]

        return K.max(relevant_prediction_tensor)

    metric.__name__ = function_name
    return metric
