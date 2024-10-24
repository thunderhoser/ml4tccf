"""Custom metrics for TC-structure parameters."""

from tensorflow.keras import backend as K
from gewittergefahr.gg_utils import error_checking


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
