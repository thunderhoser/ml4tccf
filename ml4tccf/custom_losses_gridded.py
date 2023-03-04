"""Custom loss functions for gridded predictions."""

import os
import sys
from tensorflow.keras import backend as K

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import image_filtering


def fractions_skill_score(
        half_window_size_px, use_as_loss_function, function_name=None,
        test_mode=False):
    """Creates fractions skill score (FSS) loss function.

    M = number of rows in grid
    N = number of columns in grid

    :param half_window_size_px: Number of pixels in half of smoothing window (on
        either side of center).  If this argument is K, the window size will be
        (1 + 2 * K) by (1 + 2 * K).
    :param use_as_loss_function: Boolean flag.  FSS is positively oriented
        (higher is better), but if using it as loss function, we want it to be
        negatively oriented.  Thus, if `use_as_loss_function == True`, will
        return 1 - FSS.  If `use_as_loss_function == False`, will return just
        FSS.
    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_boolean(use_as_loss_function)
    error_checking.assert_is_boolean(test_mode)

    if function_name is not None:
        error_checking.assert_is_string(function_name)

    weight_matrix = image_filtering.create_mean_conv_filter(
        half_num_rows=half_window_size_px,
        half_num_columns=half_window_size_px, num_channels=1
    )

    def loss(target_tensor, prediction_tensor):
        """Computes loss (fractions skill score).

        E = number of examples
        M = number of grid rows
        N = number of grid columns
        S = ensemble size

        :param target_tensor: E-by-M-by-N tensor of target values.
        :param prediction_tensor: E-by-M-by-N-by-S tensor of predicted values.
        :return: loss: Fractions skill score.
        """

        smoothed_target_tensor = K.conv2d(
            x=K.expand_dims(target_tensor, axis=-1),
            kernel=weight_matrix,
            padding='same', strides=(1, 1), data_format='channels_last'
        )

        smoothed_prediction_tensor = K.conv2d(
            x=K.mean(prediction_tensor, axis=-1, keepdims=True),
            kernel=weight_matrix,
            padding='same', strides=(1, 1), data_format='channels_last'
        )

        actual_mse = K.mean(
            (smoothed_target_tensor - smoothed_prediction_tensor) ** 2
        )
        reference_mse = K.mean(
            smoothed_target_tensor ** 2 + smoothed_prediction_tensor ** 2
        )
        reference_mse = K.maximum(reference_mse, K.epsilon())

        if use_as_loss_function:
            return actual_mse / reference_mse

        return 1. - actual_mse / reference_mse

    if function_name is not None:
        loss.__name__ = function_name

    return loss


def heidke_score(use_as_loss_function, function_name=None):
    """Creates Heidke-score loss function.

    :param use_as_loss_function: See doc for `fractions_skill_score`.
    :param function_name: Same.
    :return: heidke_function: Function (defined below).
    """

    error_checking.assert_is_boolean(use_as_loss_function)

    def heidke_function(target_tensor, prediction_tensor):
        """Computes Heidke score.

        :param target_tensor: See doc for `fractions_skill_score`.
        :param prediction_tensor: Same.
        :return: heidke_value: Heidke score (scalar).
        """

        mean_prediction_tensor = K.mean(prediction_tensor, axis=-1)

        num_true_positives = K.sum(target_tensor * mean_prediction_tensor)
        num_false_positives = K.sum(
            (1 - target_tensor) * mean_prediction_tensor
        )
        num_false_negatives = K.sum(
            target_tensor * (1 - mean_prediction_tensor)
        )
        num_true_negatives = K.sum(
            (1 - target_tensor) * (1 - mean_prediction_tensor)
        )

        random_num_correct = (
            (num_true_positives + num_false_positives) *
            (num_true_positives + num_false_negatives) +
            (num_false_negatives + num_true_negatives) *
            (num_false_positives + num_true_negatives)
        )
        num_examples = (
            num_true_positives + num_false_positives +
            num_false_negatives + num_true_negatives
        )
        random_num_correct = random_num_correct / num_examples

        numerator = num_true_positives + num_true_negatives - random_num_correct
        denominator = num_examples - random_num_correct + K.epsilon()

        if use_as_loss_function:
            return 1. - numerator / denominator

        return numerator / denominator

    if function_name is not None:
        heidke_function.__name__ = function_name

    return heidke_function


def peirce_score(use_as_loss_function, function_name=None):
    """Creates Peirce-score loss function.

    :param use_as_loss_function: See doc for `fractions_skill_score`.
    :param function_name: Same.
    :return: peirce_function: Function (defined below).
    """

    error_checking.assert_is_boolean(use_as_loss_function)

    def peirce_function(target_tensor, prediction_tensor):
        """Computes Peirce score.

        :param target_tensor: See doc for `fractions_skill_score`.
        :param prediction_tensor: Same.
        :return: peirce_value: Peirce score (scalar).
        """

        mean_prediction_tensor = K.mean(prediction_tensor, axis=-1)

        num_true_positives = K.sum(target_tensor * mean_prediction_tensor)
        num_false_positives = K.sum(
            (1 - target_tensor) * mean_prediction_tensor
        )
        num_false_negatives = K.sum(
            target_tensor * (1 - mean_prediction_tensor)
        )
        num_true_negatives = K.sum(
            (1 - target_tensor) * (1 - mean_prediction_tensor)
        )

        pod_value = (
            num_true_positives /
            (num_true_positives + num_false_negatives + K.epsilon())
        )
        pofd_value = (
            num_false_positives /
            (num_false_positives + num_true_negatives + K.epsilon())
        )

        if use_as_loss_function:
            return pofd_value - pod_value

        return pod_value - pofd_value

    if function_name is not None:
        peirce_function.__name__ = function_name

    return peirce_function


def gerrity_score(use_as_loss_function, function_name=None):
    """Creates Gerrity-score loss function.

    :param use_as_loss_function: See doc for `fractions_skill_score`.
    :param function_name: Same.
    :return: gerrity_function: Function (defined below).
    """

    error_checking.assert_is_boolean(use_as_loss_function)

    def gerrity_function(target_tensor, prediction_tensor):
        """Computes Gerrity score.

        :param target_tensor: See doc for `fractions_skill_score`.
        :param prediction_tensor: Same.
        :return: gerrity_value: Gerrity score (scalar).
        """

        mean_prediction_tensor = K.mean(prediction_tensor, axis=-1)

        num_true_positives = K.sum(target_tensor * mean_prediction_tensor)
        num_false_positives = K.sum(
            (1 - target_tensor) * mean_prediction_tensor
        )
        num_false_negatives = K.sum(
            target_tensor * (1 - mean_prediction_tensor)
        )
        num_true_negatives = K.sum(
            (1 - target_tensor) * (1 - mean_prediction_tensor)
        )

        event_ratio = (
            (num_false_positives + num_true_negatives) /
            (num_true_positives + num_false_negatives + K.epsilon())
        )
        num_examples = (
            num_true_positives + num_false_positives +
            num_false_negatives + num_true_negatives
        )

        numerator = (
            num_true_positives * event_ratio
            + num_true_negatives * (1. / event_ratio)
            - num_false_positives - num_false_negatives
        )

        if use_as_loss_function:
            return 1. - numerator / num_examples

        return numerator / num_examples

    if function_name is not None:
        gerrity_function.__name__ = function_name

    return gerrity_function
