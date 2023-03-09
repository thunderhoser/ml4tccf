"""Custom loss functions for gridded predictions."""

import os
import sys
import tensorflow
from tensorflow.keras import backend as K

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import image_filtering
import custom_metrics_gridded


def _log2(input_tensor):
    """Computes logarithm in base 2.

    :param input_tensor: Keras tensor.
    :return: logarithm_tensor: Keras tensor with the same shape as
        `input_tensor`.
    """

    return (
        K.log(K.maximum(input_tensor, 1e-6)) /
        K.log(tensorflow.Variable(2., dtype=tensorflow.float64))
    )


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

        :param target_tensor: E-by-M-by-N-by-1 tensor of target values.
        :param prediction_tensor: E-by-M-by-N-by-S tensor of predicted values.
        :return: loss: Fractions skill score.
        """

        smoothed_target_tensor = K.conv2d(
            x=target_tensor, kernel=weight_matrix,
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


def heidke_score(half_window_size_px, use_as_loss_function, function_name=None):
    """Creates Heidke-score loss function.

    :param half_window_size_px: See doc for `fractions_skill_score`.
    :param use_as_loss_function: Same.
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

        filtered_target_tensor = custom_metrics_gridded.apply_max_filter(
            input_tensor=target_tensor,
            half_window_size_px=half_window_size_px
        )
        filtered_prediction_tensor = custom_metrics_gridded.apply_max_filter(
            input_tensor=K.mean(prediction_tensor, axis=-1, keepdims=True),
            half_window_size_px=half_window_size_px
        )

        num_true_positives = K.sum(
            filtered_target_tensor * filtered_prediction_tensor
        )
        num_false_positives = K.sum(
            (1 - filtered_target_tensor) * filtered_prediction_tensor
        )
        num_false_negatives = K.sum(
            filtered_target_tensor * (1 - filtered_prediction_tensor)
        )
        num_true_negatives = K.sum(
            (1 - filtered_target_tensor) * (1 - filtered_prediction_tensor)
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


def peirce_score(half_window_size_px, use_as_loss_function, function_name=None):
    """Creates Peirce-score loss function.

    :param half_window_size_px: See doc for `fractions_skill_score`.
    :param use_as_loss_function: Same.
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

        filtered_target_tensor = custom_metrics_gridded.apply_max_filter(
            input_tensor=target_tensor,
            half_window_size_px=half_window_size_px
        )
        filtered_prediction_tensor = custom_metrics_gridded.apply_max_filter(
            input_tensor=K.mean(prediction_tensor, axis=-1, keepdims=True),
            half_window_size_px=half_window_size_px
        )

        num_true_positives = K.sum(
            filtered_target_tensor * filtered_prediction_tensor
        )
        num_false_positives = K.sum(
            (1 - filtered_target_tensor) * filtered_prediction_tensor
        )
        num_false_negatives = K.sum(
            filtered_target_tensor * (1 - filtered_prediction_tensor)
        )
        num_true_negatives = K.sum(
            (1 - filtered_target_tensor) * (1 - filtered_prediction_tensor)
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


def gerrity_score(half_window_size_px, use_as_loss_function,
                  function_name=None):
    """Creates Gerrity-score loss function.

    :param half_window_size_px: See doc for `fractions_skill_score`.
    :param use_as_loss_function: Same.
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

        filtered_target_tensor = custom_metrics_gridded.apply_max_filter(
            input_tensor=target_tensor,
            half_window_size_px=half_window_size_px
        )
        filtered_prediction_tensor = custom_metrics_gridded.apply_max_filter(
            input_tensor=K.mean(prediction_tensor, axis=-1, keepdims=True),
            half_window_size_px=half_window_size_px
        )

        num_true_positives = K.sum(
            filtered_target_tensor * filtered_prediction_tensor
        )
        num_false_positives = K.sum(
            (1 - filtered_target_tensor) * filtered_prediction_tensor
        )
        num_false_negatives = K.sum(
            filtered_target_tensor * (1 - filtered_prediction_tensor)
        )
        num_true_negatives = K.sum(
            (1 - filtered_target_tensor) * (1 - filtered_prediction_tensor)
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


def csi(half_window_size_px, use_as_loss_function, function_name=None):
    """Creates critical success index (CSI) loss function.

    :param half_window_size_px: See doc for `fractions_skill_score`.
    :param use_as_loss_function: Same.
    :param function_name: Same.
    :return: csi_function: Function (defined below).
    """

    error_checking.assert_is_boolean(use_as_loss_function)

    def csi_function(target_tensor, prediction_tensor):
        """Computes CSI.

        :param target_tensor: See doc for `fractions_skill_score`.
        :param prediction_tensor: Same.
        :return: csi_value: CSI (scalar).
        """

        pod_value = K.epsilon() + custom_metrics_gridded.pod(
            half_window_size_px=half_window_size_px
        )(target_tensor=target_tensor, prediction_tensor=prediction_tensor)

        success_ratio_value = K.epsilon() + custom_metrics_gridded.success_ratio(
            half_window_size_px=half_window_size_px
        )(target_tensor=target_tensor, prediction_tensor=prediction_tensor)

        csi_value = (pod_value ** -1 + success_ratio_value ** -1 - 1) ** -1

        if use_as_loss_function:
            return 1. - csi_value

        return csi_value

    if function_name is not None:
        csi_function.__name__ = function_name

    return csi_function


def iou(half_window_size_px, use_as_loss_function, function_name=None):
    """Creates intersection over union (IOU) loss function.

    :param half_window_size_px: See doc for `fractions_skill_score`.
    :param use_as_loss_function: Same.
    :param function_name: Same.
    :return: iou_function: Function (defined below).
    """

    error_checking.assert_is_boolean(use_as_loss_function)

    def iou_function(target_tensor, prediction_tensor):
        """Computes IOU.

        :param target_tensor: See doc for `fractions_skill_score`.
        :param prediction_tensor: Same.
        :return: iou_value: IOU (scalar).
        """

        filtered_target_tensor = custom_metrics_gridded.apply_max_filter(
            input_tensor=target_tensor,
            half_window_size_px=half_window_size_px
        )

        filtered_target_tensor = filtered_target_tensor[..., 0]
        mean_prediction_tensor = K.mean(prediction_tensor, axis=-1)

        intersection_tensor = K.sum(
            filtered_target_tensor * mean_prediction_tensor, axis=(1, 2)
        )
        union_tensor = (
            K.sum(filtered_target_tensor, axis=(1, 2)) +
            K.sum(mean_prediction_tensor, axis=(1, 2)) -
            intersection_tensor
        )

        iou_value = K.mean(
            intersection_tensor / (union_tensor + K.epsilon())
        )

        if use_as_loss_function:
            return 1. - iou_value

        return iou_value

    if function_name is not None:
        iou_function.__name__ = function_name

    return iou_function


def all_class_iou(half_window_size_px, use_as_loss_function,
                  function_name=None):
    """Creates all-class intersection over union (IOU) loss function.

    :param half_window_size_px: See doc for `fractions_skill_score`.
    :param use_as_loss_function: Same.
    :param function_name: Same.
    :return: all_class_iou_function: Function (defined below).
    """

    error_checking.assert_is_boolean(use_as_loss_function)

    def all_class_iou_function(target_tensor, prediction_tensor):
        """Computes IOU.

        :param target_tensor: See doc for `fractions_skill_score`.
        :param prediction_tensor: Same.
        :return: all_class_iou_value: All-class IOU (scalar).
        """

        filtered_target_tensor = custom_metrics_gridded.apply_max_filter(
            input_tensor=target_tensor,
            half_window_size_px=half_window_size_px
        )

        filtered_target_tensor = filtered_target_tensor[..., 0]
        mean_prediction_tensor = K.mean(prediction_tensor, axis=-1)

        positive_intersection_tensor = K.sum(
            filtered_target_tensor * mean_prediction_tensor, axis=(1, 2)
        )
        positive_union_tensor = (
            K.sum(filtered_target_tensor, axis=(1, 2)) +
            K.sum(mean_prediction_tensor, axis=(1, 2)) -
            positive_intersection_tensor
        )

        negative_intersection_tensor = K.sum(
            (1. - filtered_target_tensor) * (1. - mean_prediction_tensor),
            axis=(1, 2)
        )
        negative_union_tensor = (
            K.sum(1. - filtered_target_tensor, axis=(1, 2)) +
            K.sum(1. - mean_prediction_tensor, axis=(1, 2)) -
            negative_intersection_tensor
        )

        positive_iou = K.mean(
            positive_intersection_tensor / (positive_union_tensor + K.epsilon())
        )
        negative_iou = K.mean(
            negative_intersection_tensor / (negative_union_tensor + K.epsilon())
        )
        iou_value = (positive_iou + negative_iou) / 2

        if use_as_loss_function:
            return 1. - iou_value

        return iou_value

    if function_name is not None:
        all_class_iou_function.__name__ = function_name

    return all_class_iou_function


def dice_coeff(half_window_size_px, use_as_loss_function, function_name=None):
    """Creates Dice-coefficient loss function.

    :param half_window_size_px: See doc for `fractions_skill_score`.
    :param use_as_loss_function: Same.
    :param function_name: Same.
    :return: dice_coeff_function: Function (defined below).
    """

    error_checking.assert_is_boolean(use_as_loss_function)

    def dice_coeff_function(target_tensor, prediction_tensor):
        """Computes Dice coefficient.

        :param target_tensor: See doc for `fractions_skill_score`.
        :param prediction_tensor: Same.
        :return: dice_coeff_value: Dice coefficient (scalar).
        """

        filtered_target_tensor = custom_metrics_gridded.apply_max_filter(
            input_tensor=target_tensor,
            half_window_size_px=half_window_size_px
        )

        filtered_target_tensor = filtered_target_tensor[..., 0]
        mean_prediction_tensor = K.mean(prediction_tensor, axis=-1)

        positive_intersection_tensor = K.sum(
            filtered_target_tensor * mean_prediction_tensor, axis=(1, 2)
        )
        negative_intersection_tensor = K.sum(
            (1. - filtered_target_tensor) * (1. - mean_prediction_tensor),
            axis=(1, 2)
        )
        num_pixels_tensor = K.sum(
            K.ones_like(mean_prediction_tensor), axis=(1, 2)
        )

        dice_value = K.mean(
            (positive_intersection_tensor + negative_intersection_tensor) /
            num_pixels_tensor
        )

        if use_as_loss_function:
            return 1. - dice_value

        return dice_value

    if function_name is not None:
        dice_coeff_function.__name__ = function_name

    return dice_coeff_function


def brier_score(half_window_size_px, function_name=None):
    """Creates Brier-score loss function.

    :param half_window_size_px: See doc for `fractions_skill_score`.
    :param function_name: Same.
    :return: brier_function: Function (defined below).
    """

    def brier_function(target_tensor, prediction_tensor):
        """Computes Brier score.

        :param target_tensor: See doc for `fractions_skill_score`.
        :param prediction_tensor: Same.
        :return: brier_value: Brier score (scalar).
        """

        filtered_target_tensor = custom_metrics_gridded.apply_max_filter(
            input_tensor=target_tensor,
            half_window_size_px=half_window_size_px
        )

        filtered_target_tensor = filtered_target_tensor[..., 0]
        mean_prediction_tensor = K.mean(prediction_tensor, axis=-1)

        squared_error_tensor = K.sum(
            (filtered_target_tensor - mean_prediction_tensor) ** 2,
            axis=(1, 2)
        )
        num_pixels_tensor = K.sum(
            K.ones_like(mean_prediction_tensor), axis=(1, 2)
        )

        return K.mean(squared_error_tensor / num_pixels_tensor)

    if function_name is not None:
        brier_function.__name__ = function_name

    return brier_function


def cross_entropy(half_window_size_px, function_name=None):
    """Creates cross-entropy loss function.

    :param half_window_size_px: See doc for `fractions_skill_score`.
    :param function_name: Same.
    :return: xentropy_function: Function (defined below).
    """

    def xentropy_function(target_tensor, prediction_tensor):
        """Computes cross-entropy.

        :param target_tensor: See doc for `fractions_skill_score`.
        :param prediction_tensor: Same.
        :return: xentropy_value: Cross-entropy (scalar).
        """

        filtered_target_tensor = custom_metrics_gridded.apply_max_filter(
            input_tensor=target_tensor,
            half_window_size_px=half_window_size_px
        )

        filtered_target_tensor = filtered_target_tensor[..., 0]
        mean_prediction_tensor = K.mean(prediction_tensor, axis=-1)

        xentropy_tensor = K.sum(
            filtered_target_tensor * _log2(mean_prediction_tensor) +
            (1. - filtered_target_tensor) * _log2(1. - mean_prediction_tensor),
            axis=(1, 2)
        )
        num_pixels_tensor = K.sum(
            K.ones_like(mean_prediction_tensor), axis=(1, 2)
        )

        return -K.mean(xentropy_tensor / num_pixels_tensor)

    if function_name is not None:
        xentropy_function.__name__ = function_name

    return xentropy_function
