"""Custom metrics for gridded predictions."""

import numpy
from tensorflow.keras import backend as K


def apply_max_filter(input_tensor, half_window_size_px):
    """Applies maximum-filter to tensor.

    :param input_tensor: Keras tensor.
    :param half_window_size_px: Number of pixels in half of filter window (on
        either side of center).  If this argument is K, the window size will be
        (1 + 2 * K) by (1 + 2 * K).
    :return: output_tensor: Filtered version of `input_tensor`.
    """

    window_size_px = 2 * half_window_size_px + 1

    return K.pool2d(
        x=input_tensor, pool_mode='max',
        pool_size=(window_size_px, window_size_px), strides=(1, 1),
        padding='same', data_format='channels_last'
    )


def num_actual_oriented_true_positives(half_window_size_px, function_name=None):
    """Creates function to compute number of actual-oriented true positives.

    :param half_window_size_px: Number of pixels in half of filter window (on
        either side of center).  If this argument is K, the window size will be
        (1 + 2 * K) by (1 + 2 * K).
    :param function_name: Function name (string).
    :return: num_ao_true_pos_function: Function (defined below).
    """

    def num_ao_true_pos_function(target_tensor, prediction_tensor):
        """Computes number of actual-oriented true positives.

        :param target_tensor: See doc for `mean_target`.
        :param prediction_tensor: Same.
        :return: num_actual_oriented_true_positives: Scalar.
        """

        filtered_prediction_tensor = apply_max_filter(
            input_tensor=K.mean(prediction_tensor, axis=-1, keepdims=True),
            half_window_size_px=half_window_size_px
        )
        return K.sum(target_tensor * filtered_prediction_tensor[..., 0])

    if function_name is not None:
        num_ao_true_pos_function.__name__ = function_name

    return num_ao_true_pos_function


def num_prediction_oriented_true_positives(half_window_size_px,
                                           function_name=None):
    """Creates function to compute number of prediction-oriented true positives.

    :param half_window_size_px: See doc for
        `num_actual_oriented_true_positives`.
    :param function_name: Same.
    :return: num_po_true_pos_function: Function (defined below).
    """

    def num_po_true_pos_function(target_tensor, prediction_tensor):
        """Computes number of prediction-oriented true positives.

        :param target_tensor: See doc for `mean_target`.
        :param prediction_tensor: Same.
        :return: num_prediction_oriented_true_positives: Scalar.
        """

        filtered_target_tensor = apply_max_filter(
            input_tensor=K.expand_dims(target_tensor, axis=-1),
            half_window_size_px=half_window_size_px
        )
        return K.sum(
            filtered_target_tensor *
            K.mean(prediction_tensor, axis=-1, keepdims=True)
        )

    if function_name is not None:
        num_po_true_pos_function.__name__ = function_name

    return num_po_true_pos_function


def num_false_positives(half_window_size_px, function_name=None):
    """Creates function to compute number of false positives.

    :param half_window_size_px: See doc for
        `num_actual_oriented_true_positives`.
    :param function_name: Same.
    :return: num_false_pos_function: Function (defined below).
    """

    def num_false_pos_function(target_tensor, prediction_tensor):
        """Computes number of false positives.

        :param target_tensor: See doc for `mean_target`.
        :param prediction_tensor: Same.
        :return: num_false_positives: Scalar.
        """

        filtered_target_tensor = apply_max_filter(
            input_tensor=K.expand_dims(target_tensor, axis=-1),
            half_window_size_px=half_window_size_px
        )
        return K.sum(
            (1. - filtered_target_tensor) *
            K.mean(prediction_tensor, axis=-1, keepdims=True)
        )

    if function_name is not None:
        num_false_pos_function.__name__ = function_name

    return num_false_pos_function


def num_false_negatives(half_window_size_px, function_name=None):
    """Creates function to compute number of false negatives.

    :param half_window_size_px: See doc for
        `num_actual_oriented_true_positives`.
    :param function_name: Same.
    :return: num_false_neg_function: Function (defined below).
    """

    def num_false_neg_function(target_tensor, prediction_tensor):
        """Computes number of false negatives.

        :param target_tensor: See doc for `mean_target`.
        :param prediction_tensor: Same.
        :return: num_false_negatives: Scalar.
        """

        filtered_prediction_tensor = apply_max_filter(
            input_tensor=K.mean(prediction_tensor, axis=-1, keepdims=True),
            half_window_size_px=half_window_size_px
        )
        return K.sum(
            target_tensor * (1. - filtered_prediction_tensor[..., 0])
        )

    if function_name is not None:
        num_false_neg_function.__name__ = function_name

    return num_false_neg_function


def pod(half_window_size_px, function_name=None):
    """Creates function to compute probability of detection.

    :param half_window_size_px: See doc for
        `num_actual_oriented_true_positives`.
    :param function_name: Same.
    :return: pod_function: Function (defined below).
    """

    def pod_function(target_tensor, prediction_tensor):
        """Computes probability of detection.

        :param target_tensor: See doc for `mean_target`.
        :param prediction_tensor: Same.
        :return: pod: Probability of detection.
        """

        this_num_ao_true_positives = num_actual_oriented_true_positives(
            half_window_size_px=half_window_size_px
        )(target_tensor=target_tensor, prediction_tensor=prediction_tensor)

        this_num_false_negatives = num_false_negatives(
            half_window_size_px=half_window_size_px
        )(target_tensor=target_tensor, prediction_tensor=prediction_tensor)

        denominator = (
            this_num_ao_true_positives + this_num_false_negatives + K.epsilon()
        )
        return this_num_ao_true_positives / denominator

    if function_name is not None:
        pod_function.__name__ = function_name

    return pod_function


def success_ratio(half_window_size_px, function_name=None):
    """Creates function to compute success ratio.

    :param half_window_size_px: See doc for
        `num_actual_oriented_true_positives`.
    :param function_name: Same.
    :return: success_ratio_function: Function (defined below).
    """

    def success_ratio_function(target_tensor, prediction_tensor):
        """Computes success ratio.

        :param target_tensor: See doc for `mean_target`.
        :param prediction_tensor: Same.
        :return: success_ratio: Success ratio.
        """

        this_num_po_true_positives = num_prediction_oriented_true_positives(
            half_window_size_px=half_window_size_px
        )(target_tensor=target_tensor, prediction_tensor=prediction_tensor)

        this_num_false_positives = num_false_positives(
            half_window_size_px=half_window_size_px
        )(target_tensor=target_tensor, prediction_tensor=prediction_tensor)

        denominator = (
            this_num_po_true_positives + this_num_false_positives + K.epsilon()
        )
        return this_num_po_true_positives / denominator

    if function_name is not None:
        success_ratio_function.__name__ = function_name

    return success_ratio_function


def frequency_bias(half_window_size_px, function_name=None):
    """Creates function to compute frequency bias.

    :param half_window_size_px: See doc for
        `num_actual_oriented_true_positives`.
    :param function_name: Same.
    :return: frequency_bias_function: Function (defined below).
    """

    def frequency_bias_function(target_tensor, prediction_tensor):
        """Computes frequency bias.

        :param target_tensor: See doc for `mean_target`.
        :param prediction_tensor: Same.
        :return: frequency_bias: Frequency bias.
        """

        pod_value = pod(
            half_window_size_px=half_window_size_px
        )(target_tensor=target_tensor, prediction_tensor=prediction_tensor)

        success_ratio_value = success_ratio(
            half_window_size_px=half_window_size_px
        )(target_tensor=target_tensor, prediction_tensor=prediction_tensor)

        return pod_value / (success_ratio_value + K.epsilon())

    if function_name is not None:
        frequency_bias_function.__name__ = function_name

    return frequency_bias_function


def mean_target(target_tensor, prediction_tensor):
    """Computes mean target value.

    E = number of examples
    M = number of grid rows
    N = number of grid columns
    S = ensemble size

    :param target_tensor: E-by-M-by-N tensor of target values.
    :param prediction_tensor: E-by-M-by-N-by-S tensor of predicted values.
    :return: mean_target_value: Scalar.
    """

    return K.mean(target_tensor)


def mean_prediction(target_tensor, prediction_tensor):
    """Computes mean prediction.

    :param target_tensor: See doc for `mean_target`.
    :param prediction_tensor: Same.
    :return: mean_prediction: Scalar.
    """

    return K.mean(prediction_tensor)


def min_target(target_tensor, prediction_tensor):
    """Computes minimum target value.

    :param target_tensor: See doc for `mean_target`.
    :param prediction_tensor: Same.
    :return: min_target_value: Scalar.
    """

    return K.min(target_tensor)


def min_prediction(target_tensor, prediction_tensor):
    """Computes minimum prediction.

    :param target_tensor: See doc for `mean_target`.
    :param prediction_tensor: Same.
    :return: min_prediction: Scalar.
    """

    return K.min(prediction_tensor)


def max_target(target_tensor, prediction_tensor):
    """Computes maximum target value.

    :param target_tensor: See doc for `mean_target`.
    :param prediction_tensor: Same.
    :return: max_target_value: Scalar.
    """

    return K.max(target_tensor)


def max_prediction(target_tensor, prediction_tensor):
    """Computes maximum prediction.

    :param target_tensor: See doc for `mean_target`.
    :param prediction_tensor: Same.
    :return: max_prediction: Scalar.
    """

    return K.max(prediction_tensor)


def sum_of_targets(target_tensor, prediction_tensor):
    """Computes sum of target values.

    :param target_tensor: See doc for `mean_target`.
    :param prediction_tensor: Same.
    :return: sum_of_target_values: Scalar.
    """

    return K.sum(target_tensor)


def sum_of_predictions(target_tensor, prediction_tensor):
    """Computes sum of predictions.

    :param target_tensor: See doc for `mean_target`.
    :param prediction_tensor: Same.
    :return: sum_of_predictions: Scalar.
    """

    return K.sum(prediction_tensor)


def mean_predictive_stdev(target_tensor, prediction_tensor):
    """Computes mean standard deviation of predictive distribution.

    :param target_tensor: See doc for `mean_target`.
    :param prediction_tensor: Same.
    :return: mean_predictive_stdev: Scalar.
    """

    return K.mean(K.std(prediction_tensor, axis=-1))


def mean_predictive_range(target_tensor, prediction_tensor):
    """Computes mean range of predictive distribution.

    :param target_tensor: See doc for `mean_target`.
    :param prediction_tensor: Same.
    :return: mean_predictive_range: Scalar.
    """

    return K.mean(
        K.max(prediction_tensor, axis=-1) -
        K.min(prediction_tensor, axis=-1)
    )


def mean_center_of_mass_row_for_targets(target_tensor, prediction_tensor):
    """Computes mean row index at center of mass of target grid.

    :param target_tensor: See doc for `mean_target`.
    :param prediction_tensor: Same.
    :return: mean_row_index: Scalar.
    """

    row_index_matrix, _ = numpy.indices(prediction_tensor.shape[1:-1])
    row_index_matrix = numpy.expand_dims(row_index_matrix, axis=0)

    return K.mean(
        K.sum(row_index_matrix * target_tensor, axis=(1, 2)) /
        K.sum(target_tensor, axis=(1, 2))
    )


def mean_center_of_mass_column_for_targets(target_tensor, prediction_tensor):
    """Computes mean column index at center of mass of target grid.

    :param target_tensor: See doc for `mean_target`.
    :param prediction_tensor: Same.
    :return: mean_column_index: Scalar.
    """

    _, column_index_matrix = numpy.indices(prediction_tensor.shape[1:-1])
    column_index_matrix = numpy.expand_dims(column_index_matrix, axis=0)

    return K.mean(
        K.sum(column_index_matrix * target_tensor, axis=(1, 2)) /
        K.sum(target_tensor, axis=(1, 2))
    )


def mean_center_of_mass_row_for_predictions(target_tensor, prediction_tensor):
    """Computes mean row index at center of mass of prediction grid.

    :param target_tensor: See doc for `mean_target`.
    :param prediction_tensor: Same.
    :return: mean_row_index: Scalar.
    """

    mean_prediction_tensor = K.mean(prediction_tensor, axis=-1)
    row_index_matrix, _ = numpy.indices(mean_prediction_tensor.shape[1:])
    row_index_matrix = numpy.expand_dims(row_index_matrix, axis=0)

    return K.mean(
        K.sum(row_index_matrix * mean_prediction_tensor, axis=(1, 2)) /
        K.sum(mean_prediction_tensor, axis=(1, 2))
    )


def mean_center_of_mass_column_for_predictions(target_tensor,
                                               prediction_tensor):
    """Computes mean column index at center of mass of prediction grid.

    :param target_tensor: See doc for `mean_target`.
    :param prediction_tensor: Same.
    :return: mean_column_index: Scalar.
    """

    mean_prediction_tensor = K.mean(prediction_tensor, axis=-1)
    _, column_index_matrix = numpy.indices(mean_prediction_tensor.shape[1:])
    column_index_matrix = numpy.expand_dims(column_index_matrix, axis=0)

    return K.mean(
        K.sum(column_index_matrix * mean_prediction_tensor, axis=(1, 2)) /
        K.sum(mean_prediction_tensor, axis=(1, 2))
    )


def mean_center_of_mass_distance_px(target_tensor, prediction_tensor):
    """Computes mean distance between the two centers of mass.

    :param target_tensor: See doc for
        `custom_losses_scalar.mean_squared_distance_kilometres2`.
    :param prediction_tensor: Same.
    :return: mean_distance_px: Scalar.
    """

    row_index_matrix, column_index_matrix = numpy.indices(
        prediction_tensor.shape[1:-1]
    )
    row_index_matrix = numpy.expand_dims(row_index_matrix, axis=0)
    column_index_matrix = numpy.expand_dims(column_index_matrix, axis=0)

    target_row_indices = (
        K.sum(row_index_matrix * target_tensor, axis=(1, 2)) /
        K.sum(target_tensor, axis=(1, 2))
    )
    target_column_indices = (
        K.sum(column_index_matrix * target_tensor, axis=(1, 2)) /
        K.sum(target_tensor, axis=(1, 2))
    )

    mean_prediction_tensor = K.mean(prediction_tensor, axis=-1)
    predicted_row_indices = (
        K.sum(row_index_matrix * mean_prediction_tensor, axis=(1, 2)) /
        K.sum(mean_prediction_tensor, axis=(1, 2))
    )
    predicted_column_indices = (
        K.sum(column_index_matrix * mean_prediction_tensor, axis=(1, 2)) /
        K.sum(mean_prediction_tensor, axis=(1, 2))
    )

    return K.mean(K.sqrt(
        (target_row_indices - predicted_row_indices) ** 2 +
        (target_column_indices - predicted_column_indices) ** 2
    ))
