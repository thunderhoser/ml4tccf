"""Custom metrics for gridded predictions."""

import numpy
from tensorflow.keras import backend as K


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

    print(target_tensor)
    print(prediction_tensor)
    row_index_matrix, _ = numpy.indices(target_tensor.shape[1:])
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

    _, column_index_matrix = numpy.indices(target_tensor.shape[1:])
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
        target_tensor.shape[1:]
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
