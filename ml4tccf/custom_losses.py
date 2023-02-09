"""Custom loss functions."""

import tensorflow
from tensorflow.keras import backend as K


def mean_prediction(target_tensor, prediction_tensor):
    """Computes mean prediction.

    :param target_tensor: See doc for `mean_squared_distance_kilometres2`.
    :param prediction_tensor: Same.
    :return: mean_prediction: Mean prediction.
    """

    return K.mean(prediction_tensor)


def mean_predictive_stdev(target_tensor, prediction_tensor):
    """Computes mean standard deviation of predictive distribution.

    :param target_tensor: See doc for `mean_squared_distance_kilometres2`.
    :param prediction_tensor: Same.
    :return: mean_predictive_stdev: Mean stdev of predictive distribution.
    """

    return K.mean(K.std(prediction_tensor, axis=-1))


def mean_predictive_range(target_tensor, prediction_tensor):
    """Computes mean range (max less min) of predictive distribution.

    :param target_tensor: See doc for `mean_squared_distance_kilometres2`.
    :param prediction_tensor: Same.
    :return: mean_predictive_range: Mean range of predictive distribution.
    """

    return K.mean(
        K.max(prediction_tensor, axis=-1) -
        K.min(prediction_tensor, axis=-1)
    )


def mean_target(target_tensor, prediction_tensor):
    """Computes mean target value.

    :param target_tensor: See doc for `mean_squared_distance_kilometres2`.
    :param prediction_tensor: Same.
    :return: mean_target: Mean target value.
    """

    return K.mean(target_tensor[:, :2])


def mean_grid_spacing_kilometres(target_tensor, prediction_tensor):
    """Computes mean grid spacing.

    :param target_tensor: See doc for `mean_squared_distance_kilometres2`.
    :param prediction_tensor: Same.
    :return: mean_grid_spacing_km: Mean grid spacing.
    """

    return K.mean(target_tensor[:, 2])


def mean_squared_distance_kilometres2(target_tensor, prediction_tensor):
    """Computes mean squared distance btw predicted and actual TC centers.

    E = number of examples
    S = ensemble size

    :param target_tensor: E-by-3 Keras tensor.  target_tensor[:, 0] contains
        row positions of TC centers; target_tensor[:, 1] contains column
        positions of TC centers; and target_tensor[:, 2] contains grid
        spacing in km, used to convert row-column distances to actual
        distances.
    :param prediction_tensor: E-by-2-by-S Keras tensor.
        prediction_tensor[:, 0, :] contains predicted row positions of TC
        centers, and prediction_tensor[:, 1, :] contains predicted column
        positions of TC centers.
    :return: mean_squared_distance_km: Mean squared distance.
    """

    mean_prediction_tensor = K.mean(prediction_tensor, axis=-1)
    row_distances_km = (
        target_tensor[:, 2] *
        (mean_prediction_tensor[:, 0] - target_tensor[:, 0])
    )
    column_distances_km = (
        target_tensor[:, 2] *
        (mean_prediction_tensor[:, 1] - target_tensor[:, 1])
    )

    return K.mean(
        row_distances_km ** 2 + column_distances_km ** 2
    )


def mean_distance_kilometres(target_tensor, prediction_tensor):
    """Computes mean distance between predicted and actual TC centers.


    :param target_tensor: See doc for `mean_squared_distance_kilometres2`.
    :param prediction_tensor: Same.
    :return: mean_distance_km: Mean distance.
    """

    mean_prediction_tensor = K.mean(prediction_tensor, axis=-1)
    row_distances_km = (
        target_tensor[:, 2] *
        (mean_prediction_tensor[:, 0] - target_tensor[:, 0])
    )
    column_distances_km = (
        target_tensor[:, 2] *
        (mean_prediction_tensor[:, 1] - target_tensor[:, 1])
    )

    return K.mean(K.sqrt(
        row_distances_km ** 2 + column_distances_km ** 2
    ))


def crps_kilometres(target_tensor, prediction_tensor):
    """Computes CRPS between predicted and actual TC centers.

    :param target_tensor: See doc for `mean_squared_distance_kilometres2`.
    :param prediction_tensor: Same.
    :return: crps_km: CRPS.
    """

    mean_prediction_tensor = K.mean(prediction_tensor, axis=-1)
    row_distances_km = (
        target_tensor[:, 2] *
        (mean_prediction_tensor[:, 0] - target_tensor[:, 0])
    )
    column_distances_km = (
        target_tensor[:, 2] *
        (mean_prediction_tensor[:, 1] - target_tensor[:, 1])
    )
    mean_error_tensor_km = K.sqrt(
        row_distances_km ** 2 + column_distances_km ** 2
    )

    prediction_tensor_no_nan = tensorflow.where(tensorflow.is_nan(prediction_tensor), tensorflow.zeros_like(prediction_tensor), prediction_tensor)
    prediction_diff_tensor = K.abs(
        K.expand_dims(prediction_tensor_no_nan, axis=-1) -
        K.expand_dims(prediction_tensor_no_nan, axis=-2)
    )

    all_rowcol_diff_tensor = K.sqrt(
        prediction_diff_tensor[:, 0, ...] ** 2
        + prediction_diff_tensor[:, 1, ...] ** 2
    )
    all_diff_tensor_km = (
        K.expand_dims(K.expand_dims(target_tensor[:, 2], axis=-1), axis=-1)
        * all_rowcol_diff_tensor
    )
    mean_prediction_diff_tensor_km = K.mean(
        all_diff_tensor_km, axis=(-2, -1)
    )

    return K.mean(
        mean_error_tensor_km - 0.5 * mean_prediction_diff_tensor_km
    )


def crps_part1(target_tensor, prediction_tensor):
    prediction_diff_tensor = K.abs(
        K.expand_dims(prediction_tensor, axis=-1) -
        K.expand_dims(prediction_tensor, axis=-2)
    )

    return -K.mean(prediction_diff_tensor)


def crps_part2(target_tensor, prediction_tensor):
    prediction_diff_tensor = K.abs(
        K.expand_dims(prediction_tensor, axis=-1) -
        K.expand_dims(prediction_tensor, axis=-2)
    )

    all_rowcol_diff_tensor = K.sqrt(
        prediction_diff_tensor[:, 0, ...] ** 2
        + prediction_diff_tensor[:, 1, ...] ** 2
    )

    return -K.mean(all_rowcol_diff_tensor)


def crps_part3(target_tensor, prediction_tensor):
    prediction_diff_tensor = K.abs(
        K.expand_dims(prediction_tensor, axis=-1) -
        K.expand_dims(prediction_tensor, axis=-2)
    )

    all_rowcol_diff_tensor = K.sqrt(
        prediction_diff_tensor[:, 0, ...] ** 2
        + prediction_diff_tensor[:, 1, ...] ** 2
    )

    all_diff_tensor_km = (
        K.expand_dims(K.expand_dims(target_tensor[:, 2], axis=-1), axis=-1)
        * all_rowcol_diff_tensor
    )

    return -K.mean(all_diff_tensor_km)


def crps_part4(target_tensor, prediction_tensor):
    prediction_diff_tensor = K.abs(
        K.expand_dims(prediction_tensor, axis=-1) -
        K.expand_dims(prediction_tensor, axis=-2)
    )

    all_rowcol_diff_tensor = K.sqrt(
        prediction_diff_tensor[:, 0, ...] ** 2
        + prediction_diff_tensor[:, 1, ...] ** 2
    )

    all_diff_tensor_km = (
        K.expand_dims(K.expand_dims(target_tensor[:, 2], axis=-1), axis=-1)
        * all_rowcol_diff_tensor
    )

    mean_prediction_diff_tensor_km = K.mean(
        all_diff_tensor_km, axis=(-2, -1)
    )

    return -K.mean(mean_prediction_diff_tensor_km)
