"""Custom metrics for scalar predictions (x- and y-coord)."""

from tensorflow.keras import backend as K
from ml4tccf.machine_learning import custom_losses_scalar


def mean_prediction(target_tensor, prediction_tensor):
    """Computes mean prediction.

    :param target_tensor: See doc for
        `custom_losses_scalar.mean_squared_distance_kilometres2`.
    :param prediction_tensor: Same.
    :return: mean_prediction: Mean prediction.
    """

    return K.mean(prediction_tensor)


def mean_predictive_stdev(target_tensor, prediction_tensor):
    """Computes mean standard deviation of predictive distribution.

    :param target_tensor: See doc for
        `custom_losses_scalar.mean_squared_distance_kilometres2`.
    :param prediction_tensor: Same.
    :return: mean_predictive_stdev: Mean stdev of predictive distribution.
    """

    return K.mean(K.std(prediction_tensor, axis=-1))


def mean_predictive_range(target_tensor, prediction_tensor):
    """Computes mean range (max less min) of predictive distribution.

    :param target_tensor: See doc for
        `custom_losses_scalar.mean_squared_distance_kilometres2`.
    :param prediction_tensor: Same.
    :return: mean_predictive_range: Mean range of predictive distribution.
    """

    return K.mean(
        K.max(prediction_tensor, axis=-1) -
        K.min(prediction_tensor, axis=-1)
    )


def mean_target(target_tensor, prediction_tensor):
    """Computes mean target value.

    :param target_tensor: See doc for
        `custom_losses_scalar.mean_squared_distance_kilometres2`.
    :param prediction_tensor: Same.
    :return: mean_target: Mean target value.
    """

    return K.mean(target_tensor[:, :2])


def mean_grid_spacing_kilometres(target_tensor, prediction_tensor):
    """Computes mean grid spacing.

    :param target_tensor: See doc for
        `custom_losses_scalar.mean_squared_distance_kilometres2`.
    :param prediction_tensor: Same.
    :return: mean_grid_spacing_km: Mean grid spacing.
    """

    return K.mean(target_tensor[:, 2])


def mean_distance_kilometres(target_tensor, prediction_tensor):
    """Computes mean distance between predicted and actual TC centers.

    :param target_tensor: See doc for
        `custom_losses_scalar.mean_squared_distance_kilometres2`.
    :param prediction_tensor: Same.
    :return: mean_distance_km: Mean distance.
    """

    target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

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


def discretized_mean_dist_kilometres(target_tensor, prediction_tensor):
    """Computes discretized mean distance btw predicted and actual TC centers.

    :param target_tensor: See doc for
        `custom_losses_scalar.mean_squared_distance_kilometres2`.
    :param prediction_tensor: Same.
    :return: discretized_mean_distance_km: Discretized mean distance.
    """

    target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

    mean_prediction_tensor = K.mean(prediction_tensor, axis=-1)
    row_distances_km = (
        target_tensor[:, 2] *
        (mean_prediction_tensor[:, 0] - target_tensor[:, 0])
    )
    column_distances_km = (
        target_tensor[:, 2] *
        (mean_prediction_tensor[:, 1] - target_tensor[:, 1])
    )

    row_distances_km, column_distances_km = (
        custom_losses_scalar.discretize_distance_errors(
            row_distances_km=row_distances_km,
            column_distances_km=column_distances_km,
            true_center_latitudes_deg_n=target_tensor[:, 3]
        )
    )

    return K.mean(K.sqrt(
        row_distances_km ** 2 + column_distances_km ** 2
    ))


def crps_kilometres(target_tensor, prediction_tensor):
    """Computes CRPS between predicted and actual TC centers.

    :param target_tensor: See doc for
        `custom_losses_scalar.mean_squared_distance_kilometres2`.
    :param prediction_tensor: Same.
    :return: crps_km: CRPS.
    """

    target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

    mean_row_error_by_example_km = K.mean(
        K.expand_dims(target_tensor[:, 2], axis=-1) *
        K.abs(
            prediction_tensor[:, 0, :] -
            K.expand_dims(target_tensor[:, 0], axis=-1)
        ),
        axis=-1
    )

    mean_column_error_by_example_km = K.mean(
        K.expand_dims(target_tensor[:, 2], axis=-1) *
        K.abs(
            prediction_tensor[:, 1, :] -
            K.expand_dims(target_tensor[:, 1], axis=-1)
        ),
        axis=-1
    )

    mean_dist_error_by_example_km = K.sqrt(
        mean_row_error_by_example_km ** 2 +
        mean_column_error_by_example_km ** 2
    )

    prediction_diff_tensor_rowcol = K.abs(
        K.expand_dims(prediction_tensor, axis=-1) -
        K.expand_dims(prediction_tensor, axis=-2)
    )
    prediction_diff_tensor_rowcol = K.sqrt(
        prediction_diff_tensor_rowcol[:, 0, ...] ** 2
        + prediction_diff_tensor_rowcol[:, 1, ...] ** 2
    )

    grid_spacing_tensor_km = K.expand_dims(target_tensor[:, 2], axis=-1)
    grid_spacing_tensor_km = K.expand_dims(grid_spacing_tensor_km, axis=-1)
    prediction_diff_tensor_km = (
        grid_spacing_tensor_km * prediction_diff_tensor_rowcol
    )
    mean_prediction_diff_by_example_km = K.mean(
        prediction_diff_tensor_km, axis=(-2, -1)
    )

    return K.mean(
        mean_dist_error_by_example_km -
        0.5 * mean_prediction_diff_by_example_km
    )


def discretized_crps_kilometres(target_tensor, prediction_tensor):
    """Computes discretized CRPS between predicted and actual TC centers.

    :param target_tensor: See doc for
        `custom_losses_scalar.mean_squared_distance_kilometres2`.
    :param prediction_tensor: Same.
    :return: discretized_crps_km: Discretized CRPS.
    """

    target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

    mean_row_error_by_example_km = K.mean(
        K.expand_dims(target_tensor[:, 2], axis=-1) *
        K.abs(
            prediction_tensor[:, 0, :] -
            K.expand_dims(target_tensor[:, 0], axis=-1)
        ),
        axis=-1
    )

    mean_column_error_by_example_km = K.mean(
        K.expand_dims(target_tensor[:, 2], axis=-1) *
        K.abs(
            prediction_tensor[:, 1, :] -
            K.expand_dims(target_tensor[:, 1], axis=-1)
        ),
        axis=-1
    )

    mean_row_error_by_example_km, mean_column_error_by_example_km = (
        custom_losses_scalar.discretize_distance_errors(
            row_distances_km=mean_row_error_by_example_km,
            column_distances_km=mean_column_error_by_example_km,
            true_center_latitudes_deg_n=target_tensor[:, 3]
        )
    )
    mean_dist_error_by_example_km = K.sqrt(
        mean_row_error_by_example_km ** 2 +
        mean_column_error_by_example_km ** 2
    )

    prediction_diff_tensor_rowcol = K.abs(
        K.expand_dims(prediction_tensor, axis=-1) -
        K.expand_dims(prediction_tensor, axis=-2)
    )
    prediction_diff_tensor_rowcol = K.sqrt(
        prediction_diff_tensor_rowcol[:, 0, ...] ** 2
        + prediction_diff_tensor_rowcol[:, 1, ...] ** 2
    )

    grid_spacing_tensor_km = K.expand_dims(target_tensor[:, 2], axis=-1)
    grid_spacing_tensor_km = K.expand_dims(grid_spacing_tensor_km, axis=-1)
    prediction_diff_tensor_km = (
        grid_spacing_tensor_km * prediction_diff_tensor_rowcol
    )
    mean_prediction_diff_by_example_km = K.mean(
        prediction_diff_tensor_km, axis=(-2, -1)
    )

    return K.mean(
        mean_dist_error_by_example_km -
        0.5 * mean_prediction_diff_by_example_km
    )
