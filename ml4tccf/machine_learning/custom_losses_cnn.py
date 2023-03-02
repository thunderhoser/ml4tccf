"""Custom loss functions for CNN (as opposed to semantic segmentation)."""

import numpy
import tensorflow
from tensorflow.keras import backend as K

DEGREES_TO_RADIANS = numpy.pi / 180
TARGET_DISCRETIZATION_DEG = 0.1
DEG_LATITUDE_TO_KM = 60 * 1.852


def discretize_distance_errors(row_distances_km, column_distances_km,
                               true_center_latitudes_deg_n):
    """Discretizes distance errors.

    E = number of examples

    :param row_distances_km: length-E Keras tensor of row errors.
    :param column_distances_km: length-E Keras tensor of column errors.
    :param true_center_latitudes_deg_n: length-E Keras tensor of true center
        latitudes (deg north).
    :return: row_distances_km: Same as input, except that some elements may be
        set to zero.
    :return: column_distances_km: Same as input, except that some elements may
        be set to zero.
    """

    these_flags = tensorflow.where(
        K.abs(row_distances_km)
        < 0.5 * TARGET_DISCRETIZATION_DEG * DEG_LATITUDE_TO_KM,
        x=0., y=1.
    )
    these_flags = tensorflow.cast(these_flags, dtype=row_distances_km.dtype)
    row_distances_km = row_distances_km * these_flags

    latitude_cosines = K.cos(true_center_latitudes_deg_n * DEGREES_TO_RADIANS)
    these_flags = tensorflow.where(
        K.abs(column_distances_km) <
        0.5 * latitude_cosines * TARGET_DISCRETIZATION_DEG * DEG_LATITUDE_TO_KM,
        x=0., y=1.
    )
    these_flags = tensorflow.cast(these_flags, dtype=column_distances_km.dtype)
    column_distances_km = column_distances_km * these_flags

    return row_distances_km, column_distances_km


def mean_squared_distance_kilometres2(target_tensor, prediction_tensor):
    """Computes mean squared distance btw predicted and actual TC centers.

    E = number of examples
    S = ensemble size

    :param target_tensor: E-by-4 Keras tensor.  target_tensor[:, 0] contains
        row positions of TC centers; target_tensor[:, 1] contains column
        positions of TC centers; target_tensor[:, 2] contains grid spacing in
        km, used to convert row-column distances to actual distances; and
        target_tensor[:, 3] contains true TC latitudes in deg north.
    :param prediction_tensor: E-by-2-by-S Keras tensor.
        prediction_tensor[:, 0, :] contains predicted row positions of TC
        centers, and prediction_tensor[:, 1, :] contains predicted column
        positions of TC centers.
    :return: mean_squared_distance_km2: Mean squared distance.
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


def discretized_mean_sq_dist_kilometres2(target_tensor, prediction_tensor):
    """Computes discretized mean sq dist btw predicted and actual TC centers.

    E = number of examples
    S = ensemble size

    :param target_tensor: See doc for `mean_squared_distance_kilometres2`.
    :param prediction_tensor: Same.
    :return: discretized_mean_squared_distance_km2: Discretized mean squared
        distance.
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

    row_distances_km, column_distances_km = discretize_distance_errors(
        row_distances_km=row_distances_km,
        column_distances_km=column_distances_km,
        true_center_latitudes_deg_n=target_tensor[:, 3]
    )

    return K.mean(
        row_distances_km ** 2 + column_distances_km ** 2
    )


def weird_crps_kilometres2(target_tensor, prediction_tensor):
    """Computes weird CRPS between predicted and actual TC centers.

    :param target_tensor: See doc for `mean_squared_distance_kilometres2`.
    :param prediction_tensor: Same.
    :return: weird_crps_km2: Weird CRPS.
    """

    mean_prediction_tensor = K.mean(prediction_tensor, axis=-1)
    mean_row_errors_km = (
        target_tensor[:, 2] *
        K.abs(mean_prediction_tensor[:, 0] - target_tensor[:, 0])
    )
    mean_column_errors_km = (
        target_tensor[:, 2] *
        K.abs(mean_prediction_tensor[:, 1] - target_tensor[:, 1])
    )
    mean_distance_errors_km2 = (
        mean_row_errors_km ** 2 + mean_column_errors_km ** 2
    )

    prediction_diff_tensor_rowcol = K.abs(
        K.expand_dims(prediction_tensor, axis=-1) -
        K.expand_dims(prediction_tensor, axis=-2)
    )
    prediction_diff_tensor_rowcol_squared = (
        prediction_diff_tensor_rowcol[:, 0, ...] ** 2
        + prediction_diff_tensor_rowcol[:, 1, ...] ** 2
    )

    grid_spacing_tensor_km = K.expand_dims(target_tensor[:, 2], axis=-1)
    grid_spacing_tensor_km = K.expand_dims(grid_spacing_tensor_km, axis=-1)
    prediction_diff_tensor_km2 = (
        (grid_spacing_tensor_km ** 2) * prediction_diff_tensor_rowcol_squared
    )
    mean_prediction_diffs_km2 = K.mean(
        prediction_diff_tensor_km2, axis=(-2, -1)
    )

    return K.mean(
        mean_distance_errors_km2 - 0.5 * mean_prediction_diffs_km2
    )


def discretized_weird_crps_kilometres2(target_tensor, prediction_tensor):
    """Computes discretized weird CRPS between predicted and actual TC centers.

    :param target_tensor: See doc for `mean_squared_distance_kilometres2`.
    :param prediction_tensor: Same.
    :return: discretized_weird_crps_km2: Discretized weird CRPS.
    """

    mean_prediction_tensor = K.mean(prediction_tensor, axis=-1)
    mean_row_errors_km = (
        target_tensor[:, 2] *
        K.abs(mean_prediction_tensor[:, 0] - target_tensor[:, 0])
    )
    mean_column_errors_km = (
        target_tensor[:, 2] *
        K.abs(mean_prediction_tensor[:, 1] - target_tensor[:, 1])
    )

    mean_row_errors_km, mean_column_errors_km = discretize_distance_errors(
        row_distances_km=mean_row_errors_km,
        column_distances_km=mean_column_errors_km,
        true_center_latitudes_deg_n=target_tensor[:, 3]
    )
    mean_distance_errors_km2 = (
        mean_row_errors_km ** 2 + mean_column_errors_km ** 2
    )

    prediction_diff_tensor_rowcol = K.abs(
        K.expand_dims(prediction_tensor, axis=-1) -
        K.expand_dims(prediction_tensor, axis=-2)
    )
    prediction_diff_tensor_rowcol_squared = (
        prediction_diff_tensor_rowcol[:, 0, ...] ** 2
        + prediction_diff_tensor_rowcol[:, 1, ...] ** 2
    )

    grid_spacing_tensor_km = K.expand_dims(target_tensor[:, 2], axis=-1)
    grid_spacing_tensor_km = K.expand_dims(grid_spacing_tensor_km, axis=-1)
    prediction_diff_tensor_km2 = (
        (grid_spacing_tensor_km ** 2) * prediction_diff_tensor_rowcol_squared
    )
    mean_prediction_diffs_km2 = K.mean(
        prediction_diff_tensor_km2, axis=(-2, -1)
    )

    return K.mean(
        mean_distance_errors_km2 - 0.5 * mean_prediction_diffs_km2
    )


def coord_avg_crps_kilometres(target_tensor, prediction_tensor):
    """Computes coord-averaged CRPS between predicted and actual TC centers.

    :param target_tensor: See doc for `mean_squared_distance_kilometres2`.
    :param prediction_tensor: Same.
    :return: coord_averaged_crps_km: Coord-averaged CRPS.
    """

    mean_prediction_tensor = K.mean(prediction_tensor, axis=-1)

    mean_row_errors_km = (
        target_tensor[:, 2] *
        K.abs(mean_prediction_tensor[:, 0] - target_tensor[:, 0])
    )
    row_prediction_diff_tensor = K.abs(
        K.expand_dims(prediction_tensor[:, 0, :], axis=-1) -
        K.expand_dims(prediction_tensor[:, 0, :], axis=-2)
    )

    grid_spacing_tensor_km = K.expand_dims(target_tensor[:, 2], axis=-1)
    grid_spacing_tensor_km = K.expand_dims(grid_spacing_tensor_km, axis=-1)
    mean_row_prediction_diffs_km = K.mean(
        grid_spacing_tensor_km * row_prediction_diff_tensor, axis=(-2, -1)
    )

    row_crps_kilometres = K.mean(
        mean_row_errors_km - 0.5 * mean_row_prediction_diffs_km
    )

    mean_column_errors_km = (
        target_tensor[:, 2] *
        K.abs(mean_prediction_tensor[:, 1] - target_tensor[:, 1])
    )
    column_prediction_diff_tensor = K.abs(
        K.expand_dims(prediction_tensor[:, 1, :], axis=-1) -
        K.expand_dims(prediction_tensor[:, 1, :], axis=-2)
    )
    mean_column_prediction_diffs_km = K.mean(
        grid_spacing_tensor_km * column_prediction_diff_tensor, axis=(-2, -1)
    )

    column_crps_kilometres = K.mean(
        mean_column_errors_km -
        0.5 * mean_column_prediction_diffs_km
    )

    return (row_crps_kilometres + column_crps_kilometres) / 2


def discretized_coord_avg_crps_kilometres(target_tensor, prediction_tensor):
    """Computes discretized coord-avg CRPS btw predicted and actual TC centers.

    :param target_tensor: See doc for `mean_squared_distance_kilometres2`.
    :param prediction_tensor: Same.
    :return: discretized_coord_averaged_crps_km: Discretized coord-averaged
        CRPS.
    """

    mean_prediction_tensor = K.mean(prediction_tensor, axis=-1)

    mean_row_errors_km = (
        target_tensor[:, 2] *
        K.abs(mean_prediction_tensor[:, 0] - target_tensor[:, 0])
    )
    mean_column_errors_km = (
        target_tensor[:, 2] *
        K.abs(mean_prediction_tensor[:, 1] - target_tensor[:, 1])
    )
    mean_row_errors_km, mean_column_errors_km = discretize_distance_errors(
        row_distances_km=mean_row_errors_km,
        column_distances_km=mean_column_errors_km,
        true_center_latitudes_deg_n=target_tensor[:, 3]
    )

    row_prediction_diff_tensor = K.abs(
        K.expand_dims(prediction_tensor[:, 0, :], axis=-1) -
        K.expand_dims(prediction_tensor[:, 0, :], axis=-2)
    )
    grid_spacing_tensor_km = K.expand_dims(target_tensor[:, 2], axis=-1)
    grid_spacing_tensor_km = K.expand_dims(grid_spacing_tensor_km, axis=-1)
    mean_row_prediction_diffs_km = K.mean(
        grid_spacing_tensor_km * row_prediction_diff_tensor, axis=(-2, -1)
    )
    row_crps_kilometres = K.mean(
        mean_row_errors_km - 0.5 * mean_row_prediction_diffs_km
    )

    column_prediction_diff_tensor = K.abs(
        K.expand_dims(prediction_tensor[:, 1, :], axis=-1) -
        K.expand_dims(prediction_tensor[:, 1, :], axis=-2)
    )
    mean_column_prediction_diffs_km = K.mean(
        grid_spacing_tensor_km * column_prediction_diff_tensor, axis=(-2, -1)
    )
    column_crps_kilometres = K.mean(
        mean_column_errors_km -
        0.5 * mean_column_prediction_diffs_km
    )

    return (row_crps_kilometres + column_crps_kilometres) / 2
