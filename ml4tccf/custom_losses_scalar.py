"""Custom loss functions for scalar predictions (x- and y-coord)."""

import os
import sys
import numpy
import tensorflow
from tensorflow.keras import backend as K

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking

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
    prediction_diff_tensor_rowcol_squared = (
        prediction_diff_tensor_rowcol[:, 0, ...] ** 2
        + prediction_diff_tensor_rowcol[:, 1, ...] ** 2
    )

    grid_spacing_tensor_km = K.expand_dims(target_tensor[:, 2], axis=-1)
    grid_spacing_tensor_km = K.expand_dims(grid_spacing_tensor_km, axis=-1)
    prediction_diff_tensor_km2 = (
        (grid_spacing_tensor_km ** 2) * prediction_diff_tensor_rowcol_squared
    )
    mean_prediction_diff_by_example_km2 = K.mean(
        prediction_diff_tensor_km2, axis=(-2, -1)
    )

    return K.mean(
        mean_dist_error_by_example_km -
        0.5 * mean_prediction_diff_by_example_km2
    )


def discretized_weird_crps_kilometres2(target_tensor, prediction_tensor):
    """Computes discretized weird CRPS between predicted and actual TC centers.

    :param target_tensor: See doc for `mean_squared_distance_kilometres2`.
    :param prediction_tensor: Same.
    :return: discretized_weird_crps_km2: Discretized weird CRPS.
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
        discretize_distance_errors(
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
    prediction_diff_tensor_rowcol_squared = (
        prediction_diff_tensor_rowcol[:, 0, ...] ** 2
        + prediction_diff_tensor_rowcol[:, 1, ...] ** 2
    )

    grid_spacing_tensor_km = K.expand_dims(target_tensor[:, 2], axis=-1)
    grid_spacing_tensor_km = K.expand_dims(grid_spacing_tensor_km, axis=-1)
    prediction_diff_tensor_km2 = (
        (grid_spacing_tensor_km ** 2) * prediction_diff_tensor_rowcol_squared
    )
    mean_prediction_diff_by_example_km2 = K.mean(
        prediction_diff_tensor_km2, axis=(-2, -1)
    )

    return K.mean(
        mean_dist_error_by_example_km -
        0.5 * mean_prediction_diff_by_example_km2
    )


def coord_avg_crps_kilometres(target_tensor, prediction_tensor):
    """Computes coord-averaged CRPS between predicted and actual TC centers.

    :param target_tensor: See doc for `mean_squared_distance_kilometres2`.
    :param prediction_tensor: Same.
    :return: coord_averaged_crps_km: Coord-averaged CRPS.
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

    row_prediction_diff_tensor = K.abs(
        K.expand_dims(prediction_tensor[:, 0, :], axis=-1) -
        K.expand_dims(prediction_tensor[:, 0, :], axis=-2)
    )

    grid_spacing_tensor_km = K.expand_dims(target_tensor[:, 2], axis=-1)
    grid_spacing_tensor_km = K.expand_dims(grid_spacing_tensor_km, axis=-1)
    row_prediction_diff_by_example_km = K.mean(
        grid_spacing_tensor_km * row_prediction_diff_tensor, axis=(-2, -1)
    )
    row_crps_kilometres = K.mean(
        mean_row_error_by_example_km - 0.5 * row_prediction_diff_by_example_km
    )

    mean_column_error_by_example_km = K.mean(
        K.expand_dims(target_tensor[:, 2], axis=-1) *
        K.abs(
            prediction_tensor[:, 1, :] -
            K.expand_dims(target_tensor[:, 1], axis=-1)
        ),
        axis=-1
    )

    column_prediction_diff_tensor = K.abs(
        K.expand_dims(prediction_tensor[:, 1, :], axis=-1) -
        K.expand_dims(prediction_tensor[:, 1, :], axis=-2)
    )
    column_prediction_diff_by_example_km = K.mean(
        grid_spacing_tensor_km * column_prediction_diff_tensor, axis=(-2, -1)
    )

    column_crps_kilometres = K.mean(
        mean_column_error_by_example_km -
        0.5 * column_prediction_diff_by_example_km
    )

    return (row_crps_kilometres + column_crps_kilometres) / 2


def discretized_coord_avg_crps_kilometres(target_tensor, prediction_tensor):
    """Computes discretized coord-avg CRPS btw predicted and actual TC centers.

    :param target_tensor: See doc for `mean_squared_distance_kilometres2`.
    :param prediction_tensor: Same.
    :return: discretized_coord_averaged_crps_km: Discretized coord-averaged
        CRPS.
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
        discretize_distance_errors(
            row_distances_km=mean_row_error_by_example_km,
            column_distances_km=mean_column_error_by_example_km,
            true_center_latitudes_deg_n=target_tensor[:, 3]
        )
    )

    row_prediction_diff_tensor = K.abs(
        K.expand_dims(prediction_tensor[:, 0, :], axis=-1) -
        K.expand_dims(prediction_tensor[:, 0, :], axis=-2)
    )
    grid_spacing_tensor_km = K.expand_dims(target_tensor[:, 2], axis=-1)
    grid_spacing_tensor_km = K.expand_dims(grid_spacing_tensor_km, axis=-1)
    row_prediction_diff_by_example_km = K.mean(
        grid_spacing_tensor_km * row_prediction_diff_tensor, axis=(-2, -1)
    )
    row_crps_kilometres = K.mean(
        mean_row_error_by_example_km - 0.5 * row_prediction_diff_by_example_km
    )

    column_prediction_diff_tensor = K.abs(
        K.expand_dims(prediction_tensor[:, 1, :], axis=-1) -
        K.expand_dims(prediction_tensor[:, 1, :], axis=-2)
    )
    column_prediction_diff_by_example_km = K.mean(
        grid_spacing_tensor_km * column_prediction_diff_tensor, axis=(-2, -1)
    )
    column_crps_kilometres = K.mean(
        mean_column_error_by_example_km -
        0.5 * column_prediction_diff_by_example_km
    )

    return (row_crps_kilometres + column_crps_kilometres) / 2


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
        relevant_target_tensor = K.expand_dims(target_tensor, axis=-1)
        relevant_prediction_tensor = prediction_tensor
        dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(relevant_prediction_tensor)
        )

        # Turn channel weights into E-by-C tensor.
        channel_weight_tensor = K.cast(
            K.constant(channel_weights), target_tensor.dtype
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
        tensorflow.gather(prediction_tensor, indices=intensity_index, axis=-2),
        0.
    )
    predicted_r34_tensor = K.maximum(
        tensorflow.gather(prediction_tensor, indices=r34_index, axis=-2),
        0.
    )
    predicted_r50_tensor = K.maximum(
        tensorflow.gather(prediction_tensor, indices=r50_index, axis=-2),
        0.
    )
    predicted_r64_tensor = K.maximum(
        tensorflow.gather(prediction_tensor, indices=r64_index, axis=-2),
        0.
    )
    predicted_rmw_tensor = K.maximum(
        tensorflow.gather(prediction_tensor, indices=rmw_index, axis=-2),
        0.
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
        relevant_target_tensor = K.expand_dims(target_tensor, axis=-1)
        relevant_prediction_tensor = prediction_tensor
        dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(relevant_prediction_tensor)
        )

        # Turn channel weights into E-by-C tensor.
        channel_weight_tensor = K.cast(
            K.constant(channel_weights), target_tensor.dtype
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
        relevant_target_tensor = target_tensor
        relevant_prediction_tensor = K.mean(prediction_tensor, axis=-1)
        dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(relevant_prediction_tensor)
        )

        # Turn channel weights into E-by-C tensor.
        channel_weight_tensor = K.cast(
            K.constant(channel_weights), target_tensor.dtype
        )
        channel_weight_tensor = K.expand_dims(channel_weight_tensor, axis=0)

        return K.mean(
            channel_weight_tensor * dual_weight_tensor *
            (relevant_target_tensor - relevant_prediction_tensor) ** 2
        )

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
        relevant_target_tensor = target_tensor
        relevant_prediction_tensor = K.mean(prediction_tensor, axis=-1)
        dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(relevant_prediction_tensor)
        )

        # Turn channel weights into E-by-C tensor.
        channel_weight_tensor = K.cast(
            K.constant(channel_weights), target_tensor.dtype
        )
        channel_weight_tensor = K.expand_dims(channel_weight_tensor, axis=0)

        return K.mean(
            channel_weight_tensor * dual_weight_tensor *
            (relevant_target_tensor - relevant_prediction_tensor) ** 2
        )

    loss.__name__ = function_name
    return loss
