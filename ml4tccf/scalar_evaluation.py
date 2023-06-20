"""Evaluation methods for scalar predictions."""

import os
import sys
import copy
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import histograms
import longitude_conversion as lng_conversion
import file_system_utils
import error_checking
import prediction_io
import scalar_prediction_utils as prediction_utils
import neural_net_utils
import angular_utils

# TODO(thunderhoser): Maybe account for 0.1-deg discretization in error metrics?

# TODO(thunderhoser): This whole module changes xarray variables by directly
# accessing their `.values` attribute, which can fail!  I should make this
# module more robust.

TOLERANCE = 1e-6

DEGREES_TO_RADIANS = numpy.pi / 180
RADIANS_TO_DEGREES = 180. / numpy.pi
KM_TO_METRES = 1000.

X_OFFSET_NAME = 'x_offset_metres'
Y_OFFSET_NAME = 'y_offset_metres'
OFFSET_DISTANCE_NAME = 'offset_distance_metres'
OFFSET_DIRECTION_NAME = 'offset_direction_deg'
TARGET_FIELD_NAMES = [
    Y_OFFSET_NAME, X_OFFSET_NAME, OFFSET_DISTANCE_NAME, OFFSET_DIRECTION_NAME
]

TARGET_FIELD_DIM = 'target_field'
XY_OFFSET_BIN_DIM = 'xy_offset_reliability_bin'
OFFSET_DISTANCE_BIN_DIM = 'offset_distance_reliability_bin'
OFFSET_DIRECTION_BIN_DIM = 'offset_direction_reliability_bin'
BOOTSTRAP_REP_DIM = 'bootstrap_replicate'

CRPS_KEY = 'continous_rank_probability_score'
MEAN_SQUARED_ERROR_KEY = 'mean_squared_error'
MSE_SKILL_SCORE_KEY = 'mse_skill_score'
MEAN_ABSOLUTE_ERROR_KEY = 'mean_absolute_error'
MAE_SKILL_SCORE_KEY = 'mae_skill_score'
BIAS_KEY = 'bias'
CORRELATION_KEY = 'correlation'
KGE_KEY = 'kling_gupta_efficiency'
TARGET_STDEV_KEY = 'target_stdev'
PREDICTION_STDEV_KEY = 'prediction_stdev'

MEAN_DISTANCE_KEY = 'mean_distance'
MEDIAN_DISTANCE_KEY = 'median_distance'
MEAN_SQUARED_DISTANCE_KEY = 'mean_squared_distance'
MEAN_DIST_SKILL_SCORE_KEY = 'mean_distance_skill_score'
MEAN_SQ_DIST_SKILL_SCORE_KEY = 'mean_squared_distance_skill_score'

RELIABILITY_KEY = 'reliability'
RESOLUTION_KEY = 'resolution'

XY_OFFSET_MEAN_PREDICTION_KEY = 'xy_offset_mean_prediction_metres'
XY_OFFSET_MEAN_OBSERVATION_KEY = 'xy_offset_mean_observation_metres'
XY_OFFSET_BIN_CENTER_KEY = 'xy_offset_bin_center_metres'
XY_OFFSET_BIN_COUNT_KEY = 'xy_offset_bin_count'
XY_OFFSET_INV_BIN_CENTER_KEY = 'xy_offset_inv_bin_center_metres'
XY_OFFSET_INV_BIN_COUNT_KEY = 'xy_offset_inv_bin_count'

OFFSET_DIST_MEAN_PREDICTION_KEY = 'offset_distance_mean_prediction_metres'
OFFSET_DIST_MEAN_OBSERVATION_KEY = 'offset_distance_mean_observation_metres'
OFFSET_DIST_BIN_CENTER_KEY = 'offset_distance_bin_center_metres'
OFFSET_DIST_BIN_COUNT_KEY = 'offset_distance_bin_count'
OFFSET_DIST_INV_BIN_CENTER_KEY = 'offset_distance_inv_bin_center_metres'
OFFSET_DIST_INV_BIN_COUNT_KEY = 'offset_distance_inv_bin_count'

OFFSET_DIR_MEAN_PREDICTION_KEY = 'offset_direction_mean_prediction_deg'
OFFSET_DIR_MEAN_OBSERVATION_KEY = 'offset_direction_mean_observation_deg'
OFFSET_DIR_BIN_CENTER_KEY = 'offset_direction_bin_center_deg'
OFFSET_DIR_BIN_COUNT_KEY = 'offset_direction_bin_count'
OFFSET_DIR_INV_BIN_CENTER_KEY = 'offset_direction_inv_bin_center_deg'
OFFSET_DIR_INV_BIN_COUNT_KEY = 'offset_direction_inv_bin_count'

MODEL_FILE_KEY = 'model_file_name'
PREDICTION_FILES_KEY = 'prediction_file_names'
CLIMO_OFFSET_DISTANCE_KEY = 'climo_offset_distance_metres'


def _get_mean_distance(target_offset_matrix, predicted_offset_matrix):
    """Computes mean distance between prediction and target.

    E = number of examples

    :param target_offset_matrix: E-by-2 numpy array of actual offsets.
    :param predicted_offset_matrix: E-by-2 numpy array of predicted offsets.
    :return: mean_distance: Mean distance between prediction and target.
    """

    return numpy.mean(numpy.sqrt(
        (target_offset_matrix[:, 0] - predicted_offset_matrix[:, 0]) ** 2
        + (target_offset_matrix[:, 1] - predicted_offset_matrix[:, 1]) ** 2
    ))


def _get_median_distance(target_offset_matrix, predicted_offset_matrix):
    """Computes median distance between prediction and target.

    E = number of examples

    :param target_offset_matrix: E-by-2 numpy array of actual offsets.
    :param predicted_offset_matrix: E-by-2 numpy array of predicted offsets.
    :return: median_distance: Median distance between prediction and target.
    """

    return numpy.median(numpy.sqrt(
        (target_offset_matrix[:, 0] - predicted_offset_matrix[:, 0]) ** 2
        + (target_offset_matrix[:, 1] - predicted_offset_matrix[:, 1]) ** 2
    ))


def _get_mean_dist_skill_score(target_offset_matrix, predicted_offset_matrix,
                               mean_training_target_offset_matrix):
    """Computes mean-distance skill score.

    :param target_offset_matrix: See doc for `_get_mean_distance`.
    :param predicted_offset_matrix: Same.
    :param mean_training_target_offset_matrix: 1-by-2 numpy array with mean
        actual offsets in training data.
    :return: mean_distance_skill_score: Obvious.
    """

    mean_distance_actual = _get_mean_distance(
        target_offset_matrix=target_offset_matrix,
        predicted_offset_matrix=predicted_offset_matrix
    )
    mean_distance_climo = _get_mean_distance(
        target_offset_matrix=target_offset_matrix,
        predicted_offset_matrix=mean_training_target_offset_matrix
    )

    return (mean_distance_climo - mean_distance_actual) / mean_distance_climo


def _get_mean_squared_distance(target_offset_matrix, predicted_offset_matrix):
    """Computes mean squared distance between prediction and target.

    E = number of examples

    :param target_offset_matrix: E-by-2 numpy array of actual offsets.
    :param predicted_offset_matrix: E-by-2 numpy array of predicted offsets.
    :return: mean_squared_distance: Mean squared distance between prediction and
        target.
    """

    return numpy.mean(
        (target_offset_matrix[:, 0] - predicted_offset_matrix[:, 0]) ** 2
        + (target_offset_matrix[:, 1] - predicted_offset_matrix[:, 1]) ** 2
    )


def _get_mean_sq_dist_skill_score(target_offset_matrix, predicted_offset_matrix,
                                  mean_training_target_offset_matrix):
    """Computes mean-squared-distance skill score.

    :param target_offset_matrix: See doc for `_get_mean_squared_distance`.
    :param predicted_offset_matrix: Same.
    :param mean_training_target_offset_matrix: 1-by-2 numpy array with mean
        actual offsets in training data.
    :return: mean_squared_distance_skill_score: Obvious.
    """

    mean_sq_dist_actual = _get_mean_squared_distance(
        target_offset_matrix=target_offset_matrix,
        predicted_offset_matrix=predicted_offset_matrix
    )
    mean_sq_dist_climo = _get_mean_squared_distance(
        target_offset_matrix=target_offset_matrix,
        predicted_offset_matrix=mean_training_target_offset_matrix
    )

    return (mean_sq_dist_climo - mean_sq_dist_actual) / mean_sq_dist_climo


def _get_crps_one_variable(target_values, prediction_matrix, is_var_direction):
    """Computes continuous rank prob score (CRPS) for one target variable.

    E = number of examples
    S = ensemble size

    :param target_values: length-E numpy array of target (actual) values.
    :param prediction_matrix: E-by-S numpy array of predicted values.
    :param is_var_direction: Boolean flag.  If True (False), the target variable
        at hand is direction (anything else).
    :return: crps: Self-explanatory.
    """

    if is_var_direction:
        ensemble_size = prediction_matrix.shape[1]
        target_matrix = numpy.repeat(
            numpy.expand_dims(target_values, axis=-1),
            axis=-1, repeats=ensemble_size
        )
        mean_abs_error_by_example = numpy.nanmean(
            numpy.absolute(get_angular_diffs(target_matrix, prediction_matrix)),
            axis=-1
        )

        first_prediction_matrix = numpy.repeat(
            numpy.expand_dims(prediction_matrix, axis=-1),
            axis=-1, repeats=ensemble_size
        )
        second_prediction_matrix = numpy.repeat(
            numpy.expand_dims(prediction_matrix, axis=-2),
            axis=-2, repeats=ensemble_size
        )
        prediction_diff_matrix = numpy.absolute(
            get_angular_diffs(first_prediction_matrix, second_prediction_matrix)
        )
        mean_pairwise_diff_by_example = numpy.nanmean(
            prediction_diff_matrix, axis=(-2, -1)
        )

        return numpy.nanmean(
            mean_abs_error_by_example - 0.5 * mean_pairwise_diff_by_example
        )

    mean_abs_error_by_example = numpy.mean(
        numpy.absolute(
            prediction_matrix -
            numpy.expand_dims(target_values, axis=-1)
        ),
        axis=-1
    )

    prediction_diff_matrix = numpy.absolute(
        numpy.expand_dims(prediction_matrix, axis=-1) -
        numpy.expand_dims(prediction_matrix, axis=-2)
    )
    mean_pairwise_diff_by_example = numpy.mean(
        prediction_diff_matrix, axis=(-2, -1)
    )

    return numpy.mean(
        mean_abs_error_by_example - 0.5 * mean_pairwise_diff_by_example
    )


def _get_crps_euclidean(target_matrix, prediction_matrix):
    """Computes Euclidean CRPS.

    E = number of examples
    S = ensemble size

    :param target_matrix: E-by-2 numpy array of target values.
        target_matrix[:, 0] contains y-offsets, and target_matrix[:, 1] contains
        x-offsets.
    :param prediction_matrix: E-by-2-by-S numpy array of predicted values.
        prediction_matrix[:, 0, :] contains y-offsets, and
        prediction_matrix[:, 1, :] contains x-offsets.
    """

    coord_diff_matrix = (
        prediction_matrix -
        numpy.expand_dims(target_matrix, axis=-1)
    )
    distance_matrix = numpy.sqrt(
        coord_diff_matrix[:, 0, :] ** 2 +
        coord_diff_matrix[:, 1, :] ** 2
    )
    mean_dist_error_by_example = numpy.mean(distance_matrix, axis=-1)

    prediction_coord_diff_matrix = numpy.absolute(
        numpy.expand_dims(prediction_matrix, axis=-1) -
        numpy.expand_dims(prediction_matrix, axis=-2)
    )
    pairwise_distance_matrix = numpy.sqrt(
        prediction_coord_diff_matrix[:, 0, ...] ** 2 +
        prediction_coord_diff_matrix[:, 1, ...] ** 2
    )
    mean_pairwise_dist_by_example = numpy.mean(
        pairwise_distance_matrix, axis=(-2, -1)
    )

    return numpy.mean(
        mean_dist_error_by_example - 0.5 * mean_pairwise_dist_by_example
    )


def _get_mse_one_variable(target_values, predicted_values, is_var_direction):
    """Computes mean squared error (MSE) for one target variable.

    E = number of examples

    :param target_values: length-E numpy array of target (actual) values.
    :param predicted_values: length-E numpy array of predicted values.
    :param is_var_direction: Boolean flag.  If True (False), the target variable
        at hand is direction (anything else).
    :return: mean_squared_error: Self-explanatory.
    """

    if is_var_direction:
        return numpy.nanmean(
            get_angular_diffs(target_values, predicted_values) ** 2
        )

    return numpy.mean((target_values - predicted_values) ** 2)


def _get_mse_ss_one_variable(
        target_values, predicted_values, mean_training_target_value,
        is_var_direction):
    """Computes MSE skill score for one target variable.

    :param target_values: See doc for `_get_mse_one_variable`.
    :param predicted_values: Same.
    :param mean_training_target_value: Mean target value over all training
        examples.
    :param is_var_direction: See doc for `_get_mse_one_variable`.
    :return: mse_skill_score: Self-explanatory.
    """

    assert not is_var_direction

    mse_actual = _get_mse_one_variable(
        target_values=target_values, predicted_values=predicted_values,
        is_var_direction=False
    )
    mse_climo = _get_mse_one_variable(
        target_values=target_values,
        predicted_values=mean_training_target_value,
        is_var_direction=False
    )

    return (mse_climo - mse_actual) / mse_climo


def _get_mae_one_variable(target_values, predicted_values, is_var_direction):
    """Computes mean absolute error (MAE) for one target variable.

    :param target_values: See doc for `_get_mse_one_variable`.
    :param predicted_values: Same.
    :param is_var_direction: Same.
    :return: mean_absolute_error: Self-explanatory.
    """

    if is_var_direction:
        return numpy.nanmean(numpy.absolute(
            get_angular_diffs(target_values, predicted_values)
        ))

    return numpy.mean(numpy.abs(target_values - predicted_values))


def _get_mae_ss_one_variable(
        target_values, predicted_values, mean_training_target_value,
        is_var_direction):
    """Computes MAE skill score for one target variable.

    :param target_values: See doc for `_get_mse_one_variable`.
    :param predicted_values: Same.
    :param mean_training_target_value: Mean target value over all training
        examples.
    :param is_var_direction: See doc for `_get_mse_one_variable`.
    :return: mae_skill_score: Self-explanatory.
    """

    assert not is_var_direction

    mae_actual = _get_mae_one_variable(
        target_values=target_values, predicted_values=predicted_values,
        is_var_direction=False
    )
    mae_climo = _get_mae_one_variable(
        target_values=target_values,
        predicted_values=mean_training_target_value,
        is_var_direction=False
    )

    return (mae_climo - mae_actual) / mae_climo


def _get_bias_one_variable(target_values, predicted_values, is_var_direction):
    """Computes bias (mean signed error) for one target variable.

    :param target_values: See doc for `_get_mse_one_variable`.
    :param predicted_values: Same.
    :param is_var_direction: Same.
    :return: bias: Self-explanatory.
    """

    if is_var_direction:
        return numpy.nanmean(get_angular_diffs(
            target_angles_deg=target_values,
            predicted_angles_deg=predicted_values
        ))

    return numpy.mean(predicted_values - target_values)


def _get_correlation_one_variable(target_values, predicted_values,
                                  is_var_direction):
    """Computes Pearson correlation for one target variable.

    :param target_values: See doc for `_get_mse_one_variable`.
    :param predicted_values: Same.
    :param is_var_direction: Same.
    :return: correlation: Self-explanatory.
    """

    if is_var_direction:
        real_indices = numpy.where(numpy.invert(numpy.logical_or(
            numpy.isnan(target_values),
            numpy.isnan(predicted_values)
        )))[0]

        return angular_utils.corrcoef(
            target_values[real_indices], predicted_values[real_indices],
            deg=True, test=False
        )

    numerator = numpy.sum(
        (target_values - numpy.mean(target_values)) *
        (predicted_values - numpy.mean(predicted_values))
    )
    sum_squared_target_diffs = numpy.sum(
        (target_values - numpy.mean(target_values)) ** 2
    )
    sum_squared_prediction_diffs = numpy.sum(
        (predicted_values - numpy.mean(predicted_values)) ** 2
    )

    correlation = (
        numerator /
        numpy.sqrt(sum_squared_target_diffs * sum_squared_prediction_diffs)
    )

    return correlation


def _get_kge_one_variable(target_values, predicted_values, is_var_direction):
    """Computes KGE (Kling-Gupta efficiency) for one target variable.

    :param target_values: See doc for `_get_mse_one_variable`.
    :param predicted_values: Same.
    :param is_var_direction: Same.
    :return: kge: Self-explanatory.
    """

    correlation = _get_correlation_one_variable(
        target_values=target_values, predicted_values=predicted_values,
        is_var_direction=is_var_direction
    )

    if is_var_direction:
        real_indices = numpy.where(numpy.invert(numpy.logical_or(
            numpy.isnan(target_values),
            numpy.isnan(predicted_values)
        )))[0]

        mean_target_value = angular_utils.mean(
            target_values[real_indices], deg=True
        )
        mean_predicted_value = angular_utils.mean(
            predicted_values[real_indices], deg=True
        )
        stdev_target_value = angular_utils.std(
            target_values[real_indices], deg=True
        )
        stdev_predicted_value = angular_utils.std(
            predicted_values[real_indices], deg=True
        )
    else:
        mean_target_value = numpy.mean(target_values)
        mean_predicted_value = numpy.mean(predicted_values)
        stdev_target_value = numpy.std(target_values, ddof=1)
        stdev_predicted_value = numpy.std(predicted_values, ddof=1)

    variance_bias = (
        (stdev_predicted_value / mean_predicted_value) *
        (stdev_target_value / mean_target_value) ** -1
    )
    mean_bias = mean_predicted_value / mean_target_value

    kge = 1. - numpy.sqrt(
        (correlation - 1.) ** 2 +
        (variance_bias - 1.) ** 2 +
        (mean_bias - 1.) ** 2
    )

    return kge


def _get_reliability_curve_one_variable(
        target_values, predicted_values, is_var_direction,
        num_bins, min_bin_edge, max_bin_edge, invert=False):
    """Computes reliability curve for one target variable.

    B = number of bins

    :param target_values: See doc for `_get_mse_one_variable`.
    :param predicted_values: Same.
    :param is_var_direction: Same.
    :param num_bins: Number of bins (points in curve).
    :param min_bin_edge: Value at lower edge of first bin.
    :param max_bin_edge: Value at upper edge of last bin.
    :param invert: Boolean flag.  If True, will return inverted reliability
        curve, which bins by target value and relates target value to
        conditional mean prediction.  If False, will return normal reliability
        curve, which bins by predicted value and relates predicted value to
        conditional mean observation (target).
    :return: mean_predictions: length-B numpy array of x-coordinates.
    :return: mean_observations: length-B numpy array of y-coordinates.
    :return: example_counts: length-B numpy array with num examples in each bin.
    """

    if is_var_direction:
        min_bin_edge = 0.
        max_bin_edge = 360.

        real_indices = numpy.where(numpy.invert(numpy.logical_or(
            numpy.isnan(target_values),
            numpy.isnan(predicted_values)
        )))[0]

        bin_index_by_example = numpy.full(len(target_values), -9999, dtype=int)

        bin_index_by_example[real_indices] = histograms.create_histogram(
            input_values=(
                target_values[real_indices] if invert
                else predicted_values[real_indices]
            ),
            num_bins=num_bins, min_value=min_bin_edge, max_value=max_bin_edge
        )[0]
    else:
        bin_index_by_example = histograms.create_histogram(
            input_values=target_values if invert else predicted_values,
            num_bins=num_bins, min_value=min_bin_edge, max_value=max_bin_edge
        )[0]

    mean_predictions = numpy.full(num_bins, numpy.nan)
    mean_observations = numpy.full(num_bins, numpy.nan)
    example_counts = numpy.full(num_bins, -1, dtype=int)

    for i in range(num_bins):
        these_example_indices = numpy.where(bin_index_by_example == i)[0]
        example_counts[i] = len(these_example_indices)

        if is_var_direction:
            mean_predictions[i] = angular_utils.mean(
                predicted_values[these_example_indices]
            )
            mean_observations[i] = angular_utils.mean(
                target_values[these_example_indices]
            )
        else:
            mean_predictions[i] = numpy.mean(
                predicted_values[these_example_indices]
            )
            mean_observations[i] = numpy.mean(
                target_values[these_example_indices]
            )

    return mean_predictions, mean_observations, example_counts


def _get_scores_one_replicate(
        result_table_xarray, full_prediction_matrix, full_target_matrix,
        replicate_index, example_indices_in_replicate,
        num_xy_offset_bins, min_xy_offset_metres, max_xy_offset_metres,
        min_xy_offset_percentile, max_xy_offset_percentile,
        num_offset_distance_bins,
        min_offset_distance_metres, max_offset_distance_metres,
        min_offset_distance_percentile, max_offset_distance_percentile,
        num_offset_direction_bins):
    """Computes scores for one bootstrap replicate.

    E = number of examples

    :param result_table_xarray: See doc for `get_scores_all_variables`.
    :param full_prediction_matrix: E-by-4 numpy array of predictions.
        full_prediction_matrix[:, 0] contains y-offsets in metres;
        full_prediction_matrix[:, 1] contains x-offsets in metres;
        full_prediction_matrix[:, 2] contains offset magnitudes in metres;
        full_prediction_matrix[:, 3] contains offset angles in degrees.
    :param full_target_matrix: Same as `full_prediction_matrix` but for labels.
    :param replicate_index: Index of current bootstrap replicate.
    :param example_indices_in_replicate: 1-D numpy array with indices of
        examples in this bootstrap replicate.
    :param num_xy_offset_bins: See doc for `get_scores_all_variables`.
    :param min_xy_offset_metres: Same.
    :param max_xy_offset_metres: Same.
    :param min_xy_offset_percentile: Same.
    :param max_xy_offset_percentile: Same.
    :param num_offset_distance_bins: Same.
    :param min_offset_distance_metres: Same.
    :param max_offset_distance_metres: Same.
    :param min_offset_distance_percentile: Same.
    :param max_offset_distance_percentile: Same.
    :param num_offset_direction_bins: Same.
    :return: result_table_xarray: Same as input but with values filled for [i]th
        bootstrap replicate, where i = `replicate_index`.
    """

    bootstrapped_prediction_matrix = (
        full_prediction_matrix[example_indices_in_replicate, ...] + 0.
    )
    bootstrapped_target_matrix = (
        full_target_matrix[example_indices_in_replicate, ...] + 0.
    )

    t = result_table_xarray
    i = replicate_index + 0

    num_examples = bootstrapped_prediction_matrix.shape[0]
    num_target_vars = bootstrapped_prediction_matrix.shape[1]

    for j in range(num_target_vars):
        if TARGET_FIELD_NAMES[j] == OFFSET_DIRECTION_NAME:
            real_indices = numpy.where(numpy.invert(numpy.logical_or(
                numpy.isnan(bootstrapped_target_matrix[:, j]),
                numpy.isnan(bootstrapped_prediction_matrix[:, j])
            )))[0]

            t[TARGET_STDEV_KEY].values[j, i] = angular_utils.std(
                bootstrapped_target_matrix[real_indices, j], deg=True
            )
            t[PREDICTION_STDEV_KEY].values[j, i] = angular_utils.std(
                bootstrapped_prediction_matrix[real_indices, j], deg=True
            )
        else:
            t[TARGET_STDEV_KEY].values[j, i] = numpy.std(
                bootstrapped_target_matrix[:, j], ddof=1
            )
            t[PREDICTION_STDEV_KEY].values[j, i] = numpy.std(
                bootstrapped_prediction_matrix[:, j], ddof=1
            )

        t[MEAN_SQUARED_ERROR_KEY].values[j, i] = _get_mse_one_variable(
            target_values=bootstrapped_target_matrix[:, j],
            predicted_values=bootstrapped_prediction_matrix[:, j],
            is_var_direction=TARGET_FIELD_NAMES[j] == OFFSET_DIRECTION_NAME
        )
        t[MEAN_ABSOLUTE_ERROR_KEY].values[j, i] = _get_mae_one_variable(
            target_values=bootstrapped_target_matrix[:, j],
            predicted_values=bootstrapped_prediction_matrix[:, j],
            is_var_direction=TARGET_FIELD_NAMES[j] == OFFSET_DIRECTION_NAME
        )

        if TARGET_FIELD_NAMES[j] != OFFSET_DIRECTION_NAME:
            t[MSE_SKILL_SCORE_KEY].values[j, i] = _get_mse_ss_one_variable(
                target_values=bootstrapped_target_matrix[:, j],
                predicted_values=bootstrapped_prediction_matrix[:, j],
                mean_training_target_value=(
                    t.attrs[CLIMO_OFFSET_DISTANCE_KEY] if
                    TARGET_FIELD_NAMES[j] == OFFSET_DISTANCE_NAME else 0.
                ),
                is_var_direction=False
            )
            t[MAE_SKILL_SCORE_KEY].values[j, i] = _get_mae_ss_one_variable(
                target_values=bootstrapped_target_matrix[:, j],
                predicted_values=bootstrapped_prediction_matrix[:, j],
                mean_training_target_value=(
                    t.attrs[CLIMO_OFFSET_DISTANCE_KEY] if
                    TARGET_FIELD_NAMES[j] == OFFSET_DISTANCE_NAME else 0.
                ),
                is_var_direction=False
            )

        t[BIAS_KEY].values[j, i] = _get_bias_one_variable(
            target_values=bootstrapped_target_matrix[:, j],
            predicted_values=bootstrapped_prediction_matrix[:, j],
            is_var_direction=TARGET_FIELD_NAMES[j] == OFFSET_DIRECTION_NAME
        )
        t[CORRELATION_KEY].values[j, i] = _get_correlation_one_variable(
            target_values=bootstrapped_target_matrix[:, j],
            predicted_values=bootstrapped_prediction_matrix[:, j],
            is_var_direction=TARGET_FIELD_NAMES[j] == OFFSET_DIRECTION_NAME
        )
        t[KGE_KEY].values[j, i] = _get_kge_one_variable(
            target_values=bootstrapped_target_matrix[:, j],
            predicted_values=bootstrapped_prediction_matrix[:, j],
            is_var_direction=TARGET_FIELD_NAMES[j] == OFFSET_DIRECTION_NAME
        )

    xy_indices = numpy.array([
        TARGET_FIELD_NAMES.index(X_OFFSET_NAME),
        TARGET_FIELD_NAMES.index(Y_OFFSET_NAME)
    ], dtype=int)

    t[MEAN_DISTANCE_KEY].values[i] = _get_mean_distance(
        target_offset_matrix=bootstrapped_target_matrix[:, xy_indices],
        predicted_offset_matrix=bootstrapped_prediction_matrix[:, xy_indices]
    )
    t[MEDIAN_DISTANCE_KEY].values[i] = _get_median_distance(
        target_offset_matrix=bootstrapped_target_matrix[:, xy_indices],
        predicted_offset_matrix=bootstrapped_prediction_matrix[:, xy_indices]
    )
    t[MEAN_DIST_SKILL_SCORE_KEY].values[i] = _get_mean_dist_skill_score(
        target_offset_matrix=bootstrapped_target_matrix[:, xy_indices],
        predicted_offset_matrix=bootstrapped_prediction_matrix[:, xy_indices],
        mean_training_target_offset_matrix=numpy.full((1, 2), 0.)
    )
    t[MEAN_SQUARED_DISTANCE_KEY].values[i] = _get_mean_squared_distance(
        target_offset_matrix=bootstrapped_target_matrix[:, xy_indices],
        predicted_offset_matrix=bootstrapped_prediction_matrix[:, xy_indices]
    )
    t[MEAN_SQ_DIST_SKILL_SCORE_KEY].values[i] = _get_mean_sq_dist_skill_score(
        target_offset_matrix=bootstrapped_target_matrix[:, xy_indices],
        predicted_offset_matrix=bootstrapped_prediction_matrix[:, xy_indices],
        mean_training_target_offset_matrix=numpy.full((1, 2), 0.)
    )

    for j in xy_indices:
        if num_examples == 0:
            min_bin_edge = -1.
            max_bin_edge = 1.
        elif min_xy_offset_metres is not None:
            min_bin_edge = min_xy_offset_metres + 0.
            max_bin_edge = max_xy_offset_metres + 0.
        else:
            min_bin_edge = numpy.percentile(
                full_prediction_matrix[:, j], min_xy_offset_percentile
            )
            max_bin_edge = numpy.percentile(
                full_prediction_matrix[:, j], max_xy_offset_percentile
            )

        (
            t[XY_OFFSET_MEAN_PREDICTION_KEY].values[j, :, i],
            t[XY_OFFSET_MEAN_OBSERVATION_KEY].values[j, :, i],
            these_counts
        ) = _get_reliability_curve_one_variable(
            target_values=bootstrapped_target_matrix[:, j],
            predicted_values=bootstrapped_prediction_matrix[:, j],
            is_var_direction=False, num_bins=num_xy_offset_bins,
            min_bin_edge=min_bin_edge, max_bin_edge=max_bin_edge, invert=False
        )

        t[RELIABILITY_KEY].values[j, i] = _get_reliability(
            binned_mean_predictions=
            t[XY_OFFSET_MEAN_PREDICTION_KEY].values[j, :, i],
            binned_mean_observations=
            t[XY_OFFSET_MEAN_OBSERVATION_KEY].values[j, :, i],
            binned_example_counts=these_counts,
            is_var_direction=False
        )

        t[RESOLUTION_KEY].values[j, i] = _get_resolution(
            binned_mean_observations=
            t[XY_OFFSET_MEAN_OBSERVATION_KEY].values[j, :, i],
            binned_example_counts=these_counts,
            is_var_direction=False
        )

        if i == 0:
            (
                t[XY_OFFSET_BIN_CENTER_KEY].values[j, :],
                _,
                t[XY_OFFSET_BIN_COUNT_KEY].values[j, :]
            ) = _get_reliability_curve_one_variable(
                target_values=full_target_matrix[:, j],
                predicted_values=full_prediction_matrix[:, j],
                is_var_direction=False, num_bins=num_xy_offset_bins,
                min_bin_edge=min_bin_edge, max_bin_edge=max_bin_edge,
                invert=False
            )

            (
                _,
                t[XY_OFFSET_INV_BIN_CENTER_KEY].values[j, :],
                t[XY_OFFSET_INV_BIN_COUNT_KEY].values[j, :]
            ) = _get_reliability_curve_one_variable(
                target_values=full_target_matrix[:, j],
                predicted_values=full_prediction_matrix[:, j],
                is_var_direction=False, num_bins=num_xy_offset_bins,
                min_bin_edge=min_bin_edge, max_bin_edge=max_bin_edge,
                invert=True
            )

    distance_indices = numpy.array(
        [TARGET_FIELD_NAMES.index(OFFSET_DISTANCE_NAME)], dtype=int
    )

    for j in distance_indices:
        if num_examples == 0:
            min_bin_edge = 0.
            max_bin_edge = 1.
        elif min_offset_distance_metres is not None:
            min_bin_edge = min_offset_distance_metres + 0.
            max_bin_edge = max_offset_distance_metres + 0.
        else:
            min_bin_edge = numpy.percentile(
                full_prediction_matrix[:, j], min_offset_distance_percentile
            )
            max_bin_edge = numpy.percentile(
                full_prediction_matrix[:, j], max_offset_distance_percentile
            )

        (
            t[OFFSET_DIST_MEAN_PREDICTION_KEY].values[:, i],
            t[OFFSET_DIST_MEAN_OBSERVATION_KEY].values[:, i],
            these_counts
        ) = _get_reliability_curve_one_variable(
            target_values=bootstrapped_target_matrix[:, j],
            predicted_values=bootstrapped_prediction_matrix[:, j],
            is_var_direction=False, num_bins=num_offset_distance_bins,
            min_bin_edge=min_bin_edge, max_bin_edge=max_bin_edge, invert=False
        )

        t[RELIABILITY_KEY].values[j, i] = _get_reliability(
            binned_mean_predictions=
            t[OFFSET_DIST_MEAN_PREDICTION_KEY].values[:, i],
            binned_mean_observations=
            t[OFFSET_DIST_MEAN_OBSERVATION_KEY].values[:, i],
            binned_example_counts=these_counts,
            is_var_direction=False
        )

        t[RESOLUTION_KEY].values[j, i] = _get_resolution(
            binned_mean_observations=
            t[OFFSET_DIST_MEAN_OBSERVATION_KEY].values[:, i],
            binned_example_counts=these_counts,
            is_var_direction=False
        )

        if i == 0:
            (
                t[OFFSET_DIST_BIN_CENTER_KEY].values[:],
                _,
                t[OFFSET_DIST_BIN_COUNT_KEY].values[:]
            ) = _get_reliability_curve_one_variable(
                target_values=full_target_matrix[:, j],
                predicted_values=full_prediction_matrix[:, j],
                is_var_direction=False, num_bins=num_offset_distance_bins,
                min_bin_edge=min_bin_edge, max_bin_edge=max_bin_edge,
                invert=False
            )

            (
                _,
                t[OFFSET_DIST_INV_BIN_CENTER_KEY].values[:],
                t[OFFSET_DIST_INV_BIN_COUNT_KEY].values[:]
            ) = _get_reliability_curve_one_variable(
                target_values=full_target_matrix[:, j],
                predicted_values=full_prediction_matrix[:, j],
                is_var_direction=False, num_bins=num_offset_distance_bins,
                min_bin_edge=min_bin_edge, max_bin_edge=max_bin_edge,
                invert=True
            )

    direction_indices = numpy.array(
        [TARGET_FIELD_NAMES.index(OFFSET_DIRECTION_NAME)], dtype=int
    )

    for j in direction_indices:
        (
            t[OFFSET_DIR_MEAN_PREDICTION_KEY].values[:, i],
            t[OFFSET_DIR_MEAN_OBSERVATION_KEY].values[:, i],
            these_counts
        ) = _get_reliability_curve_one_variable(
            target_values=bootstrapped_target_matrix[:, j],
            predicted_values=bootstrapped_prediction_matrix[:, j],
            is_var_direction=True, num_bins=num_offset_direction_bins,
            min_bin_edge=0., max_bin_edge=360., invert=False
        )

        t[RELIABILITY_KEY].values[j, i] = _get_reliability(
            binned_mean_predictions=
            t[OFFSET_DIR_MEAN_PREDICTION_KEY].values[:, i],
            binned_mean_observations=
            t[OFFSET_DIR_MEAN_OBSERVATION_KEY].values[:, i],
            binned_example_counts=these_counts,
            is_var_direction=True
        )

        t[RESOLUTION_KEY].values[j, i] = _get_resolution(
            binned_mean_observations=
            t[OFFSET_DIR_MEAN_OBSERVATION_KEY].values[:, i],
            binned_example_counts=these_counts,
            is_var_direction=True
        )

        if i == 0:
            (
                t[OFFSET_DIR_BIN_CENTER_KEY].values[:],
                _,
                t[OFFSET_DIR_BIN_COUNT_KEY].values[:]
            ) = _get_reliability_curve_one_variable(
                target_values=full_target_matrix[:, j],
                predicted_values=full_prediction_matrix[:, j],
                is_var_direction=True, num_bins=num_offset_direction_bins,
                min_bin_edge=0., max_bin_edge=360., invert=False
            )

            (
                _,
                t[OFFSET_DIR_INV_BIN_CENTER_KEY].values[:],
                t[OFFSET_DIR_INV_BIN_COUNT_KEY].values[:]
            ) = _get_reliability_curve_one_variable(
                target_values=full_target_matrix[:, j],
                predicted_values=full_prediction_matrix[:, j],
                is_var_direction=True, num_bins=num_offset_direction_bins,
                min_bin_edge=0., max_bin_edge=360., invert=True
            )

    result_table_xarray = t
    return result_table_xarray


def _get_reliability(
        binned_mean_predictions, binned_mean_observations,
        binned_example_counts, is_var_direction):
    """Computes reliability.

    B = number of bins

    :param binned_mean_predictions: length-B numpy array of mean predictions.
    :param binned_mean_observations: length-B numpy array of mean observations.
    :param binned_example_counts: length-B numpy array of bin-membership counts.
    :param is_var_direction: Boolean flag.  If True (False), the target variable
        at hand is direction (anything else).
    :return: reliability: Obvious.
    """

    if is_var_direction:
        numerator = numpy.nansum(
            binned_example_counts *
            get_angular_diffs(binned_mean_predictions, binned_mean_observations)
            ** 2
        )
    else:
        numerator = numpy.nansum(
            binned_example_counts *
            (binned_mean_predictions - binned_mean_observations) ** 2
        )

    return numerator / numpy.sum(binned_example_counts)


def _get_resolution(
        binned_mean_observations, binned_example_counts, is_var_direction):
    """Computes resolution.

    :param binned_mean_observations: See doc for `_get_reliability.`
    :param binned_example_counts: Same.
    :param is_var_direction: Same.
    :return: resolution: Obvious.
    """

    if is_var_direction:
        sample_climatology = angular_utils.weighted_mean(
            angles=binned_mean_observations[binned_example_counts > 0],
            weights=binned_example_counts[binned_example_counts > 0],
            deg=True
        )
    else:
        sample_climatology = numpy.average(
            binned_mean_observations[binned_example_counts > 0],
            weights=binned_example_counts[binned_example_counts > 0]
        )

    numerator = numpy.nansum(
        binned_example_counts *
        (binned_mean_observations - sample_climatology) ** 2
    )
    return numerator / numpy.sum(binned_example_counts)


def get_angular_diffs(target_angles_deg, predicted_angles_deg):
    """Computes angular difference (pred minus target) for each example.

    :param target_angles_deg: numpy array of target angles.
    :param predicted_angles_deg: numpy array of predicted angles with the same
        shape.
    :return: angular_diffs_deg: numpy array of angular differences with the same
        shape as the inputs.
    """

    error_checking.assert_is_geq_numpy_array(
        target_angles_deg, 0., allow_nan=True
    )
    error_checking.assert_is_less_than_numpy_array(
        target_angles_deg, 360., allow_nan=True
    )

    error_checking.assert_is_numpy_array(
        predicted_angles_deg,
        exact_dimensions=numpy.array(target_angles_deg.shape, dtype=int)
    )
    error_checking.assert_is_geq_numpy_array(
        predicted_angles_deg, 0., allow_nan=True
    )
    error_checking.assert_is_less_than_numpy_array(
        predicted_angles_deg, 360., allow_nan=True
    )

    angular_diffs_deg = predicted_angles_deg - target_angles_deg

    angular_diffs_deg[angular_diffs_deg > 180] -= 360
    angular_diffs_deg[angular_diffs_deg < -180] += 360
    assert not numpy.any(numpy.absolute(angular_diffs_deg) > 180)

    return angular_diffs_deg


def get_offset_angles(x_offsets, y_offsets):
    """Returns angle of each offset vector.

    :param x_offsets: numpy array of x-offsets.
    :param y_offsets: numpy array of y-offsets with the same shape.
    :return: angles_deg: Angles from [0, 360) deg, with the same shape as the
        inputs.  Angles are measured counterclockwise from the vector pointing
        due east to the offset vector.
    """

    error_checking.assert_is_numpy_array_without_nan(x_offsets)
    error_checking.assert_is_numpy_array_without_nan(y_offsets)
    error_checking.assert_is_numpy_array(
        y_offsets,
        exact_dimensions=numpy.array(x_offsets.shape, dtype=int)
    )

    angles_deg = RADIANS_TO_DEGREES * numpy.arctan2(y_offsets, x_offsets)
    angles_deg = lng_conversion.convert_lng_positive_in_west(angles_deg)

    nan_indices = numpy.where(numpy.logical_and(
        numpy.absolute(x_offsets) < TOLERANCE,
        numpy.absolute(y_offsets) < TOLERANCE
    ))

    angles_deg[nan_indices] = numpy.nan
    return angles_deg


def get_scores_all_variables(
        prediction_file_names, num_bootstrap_reps,
        num_xy_offset_bins,
        min_xy_offset_metres, max_xy_offset_metres,
        min_xy_offset_percentile, max_xy_offset_percentile,
        num_offset_distance_bins,
        min_offset_distance_metres, max_offset_distance_metres,
        min_offset_distance_percentile, max_offset_distance_percentile,
        num_offset_direction_bins):
    """Computes evaluation scores for all target variables.

    :param prediction_file_names: 1-D list of paths to prediction files (will be
        read by `prediction_io.read_file`).
    :param num_bootstrap_reps: Number of replicates for statistical
        bootstrapping.
    :param num_xy_offset_bins: Number of bins in the reliability curve, used for
        both x-offset distance and y-offset distance.
    :param min_xy_offset_metres: Minimum (lowest bin edge) in the reliability
        curve, used for both x-offset distance and y-offset distance.  If you
        instead want minimum xy-offset to be a percentile over the data, make
        this argument None and use `min_xy_offset_percentile`.
    :param max_xy_offset_metres: Same as above, except maximum (highest bin
        edge).
    :param min_xy_offset_percentile: This percentile (from 0...100) determines
        the minimum in the reliability curve, used for both x-offset distance
        and y-offset distance.  If you instead want minimum xy-offset to be a
        specific value, make this argument None and use `min_xy_offset_metres`.
    :param max_xy_offset_percentile: Same as above, except maximum (highest bin
        edge).
    :param num_offset_distance_bins: Same as `num_xy_offset_bins` but for total
        Euclidean distance.
    :param min_offset_distance_metres: Same as `min_offset_distance_metres` but
        for total Euclidean distance.
    :param max_offset_distance_metres: Same as `max_offset_distance_metres` but
        for total Euclidean distance.
    :param min_offset_distance_percentile: Same as
        `min_offset_distance_percentile` but for total Euclidean distance.
    :param max_offset_distance_percentile: Same as
        `max_offset_distance_percentile` but for total Euclidean distance.
    :param num_offset_direction_bins: Same as `num_xy_offset_bins` but for
        direction of offset vector.  The min and max values in the reliability
        curve for direction will always be 0 and 360 degrees.
    :return: result_table_xarray: xarray table with results (variable and
        dimension names should make the table self-explanatory).
    """

    error_checking.assert_is_string_list(prediction_file_names)
    error_checking.assert_is_integer(num_bootstrap_reps)
    error_checking.assert_is_greater(num_bootstrap_reps, 0)
    error_checking.assert_is_integer(num_xy_offset_bins)
    error_checking.assert_is_geq(num_xy_offset_bins, 10)
    error_checking.assert_is_leq(num_xy_offset_bins, 100)
    error_checking.assert_is_integer(num_offset_distance_bins)
    error_checking.assert_is_geq(num_offset_distance_bins, 10)
    error_checking.assert_is_leq(num_offset_distance_bins, 100)
    error_checking.assert_is_integer(num_offset_direction_bins)
    error_checking.assert_is_geq(num_offset_direction_bins, 10)
    error_checking.assert_is_leq(num_offset_direction_bins, 100)

    if min_xy_offset_metres is None or max_xy_offset_metres is None:
        error_checking.assert_is_leq(min_xy_offset_percentile, 10.)
        error_checking.assert_is_geq(max_xy_offset_percentile, 90.)
    else:
        error_checking.assert_is_greater(
            max_xy_offset_metres, min_xy_offset_metres
        )

    if min_offset_distance_metres is None or max_offset_distance_metres is None:
        error_checking.assert_is_leq(min_offset_distance_percentile, 10.)
        error_checking.assert_is_geq(max_offset_distance_percentile, 90.)
    else:
        error_checking.assert_is_geq(min_offset_distance_metres, 0.)
        error_checking.assert_is_greater(
            max_offset_distance_metres, min_offset_distance_metres
        )

    num_files = len(prediction_file_names)
    ensemble_prediction_tables_xarray = [None] * num_files

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(prediction_file_names[i]))
        ensemble_prediction_tables_xarray[i] = prediction_io.read_file(
            prediction_file_names[i]
        )

    ensemble_prediction_table_xarray = prediction_utils.concat_over_examples(
        ensemble_prediction_tables_xarray
    )
    mean_prediction_table_xarray = prediction_utils.get_ensemble_mean(
        copy.deepcopy(ensemble_prediction_table_xarray)
    )

    ept = ensemble_prediction_table_xarray
    mpt = mean_prediction_table_xarray

    grid_spacings_km = mpt[prediction_utils.GRID_SPACING_KEY].values
    grid_spacings_km_2d = numpy.expand_dims(grid_spacings_km, axis=-1)

    ensemble_prediction_matrix = numpy.stack((
        grid_spacings_km_2d *
        ept[prediction_utils.PREDICTED_ROW_OFFSET_KEY].values,
        grid_spacings_km_2d *
        ept[prediction_utils.PREDICTED_COLUMN_OFFSET_KEY].values
    ), axis=-2)

    mean_prediction_matrix = numpy.transpose(numpy.vstack((
        grid_spacings_km *
        mpt[prediction_utils.PREDICTED_ROW_OFFSET_KEY].values[:, 0],
        grid_spacings_km *
        mpt[prediction_utils.PREDICTED_COLUMN_OFFSET_KEY].values[:, 0]
    )))

    target_matrix = numpy.transpose(numpy.vstack((
        grid_spacings_km * mpt[prediction_utils.ACTUAL_ROW_OFFSET_KEY].values,
        grid_spacings_km * mpt[prediction_utils.ACTUAL_COLUMN_OFFSET_KEY].values
    )))

    ensemble_prediction_matrix *= KM_TO_METRES
    mean_prediction_matrix *= KM_TO_METRES
    target_matrix *= KM_TO_METRES

    # Add vector magnitudes and directions.
    ensemble_prediction_matrix = numpy.concatenate((
        ensemble_prediction_matrix,
        numpy.sqrt(
            ensemble_prediction_matrix[:, [0], :] ** 2 +
            ensemble_prediction_matrix[:, [1], :] ** 2
        )
    ), axis=-2)

    target_matrix = numpy.concatenate((
        target_matrix,
        numpy.sqrt(target_matrix[:, [0]] ** 2 + target_matrix[:, [1]] ** 2)
    ), axis=-1)

    ensemble_prediction_matrix = numpy.concatenate((
        ensemble_prediction_matrix,
        get_offset_angles(
            x_offsets=ensemble_prediction_matrix[:, [1], :],
            y_offsets=ensemble_prediction_matrix[:, [0], :]
        )
    ), axis=-2)

    target_matrix = numpy.concatenate((
        target_matrix,
        get_offset_angles(
            x_offsets=target_matrix[:, [1]], y_offsets=target_matrix[:, [0]]
        )
    ), axis=-1)

    mean_prediction_matrix = numpy.hstack((
        mean_prediction_matrix,
        numpy.sqrt(
            mean_prediction_matrix[:, [0]] ** 2 +
            mean_prediction_matrix[:, [1]] ** 2
        )
    ))

    mean_prediction_matrix = numpy.hstack((
        mean_prediction_matrix,
        get_offset_angles(
            x_offsets=mean_prediction_matrix[:, [1]],
            y_offsets=mean_prediction_matrix[:, [0]]
        )
    ))

    num_targets = len(TARGET_FIELD_NAMES)

    these_dimensions = (num_targets, num_bootstrap_reps)
    these_dim_keys = (TARGET_FIELD_DIM, BOOTSTRAP_REP_DIM)
    main_data_dict = {
        CRPS_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        MEAN_SQUARED_ERROR_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        MSE_SKILL_SCORE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        MEAN_ABSOLUTE_ERROR_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        MAE_SKILL_SCORE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        BIAS_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        CORRELATION_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        KGE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        TARGET_STDEV_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        PREDICTION_STDEV_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        RELIABILITY_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        RESOLUTION_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        )
    }

    these_dimensions = (num_bootstrap_reps,)
    these_dim_keys = (BOOTSTRAP_REP_DIM,)
    main_data_dict.update({
        MEAN_DISTANCE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        MEDIAN_DISTANCE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        MEAN_SQUARED_DISTANCE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        MEAN_DIST_SKILL_SCORE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        MEAN_SQ_DIST_SKILL_SCORE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        )
    })

    these_dimensions = (num_targets, num_xy_offset_bins, num_bootstrap_reps)
    these_dim_keys = (TARGET_FIELD_DIM, XY_OFFSET_BIN_DIM, BOOTSTRAP_REP_DIM)
    main_data_dict.update({
        XY_OFFSET_MEAN_PREDICTION_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        XY_OFFSET_MEAN_OBSERVATION_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        )
    })

    these_dimensions = (num_targets, num_xy_offset_bins)
    these_dim_keys = (TARGET_FIELD_DIM, XY_OFFSET_BIN_DIM)
    main_data_dict.update({
        XY_OFFSET_BIN_CENTER_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        XY_OFFSET_BIN_COUNT_KEY: (
            these_dim_keys, numpy.full(these_dimensions, 0, dtype=int)
        ),
        XY_OFFSET_INV_BIN_CENTER_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        XY_OFFSET_INV_BIN_COUNT_KEY: (
            these_dim_keys, numpy.full(these_dimensions, 0, dtype=int)
        )
    })

    these_dimensions = (num_offset_distance_bins, num_bootstrap_reps)
    these_dim_keys = (OFFSET_DISTANCE_BIN_DIM, BOOTSTRAP_REP_DIM)
    main_data_dict.update({
        OFFSET_DIST_MEAN_PREDICTION_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        OFFSET_DIST_MEAN_OBSERVATION_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        )
    })

    these_dimensions = (num_offset_distance_bins,)
    these_dim_keys = (OFFSET_DISTANCE_BIN_DIM,)
    main_data_dict.update({
        OFFSET_DIST_BIN_CENTER_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        OFFSET_DIST_BIN_COUNT_KEY: (
            these_dim_keys, numpy.full(these_dimensions, 0, dtype=int)
        ),
        OFFSET_DIST_INV_BIN_CENTER_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        OFFSET_DIST_INV_BIN_COUNT_KEY: (
            these_dim_keys, numpy.full(these_dimensions, 0, dtype=int)
        )
    })

    these_dimensions = (num_offset_direction_bins, num_bootstrap_reps)
    these_dim_keys = (OFFSET_DIRECTION_BIN_DIM, BOOTSTRAP_REP_DIM)
    main_data_dict.update({
        OFFSET_DIR_MEAN_PREDICTION_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        OFFSET_DIR_MEAN_OBSERVATION_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        )
    })

    these_dimensions = (num_offset_direction_bins,)
    these_dim_keys = (OFFSET_DIRECTION_BIN_DIM,)
    main_data_dict.update({
        OFFSET_DIR_BIN_CENTER_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        OFFSET_DIR_BIN_COUNT_KEY: (
            these_dim_keys, numpy.full(these_dimensions, 0, dtype=int)
        ),
        OFFSET_DIR_INV_BIN_CENTER_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        OFFSET_DIR_INV_BIN_COUNT_KEY: (
            these_dim_keys, numpy.full(these_dimensions, 0, dtype=int)
        )
    })

    metadata_dict = {
        TARGET_FIELD_DIM: TARGET_FIELD_NAMES,
        XY_OFFSET_BIN_DIM: numpy.linspace(
            0, num_xy_offset_bins - 1, num=num_xy_offset_bins, dtype=int
        ),
        OFFSET_DISTANCE_BIN_DIM: numpy.linspace(
            0, num_offset_distance_bins - 1, num=num_offset_distance_bins,
            dtype=int
        ),
        OFFSET_DIRECTION_BIN_DIM: numpy.linspace(
            0, num_offset_direction_bins - 1, num=num_offset_direction_bins,
            dtype=int
        ),
        BOOTSTRAP_REP_DIM: numpy.linspace(
            0, num_bootstrap_reps - 1, num=num_bootstrap_reps, dtype=int
        )
    }

    result_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )

    model_file_name = (
        mean_prediction_table_xarray.attrs[prediction_utils.MODEL_FILE_KEY]
    )

    # TODO(thunderhoser): This is a HACK to deal with ensembles.
    model_file_name = model_file_name.split(' ')[0]
    model_metafile_name = neural_net_utils.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    model_metadata_dict = neural_net_utils.read_metafile(model_metafile_name)
    training_option_dict = model_metadata_dict[
        neural_net_utils.TRAINING_OPTIONS_KEY
    ]

    # TODO(thunderhoser): Need constant!
    climo_xy_offset_metres = numpy.mean(1000 * grid_spacings_km) * (
        training_option_dict[neural_net_utils.DATA_AUG_MEAN_TRANS_KEY]
    )
    climo_offset_distance_metres = numpy.sqrt(2 * (climo_xy_offset_metres ** 2))

    result_table_xarray.attrs[MODEL_FILE_KEY] = model_file_name
    result_table_xarray.attrs[PREDICTION_FILES_KEY] = prediction_file_names
    result_table_xarray.attrs[CLIMO_OFFSET_DISTANCE_KEY] = (
        climo_offset_distance_metres
    )

    num_examples = target_matrix.shape[0]
    ensemble_size = ensemble_prediction_matrix.shape[-1]
    example_indices = numpy.linspace(
        0, num_examples - 1, num=num_examples, dtype=int
    )

    for i in range(num_bootstrap_reps):
        if num_bootstrap_reps == 1:
            these_indices = example_indices
        else:
            these_indices = numpy.random.choice(
                example_indices, size=num_examples, replace=True
            )

        if ensemble_size > 1:
            print((
                'Computing CRPS for {0:d}th of {1:d} bootstrap replicates...'
            ).format(
                i + 1, num_bootstrap_reps
            ))

            epm = ensemble_prediction_matrix

            for j in range(num_targets):
                if TARGET_FIELD_NAMES[j] == OFFSET_DISTANCE_NAME:
                    xy_indices = numpy.array([
                        TARGET_FIELD_NAMES.index(X_OFFSET_NAME),
                        TARGET_FIELD_NAMES.index(Y_OFFSET_NAME)
                    ], dtype=int)

                    result_table_xarray[CRPS_KEY].values[j, i] = (
                        _get_crps_euclidean(
                            target_matrix=
                            target_matrix[:, xy_indices][these_indices, ...],
                            prediction_matrix=
                            epm[:, xy_indices, :][these_indices, ...]
                        )
                    )
                else:
                    result_table_xarray[CRPS_KEY].values[j, i] = (
                        _get_crps_one_variable(
                            target_values=
                            target_matrix[:, j][these_indices, ...],
                            prediction_matrix=epm[:, j, :][these_indices, ...],
                            is_var_direction=
                            TARGET_FIELD_NAMES[j] == OFFSET_DIRECTION_NAME
                        )
                    )

        print((
            'Computing scores for {0:d}th of {1:d} bootstrap replicates...'
        ).format(
            i + 1, num_bootstrap_reps
        ))

        result_table_xarray = _get_scores_one_replicate(
            result_table_xarray=result_table_xarray,
            full_prediction_matrix=mean_prediction_matrix,
            full_target_matrix=target_matrix,
            replicate_index=i, example_indices_in_replicate=these_indices,
            num_xy_offset_bins=num_xy_offset_bins,
            min_xy_offset_metres=min_xy_offset_metres,
            max_xy_offset_metres=max_xy_offset_metres,
            min_xy_offset_percentile=min_xy_offset_percentile,
            max_xy_offset_percentile=max_xy_offset_percentile,
            num_offset_distance_bins=num_offset_distance_bins,
            min_offset_distance_metres=min_offset_distance_metres,
            max_offset_distance_metres=max_offset_distance_metres,
            min_offset_distance_percentile=min_offset_distance_percentile,
            max_offset_distance_percentile=max_offset_distance_percentile,
            num_offset_direction_bins=num_offset_direction_bins
        )

    return result_table_xarray


def read_file(netcdf_file_name):
    """Reads evaluation results from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: result_table_xarray: xarray table in format returned by
        `get_scores_all_variables`.  Variable names and metadata should make
        this table self-explanatory.
    """

    result_table_xarray = xarray.open_dataset(netcdf_file_name)
    result_table_xarray.attrs[PREDICTION_FILES_KEY] = (
        result_table_xarray.attrs[PREDICTION_FILES_KEY].split()
    )

    return result_table_xarray


def write_file(result_table_xarray, netcdf_file_name):
    """Writes evaluation results to NetCDF file.

    :param result_table_xarray: xarray table in format returned by
        `get_scores_all_variables`.  Variable names and metadata should make
        this table self-explanatory.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    result_table_xarray.attrs[PREDICTION_FILES_KEY] = ' '.join(
        result_table_xarray.attrs[PREDICTION_FILES_KEY]
    )

    result_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )
