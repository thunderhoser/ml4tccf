"""Helper methods for computing spread-skill relationship."""

import numpy
import xarray
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4tccf.io import prediction_io
from ml4tccf.utils import scalar_evaluation
from ml4tccf.utils import scalar_prediction_utils as prediction_utils
from ml4tccf.outside_code import angular_utils

KM_TO_METRES = 1000.

MEAN_PREDICTION_STDEVS_KEY = 'mean_prediction_stdevs'
BIN_EDGE_PREDICTION_STDEVS_KEY = 'bin_edge_prediction_stdevs'
RMSE_VALUES_KEY = 'rmse_values'
SPREAD_SKILL_RELIABILITY_KEY = 'spread_skill_reliability'
SPREAD_SKILL_RATIO_KEY = 'spread_skill_ratio'
EXAMPLE_COUNTS_KEY = 'example_counts'
MEAN_MEAN_PREDICTIONS_KEY = 'mean_mean_predictions'
MEAN_TARGET_VALUES_KEY = 'mean_target_values'

MEAN_PREDICTION_STDISTDEVS_KEY = 'mean_prediction_stdistdevs'
BIN_EDGE_PREDICTION_STDISTDEVS_KEY = 'bin_edge_prediction_stdistdevs'
RMSD_VALUES_KEY = 'rmsd_values'
MEAN_MEAN_PREDICTED_DISTANCES_KEY = 'mean_mean_predicted_distances'
MEAN_TARGET_DISTANCES_KEY = 'mean_target_distances'

X_OFFSET_NAME = 'x_offset_metres'
Y_OFFSET_NAME = 'y_offset_metres'
OFFSET_DIRECTION_NAME = 'offset_direction_deg'
OFFSET_DISTANCE_NAME = 'offset_distance_metres'
TARGET_FIELD_NAMES = [
    Y_OFFSET_NAME, X_OFFSET_NAME, OFFSET_DIRECTION_NAME, OFFSET_DISTANCE_NAME
]

TARGET_FIELD_DIM = 'target_field'
XY_OFFSET_BIN_DIM = 'xy_offset_spread_bin'
OFFSET_DISTANCE_BIN_DIM = 'offset_distance_spread_bin'
OFFSET_DIRECTION_BIN_DIM = 'offset_direction_spread_bin'
XY_OFFSET_BIN_EDGE_DIM = 'xy_offset_spread_bin_edge'
OFFSET_DISTANCE_BIN_EDGE_DIM = 'offset_distance_spread_bin_edge'
OFFSET_DIRECTION_BIN_EDGE_DIM = 'offset_direction_spread_bin_edge'

XY_MEAN_STDEV_KEY = 'xy_mean_prediction_stdev_metres'
XY_BIN_EDGE_KEY = 'xy_spread_bin_edge_metres'
XY_RMSE_KEY = 'xy_rmse_metres'
XY_SSREL_KEY = 'xy_spread_skill_relia_metres'
XY_SSRAT_KEY = 'xy_spread_skill_ratio'
XY_EXAMPLE_COUNT_KEY = 'xy_example_count'
XY_MEAN_MEAN_PREDICTION_KEY = 'xy_mean_mean_prediction_metres'
XY_MEAN_TARGET_KEY = 'xy_mean_target_metres'

OFFSET_DIR_MEAN_STDEV_KEY = 'offset_dir_mean_prediction_stdev_deg'
OFFSET_DIR_BIN_EDGE_KEY = 'offset_dir_spread_bin_edge_deg'
OFFSET_DIR_RMSE_KEY = 'offset_dir_rmse_deg'
OFFSET_DIR_SSREL_KEY = 'offset_dir_spread_skill_relia_deg'
OFFSET_DIR_SSRAT_KEY = 'offset_dir_spread_skill_ratio'
OFFSET_DIR_EXAMPLE_COUNT_KEY = 'offset_dir_example_count'
OFFSET_DIR_MEAN_MEAN_PREDICTION_KEY = 'offset_dir_mean_mean_prediction_deg'
OFFSET_DIR_MEAN_TARGET_KEY = 'offset_dir_mean_target_deg'

OFFSET_DIST_MEAN_STDIST_KEY = 'offset_dist_mean_prediction_stdist_metres'
OFFSET_DIST_BIN_EDGE_KEY = 'offset_dist_spread_bin_edge_metres'
OFFSET_DIST_RMSD_KEY = 'offset_dist_rmsd_metres'
OFFSET_DIST_SSREL_KEY = 'offset_dist_spread_skill_relia_metres'
OFFSET_DIST_SSRAT_KEY = 'offset_dist_spread_skill_ratio'
OFFSET_DIST_EXAMPLE_COUNT_KEY = 'offset_dist_example_count'
OFFSET_DIST_MEAN_MEAN_PRED_DIST_KEY = (
    'offset_dist_mean_mean_predicted_dist_metres'
)
OFFSET_DIST_MEAN_TARGET_DIST_KEY = 'offset_dist_mean_target_dist_metres'

MODEL_FILE_KEY = 'model_file_name'
ISOTONIC_MODEL_FILE_KEY = 'isotonic_model_file_name'
PREDICTION_FILES_KEY = 'prediction_file_names'


def _get_predictive_stdistdevs(prediction_matrix):
    """Computes standard distance deviation of each predictive distribution.

    E = number of examples
    S = number of ensemble members

    :param prediction_matrix: E-by-2-by-S numpy array of predicted values,
        where prediction_matrix[:, 0, :] contains x-displacements and
        prediction_matrix[:, 1, :] contains y-displacements.
    :return: predictive_stdistdevs: length-E numpy array of standard distance
        deviations.
    """

    ensemble_size = prediction_matrix.shape[-1]

    mean_prediction_matrix = numpy.mean(prediction_matrix, axis=-1)
    prediction_anomaly_matrix = (
        prediction_matrix - numpy.expand_dims(mean_prediction_matrix, axis=-1)
    )
    prediction_sq_euc_anomaly_matrix = (
        prediction_anomaly_matrix[:, 0, :] ** 2 +
        prediction_anomaly_matrix[:, 1, :] ** 2
    )
    return numpy.sqrt(
        numpy.sum(prediction_sq_euc_anomaly_matrix, axis=1) /
        (ensemble_size - 1)
    )


def _get_angular_predictive_stdevs(prediction_matrix):
    """Computes predictive stdevs for offset angles.

    E = number of examples
    S = number of ensemble members

    :param prediction_matrix: E-by-S numpy array of predicted offset angles.
    :return: predictive_stdevs: length-E numpy array of standard deviations.
    """

    num_examples = prediction_matrix.shape[0]
    predictive_stdevs = numpy.full(num_examples, numpy.nan)

    for i in range(num_examples):
        real_indices = numpy.where(
            numpy.invert(numpy.isnan(prediction_matrix[i, :]))
        )[0]

        if len(real_indices) >= 2:
            predictive_stdevs[i] = angular_utils.std(
                prediction_matrix[i, real_indices], deg=True
            )

    return predictive_stdevs


def _get_results_euclidean(
        target_matrix, prediction_matrix, bin_edge_prediction_stdistdevs):
    """Computes spread-skill relationship for Euclidean offset.

    E = number of examples
    S = number of ensemble members
    B = number of bins

    :param target_matrix: E-by-2 numpy array of actual values, where
        target_matrix[:, 0] contains x-displacements and target_matrix[:, 1]
        contains y-displacements.
    :param prediction_matrix: E-by-2-by-S numpy array of predicted values,
        where prediction_matrix[:, 0, :] contains x-displacements and
        prediction_matrix[:, 1, :] contains y-displacements.
    :param bin_edge_prediction_stdistdevs: length-(B - 1) numpy array of bin
        cutoffs.  Each is a standard distance deviation for the predictive
        distribution.  Ultimately, there will be B + 1 edges; this method will
        use 0 as the lowest edge and inf as the highest edge.
    :return: result_dict: Dictionary with the following keys.
    result_dict['mean_prediction_stdistdevs']: length-B numpy array, where the
        [i]th entry is the mean standard distance deviation of predictive
        distributions in the [i]th bin.
    result_dict['bin_edge_prediction_stdistdevs']: length-(B + 1) numpy array,
        where the [i]th and [i + 1]th entries are the edges for the [i]th bin.
    result_dict['rmsd_values']: length-B numpy array, where the [i]th
        entry is the root mean squared distance of mean predictions in the [i]th
        bin.
    result_dict['spread_skill_reliability']: Spread-skill reliability (SSREL).
    result_dict['spread_skill_ratio']: Spread-skill ratio (SSRAT).
    result_dict['example_counts']: length-B numpy array of corresponding example
        counts.
    result_dict['mean_mean_predicted_distances']: length-B numpy array, where
        the [i]th entry is the mean mean predicted distance (over data samples
        and ensemble members) for the [i]th bin.
    result_dict['mean_predicted_targets']: length-B numpy array, where the [i]th
        entry is the mean actual distance (over data samples and ensemble
        members) for the [i]th bin.
    """

    # Check input args.
    error_checking.assert_is_numpy_array_without_nan(target_matrix)
    error_checking.assert_is_numpy_array(target_matrix, num_dimensions=2)

    error_checking.assert_is_numpy_array_without_nan(prediction_matrix)
    error_checking.assert_is_numpy_array(prediction_matrix, num_dimensions=3)

    num_examples = target_matrix.shape[0]
    num_ensemble_members = prediction_matrix.shape[2]
    error_checking.assert_is_greater(num_ensemble_members, 1)

    these_dim = numpy.array([num_examples, 2], dtype=int)
    error_checking.assert_is_numpy_array(
        target_matrix, exact_dimensions=these_dim
    )

    these_dim = numpy.array([num_examples, 2, num_ensemble_members], dtype=int)
    error_checking.assert_is_numpy_array(
        prediction_matrix, exact_dimensions=these_dim
    )

    error_checking.assert_is_numpy_array(
        bin_edge_prediction_stdistdevs, num_dimensions=1
    )
    error_checking.assert_is_greater_numpy_array(
        bin_edge_prediction_stdistdevs, 0.
    )
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(bin_edge_prediction_stdistdevs), 0.
    )

    bin_edge_prediction_stdistdevs = numpy.concatenate((
        numpy.array([0.]),
        bin_edge_prediction_stdistdevs,
        numpy.array([numpy.inf])
    ))

    num_bins = len(bin_edge_prediction_stdistdevs) - 1
    assert num_bins >= 2

    # Do actual stuff.
    mean_prediction_matrix = numpy.mean(prediction_matrix, axis=-1)
    squared_errors = (
        (mean_prediction_matrix[:, 0] - target_matrix[:, 0]) ** 2 +
        (mean_prediction_matrix[:, 1] - target_matrix[:, 1]) ** 2
    )
    predictive_stdistdevs = _get_predictive_stdistdevs(prediction_matrix)

    mean_prediction_stdistdevs = numpy.full(num_bins, numpy.nan)
    rmsd_values = numpy.full(num_bins, numpy.nan)
    example_counts = numpy.full(num_bins, 0, dtype=int)
    mean_mean_predicted_distances = numpy.full(num_bins, numpy.nan)
    mean_target_distances = numpy.full(num_bins, numpy.nan)

    for k in range(num_bins):
        these_indices = numpy.where(numpy.logical_and(
            predictive_stdistdevs >= bin_edge_prediction_stdistdevs[k],
            predictive_stdistdevs < bin_edge_prediction_stdistdevs[k + 1]
        ))[0]

        mean_prediction_stdistdevs[k] = numpy.sqrt(numpy.mean(
            predictive_stdistdevs[these_indices] ** 2
        ))
        rmsd_values[k] = numpy.sqrt(numpy.mean(
            squared_errors[these_indices]
        ))
        example_counts[k] = len(these_indices)

        mean_mean_predicted_distances[k] = numpy.sqrt(numpy.mean(
            mean_prediction_matrix[these_indices, 0] ** 2 +
            mean_prediction_matrix[these_indices, 1] ** 2
        ))
        mean_target_distances[k] = numpy.sqrt(numpy.mean(
            target_matrix[these_indices, 0] ** 2 +
            target_matrix[these_indices, 1] ** 2
        ))

    these_diffs = numpy.absolute(mean_prediction_stdistdevs - rmsd_values)
    these_diffs[numpy.isnan(these_diffs)] = 0.
    spread_skill_reliability = numpy.average(
        these_diffs, weights=example_counts
    )

    this_numer = numpy.sqrt(numpy.mean(predictive_stdistdevs ** 2))
    this_denom = numpy.sqrt(numpy.mean(squared_errors))
    spread_skill_ratio = this_numer / this_denom

    return {
        MEAN_PREDICTION_STDISTDEVS_KEY: mean_prediction_stdistdevs,
        BIN_EDGE_PREDICTION_STDISTDEVS_KEY: bin_edge_prediction_stdistdevs,
        RMSD_VALUES_KEY: rmsd_values,
        SPREAD_SKILL_RELIABILITY_KEY: spread_skill_reliability,
        SPREAD_SKILL_RATIO_KEY: spread_skill_ratio,
        EXAMPLE_COUNTS_KEY: example_counts,
        MEAN_MEAN_PREDICTED_DISTANCES_KEY: mean_mean_predicted_distances,
        MEAN_TARGET_DISTANCES_KEY: mean_target_distances
    }


def _get_results_one_var(
        target_values, prediction_matrix, bin_edge_prediction_stdevs,
        is_var_direction):
    """Computes spread-skill relationship for one target variable.

    E = number of examples
    S = number of ensemble members
    B = number of bins

    :param target_values: length-E numpy array of actual values.
    :param prediction_matrix: E-by-S numpy array of predicted values.
    :param bin_edge_prediction_stdevs: length-(B - 1) numpy array of bin
        cutoffs.  Each is a standard deviation for the predictive distribution.
        Ultimately, there will be B + 1 edges; this method will use 0 as the
        lowest edge and inf as the highest edge.
    :param is_var_direction: Boolean flag.  If True (False), this method assumes
        that the target variable is direction (x or y).
    :return: result_dict: Dictionary with the following keys.
    result_dict['mean_prediction_stdevs']: length-B numpy array, where the [i]th
        entry is the mean standard deviation of predictive distributions in the
        [i]th bin.
    result_dict['bin_edge_prediction_stdevs']: length-(B + 1) numpy array,
        where the [i]th and [i + 1]th entries are the edges for the [i]th bin.
    result_dict['rmse_values']: length-B numpy array, where the [i]th
        entry is the root mean squared error of mean predictions in the [i]th
        bin.
    result_dict['spread_skill_reliability']: Spread-skill reliability (SSREL).
    result_dict['spread_skill_ratio']: Spread-skill ratio (SSRAT).
    result_dict['example_counts']: length-B numpy array of corresponding example
        counts.
    result_dict['mean_mean_predictions']: length-B numpy array, where the
        [i]th entry is the mean mean prediction for the [i]th bin.
    result_dict['mean_target_values']: length-B numpy array, where the [i]th
        entry is the mean target value for the [i]th bin.
    """

    # Check input args.
    error_checking.assert_is_numpy_array(target_values, num_dimensions=1)
    error_checking.assert_is_numpy_array(prediction_matrix, num_dimensions=2)

    if not is_var_direction:
        error_checking.assert_is_numpy_array_without_nan(target_values)
        error_checking.assert_is_numpy_array_without_nan(prediction_matrix)

    num_examples = len(target_values)
    num_ensemble_members = prediction_matrix.shape[1]
    error_checking.assert_is_greater(num_ensemble_members, 1)

    these_dim = numpy.array([num_examples, num_ensemble_members], dtype=int)
    error_checking.assert_is_numpy_array(
        prediction_matrix, exact_dimensions=these_dim
    )

    error_checking.assert_is_numpy_array(
        bin_edge_prediction_stdevs, num_dimensions=1
    )
    error_checking.assert_is_greater_numpy_array(bin_edge_prediction_stdevs, 0.)
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(bin_edge_prediction_stdevs), 0.
    )

    bin_edge_prediction_stdevs = numpy.concatenate((
        numpy.array([0.]),
        bin_edge_prediction_stdevs,
        numpy.array([numpy.inf])
    ))

    num_bins = len(bin_edge_prediction_stdevs) - 1
    assert num_bins >= 2

    error_checking.assert_is_boolean(is_var_direction)

    # Do actual stuff.
    if is_var_direction:
        predictive_stdevs = _get_angular_predictive_stdevs(prediction_matrix)
        mean_predictions = numpy.full(num_examples, numpy.nan)
        squared_errors = numpy.full(num_examples, numpy.nan)

        for i in range(num_examples):
            real_indices = numpy.where(
                numpy.invert(numpy.isnan(prediction_matrix[i, :]))
            )[0]

            if len(real_indices) >= 1:
                mean_predictions[i] = angular_utils.mean(
                    prediction_matrix[i, real_indices], deg=True
                )

                if not numpy.isnan(target_values[i]):
                    squared_errors[i] = scalar_evaluation.get_angular_diffs(
                        target_angles_deg=target_values[[i]],
                        predicted_angles_deg=mean_predictions[[i]]
                    )[0] ** 2
    else:
        mean_predictions = numpy.mean(prediction_matrix, axis=1)
        predictive_stdevs = numpy.std(prediction_matrix, axis=1, ddof=1)
        squared_errors = (mean_predictions - target_values) ** 2

    mean_prediction_stdevs = numpy.full(num_bins, numpy.nan)
    rmse_values = numpy.full(num_bins, numpy.nan)
    example_counts = numpy.full(num_bins, 0, dtype=int)
    mean_mean_predictions = numpy.full(num_bins, numpy.nan)
    mean_target_values = numpy.full(num_bins, numpy.nan)

    for k in range(num_bins):
        these_indices = numpy.where(numpy.logical_and(
            predictive_stdevs >= bin_edge_prediction_stdevs[k],
            predictive_stdevs < bin_edge_prediction_stdevs[k + 1]
        ))[0]

        mean_prediction_stdevs[k] = numpy.sqrt(numpy.mean(
            predictive_stdevs[these_indices] ** 2
        ))
        rmse_values[k] = numpy.sqrt(numpy.nanmean(
            squared_errors[these_indices]
        ))
        example_counts[k] = len(these_indices)

        if is_var_direction:
            mean_mean_predictions[k] = angular_utils.mean(
                mean_predictions[these_indices], deg=True
            )

            these_target_values = target_values[these_indices][
                numpy.invert(numpy.isnan(target_values[these_indices]))
            ]
            mean_target_values[k] = angular_utils.mean(
                these_target_values, deg=True
            )
        else:
            mean_mean_predictions[k] = numpy.mean(
                mean_predictions[these_indices]
            )
            mean_target_values[k] = numpy.mean(target_values[these_indices])

    these_diffs = numpy.absolute(mean_prediction_stdevs - rmse_values)
    these_diffs[numpy.isnan(these_diffs)] = 0.
    spread_skill_reliability = numpy.average(
        these_diffs, weights=example_counts
    )

    nan_flags = numpy.logical_or(
        numpy.isnan(predictive_stdevs),
        numpy.isnan(squared_errors)
    )
    real_indices = numpy.where(numpy.invert(nan_flags))[0]

    this_numer = numpy.sqrt(numpy.mean(predictive_stdevs[real_indices] ** 2))
    this_denom = numpy.sqrt(numpy.mean(squared_errors[real_indices]))
    spread_skill_ratio = this_numer / this_denom

    return {
        MEAN_PREDICTION_STDEVS_KEY: mean_prediction_stdevs,
        BIN_EDGE_PREDICTION_STDEVS_KEY: bin_edge_prediction_stdevs,
        RMSE_VALUES_KEY: rmse_values,
        SPREAD_SKILL_RELIABILITY_KEY: spread_skill_reliability,
        SPREAD_SKILL_RATIO_KEY: spread_skill_ratio,
        EXAMPLE_COUNTS_KEY: example_counts,
        MEAN_MEAN_PREDICTIONS_KEY: mean_mean_predictions,
        MEAN_TARGET_VALUES_KEY: mean_target_values
    }


def get_results_all_vars(
        prediction_file_names, num_xy_offset_bins,
        bin_min_xy_offset_metres, bin_max_xy_offset_metres,
        bin_min_xy_offset_percentile, bin_max_xy_offset_percentile,
        num_offset_distance_bins,
        bin_min_offset_distance_metres, bin_max_offset_distance_metres,
        bin_min_offset_distance_percentile, bin_max_offset_distance_percentile,
        num_offset_direction_bins,
        bin_min_offset_direction_deg, bin_max_offset_direction_deg,
        bin_min_offset_dir_percentile, bin_max_offset_dir_percentile):
    """Computes spread-skill relationship for each target variable.

    :param prediction_file_names: 1-D list of paths to prediction files (will be
        read by `prediction_io.read_file`).
    :param num_xy_offset_bins: Number of spread bins for both x-offset and
        y-offset.
    :param bin_min_xy_offset_metres: Lower edge of lowest bin for both x-offset
        and y-offset.  If you want to choose bin edges by specifying percentiles
        over the data, rather than direct values, make this argument None and
        use `bin_min_xy_offset_percentile`.
    :param bin_max_xy_offset_metres: Same as above but for upper edge of highest
        bin.
    :param bin_min_xy_offset_percentile: Determines lower edge of lowest bin for
        both x-offset and y-offset.  Specifically, this bin edge will be the
        [q]th percentile over all spread values for x- or y-offset in the data,
        where q = `bin_min_xy_offset_percentile`.  If you want to choose bin
        edges directly, make this argument None and use
        `bin_min_xy_offset_metres`.
    :param bin_max_xy_offset_percentile: Same as above but for upper edge of
        highest bin.
    :param num_offset_distance_bins: Number of spread bins for total (Euclidean)
        offset distance.
    :param bin_min_offset_distance_metres: Lower edge of lowest bin for total
        (Euclidean) offset distance.  If you want to choose bin edges by
        percentiles over the data, rather than direct values, make this argument
        None and use `bin_min_offset_distance_percentile`.
    :param bin_max_offset_distance_metres: Same as above but for upper edge of
        highest bin.
    :param bin_min_offset_distance_percentile: Determines lower edge of lowest
        bin for total (Euclidean) offset distance.  Specifically, this bin edge
        will be the [q]th percentile over all Euclidean spread values in the
        data, where q = `bin_min_offset_distance_percentile`.  If you want to
        choose bin edges directly, make this argument None and use
        `bin_min_offset_distance_metres`.
    :param bin_max_offset_distance_percentile: Same as above but for upper edge
        of highest bin.
    :param num_offset_direction_bins: Number of spread bins for offset direction
        (angle).
    :param bin_min_offset_direction_deg: Lower edge of lowest bin for offset
        direction (angle).  If you want to choose bin edges by percentiles over
        the data, rather than direct values, make this argument None and use
        `bin_min_offset_dir_percentile`.
    :param bin_max_offset_direction_deg: Same as above but for upper edge of
        highest bin.
    :param bin_min_offset_dir_percentile: Determines lower edge of lowest bin
        for offset direction (angle).  Specifically, this bin edge will be the
        [q]th percentile over all spread values in the data, where
        q = `bin_min_offset_dir_percentile`.  If you want to choose bin edges
        directly, make this argument None and use
        `bin_min_offset_direction_deg`.
    :param bin_max_offset_dir_percentile: Same as above but for upper edge of
        highest bin.
    :return: result_table_xarray: xarray table with results (variable and
        dimension names should make the table self-explanatory).
    """

    # Check input args.
    error_checking.assert_is_string_list(prediction_file_names)
    error_checking.assert_is_integer(num_xy_offset_bins)
    error_checking.assert_is_geq(num_xy_offset_bins, 10)
    error_checking.assert_is_leq(num_xy_offset_bins, 100)
    error_checking.assert_is_integer(num_offset_distance_bins)
    error_checking.assert_is_geq(num_offset_distance_bins, 10)
    error_checking.assert_is_leq(num_offset_distance_bins, 100)
    error_checking.assert_is_integer(num_offset_direction_bins)
    error_checking.assert_is_geq(num_offset_direction_bins, 10)
    error_checking.assert_is_leq(num_offset_direction_bins, 100)

    if bin_min_xy_offset_metres is None or bin_max_xy_offset_metres is None:
        error_checking.assert_is_leq(bin_min_xy_offset_percentile, 10.)
        error_checking.assert_is_geq(bin_max_xy_offset_percentile, 90.)
    else:
        error_checking.assert_is_greater(
            bin_max_xy_offset_metres, bin_min_xy_offset_metres
        )

    if (
            bin_min_offset_distance_metres is None or
            bin_max_offset_distance_metres is None
    ):
        error_checking.assert_is_leq(bin_min_offset_distance_percentile, 10.)
        error_checking.assert_is_geq(bin_max_offset_distance_percentile, 90.)
    else:
        error_checking.assert_is_geq(bin_min_offset_distance_metres, 0.)
        error_checking.assert_is_greater(
            bin_max_offset_distance_metres, bin_min_offset_distance_metres
        )

    if (
            bin_min_offset_direction_deg is None or
            bin_max_offset_direction_deg is None
    ):
        error_checking.assert_is_leq(bin_min_offset_dir_percentile, 10.)
        error_checking.assert_is_geq(bin_max_offset_dir_percentile, 90.)
    else:
        error_checking.assert_is_geq(bin_min_offset_direction_deg, 0.)
        error_checking.assert_is_less_than(bin_max_offset_direction_deg, 360.)
        error_checking.assert_is_greater(
            bin_max_offset_direction_deg, bin_min_offset_direction_deg
        )

    # Do actual stuff.
    num_files = len(prediction_file_names)
    prediction_tables_xarray = [None] * num_files

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(prediction_file_names[i]))
        prediction_tables_xarray[i] = prediction_io.read_file(
            prediction_file_names[i]
        )

    prediction_table_xarray = prediction_utils.concat_over_examples(
        prediction_tables_xarray
    )
    pt = prediction_table_xarray

    grid_spacings_km = pt[prediction_utils.GRID_SPACING_KEY].values

    prediction_matrix = numpy.stack((
        pt[prediction_utils.PREDICTED_ROW_OFFSET_KEY].values,
        pt[prediction_utils.PREDICTED_COLUMN_OFFSET_KEY].values
    ), axis=1)
    prediction_matrix *= numpy.expand_dims(grid_spacings_km, axis=(1, 2))

    target_matrix = numpy.transpose(numpy.vstack((
        grid_spacings_km * pt[prediction_utils.ACTUAL_ROW_OFFSET_KEY].values,
        grid_spacings_km * pt[prediction_utils.ACTUAL_COLUMN_OFFSET_KEY].values
    )))

    prediction_matrix *= KM_TO_METRES
    target_matrix *= KM_TO_METRES

    # Add vector directions.
    prediction_matrix = numpy.concatenate((
        prediction_matrix,
        scalar_evaluation.get_offset_angles(
            x_offsets=prediction_matrix[:, [1], :],
            y_offsets=prediction_matrix[:, [0], :]
        )
    ), axis=1)

    target_matrix = numpy.hstack((
        target_matrix,
        scalar_evaluation.get_offset_angles(
            x_offsets=target_matrix[:, [1]],
            y_offsets=target_matrix[:, [0]]
        )
    ))

    num_targets = len(TARGET_FIELD_NAMES)

    these_dim_no_bins = num_targets
    these_dim_no_edge = (num_targets, num_xy_offset_bins)
    these_dim_with_edge = (num_targets, num_xy_offset_bins + 1)

    these_dim_keys_no_bins = (TARGET_FIELD_DIM,)
    these_dim_keys_no_edge = (TARGET_FIELD_DIM, XY_OFFSET_BIN_DIM)
    these_dim_keys_with_edge = (TARGET_FIELD_DIM, XY_OFFSET_BIN_EDGE_DIM)

    main_data_dict = {
        XY_MEAN_STDEV_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
        ),
        XY_BIN_EDGE_KEY: (
            these_dim_keys_with_edge, numpy.full(these_dim_with_edge, numpy.nan)
        ),
        XY_RMSE_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
        ),
        XY_SSREL_KEY: (
            these_dim_keys_no_bins, numpy.full(these_dim_no_bins, numpy.nan)
        ),
        XY_SSRAT_KEY: (
            these_dim_keys_no_bins, numpy.full(these_dim_no_bins, numpy.nan)
        ),
        XY_EXAMPLE_COUNT_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, -1, dtype=int)
        ),
        XY_MEAN_MEAN_PREDICTION_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
        ),
        XY_MEAN_TARGET_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
        )
    }

    these_dim_no_bins = num_targets
    these_dim_no_edge = (num_targets, num_offset_direction_bins)
    these_dim_with_edge = (num_targets, num_offset_direction_bins + 1)

    these_dim_keys_no_bins = (TARGET_FIELD_DIM,)
    these_dim_keys_no_edge = (TARGET_FIELD_DIM, OFFSET_DIRECTION_BIN_DIM)
    these_dim_keys_with_edge = (TARGET_FIELD_DIM, OFFSET_DIRECTION_BIN_EDGE_DIM)

    main_data_dict.update({
        OFFSET_DIR_MEAN_STDEV_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
        ),
        OFFSET_DIR_BIN_EDGE_KEY: (
            these_dim_keys_with_edge, numpy.full(these_dim_with_edge, numpy.nan)
        ),
        OFFSET_DIR_RMSE_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
        ),
        OFFSET_DIR_SSREL_KEY: (
            these_dim_keys_no_bins, numpy.full(these_dim_no_bins, numpy.nan)
        ),
        OFFSET_DIR_SSRAT_KEY: (
            these_dim_keys_no_bins, numpy.full(these_dim_no_bins, numpy.nan)
        ),
        OFFSET_DIR_EXAMPLE_COUNT_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, -1, dtype=int)
        ),
        OFFSET_DIR_MEAN_MEAN_PREDICTION_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
        ),
        OFFSET_DIR_MEAN_TARGET_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
        )
    })

    these_dim_no_bins = num_targets
    these_dim_no_edge = (num_targets, num_offset_distance_bins)
    these_dim_with_edge = (num_targets, num_offset_distance_bins + 1)

    these_dim_keys_no_bins = (TARGET_FIELD_DIM,)
    these_dim_keys_no_edge = (TARGET_FIELD_DIM, OFFSET_DISTANCE_BIN_DIM)
    these_dim_keys_with_edge = (TARGET_FIELD_DIM, OFFSET_DISTANCE_BIN_EDGE_DIM)

    main_data_dict.update({
        OFFSET_DIST_MEAN_STDIST_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
        ),
        OFFSET_DIST_BIN_EDGE_KEY: (
            these_dim_keys_with_edge, numpy.full(these_dim_with_edge, numpy.nan)
        ),
        OFFSET_DIST_RMSD_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
        ),
        OFFSET_DIST_SSREL_KEY: (
            these_dim_keys_no_bins, numpy.full(these_dim_no_bins, numpy.nan)
        ),
        OFFSET_DIST_SSRAT_KEY: (
            these_dim_keys_no_bins, numpy.full(these_dim_no_bins, numpy.nan)
        ),
        OFFSET_DIST_EXAMPLE_COUNT_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, -1, dtype=int)
        ),
        OFFSET_DIST_MEAN_MEAN_PRED_DIST_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
        ),
        OFFSET_DIST_MEAN_TARGET_DIST_KEY: (
            these_dim_keys_no_edge, numpy.full(these_dim_no_edge, numpy.nan)
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
        )
    }

    result_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )
    result_table_xarray.attrs[MODEL_FILE_KEY] = (
        prediction_table_xarray.attrs[prediction_utils.MODEL_FILE_KEY]
    )
    result_table_xarray.attrs[ISOTONIC_MODEL_FILE_KEY] = (
        prediction_table_xarray.attrs[prediction_utils.ISOTONIC_MODEL_FILE_KEY]
    )
    result_table_xarray.attrs[PREDICTION_FILES_KEY] = ' '.join([
        '{0:s}'.format(f) for f in prediction_file_names
    ])

    xy_indices = numpy.array([
        TARGET_FIELD_NAMES.index(X_OFFSET_NAME),
        TARGET_FIELD_NAMES.index(Y_OFFSET_NAME)
    ], dtype=int)

    for j in xy_indices:
        print('Computing spread-skill relationship for "{0:s}"...'.format(
            TARGET_FIELD_NAMES[j]
        ))

        if bin_min_xy_offset_metres is None or bin_max_xy_offset_metres is None:
            these_stdevs = numpy.std(
                prediction_matrix[:, j, :], ddof=1, axis=-1
            )

            these_bin_edges = numpy.linspace(
                numpy.percentile(these_stdevs, bin_min_xy_offset_percentile),
                numpy.percentile(these_stdevs, bin_max_xy_offset_percentile),
                num=num_xy_offset_bins + 1, dtype=float
            )[1:-1]
        else:
            these_bin_edges = numpy.linspace(
                bin_min_xy_offset_metres, bin_max_xy_offset_metres,
                num=num_xy_offset_bins + 1, dtype=float
            )[1:-1]

        this_result_dict = _get_results_one_var(
            target_values=target_matrix[:, j],
            prediction_matrix=prediction_matrix[:, j, :],
            bin_edge_prediction_stdevs=these_bin_edges,
            is_var_direction=False
        )

        result_table_xarray[XY_MEAN_STDEV_KEY].values[j, :] = (
            this_result_dict[MEAN_PREDICTION_STDEVS_KEY]
        )
        result_table_xarray[XY_BIN_EDGE_KEY].values[j, :] = (
            this_result_dict[BIN_EDGE_PREDICTION_STDEVS_KEY]
        )
        result_table_xarray[XY_RMSE_KEY].values[j, :] = (
            this_result_dict[RMSE_VALUES_KEY]
        )
        result_table_xarray[XY_SSREL_KEY].values[j] = (
            this_result_dict[SPREAD_SKILL_RELIABILITY_KEY]
        )
        result_table_xarray[XY_SSRAT_KEY].values[j] = (
            this_result_dict[SPREAD_SKILL_RATIO_KEY]
        )
        result_table_xarray[XY_EXAMPLE_COUNT_KEY].values[j, :] = (
            this_result_dict[EXAMPLE_COUNTS_KEY]
        )
        result_table_xarray[XY_MEAN_MEAN_PREDICTION_KEY].values[j, :] = (
            this_result_dict[MEAN_MEAN_PREDICTIONS_KEY]
        )
        result_table_xarray[XY_MEAN_TARGET_KEY].values[j, :] = (
            this_result_dict[MEAN_TARGET_VALUES_KEY]
        )

    direction_indices = numpy.array(
        [TARGET_FIELD_NAMES.index(OFFSET_DIRECTION_NAME)], dtype=int
    )

    for j in direction_indices:
        print('Computing spread-skill relationship for "{0:s}"...'.format(
            TARGET_FIELD_NAMES[j]
        ))

        if (
                bin_min_offset_direction_deg is None or
                bin_max_offset_direction_deg is None
        ):
            these_stdevs = _get_angular_predictive_stdevs(
                prediction_matrix[:, j, :]
            )

            these_bin_edges = numpy.linspace(
                numpy.nanpercentile(
                    these_stdevs, bin_min_offset_dir_percentile
                ),
                numpy.nanpercentile(
                    these_stdevs, bin_max_offset_dir_percentile
                ),
                num=num_offset_direction_bins + 1, dtype=float
            )[1:-1]
        else:
            these_bin_edges = numpy.linspace(
                bin_min_offset_direction_deg, bin_max_offset_direction_deg,
                num=num_offset_direction_bins + 1, dtype=float
            )[1:-1]

        this_result_dict = _get_results_one_var(
            target_values=target_matrix[:, j],
            prediction_matrix=prediction_matrix[:, j, :],
            bin_edge_prediction_stdevs=these_bin_edges,
            is_var_direction=True
        )

        result_table_xarray[OFFSET_DIR_MEAN_STDEV_KEY].values[j, :] = (
            this_result_dict[MEAN_PREDICTION_STDEVS_KEY]
        )
        result_table_xarray[OFFSET_DIR_BIN_EDGE_KEY].values[j, :] = (
            this_result_dict[BIN_EDGE_PREDICTION_STDEVS_KEY]
        )
        result_table_xarray[OFFSET_DIR_RMSE_KEY].values[j, :] = (
            this_result_dict[RMSE_VALUES_KEY]
        )
        result_table_xarray[OFFSET_DIR_SSREL_KEY].values[j] = (
            this_result_dict[SPREAD_SKILL_RELIABILITY_KEY]
        )
        result_table_xarray[OFFSET_DIR_SSRAT_KEY].values[j] = (
            this_result_dict[SPREAD_SKILL_RATIO_KEY]
        )
        result_table_xarray[OFFSET_DIR_EXAMPLE_COUNT_KEY].values[j, :] = (
            this_result_dict[EXAMPLE_COUNTS_KEY]
        )
        result_table_xarray[
            OFFSET_DIR_MEAN_MEAN_PREDICTION_KEY
        ].values[j, :] = this_result_dict[MEAN_MEAN_PREDICTIONS_KEY]

        result_table_xarray[OFFSET_DIR_MEAN_TARGET_KEY].values[j, :] = (
            this_result_dict[MEAN_TARGET_VALUES_KEY]
        )

    distance_indices = numpy.array(
        [TARGET_FIELD_NAMES.index(OFFSET_DISTANCE_NAME)], dtype=int
    )

    for j in distance_indices:
        print('Computing spread-skill relationship for "{0:s}"...'.format(
            TARGET_FIELD_NAMES[j]
        ))

        if (
                bin_min_offset_distance_metres is None or
                bin_max_offset_distance_metres is None
        ):
            these_stdistdevs = _get_predictive_stdistdevs(
                prediction_matrix[:, xy_indices, :]
            )

            these_bin_edges = numpy.linspace(
                numpy.percentile(
                    these_stdistdevs, bin_min_offset_distance_percentile
                ),
                numpy.percentile(
                    these_stdistdevs, bin_max_offset_distance_percentile
                ),
                num=num_offset_distance_bins + 1, dtype=float
            )[1:-1]
        else:
            these_bin_edges = numpy.linspace(
                bin_min_offset_distance_metres, bin_max_offset_distance_metres,
                num=num_offset_distance_bins + 1, dtype=float
            )[1:-1]

        this_result_dict = _get_results_euclidean(
            target_matrix=target_matrix[:, xy_indices],
            prediction_matrix=prediction_matrix[:, xy_indices, :],
            bin_edge_prediction_stdistdevs=these_bin_edges,
        )

        result_table_xarray[OFFSET_DIST_MEAN_STDIST_KEY].values[j, :] = (
            this_result_dict[MEAN_PREDICTION_STDISTDEVS_KEY]
        )
        result_table_xarray[OFFSET_DIST_BIN_EDGE_KEY].values[j, :] = (
            this_result_dict[BIN_EDGE_PREDICTION_STDISTDEVS_KEY]
        )
        result_table_xarray[OFFSET_DIST_RMSD_KEY].values[j, :] = (
            this_result_dict[RMSD_VALUES_KEY]
        )
        result_table_xarray[OFFSET_DIST_SSREL_KEY].values[j] = (
            this_result_dict[SPREAD_SKILL_RELIABILITY_KEY]
        )
        result_table_xarray[OFFSET_DIST_SSRAT_KEY].values[j] = (
            this_result_dict[SPREAD_SKILL_RATIO_KEY]
        )
        result_table_xarray[OFFSET_DIST_EXAMPLE_COUNT_KEY].values[j, :] = (
            this_result_dict[EXAMPLE_COUNTS_KEY]
        )
        result_table_xarray[
            OFFSET_DIST_MEAN_MEAN_PRED_DIST_KEY
        ].values[j, :] = this_result_dict[MEAN_MEAN_PREDICTED_DISTANCES_KEY]

        result_table_xarray[OFFSET_DIST_MEAN_TARGET_DIST_KEY].values[j, :] = (
            this_result_dict[MEAN_TARGET_DISTANCES_KEY]
        )

    return result_table_xarray


def write_results(result_table_xarray, netcdf_file_name):
    """Writes spread-vs.-skill results to NetCDF file.

    :param result_table_xarray: xarray table in format returned by
        `get_results_all_vars`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    result_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def read_results(netcdf_file_name):
    """Reads spread-vs.-skill results from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: result_table_xarray: xarray table.  Documentation in the
        xarray table should make values self-explanatory.
    """

    result_table_xarray = xarray.open_dataset(netcdf_file_name)
    if ISOTONIC_MODEL_FILE_KEY not in result_table_xarray.attrs:
        result_table_xarray.attrs[ISOTONIC_MODEL_FILE_KEY] = None

    return result_table_xarray
