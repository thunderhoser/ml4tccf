"""Methods for training and applying neural nets."""

import os
import sys
import copy
import random
import pickle
import numpy
import xarray
import keras
import tensorflow.keras as tf_keras
from pyproj import Geod

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import number_rounding
import time_conversion
import file_system_utils
import error_checking
import cira_ir_example_io
import satellite_io
import misc_utils
import satellite_utils
import image_filtering
import custom_losses_scalar
import custom_metrics_scalar
import custom_losses_gridded
import custom_metrics_gridded
import cnn_architecture
import u_net_architecture
import accum_grad_optimizer

TOLERANCE = 1e-6

DEG_LATITUDE_TO_KM = 60 * 1.852
DEGREES_TO_RADIANS = numpy.pi / 180
TARGET_DISCRETIZATION_DEG = 0.1

CIRA_IR_TIME_DIM = 'satellite_valid_time_unix_sec'
CIRA_IR_BRIGHTNESS_TEMP_KEY = 'satellite_predictors_gridded'
CIRA_IR_GRID_LATITUDE_KEY = 'satellite_grid_latitude_deg_n'
CIRA_IR_GRID_LONGITUDE_KEY = 'satellite_grid_longitude_deg_e'

METRIC_FUNCTION_LIST_SCALAR = [
    custom_losses_scalar.mean_squared_distance_kilometres2,
    custom_losses_scalar.weird_crps_kilometres2,
    custom_losses_scalar.coord_avg_crps_kilometres,
    custom_losses_scalar.discretized_mean_sq_dist_kilometres2,
    custom_losses_scalar.discretized_weird_crps_kilometres2,
    custom_losses_scalar.discretized_coord_avg_crps_kilometres,
    custom_metrics_scalar.mean_distance_kilometres,
    custom_metrics_scalar.mean_prediction,
    custom_metrics_scalar.mean_predictive_stdev,
    custom_metrics_scalar.mean_predictive_range,
    custom_metrics_scalar.mean_target,
    custom_metrics_scalar.mean_grid_spacing_kilometres,
    custom_metrics_scalar.crps_kilometres,
    custom_metrics_scalar.discretized_mean_dist_kilometres,
    custom_metrics_scalar.discretized_crps_kilometres
]

METRIC_FUNCTION_DICT_SCALAR = {
    'mean_squared_distance_kilometres2':
        custom_losses_scalar.mean_squared_distance_kilometres2,
    'weird_crps_kilometres2': custom_losses_scalar.weird_crps_kilometres2,
    'coord_avg_crps_kilometres': custom_losses_scalar.coord_avg_crps_kilometres,
    'discretized_mean_sq_dist_kilometres2':
        custom_losses_scalar.discretized_mean_sq_dist_kilometres2,
    'discretized_weird_crps_kilometres2':
        custom_losses_scalar.discretized_weird_crps_kilometres2,
    'discretized_coord_avg_crps_kilometres':
        custom_losses_scalar.discretized_coord_avg_crps_kilometres,
    'mean_distance_kilometres': custom_metrics_scalar.mean_distance_kilometres,
    'mean_prediction': custom_metrics_scalar.mean_prediction,
    'mean_predictive_stdev': custom_metrics_scalar.mean_predictive_stdev,
    'mean_predictive_range': custom_metrics_scalar.mean_predictive_range,
    'mean_target': custom_metrics_scalar.mean_target,
    'mean_grid_spacing_kilometres':
        custom_metrics_scalar.mean_grid_spacing_kilometres,
    'crps_kilometres': custom_metrics_scalar.crps_kilometres,
    'discretized_mean_dist_kilometres':
        custom_metrics_scalar.discretized_mean_dist_kilometres,
    'discretized_crps_kilometres':
        custom_metrics_scalar.discretized_crps_kilometres
}

METRIC_FUNCTION_LIST_GRIDDED = []
METRIC_FUNCTION_DICT_GRIDDED = {}

THESE_HALF_WINDOW_SIZES_PX = numpy.array([3, 6], dtype=int)
THESE_WINDOW_SIZE_STRINGS = ['7by7', '13by13']

for m in range(len(THESE_HALF_WINDOW_SIZES_PX)):
    this_function_name = 'fss_{0:s}'.format(THESE_WINDOW_SIZE_STRINGS[m])
    this_function = custom_losses_gridded.fractions_skill_score(
        half_window_size_px=THESE_HALF_WINDOW_SIZES_PX[m],
        use_as_loss_function=False, function_name=this_function_name
    )

    METRIC_FUNCTION_LIST_GRIDDED.append(this_function)
    METRIC_FUNCTION_DICT_GRIDDED[this_function_name] = this_function

    this_function_name = 'peirce_{0:s}'.format(THESE_WINDOW_SIZE_STRINGS[m])
    this_function = custom_losses_gridded.peirce_score(
        half_window_size_px=THESE_HALF_WINDOW_SIZES_PX[m],
        use_as_loss_function=False, function_name=this_function_name
    )

    METRIC_FUNCTION_LIST_GRIDDED.append(this_function)
    METRIC_FUNCTION_DICT_GRIDDED[this_function_name] = this_function

    this_function_name = 'csi_{0:s}'.format(THESE_WINDOW_SIZE_STRINGS[m])
    this_function = custom_losses_gridded.csi(
        half_window_size_px=THESE_HALF_WINDOW_SIZES_PX[m],
        use_as_loss_function=False, function_name=this_function_name
    )

    METRIC_FUNCTION_LIST_GRIDDED.append(this_function)
    METRIC_FUNCTION_DICT_GRIDDED[this_function_name] = this_function

    this_function_name = 'brier_{0:s}'.format(THESE_WINDOW_SIZE_STRINGS[m])
    this_function = custom_losses_gridded.brier_score(
        half_window_size_px=THESE_HALF_WINDOW_SIZES_PX[m],
        function_name=this_function_name
    )

    METRIC_FUNCTION_LIST_GRIDDED.append(this_function)
    METRIC_FUNCTION_DICT_GRIDDED[this_function_name] = this_function

    this_function_name = 'xentropy_{0:s}'.format(THESE_WINDOW_SIZE_STRINGS[m])
    this_function = custom_losses_gridded.cross_entropy(
        half_window_size_px=THESE_HALF_WINDOW_SIZES_PX[m],
        function_name=this_function_name
    )

    METRIC_FUNCTION_LIST_GRIDDED.append(this_function)
    METRIC_FUNCTION_DICT_GRIDDED[this_function_name] = this_function

METRIC_FUNCTION_LIST_GRIDDED += [
    custom_metrics_gridded.mean_target,
    custom_metrics_gridded.mean_prediction,
    custom_metrics_gridded.min_target,
    custom_metrics_gridded.min_prediction,
    custom_metrics_gridded.max_target,
    custom_metrics_gridded.max_prediction,
    custom_metrics_gridded.sum_of_targets,
    custom_metrics_gridded.sum_of_predictions,
    custom_metrics_gridded.mean_predictive_stdev,
    custom_metrics_gridded.mean_predictive_range,
    custom_metrics_gridded.mean_center_of_mass_row_for_targets,
    custom_metrics_gridded.mean_center_of_mass_column_for_targets,
    custom_metrics_gridded.mean_center_of_mass_row_for_predictions,
    custom_metrics_gridded.mean_center_of_mass_column_for_predictions,
    custom_metrics_gridded.mean_center_of_mass_distance_px
]

METRIC_FUNCTION_DICT_GRIDDED.update({
    'mean_target': custom_metrics_gridded.mean_target,
    'mean_prediction': custom_metrics_gridded.mean_prediction,
    'min_target': custom_metrics_gridded.min_target,
    'min_prediction': custom_metrics_gridded.min_prediction,
    'max_target': custom_metrics_gridded.max_target,
    'max_prediction': custom_metrics_gridded.max_prediction,
    'sum_of_targets': custom_metrics_gridded.sum_of_targets,
    'sum_of_predictions': custom_metrics_gridded.sum_of_predictions,
    'mean_predictive_stdev': custom_metrics_gridded.mean_predictive_stdev,
    'mean_predictive_range': custom_metrics_gridded.mean_predictive_range,
    'mean_center_of_mass_row_for_targets':
        custom_metrics_gridded.mean_center_of_mass_row_for_targets,
    'mean_center_of_mass_column_for_targets':
        custom_metrics_gridded.mean_center_of_mass_column_for_targets,
    'mean_center_of_mass_row_for_predictions':
        custom_metrics_gridded.mean_center_of_mass_row_for_predictions,
    'mean_center_of_mass_column_for_predictions':
        custom_metrics_gridded.mean_center_of_mass_column_for_predictions,
    'mean_center_of_mass_distance_px':
        custom_metrics_gridded.mean_center_of_mass_distance_px
})

METRES_TO_KM = 0.001
MINUTES_TO_SECONDS = 60
DAYS_TO_SECONDS = 86400
INTERVAL_BETWEEN_TARGET_TIMES_SEC = 21600

DATE_FORMAT = satellite_io.DATE_FORMAT
TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H%M'

SATELLITE_DIRECTORY_KEY = 'satellite_dir_name'
YEARS_KEY = 'years'
LAG_TIMES_KEY = 'lag_times_minutes'
HIGH_RES_WAVELENGTHS_KEY = 'high_res_wavelengths_microns'
LOW_RES_WAVELENGTHS_KEY = 'low_res_wavelengths_microns'
BATCH_SIZE_KEY = 'num_examples_per_batch'
MAX_EXAMPLES_PER_CYCLONE_KEY = 'max_examples_per_cyclone'
NUM_GRID_ROWS_KEY = 'num_rows_low_res'
NUM_GRID_COLUMNS_KEY = 'num_columns_low_res'
DATA_AUG_NUM_TRANS_KEY = 'data_aug_num_translations'
DATA_AUG_MEAN_TRANS_KEY = 'data_aug_mean_translation_low_res_px'
DATA_AUG_STDEV_TRANS_KEY = 'data_aug_stdev_translation_low_res_px'
LAG_TIME_TOLERANCE_KEY = 'lag_time_tolerance_sec'
MAX_MISSING_LAG_TIMES_KEY = 'max_num_missing_lag_times'
MAX_INTERP_GAP_KEY = 'max_interp_gap_sec'
SENTINEL_VALUE_KEY = 'sentinel_value'
SEMANTIC_SEG_FLAG_KEY = 'semantic_segmentation_flag'
TARGET_SMOOOTHER_STDEV_KEY = 'target_smoother_stdev_km'

DEFAULT_GENERATOR_OPTION_DICT = {
    HIGH_RES_WAVELENGTHS_KEY: None,
    LOW_RES_WAVELENGTHS_KEY: None,
    BATCH_SIZE_KEY: 8,
    MAX_EXAMPLES_PER_CYCLONE_KEY: 1,
    DATA_AUG_NUM_TRANS_KEY: 8,
    DATA_AUG_MEAN_TRANS_KEY: 15.,
    DATA_AUG_STDEV_TRANS_KEY: 7.5,
    MAX_INTERP_GAP_KEY: 0,
    SENTINEL_VALUE_KEY: -10.
}

NUM_EPOCHS_KEY = 'num_epochs'
NUM_TRAINING_BATCHES_KEY = 'num_training_batches_per_epoch'
TRAINING_OPTIONS_KEY = 'training_option_dict'
NUM_VALIDATION_BATCHES_KEY = 'num_validation_batches_per_epoch'
VALIDATION_OPTIONS_KEY = 'validation_option_dict'
LOSS_FUNCTION_KEY = 'loss_function_string'
OPTIMIZER_FUNCTION_KEY = 'optimizer_function_string'
PLATEAU_PATIENCE_KEY = 'plateau_patience_epochs'
PLATEAU_LR_MUTIPLIER_KEY = 'plateau_learning_rate_multiplier'
EARLY_STOPPING_PATIENCE_KEY = 'early_stopping_patience_epochs'
ARCHITECTURE_KEY = 'architecture_dict'
IS_MODEL_BNN_KEY = 'is_model_bnn'

METADATA_KEYS = [
    NUM_EPOCHS_KEY, NUM_TRAINING_BATCHES_KEY, TRAINING_OPTIONS_KEY,
    NUM_VALIDATION_BATCHES_KEY, VALIDATION_OPTIONS_KEY,
    LOSS_FUNCTION_KEY, OPTIMIZER_FUNCTION_KEY,
    PLATEAU_PATIENCE_KEY, PLATEAU_LR_MUTIPLIER_KEY, EARLY_STOPPING_PATIENCE_KEY,
    ARCHITECTURE_KEY, IS_MODEL_BNN_KEY
]

BIDIRECTIONAL_REFLECTANCES_KEY = 'bidirectional_reflectance_matrix'
BRIGHTNESS_TEMPS_KEY = 'brightness_temp_matrix_kelvins'
GRID_SPACINGS_KEY = 'grid_spacings_low_res_km'
CENTER_LATITUDES_KEY = 'cyclone_center_latitudes_deg_n'
TARGET_TIMES_KEY = 'target_times_unix_sec'
HIGH_RES_LATITUDES_KEY = 'high_res_latitude_matrix_deg_n'
HIGH_RES_LONGITUDES_KEY = 'high_res_longitude_matrix_deg_e'
LOW_RES_LATITUDES_KEY = 'low_res_latitude_matrix_deg_n'
LOW_RES_LONGITUDES_KEY = 'low_res_longitude_matrix_deg_e'
PREDICTOR_MATRICES_KEY = 'predictor_matrices'
TARGET_MATRIX_KEY = 'target_matrix_low_res_px'


def _get_random_signs(array_length):
    """Returns array of random signs (1 for positive, -1 for negative).

    :param array_length: Number of random signs desired.
    :return: sign_array: numpy array of integers in {-1, 1}.
    """

    return 2 * numpy.random.randint(low=0, high=2, size=array_length) - 1


def _check_generator_args(option_dict):
    """Error-checks input arguments for generator.

    :param option_dict: See doc for `data_generator`.
    :return: option_dict: Same as input, except defaults may have been added.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_GENERATOR_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    error_checking.assert_is_string(option_dict[SATELLITE_DIRECTORY_KEY])

    error_checking.assert_is_integer_numpy_array(option_dict[YEARS_KEY])
    error_checking.assert_is_numpy_array(
        option_dict[YEARS_KEY], num_dimensions=1
    )
    option_dict[YEARS_KEY] = numpy.unique(option_dict[YEARS_KEY])

    # TODO(thunderhoser): I might eventually get data with 10- or 15-min time
    # steps, which will make things more interesting.
    error_checking.assert_is_integer_numpy_array(option_dict[LAG_TIMES_KEY])
    error_checking.assert_is_geq_numpy_array(option_dict[LAG_TIMES_KEY], 0)
    assert numpy.all(numpy.mod(option_dict[LAG_TIMES_KEY], 30) == 0)
    option_dict[LAG_TIMES_KEY] = numpy.sort(option_dict[LAG_TIMES_KEY])[::-1]

    error_checking.assert_is_numpy_array(
        option_dict[HIGH_RES_WAVELENGTHS_KEY], num_dimensions=1
    )
    error_checking.assert_is_greater_numpy_array(
        option_dict[HIGH_RES_WAVELENGTHS_KEY], 0.
    )

    error_checking.assert_is_numpy_array(
        option_dict[LOW_RES_WAVELENGTHS_KEY], num_dimensions=1
    )
    error_checking.assert_is_greater_numpy_array(
        option_dict[LOW_RES_WAVELENGTHS_KEY], 0.
    )
    error_checking.assert_is_greater(
        len(option_dict[LOW_RES_WAVELENGTHS_KEY]), 0
    )

    error_checking.assert_is_integer(option_dict[BATCH_SIZE_KEY])
    error_checking.assert_is_geq(option_dict[BATCH_SIZE_KEY], 1)

    error_checking.assert_is_integer(option_dict[MAX_EXAMPLES_PER_CYCLONE_KEY])
    error_checking.assert_is_geq(option_dict[MAX_EXAMPLES_PER_CYCLONE_KEY], 1)
    error_checking.assert_is_geq(
        option_dict[BATCH_SIZE_KEY],
        option_dict[MAX_EXAMPLES_PER_CYCLONE_KEY]
    )

    error_checking.assert_is_integer(option_dict[NUM_GRID_ROWS_KEY])
    error_checking.assert_is_geq(option_dict[NUM_GRID_ROWS_KEY], 10)
    assert numpy.mod(option_dict[NUM_GRID_ROWS_KEY], 2) == 0

    error_checking.assert_is_integer(option_dict[NUM_GRID_COLUMNS_KEY])
    error_checking.assert_is_geq(option_dict[NUM_GRID_COLUMNS_KEY], 10)
    assert numpy.mod(option_dict[NUM_GRID_COLUMNS_KEY], 2) == 0

    error_checking.assert_is_integer(option_dict[DATA_AUG_NUM_TRANS_KEY])
    error_checking.assert_is_greater(option_dict[DATA_AUG_NUM_TRANS_KEY], 0)
    error_checking.assert_is_greater(option_dict[DATA_AUG_MEAN_TRANS_KEY], 0.)
    error_checking.assert_is_greater(option_dict[DATA_AUG_STDEV_TRANS_KEY], 0.)

    error_checking.assert_is_integer(option_dict[LAG_TIME_TOLERANCE_KEY])
    error_checking.assert_is_geq(option_dict[LAG_TIME_TOLERANCE_KEY], 0)
    error_checking.assert_is_integer(option_dict[MAX_MISSING_LAG_TIMES_KEY])
    error_checking.assert_is_geq(option_dict[MAX_MISSING_LAG_TIMES_KEY], 0)
    error_checking.assert_is_integer(option_dict[MAX_INTERP_GAP_KEY])
    error_checking.assert_is_geq(option_dict[MAX_INTERP_GAP_KEY], 0)
    error_checking.assert_is_not_nan(option_dict[SENTINEL_VALUE_KEY])

    error_checking.assert_is_boolean(option_dict[SEMANTIC_SEG_FLAG_KEY])

    if option_dict[SEMANTIC_SEG_FLAG_KEY]:
        error_checking.assert_is_greater(
            option_dict[TARGET_SMOOOTHER_STDEV_KEY], 0.
        )
    else:
        option_dict[TARGET_SMOOOTHER_STDEV_KEY] = None

    return option_dict


def _date_in_time_period(date_string, start_time_unix_sec, end_time_unix_sec):
    """Determines whether or not date is in time period.

    :param date_string: Date (format "yyyy-mm-dd").
    :param start_time_unix_sec: Start of time period.
    :param end_time_unix_sec: End of time period.
    :return: result_flag: Boolean flag.
    """

    start_date_string = time_conversion.unix_sec_to_string(
        start_time_unix_sec, DATE_FORMAT
    )
    if start_date_string == date_string:
        return True

    end_date_string = time_conversion.unix_sec_to_string(
        end_time_unix_sec, DATE_FORMAT
    )
    if end_date_string == date_string:
        return True

    date_start_time_unix_sec = time_conversion.string_to_unix_sec(
        date_string, DATE_FORMAT
    )
    if start_time_unix_sec <= date_start_time_unix_sec <= end_time_unix_sec:
        return True

    date_end_time_unix_sec = date_start_time_unix_sec + DAYS_TO_SECONDS - 1
    if start_time_unix_sec <= date_end_time_unix_sec <= end_time_unix_sec:
        return True

    return False


def _decide_files_to_read_one_cyclone(
        satellite_file_names, target_times_unix_sec,
        lag_times_minutes, lag_time_tolerance_sec, max_interp_gap_sec):
    """Decides which satellite files to read for one tropical cyclone.

    :param satellite_file_names: 1-D list of paths to input files (will be read
        by `satellite_io.read_file`).
    :param target_times_unix_sec: See doc for
        `_read_satellite_data_one_cyclone`.
    :param lag_times_minutes: Same.
    :param lag_time_tolerance_sec: Same.
    :param max_interp_gap_sec: Same.
    :return: desired_file_to_times_dict: Dictionary, where each key is the path
        to a desired file and the corresponding value is a length-2 list:
        [start_times_unix_sec, end_times_unix_sec].  Each list item is a 1-D
        numpy array; the two arrays have the same length; and they contain
        start/end times to be read.
    """

    lag_times_sec = MINUTES_TO_SECONDS * lag_times_minutes
    num_target_times = len(target_times_unix_sec)

    desired_start_times_unix_sec = numpy.full(num_target_times, -1, dtype=int)
    desired_end_times_unix_sec = numpy.full(num_target_times, -2, dtype=int)

    offset_sec = max([
        lag_time_tolerance_sec, max_interp_gap_sec
    ])

    for i in range(num_target_times):
        desired_start_times_unix_sec[i] = (
            target_times_unix_sec[i] - numpy.max(lag_times_sec)
        )
        desired_end_times_unix_sec[i] = (
            target_times_unix_sec[i] - numpy.min(lag_times_sec)
        )

        if desired_start_times_unix_sec[i] != target_times_unix_sec[i]:
            desired_start_times_unix_sec[i] -= offset_sec
        if desired_end_times_unix_sec[i] != target_times_unix_sec[i]:
            desired_end_times_unix_sec[i] += offset_sec

    satellite_file_date_strings = [
        satellite_io.file_name_to_date(f) for f in satellite_file_names
    ]
    desired_file_to_times_dict = dict()

    for j in range(len(satellite_file_names)):
        for i in range(num_target_times):
            if not _date_in_time_period(
                    date_string=satellite_file_date_strings[j],
                    start_time_unix_sec=desired_start_times_unix_sec[i],
                    end_time_unix_sec=desired_end_times_unix_sec[i]
            ):
                continue

            if satellite_file_names[j] not in desired_file_to_times_dict:
                desired_file_to_times_dict[satellite_file_names[j]] = [
                    desired_start_times_unix_sec[[i]],
                    desired_end_times_unix_sec[[i]]
                ]

                continue

            d = desired_file_to_times_dict

            d[satellite_file_names[j]][0] = numpy.concatenate((
                d[satellite_file_names[j]][0],
                desired_start_times_unix_sec[[i]]
            ))
            d[satellite_file_names[j]][1] = numpy.concatenate((
                d[satellite_file_names[j]][1],
                desired_end_times_unix_sec[[i]]
            ))

            desired_file_to_times_dict = d

    return desired_file_to_times_dict


def _read_brightness_temp_1file_cira_ir(
        example_table_xarray, table_rows_by_example, lag_times_sec,
        num_grid_rows=None, num_grid_columns=None):
    """Reads brightness temperatures from one CIRA IR file.

    T = number of target times
    L = number of lag times
    m = number of rows in grid
    n = number of columns in grid

    :param example_table_xarray: xarray table.  Variable names and metadata
        therein should make the table self-explanatory.
    :param table_rows_by_example: length-E list, where each element is a 1-D
        numpy array of indices to satellite times needed for the given example.
        These are row indices into `example_table_xarray`.
    :param lag_times_sec: 1-D numpy array of lag times.
    :param num_grid_rows: Number of rows to keep in grid.  If None, will keep
        all rows.
    :param num_grid_columns: Same but for columns.
    :return: brightness_temp_matrix_kelvins: T-by-m-by-n-by-L-by-1 numpy array
        of brightness temperatures.
    :return: grid_latitude_matrix_deg_n: numpy array of latitudes (deg north).
        If regular grids, array shape will be T x m x L.  If irregular grids,
        array shape will be T x m x n x L.
    :return: grid_longitude_matrix_deg_e: numpy array of longitudes (deg east).
        If regular grids, array shape will be T x n x L.  If irregular grids,
        array shape will be T x m x n x L.
    """

    xt = example_table_xarray

    num_examples = len(table_rows_by_example)
    num_lag_times = len(lag_times_sec)
    num_grid_rows_orig = xt[CIRA_IR_BRIGHTNESS_TEMP_KEY].values.shape[1]
    num_grid_columns_orig = xt[CIRA_IR_BRIGHTNESS_TEMP_KEY].values.shape[2]

    these_dim = (
        num_examples, num_grid_rows_orig, num_grid_columns_orig,
        num_lag_times, 1
    )
    brightness_temp_matrix = numpy.full(these_dim, numpy.nan)

    regular_grids = len(xt[CIRA_IR_GRID_LATITUDE_KEY].values.shape) == 2

    if regular_grids:
        grid_latitude_matrix_deg_n = numpy.full(
            (num_examples, num_grid_rows_orig, num_lag_times), numpy.nan
        )
        grid_longitude_matrix_deg_e = numpy.full(
            (num_examples, num_grid_columns_orig, num_lag_times), numpy.nan
        )
    else:
        dimensions = (
            num_examples, num_grid_rows_orig, num_grid_columns_orig,
            num_lag_times
        )
        grid_latitude_matrix_deg_n = numpy.full(dimensions, numpy.nan)
        grid_longitude_matrix_deg_e = numpy.full(dimensions, numpy.nan)

    bad_example_indices = []

    for i in range(num_examples):
        for j in range(len(lag_times_sec)):
            k = table_rows_by_example[i][j]

            try:
                these_latitudes_deg_n = (
                    xt[CIRA_IR_GRID_LATITUDE_KEY].values[k, ...]
                )
                these_longitudes_deg_e = (
                    xt[CIRA_IR_GRID_LONGITUDE_KEY].values[k, ...]
                )

                if regular_grids:
                    assert misc_utils.is_regular_grid_valid(
                        latitudes_deg_n=these_latitudes_deg_n,
                        longitudes_deg_e=these_longitudes_deg_e
                    )[0]
            except:
                bad_example_indices.append(i)

            if i in bad_example_indices:
                break

            grid_latitude_matrix_deg_n[i, ..., j] = these_latitudes_deg_n
            grid_longitude_matrix_deg_e[i, ..., j] = these_longitudes_deg_e
            brightness_temp_matrix[i, ..., j, 0] = (
                xt[CIRA_IR_BRIGHTNESS_TEMP_KEY].values[k, ..., 0]
            )

    good_example_flags = numpy.full(num_examples, True, dtype=bool)
    good_example_flags[bad_example_indices] = False
    good_example_indices = numpy.where(good_example_flags)[0]

    brightness_temp_matrix = brightness_temp_matrix[good_example_indices, ...]
    grid_latitude_matrix_deg_n = grid_latitude_matrix_deg_n[
        good_example_indices, ...
    ]
    grid_longitude_matrix_deg_e = grid_longitude_matrix_deg_e[
        good_example_indices, ...
    ]

    if num_grid_rows is not None:
        error_checking.assert_is_less_than(num_grid_rows, num_grid_rows_orig)

        first_index = int(numpy.round(
            num_grid_rows_orig / 2 - num_grid_rows / 2
        ))
        last_index = int(numpy.round(
            num_grid_rows_orig / 2 + num_grid_rows / 2
        ))

        brightness_temp_matrix = (
            brightness_temp_matrix[:, first_index:last_index, ...]
        )
        grid_latitude_matrix_deg_n = (
            grid_latitude_matrix_deg_n[:, first_index:last_index, ...]
        )
        grid_longitude_matrix_deg_e = (
            grid_longitude_matrix_deg_e[:, first_index:last_index, ...]
        )

    if num_grid_columns is not None:
        error_checking.assert_is_less_than(
            num_grid_columns, num_grid_columns_orig
        )

        first_index = int(numpy.round(
            num_grid_columns_orig / 2 - num_grid_columns / 2
        ))
        last_index = int(numpy.round(
            num_grid_columns_orig / 2 + num_grid_columns / 2
        ))

        brightness_temp_matrix = (
            brightness_temp_matrix[:, :, first_index:last_index, ...]
        )
        grid_latitude_matrix_deg_n = (
            grid_latitude_matrix_deg_n[:, :, first_index:last_index, ...]
        )
        grid_longitude_matrix_deg_e = (
            grid_longitude_matrix_deg_e[:, :, first_index:last_index, ...]
        )

    return (
        brightness_temp_matrix, grid_latitude_matrix_deg_n,
        grid_longitude_matrix_deg_e
    )


def _read_satellite_data_1cyclone_cira_ir(
        input_file_name, lag_times_minutes, num_grid_rows, num_grid_columns,
        return_coords, num_target_times=None, target_times_unix_sec=None):
    """Reads satellite data for one cyclone from CIRA IR dataset.

    :param input_file_name: See doc for `_read_satellite_data_one_cyclone`.
    :param lag_times_minutes: Same.
    :param num_grid_rows: Same.
    :param num_grid_columns: Same.
    :param return_coords: Same.
    :param num_target_times: Same.
    :param target_times_unix_sec: Same.
    :return: data_dict: Same.
    :raises: ValueError: if `target_times_unix_sec is not None` and any desired
        time cannot be found.
    """

    # lag_times_minutes = numpy.sort(lag_times_minutes)[::-1]  # Already done in _check_generator_args.
    lag_times_sec = lag_times_minutes * MINUTES_TO_SECONDS

    need_exact_target_times = target_times_unix_sec is not None

    all_times_unix_sec = cira_ir_example_io.read_file(input_file_name).coords[
        CIRA_IR_TIME_DIM
    ].values

    if not need_exact_target_times:
        first_synoptic_times_unix_sec = number_rounding.floor_to_nearest(
            all_times_unix_sec, INTERVAL_BETWEEN_TARGET_TIMES_SEC
        )
        second_synoptic_times_unix_sec = number_rounding.ceiling_to_nearest(
            all_times_unix_sec, INTERVAL_BETWEEN_TARGET_TIMES_SEC
        )
        synoptic_times_unix_sec = numpy.concatenate((
            first_synoptic_times_unix_sec, second_synoptic_times_unix_sec
        ))

        good_indices = numpy.array([
            numpy.argmin(numpy.absolute(st - all_times_unix_sec))
            for st in synoptic_times_unix_sec
        ], dtype=int)

        time_diffs_sec = numpy.absolute(
            all_times_unix_sec[good_indices] - synoptic_times_unix_sec
        )
        good_subindices = numpy.where(time_diffs_sec <= 900)[0]
        good_indices = good_indices[good_subindices]
        target_times_unix_sec = all_times_unix_sec[good_indices]

        if len(target_times_unix_sec) == 0:
            return None

    orig_target_times_unix_sec = target_times_unix_sec + 0
    target_times_unix_sec = []
    table_rows_by_target_time = []

    for i in range(len(orig_target_times_unix_sec)):
        these_predictor_times_unix_sec = (
            orig_target_times_unix_sec[i] - lag_times_sec
        )

        if not numpy.all(numpy.isin(
                element=these_predictor_times_unix_sec,
                test_elements=all_times_unix_sec
        )):
            if not need_exact_target_times:
                continue

            missing_flags = numpy.invert(numpy.isin(
                element=these_predictor_times_unix_sec,
                test_elements=all_times_unix_sec
            ))
            missing_time_strings = [
                time_conversion.unix_sec_to_string(
                    t, TIME_FORMAT_FOR_LOG_MESSAGES
                )
                for t in these_predictor_times_unix_sec[missing_flags]
            ]

            error_string = (
                'File "{0:s}" is missing the following times:\n{1:s}'
            ).format(
                input_file_name, str(missing_time_strings)
            )

            raise ValueError(error_string)

        these_rows = numpy.array([
            numpy.where(all_times_unix_sec == t)[0][0]
            for t in these_predictor_times_unix_sec
        ], dtype=int)

        target_times_unix_sec.append(orig_target_times_unix_sec[i])
        table_rows_by_target_time.append(these_rows)

    target_times_unix_sec = numpy.array(target_times_unix_sec, dtype=int)

    if (
            not need_exact_target_times
            and num_target_times < len(target_times_unix_sec)
    ):
        example_indices = numpy.linspace(
            0, len(target_times_unix_sec) - 1, num=len(target_times_unix_sec),
            dtype=int
        )
        example_indices = numpy.random.choice(
            example_indices, size=num_target_times, replace=False
        )

        target_times_unix_sec = target_times_unix_sec[example_indices]
        table_rows_by_target_time = [
            table_rows_by_target_time[k] for k in example_indices
        ]

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    example_table_xarray = cira_ir_example_io.read_file(input_file_name)

    (
        brightness_temp_matrix_kelvins,
        grid_latitude_matrix_deg_n,
        grid_longitude_matrix_deg_e
    ) = _read_brightness_temp_1file_cira_ir(
        example_table_xarray=example_table_xarray,
        table_rows_by_example=table_rows_by_target_time,
        lag_times_sec=lag_times_sec,
        num_grid_rows=num_grid_rows,
        num_grid_columns=num_grid_columns
    )

    assert not numpy.any(numpy.isnan(brightness_temp_matrix_kelvins))
    regular_grids = len(grid_latitude_matrix_deg_n.shape) == 3

    i_start = int(numpy.round(
        float(num_grid_rows) / 2 - 1
    ))
    i_end = i_start + 1
    j_start = int(numpy.round(
        float(num_grid_columns) / 2 - 1
    ))
    j_end = j_start + 1

    num_examples = brightness_temp_matrix_kelvins.shape[0]
    grid_spacings_km = numpy.full(num_examples, numpy.nan)
    cyclone_center_latitudes_deg_n = numpy.full(num_examples, numpy.nan)

    geodesic_object = Geod(ellps='WGS84')

    for i in range(num_examples):
        if regular_grids:
            start_latitude_deg_n = grid_latitude_matrix_deg_n[i, i_start, -1]
            end_latitude_deg_n = grid_latitude_matrix_deg_n[i, i_end, -1]
            start_longitude_deg_e = grid_longitude_matrix_deg_e[i, j_start, -1]
            end_longitude_deg_e = grid_longitude_matrix_deg_e[i, j_end, -1]
        else:
            start_latitude_deg_n = grid_latitude_matrix_deg_n[
                i, i_start, j_start, -1
            ]
            end_latitude_deg_n = grid_latitude_matrix_deg_n[
                i, i_end, j_start, -1
            ]
            start_longitude_deg_e = grid_longitude_matrix_deg_e[
                i, i_start, j_start, -1
            ]
            end_longitude_deg_e = grid_longitude_matrix_deg_e[
                i, i_start, j_end, -1
            ]

        first_distance_metres = geodesic_object.inv(
            lons1=start_longitude_deg_e, lats1=start_latitude_deg_n,
            lons2=start_longitude_deg_e, lats2=end_latitude_deg_n
        )[2]
        second_distance_metres = geodesic_object.inv(
            lons1=start_longitude_deg_e, lats1=start_latitude_deg_n,
            lons2=end_longitude_deg_e, lats2=start_latitude_deg_n
        )[2]
        grid_spacings_km[i] = (
            0.5 * METRES_TO_KM *
            (first_distance_metres + second_distance_metres)
        )

        cyclone_center_latitudes_deg_n[i] = (
            0.5 * (start_latitude_deg_n + end_latitude_deg_n)
        )

    return {
        BIDIRECTIONAL_REFLECTANCES_KEY: None,
        BRIGHTNESS_TEMPS_KEY: brightness_temp_matrix_kelvins,
        GRID_SPACINGS_KEY: grid_spacings_km,
        CENTER_LATITUDES_KEY: cyclone_center_latitudes_deg_n,
        TARGET_TIMES_KEY: target_times_unix_sec,
        HIGH_RES_LATITUDES_KEY: None,
        HIGH_RES_LONGITUDES_KEY: None,
        LOW_RES_LATITUDES_KEY:
            grid_latitude_matrix_deg_n if return_coords else None,
        LOW_RES_LONGITUDES_KEY:
            grid_longitude_matrix_deg_e if return_coords else None
    }


def _read_satellite_data_1cyclone_simple(
        input_file_names, lag_times_minutes, num_rows_low_res,
        num_columns_low_res, return_coords, num_target_times=None,
        target_times_unix_sec=None):
    """Reads satellite data for one cyclone.

    :param input_file_names: See doc for `_read_satellite_data_one_cyclone`.
    :param lag_times_minutes: Same.
    :param low_res_wavelengths_microns: Same.
    :param num_rows_low_res: Same.
    :param num_columns_low_res: Same.
    :param return_coords: Same.
    :param num_target_times: Same.
    :param target_times_unix_sec: Same.
    :return: data_dict: Same.
    """

    if num_target_times is not None:
        target_times_unix_sec = numpy.concatenate([
            xarray.open_zarr(f).coords[satellite_utils.TIME_DIM].values
            for f in input_file_names
        ])
        target_times_unix_sec = target_times_unix_sec[
            numpy.mod(target_times_unix_sec, INTERVAL_BETWEEN_TARGET_TIMES_SEC)
            == 0
        ]

        if len(target_times_unix_sec) == 0:
            return None

        if num_target_times < len(target_times_unix_sec):
            target_times_unix_sec = numpy.random.choice(
                target_times_unix_sec, size=num_target_times, replace=False
            )

    # TODO(thunderhoser): This could be simplified more.
    desired_file_to_times_dict = _decide_files_to_read_one_cyclone(
        satellite_file_names=input_file_names,
        target_times_unix_sec=target_times_unix_sec,
        lag_times_minutes=lag_times_minutes,
        lag_time_tolerance_sec=0,
        max_interp_gap_sec=0
    )

    desired_file_names = list(desired_file_to_times_dict.keys())
    num_files = len(desired_file_names)
    orig_satellite_tables_xarray = [None] * num_files

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(desired_file_names[i]))
        orig_satellite_tables_xarray[i] = satellite_io.read_file(
            desired_file_names[i]
        )

        # TODO(thunderhoser): This could be simplified more.
        orig_satellite_tables_xarray[i] = (
            satellite_utils.subset_to_multiple_time_windows(
                satellite_table_xarray=orig_satellite_tables_xarray[i],
                start_times_unix_sec=
                desired_file_to_times_dict[desired_file_names[i]][0],
                end_times_unix_sec=
                desired_file_to_times_dict[desired_file_names[i]][1]
            )
        )

        orig_satellite_tables_xarray[i] = satellite_utils.subset_grid(
            satellite_table_xarray=orig_satellite_tables_xarray[i],
            num_rows_to_keep=num_rows_low_res,
            num_columns_to_keep=num_columns_low_res,
            for_high_res=False
        )

    satellite_table_xarray = satellite_utils.concat_over_time(
        orig_satellite_tables_xarray
    )
    del orig_satellite_tables_xarray

    num_target_times = len(target_times_unix_sec)
    num_lag_times = len(lag_times_minutes)

    t = satellite_table_xarray
    this_num_rows = (
        t[satellite_utils.BRIGHTNESS_TEMPERATURE_KEY].values.shape[1]
    )
    this_num_columns = (
        t[satellite_utils.BRIGHTNESS_TEMPERATURE_KEY].values.shape[2]
    )
    this_num_wavelengths = (
        t[satellite_utils.BRIGHTNESS_TEMPERATURE_KEY].values.shape[3]
    )
    these_dim = (
        num_target_times, this_num_rows, this_num_columns, num_lag_times,
        this_num_wavelengths
    )
    brightness_temp_matrix_kelvins = numpy.full(these_dim, numpy.nan)

    if return_coords:
        low_res_latitude_matrix_deg_n = numpy.full(
            (num_target_times, this_num_rows, num_lag_times), numpy.nan
        )
        low_res_longitude_matrix_deg_e = numpy.full(
            (num_target_times, this_num_columns, num_lag_times), numpy.nan
        )
    else:
        low_res_latitude_matrix_deg_n = None
        low_res_longitude_matrix_deg_e = None

    grid_spacings_low_res_km = numpy.full(num_target_times, numpy.nan)
    cyclone_center_latitudes_deg_n = numpy.full(num_target_times, numpy.nan)

    lag_times_sec = MINUTES_TO_SECONDS * lag_times_minutes
    target_time_success_flags = numpy.full(num_target_times, 0, dtype=bool)

    for i in range(num_target_times):
        print('Finding satellite data for target time {0:s}...'.format(
            time_conversion.unix_sec_to_string(
                target_times_unix_sec[i], TIME_FORMAT_FOR_LOG_MESSAGES
            )
        ))

        these_desired_times_unix_sec = numpy.sort(
            target_times_unix_sec[i] - lag_times_sec
        )
        these_tolerances_sec = numpy.full(
            len(these_desired_times_unix_sec), 0, dtype=int
        )
        these_max_gaps_sec = numpy.full(
            len(these_desired_times_unix_sec), 0, dtype=int
        )

        try:
            # TODO(thunderhoser): This could be simplified more.
            new_table_xarray = satellite_utils.subset_times(
                satellite_table_xarray=satellite_table_xarray,
                desired_times_unix_sec=these_desired_times_unix_sec,
                tolerances_sec=these_tolerances_sec,
                max_num_missing_times=0,
                max_interp_gaps_sec=these_max_gaps_sec
            )
            target_time_success_flags[i] = True
        except ValueError:
            continue

        t = new_table_xarray
        this_bt_matrix_kelvins = numpy.swapaxes(
            t[satellite_utils.BRIGHTNESS_TEMPERATURE_KEY].values, 0, 1
        )
        brightness_temp_matrix_kelvins[i, ...] = numpy.swapaxes(
            this_bt_matrix_kelvins, 1, 2
        )

        these_x_diffs_metres = numpy.diff(
            t[satellite_utils.X_COORD_LOW_RES_KEY].values[-1, :]
        )
        these_y_diffs_metres = numpy.diff(
            t[satellite_utils.Y_COORD_LOW_RES_KEY].values[-1, :]
        )
        grid_spacings_low_res_km[i] = METRES_TO_KM * numpy.mean(
            numpy.concatenate((these_x_diffs_metres, these_y_diffs_metres))
        )

        cyclone_center_latitudes_deg_n[i] = numpy.median(
            t[satellite_utils.LATITUDE_LOW_RES_KEY].values[-1, :]
        )

        if return_coords:
            low_res_latitude_matrix_deg_n[i, ...] = numpy.swapaxes(
                t[satellite_utils.LATITUDE_LOW_RES_KEY].values, 0, 1
            )
            low_res_longitude_matrix_deg_e[i, ...] = numpy.swapaxes(
                t[satellite_utils.LONGITUDE_LOW_RES_KEY].values, 0, 1
            )

    good_indices = numpy.where(target_time_success_flags)[0]
    target_times_unix_sec = target_times_unix_sec[good_indices]
    brightness_temp_matrix_kelvins = (
        brightness_temp_matrix_kelvins[good_indices, ...]
    )
    grid_spacings_low_res_km = grid_spacings_low_res_km[good_indices]
    cyclone_center_latitudes_deg_n = (
        cyclone_center_latitudes_deg_n[good_indices]
    )

    assert not numpy.any(numpy.isnan(brightness_temp_matrix_kelvins))

    if return_coords:
        low_res_latitude_matrix_deg_n = low_res_latitude_matrix_deg_n[
            good_indices, ...
        ]
        low_res_longitude_matrix_deg_e = low_res_longitude_matrix_deg_e[
            good_indices, ...
        ]

    return {
        BIDIRECTIONAL_REFLECTANCES_KEY: None,
        BRIGHTNESS_TEMPS_KEY: brightness_temp_matrix_kelvins,
        GRID_SPACINGS_KEY: grid_spacings_low_res_km,
        CENTER_LATITUDES_KEY: cyclone_center_latitudes_deg_n,
        TARGET_TIMES_KEY: target_times_unix_sec,
        HIGH_RES_LATITUDES_KEY: None,
        HIGH_RES_LONGITUDES_KEY: None,
        LOW_RES_LATITUDES_KEY: low_res_latitude_matrix_deg_n,
        LOW_RES_LONGITUDES_KEY: low_res_longitude_matrix_deg_e
    }


def _read_satellite_data_one_cyclone(
        input_file_names, lag_times_minutes, lag_time_tolerance_sec,
        max_num_missing_lag_times, max_interp_gap_sec,
        high_res_wavelengths_microns, low_res_wavelengths_microns,
        num_rows_low_res, num_columns_low_res, sentinel_value, return_coords,
        num_target_times=None, target_times_unix_sec=None):
    """Reads satellite data for one cyclone.

    T = number of target times
    L = number of lag times
    W = number of high-res wavelengths
    w = number of low-res wavelengths
    M = number of rows in high-res grid
    m = number of rows in low-res grid = M/4
    N = number of columns in high-res grid
    n = number of columns in low-res grid = M/4

    Only one of the arguments `num_target_times` and `target_times_unix_sec`
    will be used.

    :param input_file_names: 1-D list of paths to input files (will be read by
        `satellite_io.read_file`).
    :param lag_times_minutes: length-L numpy array of lag times for predictors.
    :param lag_time_tolerance_sec: Tolerance for lag times (tolerance for target
        time is always zero).
    :param max_num_missing_lag_times: Max number of missing lag times (i.e.,
        predictor times) per example.
    :param max_interp_gap_sec: Maximum gap (in seconds) over which to
        interpolate for missing lag times.
    :param high_res_wavelengths_microns: length-W numpy array of desired
        wavelengths for high-resolution data.
    :param low_res_wavelengths_microns: length-w numpy array of desired
        wavelengths for low-resolution data.
    :param num_rows_low_res: m in the above discussion.
    :param num_columns_low_res: n in the above discussion.
    :param sentinel_value: NaN's will be replaced with this value.
    :param return_coords: Boolean flag.  If True, will return coordinates.
    :param num_target_times: T in the above definitions.  If this argument is
        None, `target_times_unix_sec` will be used, instead.
    :param target_times_unix_sec: length-T numpy array of target times.  If this
        argument is None, `num_target_times` will be used, instead.

    :return: data_dict: Dictionary with the following keys.  If
        `return_coords == False`, the last 4 keys will be None.

    data_dict["bidirectional_reflectance_matrix"]: T-by-M-by-N-by-L-by-W numpy
        array of reflectance values (unitless).
    data_dict["brightness_temp_matrix_kelvins"]: T-by-m-by-n-by-L-by-w numpy
        array of brightness temperatures.
    data_dict["grid_spacings_low_res_km"]: length-T numpy array of grid spacings
        for low-resolution data.
    data_dict["cyclone_center_latitudes_deg_n"]: length-T numpy array of
        center latitudes (deg north).
    data_dict["target_times_unix_sec"]: length-T numpy array of target times.
    data_dict["high_res_latitude_matrix_deg_n"]: T-by-M-by-L numpy array of
        latitudes (deg north).
    data_dict["high_res_longitude_matrix_deg_e"]: T-by-N-by-L numpy array of
        longitudes (deg east).
    data_dict["low_res_latitude_matrix_deg_n"]: T-by-m-by-L numpy array of
        latitudes (deg north).
    data_dict["low_res_longitude_matrix_deg_e"]: T-by-n-by-L numpy array of
        longitudes (deg east).
    """

    # TODO(thunderhoser): If I ever start using visible data, I will need to
    # choose target times during the day here.

    if num_target_times is not None:
        target_times_unix_sec = numpy.concatenate([
            xarray.open_zarr(f).coords[satellite_utils.TIME_DIM].values
            for f in input_file_names
        ])
        target_times_unix_sec = target_times_unix_sec[
            numpy.mod(
                target_times_unix_sec, INTERVAL_BETWEEN_TARGET_TIMES_SEC
            ) == 0
        ]

        if len(target_times_unix_sec) == 0:
            return None

        if num_target_times < len(target_times_unix_sec):
            target_times_unix_sec = numpy.random.choice(
                target_times_unix_sec, size=num_target_times, replace=False
            )

    desired_file_to_times_dict = _decide_files_to_read_one_cyclone(
        satellite_file_names=input_file_names,
        target_times_unix_sec=target_times_unix_sec,
        lag_times_minutes=lag_times_minutes,
        lag_time_tolerance_sec=lag_time_tolerance_sec,
        max_interp_gap_sec=max_interp_gap_sec
    )

    desired_file_names = list(desired_file_to_times_dict.keys())
    num_files = len(desired_file_names)
    orig_satellite_tables_xarray = [None] * num_files

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(desired_file_names[i]))
        orig_satellite_tables_xarray[i] = satellite_io.read_file(
            desired_file_names[i]
        )

        orig_satellite_tables_xarray[i] = (
            satellite_utils.subset_to_multiple_time_windows(
                satellite_table_xarray=orig_satellite_tables_xarray[i],
                start_times_unix_sec=
                desired_file_to_times_dict[desired_file_names[i]][0],
                end_times_unix_sec=
                desired_file_to_times_dict[desired_file_names[i]][1]
            )
        )

        orig_satellite_tables_xarray[i] = satellite_utils.subset_wavelengths(
            satellite_table_xarray=orig_satellite_tables_xarray[i],
            wavelengths_to_keep_microns=low_res_wavelengths_microns,
            for_high_res=False
        )

        if satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY in list(
                orig_satellite_tables_xarray[i].data_vars
        ):
            orig_satellite_tables_xarray[i] = (
                satellite_utils.subset_wavelengths(
                    satellite_table_xarray=orig_satellite_tables_xarray[i],
                    wavelengths_to_keep_microns=high_res_wavelengths_microns,
                    for_high_res=True
                )
            )

        orig_satellite_tables_xarray[i] = satellite_utils.subset_grid(
            satellite_table_xarray=orig_satellite_tables_xarray[i],
            num_rows_to_keep=num_rows_low_res,
            num_columns_to_keep=num_columns_low_res,
            for_high_res=False
        )

        if len(high_res_wavelengths_microns) > 0:
            orig_satellite_tables_xarray[i] = satellite_utils.subset_grid(
                satellite_table_xarray=orig_satellite_tables_xarray[i],
                num_rows_to_keep=4 * num_rows_low_res,
                num_columns_to_keep=4 * num_columns_low_res,
                for_high_res=True
            )

    satellite_table_xarray = satellite_utils.concat_over_time(
        orig_satellite_tables_xarray
    )
    del orig_satellite_tables_xarray

    num_target_times = len(target_times_unix_sec)
    num_lag_times = len(lag_times_minutes)

    t = satellite_table_xarray
    this_num_rows = (
        t[satellite_utils.BRIGHTNESS_TEMPERATURE_KEY].values.shape[1]
    )
    this_num_columns = (
        t[satellite_utils.BRIGHTNESS_TEMPERATURE_KEY].values.shape[2]
    )
    this_num_wavelengths = len(low_res_wavelengths_microns)
    these_dim = (
        num_target_times, this_num_rows, this_num_columns, num_lag_times,
        this_num_wavelengths
    )
    brightness_temp_matrix_kelvins = numpy.full(these_dim, numpy.nan)

    if return_coords:
        low_res_latitude_matrix_deg_n = numpy.full(
            (num_target_times, this_num_rows, num_lag_times), numpy.nan
        )
        low_res_longitude_matrix_deg_e = numpy.full(
            (num_target_times, this_num_columns, num_lag_times), numpy.nan
        )
    else:
        low_res_latitude_matrix_deg_n = None
        low_res_longitude_matrix_deg_e = None

    if len(high_res_wavelengths_microns) > 0:
        t = satellite_table_xarray
        this_num_rows = (
            t[satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY].values.shape[1]
        )
        this_num_columns = (
            t[satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY].values.shape[2]
        )
        this_num_wavelengths = len(high_res_wavelengths_microns)

        these_dim = (
            num_target_times, this_num_rows, this_num_columns, num_lag_times,
            this_num_wavelengths
        )
        bidirectional_reflectance_matrix = numpy.full(these_dim, numpy.nan)

        if return_coords:
            high_res_latitude_matrix_deg_n = numpy.full(
                (num_target_times, this_num_rows, num_lag_times), numpy.nan
            )
            high_res_longitude_matrix_deg_e = numpy.full(
                (num_target_times, this_num_columns, num_lag_times), numpy.nan
            )
        else:
            high_res_latitude_matrix_deg_n = None
            high_res_longitude_matrix_deg_e = None
    else:
        bidirectional_reflectance_matrix = None
        high_res_latitude_matrix_deg_n = None
        high_res_longitude_matrix_deg_e = None

    grid_spacings_low_res_km = numpy.full(num_target_times, numpy.nan)
    cyclone_center_latitudes_deg_n = numpy.full(num_target_times, numpy.nan)

    lag_times_sec = MINUTES_TO_SECONDS * lag_times_minutes
    target_time_success_flags = numpy.full(num_target_times, 0, dtype=bool)

    for i in range(num_target_times):
        print('Finding satellite data for target time {0:s}...'.format(
            time_conversion.unix_sec_to_string(
                target_times_unix_sec[i], TIME_FORMAT_FOR_LOG_MESSAGES
            )
        ))

        these_desired_times_unix_sec = numpy.sort(
            target_times_unix_sec[i] - lag_times_sec
        )
        these_tolerances_sec = numpy.full(
            len(these_desired_times_unix_sec), lag_time_tolerance_sec
        )
        these_tolerances_sec[
            these_desired_times_unix_sec == target_times_unix_sec[i]
        ] = 0
        these_max_gaps_sec = numpy.full(
            len(these_desired_times_unix_sec), max_interp_gap_sec
        )
        these_max_gaps_sec[
            these_desired_times_unix_sec == target_times_unix_sec[i]
        ] = 0

        try:
            new_table_xarray = satellite_utils.subset_times(
                satellite_table_xarray=satellite_table_xarray,
                desired_times_unix_sec=these_desired_times_unix_sec,
                tolerances_sec=these_tolerances_sec,
                max_num_missing_times=max_num_missing_lag_times,
                max_interp_gaps_sec=these_max_gaps_sec
            )
            target_time_success_flags[i] = True
        except ValueError:
            continue

        t = new_table_xarray
        this_bt_matrix_kelvins = numpy.swapaxes(
            t[satellite_utils.BRIGHTNESS_TEMPERATURE_KEY].values, 0, 1
        )
        brightness_temp_matrix_kelvins[i, ...] = numpy.swapaxes(
            this_bt_matrix_kelvins, 1, 2
        )

        these_x_diffs_metres = numpy.diff(
            t[satellite_utils.X_COORD_LOW_RES_KEY].values[-1, :]
        )
        these_y_diffs_metres = numpy.diff(
            t[satellite_utils.Y_COORD_LOW_RES_KEY].values[-1, :]
        )
        grid_spacings_low_res_km[i] = METRES_TO_KM * numpy.mean(
            numpy.concatenate((these_x_diffs_metres, these_y_diffs_metres))
        )

        cyclone_center_latitudes_deg_n[i] = numpy.median(
            t[satellite_utils.LATITUDE_LOW_RES_KEY].values[-1, :]
        )

        if return_coords:
            low_res_latitude_matrix_deg_n[i, ...] = numpy.swapaxes(
                t[satellite_utils.LATITUDE_LOW_RES_KEY].values, 0, 1
            )
            low_res_longitude_matrix_deg_e[i, ...] = numpy.swapaxes(
                t[satellite_utils.LONGITUDE_LOW_RES_KEY].values, 0, 1
            )

        if len(high_res_wavelengths_microns) == 0:
            continue

        this_refl_matrix_kelvins = numpy.swapaxes(
            t[satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY].values, 0, 1
        )
        bidirectional_reflectance_matrix[i, ...] = numpy.swapaxes(
            this_refl_matrix_kelvins, 1, 2
        )

        if return_coords:
            high_res_latitude_matrix_deg_n[i, ...] = numpy.swapaxes(
                t[satellite_utils.LATITUDE_HIGH_RES_KEY].values, 0, 1
            )
            high_res_longitude_matrix_deg_e[i, ...] = numpy.swapaxes(
                t[satellite_utils.LONGITUDE_HIGH_RES_KEY].values, 0, 1
            )

    good_indices = numpy.where(target_time_success_flags)[0]
    target_times_unix_sec = target_times_unix_sec[good_indices]

    if len(high_res_wavelengths_microns) > 0:
        bidirectional_reflectance_matrix = (
            bidirectional_reflectance_matrix[good_indices, ...]
        )
        bidirectional_reflectance_matrix[
            numpy.isnan(bidirectional_reflectance_matrix)
        ] = sentinel_value

        if return_coords:
            high_res_latitude_matrix_deg_n = high_res_latitude_matrix_deg_n[
                good_indices, ...
            ]
            high_res_longitude_matrix_deg_e = high_res_longitude_matrix_deg_e[
                good_indices, ...
            ]

    brightness_temp_matrix_kelvins = (
        brightness_temp_matrix_kelvins[good_indices, ...]
    )
    brightness_temp_matrix_kelvins[
        numpy.isnan(brightness_temp_matrix_kelvins)
    ] = sentinel_value

    grid_spacings_low_res_km = grid_spacings_low_res_km[good_indices]
    cyclone_center_latitudes_deg_n = (
        cyclone_center_latitudes_deg_n[good_indices]
    )

    if return_coords:
        low_res_latitude_matrix_deg_n = low_res_latitude_matrix_deg_n[
            good_indices, ...
        ]
        low_res_longitude_matrix_deg_e = low_res_longitude_matrix_deg_e[
            good_indices, ...
        ]

    return {
        BIDIRECTIONAL_REFLECTANCES_KEY: bidirectional_reflectance_matrix,
        BRIGHTNESS_TEMPS_KEY: brightness_temp_matrix_kelvins,
        GRID_SPACINGS_KEY: grid_spacings_low_res_km,
        CENTER_LATITUDES_KEY: cyclone_center_latitudes_deg_n,
        TARGET_TIMES_KEY: target_times_unix_sec,
        HIGH_RES_LATITUDES_KEY: high_res_latitude_matrix_deg_n,
        HIGH_RES_LONGITUDES_KEY: high_res_longitude_matrix_deg_e,
        LOW_RES_LATITUDES_KEY: low_res_latitude_matrix_deg_n,
        LOW_RES_LONGITUDES_KEY: low_res_longitude_matrix_deg_e
    }


# def _translate_images_fast_attempt(
#         image_matrix, row_translation_px, column_translation_px, padding_value):
#     """Translates set of images in both the x- and y-directions.
#
#     :param image_matrix: numpy array, where the second (third) axis is the row
#         (column) dimension.
#     :param row_translation_px: Will translate each image by this many rows.
#     :param column_translation_px: Will translate each image by this many
#         columns.
#     :param padding_value: Padded pixels will be filled with this value.
#     :return: translated_image_matrix: Same as input but after translation.
#     """
#
#     transform_object = skimage.transform.AffineTransform(
#         translation=[column_translation_px, row_translation_px]
#     )
#
#     num_examples = image_matrix.shape[0]
#     num_channels = image_matrix.shape[3]
#     translated_image_matrix = image_matrix + 0.
#
#     for i in range(num_examples):
#         for j in range(num_channels):
#             translated_image_matrix[i, ..., j] = skimage.transform.warp(
#                 translated_image_matrix[i, ..., j], transform_object.inverse
#             )
#
#     return translated_image_matrix


def _translate_images(image_matrix, row_translation_px, column_translation_px,
                      padding_value):
    """Translates set of images in both the x- and y-directions.

    :param image_matrix: numpy array, where the second (third) axis is the row
        (column) dimension.
    :param row_translation_px: Will translate each image by this many rows.
    :param column_translation_px: Will translate each image by this many
        columns.
    :param padding_value: Padded pixels will be filled with this value.
    :return: translated_image_matrix: Same as input but after translation.
    """

    num_rows = image_matrix.shape[1]
    num_columns = image_matrix.shape[2]

    num_padded_columns_at_left = max([column_translation_px, 0])
    num_padded_columns_at_right = max([-column_translation_px, 0])
    num_padded_rows_at_top = max([row_translation_px, 0])
    num_padded_rows_at_bottom = max([-row_translation_px, 0])

    padding_arg = (
        (0, 0),
        (num_padded_rows_at_top, num_padded_rows_at_bottom),
        (num_padded_columns_at_left, num_padded_columns_at_right)
    )

    num_dimensions = len(image_matrix.shape)
    for _ in range(3, num_dimensions):
        padding_arg += ((0, 0),)

    translated_image_matrix = numpy.pad(
        image_matrix, pad_width=padding_arg, mode='constant',
        constant_values=padding_value
    )

    if column_translation_px >= 0:
        translated_image_matrix = (
            translated_image_matrix[:, :, :num_columns, ...]
        )
    else:
        translated_image_matrix = (
            translated_image_matrix[:, :, -num_columns:, ...]
        )

    if row_translation_px >= 0:
        translated_image_matrix = translated_image_matrix[:, :num_rows, ...]
    else:
        translated_image_matrix = translated_image_matrix[:, -num_rows:, ...]

    return translated_image_matrix


def _augment_data(
        bidirectional_reflectance_matrix, brightness_temp_matrix_kelvins,
        num_translations_per_example, mean_translation_low_res_px,
        stdev_translation_low_res_px, sentinel_value):
    """Augments data via translation.

    E = number of examples
    T = number of translations per example
    L = number of lag times
    W = number of high-res wavelengths
    w = number of low-res wavelengths
    M = number of rows in high-res grid
    m = number of rows in low-res grid = M/4
    N = number of columns in high-res grid
    n = number of columns in low-res grid = M/4

    :param bidirectional_reflectance_matrix: E-by-M-by-N-by-L-by-W numpy array
        of reflectance values (unitless).  This may also be None.
    :param brightness_temp_matrix_kelvins: E-by-m-by-n-by-L-by-w numpy array
        of brightness temperatures.
    :param num_translations_per_example: T in the above discussion.
    :param mean_translation_low_res_px: Mean translation distance (in units of
        low-resolution pixels).
    :param stdev_translation_low_res_px: Standard deviation of translation
        distance (in units of low-resolution pixels).
    :param sentinel_value: Sentinel value (used for padded pixels around edge).
    :return: bidirectional_reflectance_matrix: ET-by-M-by-N-by-L-by-W numpy
        array of reflectance values (unitless).  This may also be None.
    :return: brightness_temp_matrix_kelvins: ET-by-m-by-n-by-L-by-w numpy array
        of brightness temperatures.
    :return: row_translations_low_res_px: length-(ET) numpy array of translation
        distances applied (in units of low-resolution pixels).
    :return: column_translations_low_res_px: Same but for columns.
    """

    # Housekeeping.
    num_examples_orig = brightness_temp_matrix_kelvins.shape[0]
    num_examples_new = num_examples_orig * num_translations_per_example

    row_translations_low_res_px, column_translations_low_res_px = (
        get_translation_distances(
            mean_translation_px=mean_translation_low_res_px,
            stdev_translation_px=stdev_translation_low_res_px,
            num_translations=num_examples_new
        )
    )

    # Do actual stuff.
    new_bt_matrix_kelvins = numpy.full(
        (num_examples_new,) + brightness_temp_matrix_kelvins.shape[1:],
        numpy.nan
    )

    if bidirectional_reflectance_matrix is None:
        new_reflectance_matrix = None
    else:
        new_reflectance_matrix = numpy.full(
            (num_examples_new,) + bidirectional_reflectance_matrix.shape[1:],
            numpy.nan
        )

    for i in range(num_examples_orig):
        first_index = i * num_translations_per_example
        last_index = first_index + num_translations_per_example

        for j in range(first_index, last_index):
            new_bt_matrix_kelvins[j, ...] = _translate_images(
                image_matrix=brightness_temp_matrix_kelvins[[i], ...],
                row_translation_px=row_translations_low_res_px[j],
                column_translation_px=column_translations_low_res_px[j],
                padding_value=sentinel_value
            )[0, ...]

            if new_reflectance_matrix is None:
                continue

            new_reflectance_matrix[j, ...] = _translate_images(
                image_matrix=bidirectional_reflectance_matrix[[i], ...],
                row_translation_px=4 * row_translations_low_res_px[j],
                column_translation_px=4 * column_translations_low_res_px[j],
                padding_value=sentinel_value
            )[0, ...]

    return (
        new_reflectance_matrix, new_bt_matrix_kelvins,
        row_translations_low_res_px, column_translations_low_res_px
    )


def _augment_data_specific_trans(
        bidirectional_reflectance_matrix, brightness_temp_matrix_kelvins,
        row_translations_low_res_px, column_translations_low_res_px,
        sentinel_value):
    """Augments data via specific (not random) translations.

    E = number of examples

    :param bidirectional_reflectance_matrix: See doc for `_augment_data`.
    :param brightness_temp_matrix_kelvins: Same.
    :param row_translations_low_res_px: length-E numpy array of row
        translations.  The [i]th example will be shifted
        row_translations_low_res_px[i] rows up (towards the north).
    :param column_translations_low_res_px: length-E numpy array of column
        translations.  The [i]th example will be shifted
        column_translations_low_res_px[i] columns towards the right (east).
    :param sentinel_value: Sentinel value (used for padded pixels around edge).
    :return: bidirectional_reflectance_matrix: numpy array of translated images
        with same size as input.
    :return: brightness_temp_matrix_kelvins: numpy array of translated images
        with same size as input.
    """

    num_examples = brightness_temp_matrix_kelvins.shape[0]

    for i in range(num_examples):
        brightness_temp_matrix_kelvins[i, ...] = _translate_images(
            image_matrix=brightness_temp_matrix_kelvins[[i], ...],
            row_translation_px=row_translations_low_res_px[i],
            column_translation_px=column_translations_low_res_px[i],
            padding_value=sentinel_value
        )[0, ...]

        if bidirectional_reflectance_matrix is None:
            continue

        bidirectional_reflectance_matrix[i, ...] = _translate_images(
            image_matrix=bidirectional_reflectance_matrix[[i], ...],
            row_translation_px=4 * row_translations_low_res_px[i],
            column_translation_px=4 * column_translations_low_res_px[i],
            padding_value=sentinel_value
        )[0, ...]

    return bidirectional_reflectance_matrix, brightness_temp_matrix_kelvins


def _subset_grid_after_data_aug(data_matrix, num_rows_to_keep,
                                num_columns_to_keep, for_high_res):
    """Subsets grid after data augmentation (more cropping than first time).

    E = number of examples
    M = number of rows in original grid
    N = number of columns in original grid
    m = number of rows in subset grid
    n = number of columns in subset grid
    W = number of wavelengths in grid

    :param data_matrix: E-by-M-by-N-by-W numpy array of data.
    :param num_rows_to_keep: m in the above discussion.
    :param num_columns_to_keep: n in the above discussion.
    :param for_high_res: Boolean flag.
    :return: data_matrix: E-by-m-by-n-by-W numpy array of data (subset of
        input).
    """

    num_examples = data_matrix.shape[0]
    num_rows = data_matrix.shape[1]
    num_columns = data_matrix.shape[2]
    num_wavelengths = data_matrix.shape[3]

    row_dim = (
        satellite_utils.HIGH_RES_ROW_DIM if for_high_res
        else satellite_utils.LOW_RES_ROW_DIM
    )
    column_dim = (
        satellite_utils.HIGH_RES_COLUMN_DIM if for_high_res
        else satellite_utils.LOW_RES_COLUMN_DIM
    )
    wavelength_dim = (
        satellite_utils.HIGH_RES_WAVELENGTH_DIM if for_high_res
        else satellite_utils.LOW_RES_WAVELENGTH_DIM
    )
    main_data_key = (
        satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY if for_high_res
        else satellite_utils.BRIGHTNESS_TEMPERATURE_KEY
    )

    metadata_dict = {
        satellite_utils.TIME_DIM: numpy.linspace(
            0, num_examples - 1, num=num_examples, dtype=int
        ),
        row_dim: numpy.linspace(
            0, num_rows - 1, num=num_rows, dtype=int
        ),
        column_dim: numpy.linspace(
            0, num_columns - 1, num=num_columns, dtype=int
        ),
        wavelength_dim: numpy.linspace(
            1, num_wavelengths, num=num_wavelengths, dtype=float
        )
    }

    main_data_dict = {
        main_data_key: (
            (satellite_utils.TIME_DIM, row_dim, column_dim, wavelength_dim),
            data_matrix
        )
    }
    dummy_satellite_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )

    dummy_satellite_table_xarray = satellite_utils.subset_grid(
        satellite_table_xarray=dummy_satellite_table_xarray,
        num_rows_to_keep=num_rows_to_keep,
        num_columns_to_keep=num_columns_to_keep,
        for_high_res=for_high_res
    )
    return dummy_satellite_table_xarray[main_data_key].values


def _write_metafile(
        pickle_file_name, num_epochs, num_training_batches_per_epoch,
        training_option_dict, num_validation_batches_per_epoch,
        validation_option_dict, loss_function_string, optimizer_function_string,
        plateau_patience_epochs, plateau_learning_rate_multiplier,
        early_stopping_patience_epochs, architecture_dict, is_model_bnn):
    """Writes metadata to Pickle file.

    :param pickle_file_name: Path to output file.
    :param num_epochs: See doc for `train_model`.
    :param num_training_batches_per_epoch: Same.
    :param training_option_dict: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validation_option_dict: Same.
    :param loss_function_string: Same.
    :param optimizer_function_string: Same.
    :param plateau_patience_epochs: Same.
    :param plateau_learning_rate_multiplier: Same.
    :param early_stopping_patience_epochs: Same.
    :param architecture_dict: Same.
    :param is_model_bnn: Same.
    """

    metadata_dict = {
        NUM_EPOCHS_KEY: num_epochs,
        NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        TRAINING_OPTIONS_KEY: training_option_dict,
        NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch,
        VALIDATION_OPTIONS_KEY: validation_option_dict,
        LOSS_FUNCTION_KEY: loss_function_string,
        OPTIMIZER_FUNCTION_KEY: optimizer_function_string,
        PLATEAU_PATIENCE_KEY: plateau_patience_epochs,
        PLATEAU_LR_MUTIPLIER_KEY: plateau_learning_rate_multiplier,
        EARLY_STOPPING_PATIENCE_KEY: early_stopping_patience_epochs,
        ARCHITECTURE_KEY: architecture_dict,
        IS_MODEL_BNN_KEY: is_model_bnn
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(metadata_dict, pickle_file_handle)
    pickle_file_handle.close()


def _grid_coords_3d_to_4d(latitude_matrix_deg_n, longitude_matrix_deg_e):
    """Converts grid coordinates from 3-D matrices to 4-D matrices.

    T = number of target times
    M = number of rows per grid
    N = number of columns per grid
    L = number of lag times

    :param latitude_matrix_deg_n: T-by-M-by-L numpy array of latitudes (deg
        north).
    :param longitude_matrix_deg_e: T-by-N-by-L numpy array of longitudes (deg
        east).
    :return: latitude_matrix_deg_n: T-by-M-by-N-by-L numpy array of latitudes
        (deg north).
    :return: longitude_matrix_deg_e: T-by-M-by-N-by-L numpy array of longitudes
        (deg east).
    """

    num_target_times = latitude_matrix_deg_n.shape[0]
    num_lag_times = latitude_matrix_deg_n.shape[-1]

    latitude_matrix_deg_n = numpy.stack(
        [
            numpy.stack(
                [
                    numpy.meshgrid(
                        longitude_matrix_deg_e[i, :, j],
                        latitude_matrix_deg_n[i, :, j]
                    )[1]
                    for j in range(num_lag_times)
                ],
                axis=-1
            )
            for i in range(num_target_times)
        ],
        axis=0
    )

    longitude_matrix_deg_e = numpy.stack(
        [
            numpy.stack(
                [
                    numpy.meshgrid(
                        longitude_matrix_deg_e[i, :, j],
                        latitude_matrix_deg_n[i, :, j]
                    )[0]
                    for j in range(num_lag_times)
                ],
                axis=-1
            )
            for i in range(num_target_times)
        ],
        axis=0
    )

    return latitude_matrix_deg_n, longitude_matrix_deg_e


def _make_targets_for_semantic_seg(
        row_translations_px, column_translations_px,
        grid_spacings_km, cyclone_center_latitudes_deg_n,
        gaussian_smoother_stdev_km, num_grid_rows, num_grid_columns):
    """Creates targets for semantic segmentation.

    E = number of examples
    M = number of rows in full image grid (must be even number)
    N = number of columns in full image grid (must be even number)

    :param row_translations_px: length-E numpy array of translation distances
        (pixel units) in +y-direction, or north.
    :param column_translations_px: length-E numpy array of translation distances
        (pixel units) in +x-direction, or east.
    :param grid_spacings_km: length-E numpy array of grid spacings.
    :param cyclone_center_latitudes_deg_n: length-E numpy array of latitudes at
        actual TC centers (deg north).
    :param gaussian_smoother_stdev_km: Standard-deviation distance for Gaussian
        smoother.
    :param num_grid_rows: M in the above discussion.
    :param num_grid_columns: N in the above discussion.
    :return: target_matrix: E-by-M-by-N-by-1 numpy array of "probabilities" in
        range 0...1.  The sum over each grid is exactly 1.
    """

    num_examples = len(row_translations_px)
    target_matrix = numpy.full(
        (num_examples, num_grid_rows, num_grid_columns), 0.
    )

    first_row_index_default = int(numpy.round(0.5 * num_grid_rows - 1))
    last_row_index_default = int(numpy.round(0.5 * num_grid_rows + 1))
    first_column_index_default = int(numpy.round(0.5 * num_grid_columns - 1))
    last_column_index_default = int(numpy.round(0.5 * num_grid_columns + 1))

    for k in range(num_examples):
        i_start = first_row_index_default + row_translations_px[k]
        i_end = last_row_index_default + row_translations_px[k]
        j_start = first_column_index_default + column_translations_px[k]
        j_end = last_column_index_default + column_translations_px[k]

        target_matrix[k, i_start:i_end, j_start:j_end] = 1.

        target_matrix[k, ...] = image_filtering.undo_target_discretization(
            integer_target_matrix=target_matrix[k, ...].astype(int),
            grid_spacing_km=grid_spacings_km[k],
            cyclone_center_latitude_deg_n=cyclone_center_latitudes_deg_n[k]
        )

        target_matrix[k, ...] = image_filtering.smooth_targets_with_gaussian(
            target_matrix=target_matrix[k, ...],
            grid_spacing_km=grid_spacings_km[k],
            stdev_distance_km=gaussian_smoother_stdev_km
        )

    return numpy.expand_dims(target_matrix, axis=-1)


def get_translation_distances(
        mean_translation_px, stdev_translation_px, num_translations):
    """Samples translation distances from normal distribution.

    T = number of translations

    :param mean_translation_px: Mean translation distance (pixels).
    :param stdev_translation_px: Standard deviation of translation distance
        (pixels).
    :param num_translations: T in the above discussion.
    :return: row_translations_px: length-T numpy array of translation distances
        (pixels).
    :return: column_translations_px: Same but for columns.
    """

    error_checking.assert_is_greater(mean_translation_px, 0.)
    error_checking.assert_is_greater(stdev_translation_px, 0.)
    error_checking.assert_is_integer(num_translations)
    error_checking.assert_is_greater(num_translations, 0)

    euclidean_translations_low_res_px = numpy.random.normal(
        loc=mean_translation_px, scale=stdev_translation_px,
        size=num_translations
    )
    euclidean_translations_low_res_px = numpy.maximum(
        euclidean_translations_low_res_px, 0.
    )
    translation_directions_rad = numpy.random.uniform(
        low=0., high=2 * numpy.pi - 1e-6, size=num_translations
    )

    row_translations_low_res_px = (
        euclidean_translations_low_res_px *
        numpy.sin(translation_directions_rad)
    )
    column_translations_low_res_px = (
        euclidean_translations_low_res_px *
        numpy.cos(translation_directions_rad)
    )

    # these_flags = numpy.logical_and(
    #     row_translations_low_res_px < 0, row_translations_low_res_px >= -0.5
    # )
    # row_translations_low_res_px[these_flags] = -1.
    #
    # these_flags = numpy.logical_and(
    #     row_translations_low_res_px > 0, row_translations_low_res_px <= 0.5
    # )
    # row_translations_low_res_px[these_flags] = 1.
    #
    # these_flags = numpy.logical_and(
    #     column_translations_low_res_px < 0,
    #     column_translations_low_res_px >= -0.5
    # )
    # column_translations_low_res_px[these_flags] = -1.
    #
    # these_flags = numpy.logical_and(
    #     column_translations_low_res_px > 0,
    #     column_translations_low_res_px <= 0.5
    # )
    # column_translations_low_res_px[these_flags] = 1.

    row_translations_low_res_px = (
        numpy.round(row_translations_low_res_px).astype(int)
    )
    column_translations_low_res_px = (
        numpy.round(column_translations_low_res_px).astype(int)
    )

    error_checking.assert_is_geq_numpy_array(
        numpy.absolute(row_translations_low_res_px), 0
    )
    error_checking.assert_is_geq_numpy_array(
        numpy.absolute(column_translations_low_res_px), 0
    )

    return row_translations_low_res_px, column_translations_low_res_px


def combine_lag_times_and_wavelengths(satellite_data_matrix):
    """Combines lag times and wavelengths into one axis.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    L = number of lag times
    W = number of wavelengths

    :param satellite_data_matrix: E-by-M-by-N-by-L-by-W numpy array of satellite
        data (either brightness temperatures or bidirectional reflectances).
    :return: satellite_data_matrix: Same as input but with shape
        E-by-M-by-N-by-(L * W).
    """

    error_checking.assert_is_numpy_array_without_nan(satellite_data_matrix)
    error_checking.assert_is_numpy_array(
        satellite_data_matrix, num_dimensions=5
    )

    these_dim = satellite_data_matrix.shape[:-2] + (
        numpy.prod(satellite_data_matrix.shape[-2:]),
    )
    return numpy.reshape(satellite_data_matrix, these_dim)


def separate_lag_times_and_wavelengths(satellite_data_matrix, num_lag_times):
    """Separates lag times and wavelengths into different axes.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    L = number of lag times
    W = number of wavelengths

    :param satellite_data_matrix: E-by-M-by-N-by-(L * W) numpy array of
        satellite data (either brightness temperatures or bidirectional
        reflectances).
    :param num_lag_times: Number of lag times.
    :return: satellite_data_matrix: Same as input but with shape
        E-by-M-by-N-by-L-by-W.
    """

    error_checking.assert_is_numpy_array_without_nan(satellite_data_matrix)
    error_checking.assert_is_numpy_array(
        satellite_data_matrix, num_dimensions=4
    )

    error_checking.assert_is_integer(num_lag_times)
    error_checking.assert_is_geq(num_lag_times, 1)

    last_axis_length = satellite_data_matrix.shape[-1]
    num_wavelengths_float = float(last_axis_length) / num_lag_times
    assert numpy.isclose(
        num_wavelengths_float, numpy.round(num_wavelengths_float),
        atol=TOLERANCE
    )

    num_wavelengths = int(numpy.round(num_wavelengths_float))
    these_dim = satellite_data_matrix.shape[:-1] + (
        num_lag_times, num_wavelengths
    )

    return numpy.reshape(satellite_data_matrix, these_dim)


def create_data(option_dict, cyclone_id_string, num_target_times):
    """Creates input data for neural net (without being a generator).

    E = batch size = number of examples after data augmentation
    L = number of lag times for predictors
    W = number of high-res wavelengths
    w = number of low-res wavelengths
    M = number of rows in high-res grid
    m = number of rows in low-res grid = M/4
    N = number of columns in high-res grid
    n = number of columns in low-res grid = M/4

    :param option_dict: See doc for `data_generator`.
    :param cyclone_id_string: Will create data for this cyclone.
    :param num_target_times: Will create data for this number of target times
        for the given cyclone.

    :return: data_dict: Dictionary with the following keys.
    data_dict["predictor_matrices"]: See doc for `data_generator`.
    data_dict["target_matrix"]: Same.
    data_dict["target_times_unix_sec"]: length-E numpy array of target times.
    data_dict["grid_spacings_low_res_km"]: length-E numpy array of grid
        spacings.
    data_dict["cyclone_center_latitudes_deg_n"]: length-E numpy array of true
        TC-center latitudes (deg north).
    data_dict["high_res_latitude_matrix_deg_n"]: E-by-M-by-L numpy array of
        latitudes (deg north).
    data_dict["high_res_longitude_matrix_deg_e"]: E-by-N-by-L numpy array of
        longitudes (deg east).
    data_dict["low_res_latitude_matrix_deg_n"]: E-by-m-by-L numpy array of
        latitudes (deg north).
    data_dict["low_res_longitude_matrix_deg_e"]: E-by-n-by-L numpy array of
        longitudes (deg east).
    """

    option_dict = _check_generator_args(option_dict)
    error_checking.assert_is_integer(num_target_times)
    error_checking.assert_is_greater(num_target_times, 0)

    satellite_dir_name = option_dict[SATELLITE_DIRECTORY_KEY]
    lag_times_minutes = option_dict[LAG_TIMES_KEY]
    high_res_wavelengths_microns = option_dict[HIGH_RES_WAVELENGTHS_KEY]
    low_res_wavelengths_microns = option_dict[LOW_RES_WAVELENGTHS_KEY]
    num_rows_low_res = option_dict[NUM_GRID_ROWS_KEY]
    num_columns_low_res = option_dict[NUM_GRID_COLUMNS_KEY]
    data_aug_num_translations = option_dict[DATA_AUG_NUM_TRANS_KEY]
    data_aug_mean_translation_low_res_px = option_dict[DATA_AUG_MEAN_TRANS_KEY]
    data_aug_stdev_translation_low_res_px = (
        option_dict[DATA_AUG_STDEV_TRANS_KEY]
    )
    lag_time_tolerance_sec = option_dict[LAG_TIME_TOLERANCE_KEY]
    max_num_missing_lag_times = option_dict[MAX_MISSING_LAG_TIMES_KEY]
    max_interp_gap_sec = option_dict[MAX_INTERP_GAP_KEY]
    sentinel_value = option_dict[SENTINEL_VALUE_KEY]
    semantic_segmentation_flag = option_dict[SEMANTIC_SEG_FLAG_KEY]
    target_smoother_stdev_km = option_dict[TARGET_SMOOOTHER_STDEV_KEY]

    orig_num_rows_low_res = num_rows_low_res + 0
    orig_num_columns_low_res = num_columns_low_res + 0

    num_extra_rowcols = 2 * int(numpy.ceil(
        data_aug_mean_translation_low_res_px +
        5 * data_aug_stdev_translation_low_res_px
    ))
    num_rows_low_res += num_extra_rowcols
    num_columns_low_res += num_extra_rowcols

    satellite_file_names = satellite_io.find_files_one_cyclone(
        directory_name=satellite_dir_name, cyclone_id_string=cyclone_id_string,
        raise_error_if_all_missing=True
    )

    data_dict = _read_satellite_data_one_cyclone(
        input_file_names=satellite_file_names,
        num_target_times=num_target_times,
        lag_times_minutes=lag_times_minutes,
        lag_time_tolerance_sec=lag_time_tolerance_sec,
        max_num_missing_lag_times=max_num_missing_lag_times,
        max_interp_gap_sec=max_interp_gap_sec,
        high_res_wavelengths_microns=high_res_wavelengths_microns,
        low_res_wavelengths_microns=low_res_wavelengths_microns,
        num_rows_low_res=num_rows_low_res,
        num_columns_low_res=num_columns_low_res,
        sentinel_value=sentinel_value, return_coords=True
    )

    if data_dict is None:
        return None

    bidirectional_reflectance_matrix = data_dict[BIDIRECTIONAL_REFLECTANCES_KEY]
    brightness_temp_matrix_kelvins = data_dict[BRIGHTNESS_TEMPS_KEY]
    grid_spacings_km = data_dict[GRID_SPACINGS_KEY]
    cyclone_center_latitudes_deg_n = data_dict[CENTER_LATITUDES_KEY]
    target_times_unix_sec = data_dict[TARGET_TIMES_KEY]
    low_res_latitude_matrix_deg_n = data_dict[LOW_RES_LATITUDES_KEY]
    low_res_longitude_matrix_deg_e = data_dict[LOW_RES_LONGITUDES_KEY]
    high_res_latitude_matrix_deg_n = data_dict[HIGH_RES_LATITUDES_KEY]
    high_res_longitude_matrix_deg_e = data_dict[HIGH_RES_LONGITUDES_KEY]

    if brightness_temp_matrix_kelvins is None:
        return None
    if brightness_temp_matrix_kelvins.size == 0:
        return None

    brightness_temp_matrix_kelvins = combine_lag_times_and_wavelengths(
        brightness_temp_matrix_kelvins
    )
    if bidirectional_reflectance_matrix is not None:
        bidirectional_reflectance_matrix = combine_lag_times_and_wavelengths(
            bidirectional_reflectance_matrix
        )

    (
        bidirectional_reflectance_matrix, brightness_temp_matrix_kelvins,
        row_translations_low_res_px, column_translations_low_res_px
    ) = _augment_data(
        bidirectional_reflectance_matrix=bidirectional_reflectance_matrix,
        brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
        num_translations_per_example=data_aug_num_translations,
        mean_translation_low_res_px=data_aug_mean_translation_low_res_px,
        stdev_translation_low_res_px=data_aug_stdev_translation_low_res_px,
        sentinel_value=sentinel_value
    )

    brightness_temp_matrix_kelvins = _subset_grid_after_data_aug(
        data_matrix=brightness_temp_matrix_kelvins,
        num_rows_to_keep=orig_num_rows_low_res,
        num_columns_to_keep=orig_num_columns_low_res,
        for_high_res=False
    )

    low_res_latitude_matrix_deg_n, low_res_longitude_matrix_deg_e = (
        _grid_coords_3d_to_4d(
            latitude_matrix_deg_n=low_res_latitude_matrix_deg_n,
            longitude_matrix_deg_e=low_res_longitude_matrix_deg_e
        )
    )

    low_res_latitude_matrix_deg_n = _subset_grid_after_data_aug(
        data_matrix=low_res_latitude_matrix_deg_n,
        num_rows_to_keep=orig_num_rows_low_res,
        num_columns_to_keep=orig_num_columns_low_res,
        for_high_res=False
    )[:, :, 0, :]

    low_res_longitude_matrix_deg_e = _subset_grid_after_data_aug(
        data_matrix=low_res_longitude_matrix_deg_e,
        num_rows_to_keep=orig_num_rows_low_res,
        num_columns_to_keep=orig_num_columns_low_res,
        for_high_res=False
    )[:, 0, :, :]

    if bidirectional_reflectance_matrix is not None:
        bidirectional_reflectance_matrix = _subset_grid_after_data_aug(
            data_matrix=bidirectional_reflectance_matrix,
            num_rows_to_keep=orig_num_rows_low_res * 4,
            num_columns_to_keep=orig_num_columns_low_res * 4,
            for_high_res=True
        )

        high_res_latitude_matrix_deg_n, high_res_longitude_matrix_deg_e = (
            _grid_coords_3d_to_4d(
                latitude_matrix_deg_n=high_res_latitude_matrix_deg_n,
                longitude_matrix_deg_e=high_res_longitude_matrix_deg_e
            )
        )

        high_res_latitude_matrix_deg_n = _subset_grid_after_data_aug(
            data_matrix=high_res_latitude_matrix_deg_n,
            num_rows_to_keep=orig_num_rows_low_res * 4,
            num_columns_to_keep=orig_num_columns_low_res * 4,
            for_high_res=True
        )[:, :, 0, :]

        high_res_longitude_matrix_deg_e = _subset_grid_after_data_aug(
            data_matrix=high_res_longitude_matrix_deg_e,
            num_rows_to_keep=orig_num_rows_low_res * 4,
            num_columns_to_keep=orig_num_columns_low_res * 4,
            for_high_res=True
        )[:, 0, :, :]

    grid_spacings_km = numpy.repeat(
        grid_spacings_km, repeats=data_aug_num_translations
    )
    cyclone_center_latitudes_deg_n = numpy.repeat(
        cyclone_center_latitudes_deg_n, repeats=data_aug_num_translations
    )
    target_times_unix_sec = numpy.repeat(
        target_times_unix_sec, repeats=data_aug_num_translations
    )
    low_res_latitude_matrix_deg_n = numpy.repeat(
        low_res_latitude_matrix_deg_n, repeats=data_aug_num_translations,
        axis=0
    )
    low_res_longitude_matrix_deg_e = numpy.repeat(
        low_res_longitude_matrix_deg_e, repeats=data_aug_num_translations,
        axis=0
    )

    if bidirectional_reflectance_matrix is not None:
        high_res_latitude_matrix_deg_n = numpy.repeat(
            high_res_latitude_matrix_deg_n, repeats=data_aug_num_translations,
            axis=0
        )
        high_res_longitude_matrix_deg_e = numpy.repeat(
            high_res_longitude_matrix_deg_e, repeats=data_aug_num_translations,
            axis=0
        )

    predictor_matrices = [brightness_temp_matrix_kelvins]
    if bidirectional_reflectance_matrix is not None:
        predictor_matrices.insert(0, bidirectional_reflectance_matrix)

    if semantic_segmentation_flag:
        target_matrix = _make_targets_for_semantic_seg(
            row_translations_px=row_translations_low_res_px,
            column_translations_px=column_translations_low_res_px,
            grid_spacings_km=grid_spacings_km,
            cyclone_center_latitudes_deg_n=cyclone_center_latitudes_deg_n,
            gaussian_smoother_stdev_km=target_smoother_stdev_km,
            num_grid_rows=orig_num_rows_low_res,
            num_grid_columns=orig_num_columns_low_res
        )
    else:
        target_matrix = numpy.transpose(numpy.vstack((
            row_translations_low_res_px, column_translations_low_res_px,
            grid_spacings_km, cyclone_center_latitudes_deg_n
        )))

    predictor_matrices = [p.astype('float16') for p in predictor_matrices]

    return {
        PREDICTOR_MATRICES_KEY: predictor_matrices,
        TARGET_MATRIX_KEY: target_matrix,
        TARGET_TIMES_KEY: target_times_unix_sec,
        GRID_SPACINGS_KEY: grid_spacings_km,
        CENTER_LATITUDES_KEY: cyclone_center_latitudes_deg_n,
        HIGH_RES_LATITUDES_KEY: high_res_latitude_matrix_deg_n,
        HIGH_RES_LONGITUDES_KEY: high_res_longitude_matrix_deg_e,
        LOW_RES_LATITUDES_KEY: low_res_latitude_matrix_deg_n,
        LOW_RES_LONGITUDES_KEY: low_res_longitude_matrix_deg_e
    }


def create_data_specific_trans(
        option_dict, cyclone_id_string, target_times_unix_sec,
        row_translations_low_res_px, column_translations_low_res_px):
    """Creates NN-input data with specific (instead of random) translations.

    E = batch size = number of examples after data augmentation
    L = number of lag times for predictors
    W = number of high-res wavelengths
    w = number of low-res wavelengths
    M = number of rows in high-res grid
    m = number of rows in low-res grid = M/4
    N = number of columns in high-res grid
    n = number of columns in low-res grid = M/4

    :param option_dict: See doc for `data_generator`.
    :param cyclone_id_string: Will create data for this cyclone.
    :param target_times_unix_sec: length-E numpy array of target times.
    :param row_translations_low_res_px: length-E numpy array of row
        translations.  The [i]th example will be shifted
        row_translations_low_res_px[i] rows up (towards the north).
    :param column_translations_low_res_px: length-E numpy array of column
        translations.  The [i]th example will be shifted
        column_translations_low_res_px[i] columns towards the right (east).

    :return: data_dict: See documentation for `create_data`.
    """

    # Check input args.
    option_dict = _check_generator_args(option_dict)

    error_checking.assert_is_integer_numpy_array(target_times_unix_sec)
    error_checking.assert_is_numpy_array(
        target_times_unix_sec, num_dimensions=1
    )

    num_examples = len(target_times_unix_sec)
    expected_dim = numpy.array([num_examples], dtype=int)

    error_checking.assert_is_integer_numpy_array(row_translations_low_res_px)
    error_checking.assert_is_numpy_array(
        row_translations_low_res_px, exact_dimensions=expected_dim
    )
    # error_checking.assert_is_greater_numpy_array(
    #     numpy.absolute(row_translations_low_res_px), 0
    # )

    error_checking.assert_is_integer_numpy_array(column_translations_low_res_px)
    error_checking.assert_is_numpy_array(
        column_translations_low_res_px, exact_dimensions=expected_dim
    )
    # error_checking.assert_is_greater_numpy_array(
    #     numpy.absolute(row_translations_low_res_px) +
    #     numpy.absolute(column_translations_low_res_px),
    #     0
    # )

    # Do actual stuff.
    satellite_dir_name = option_dict[SATELLITE_DIRECTORY_KEY]
    lag_times_minutes = option_dict[LAG_TIMES_KEY]
    high_res_wavelengths_microns = option_dict[HIGH_RES_WAVELENGTHS_KEY]
    low_res_wavelengths_microns = option_dict[LOW_RES_WAVELENGTHS_KEY]
    num_rows_low_res = option_dict[NUM_GRID_ROWS_KEY]
    num_columns_low_res = option_dict[NUM_GRID_COLUMNS_KEY]
    lag_time_tolerance_sec = option_dict[LAG_TIME_TOLERANCE_KEY]
    max_num_missing_lag_times = option_dict[MAX_MISSING_LAG_TIMES_KEY]
    max_interp_gap_sec = option_dict[MAX_INTERP_GAP_KEY]
    sentinel_value = option_dict[SENTINEL_VALUE_KEY]
    semantic_segmentation_flag = option_dict[SEMANTIC_SEG_FLAG_KEY]
    target_smoother_stdev_km = option_dict[TARGET_SMOOOTHER_STDEV_KEY]

    orig_num_rows_low_res = num_rows_low_res + 0
    orig_num_columns_low_res = num_columns_low_res + 0

    num_extra_rowcols = 2 * max([
        numpy.max(numpy.absolute(row_translations_low_res_px)),
        numpy.max(numpy.absolute(column_translations_low_res_px))
    ])
    num_rows_low_res += num_extra_rowcols
    num_columns_low_res += num_extra_rowcols

    satellite_file_names = satellite_io.find_files_one_cyclone(
        directory_name=satellite_dir_name, cyclone_id_string=cyclone_id_string,
        raise_error_if_all_missing=True
    )

    unique_target_times_unix_sec = numpy.unique(target_times_unix_sec)

    data_dict = _read_satellite_data_one_cyclone(
        input_file_names=satellite_file_names,
        target_times_unix_sec=unique_target_times_unix_sec,
        lag_times_minutes=lag_times_minutes,
        lag_time_tolerance_sec=lag_time_tolerance_sec,
        max_num_missing_lag_times=max_num_missing_lag_times,
        max_interp_gap_sec=max_interp_gap_sec,
        high_res_wavelengths_microns=high_res_wavelengths_microns,
        low_res_wavelengths_microns=low_res_wavelengths_microns,
        num_rows_low_res=num_rows_low_res,
        num_columns_low_res=num_columns_low_res,
        sentinel_value=sentinel_value, return_coords=True
    )

    if data_dict is None:
        return None

    bidirectional_reflectance_matrix = data_dict[BIDIRECTIONAL_REFLECTANCES_KEY]
    brightness_temp_matrix_kelvins = data_dict[BRIGHTNESS_TEMPS_KEY]
    grid_spacings_km = data_dict[GRID_SPACINGS_KEY]
    cyclone_center_latitudes_deg_n = data_dict[CENTER_LATITUDES_KEY]
    low_res_latitude_matrix_deg_n = data_dict[LOW_RES_LATITUDES_KEY]
    low_res_longitude_matrix_deg_e = data_dict[LOW_RES_LONGITUDES_KEY]
    high_res_latitude_matrix_deg_n = data_dict[HIGH_RES_LATITUDES_KEY]
    high_res_longitude_matrix_deg_e = data_dict[HIGH_RES_LONGITUDES_KEY]

    if brightness_temp_matrix_kelvins is None:
        return None
    if brightness_temp_matrix_kelvins.size == 0:
        return None

    reconstruction_indices = numpy.array([
        numpy.where(unique_target_times_unix_sec == t)[0][0]
        for t in target_times_unix_sec
    ], dtype=int)

    brightness_temp_matrix_kelvins = (
        brightness_temp_matrix_kelvins[reconstruction_indices, ...]
    )
    grid_spacings_km = grid_spacings_km[reconstruction_indices]
    cyclone_center_latitudes_deg_n = (
        cyclone_center_latitudes_deg_n[reconstruction_indices]
    )
    low_res_latitude_matrix_deg_n = (
        low_res_latitude_matrix_deg_n[reconstruction_indices, ...]
    )
    low_res_longitude_matrix_deg_e = (
        low_res_longitude_matrix_deg_e[reconstruction_indices, ...]
    )

    brightness_temp_matrix_kelvins = combine_lag_times_and_wavelengths(
        brightness_temp_matrix_kelvins
    )

    if bidirectional_reflectance_matrix is not None:
        bidirectional_reflectance_matrix = (
            bidirectional_reflectance_matrix[reconstruction_indices, ...]
        )
        high_res_latitude_matrix_deg_n = (
            high_res_latitude_matrix_deg_n[reconstruction_indices, ...]
        )
        high_res_longitude_matrix_deg_e = (
            high_res_longitude_matrix_deg_e[reconstruction_indices, ...]
        )

        bidirectional_reflectance_matrix = combine_lag_times_and_wavelengths(
            bidirectional_reflectance_matrix
        )

    (
        bidirectional_reflectance_matrix, brightness_temp_matrix_kelvins
    ) = _augment_data_specific_trans(
        bidirectional_reflectance_matrix=bidirectional_reflectance_matrix,
        brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
        row_translations_low_res_px=row_translations_low_res_px,
        column_translations_low_res_px=column_translations_low_res_px,
        sentinel_value=sentinel_value
    )

    brightness_temp_matrix_kelvins = _subset_grid_after_data_aug(
        data_matrix=brightness_temp_matrix_kelvins,
        num_rows_to_keep=orig_num_rows_low_res,
        num_columns_to_keep=orig_num_columns_low_res,
        for_high_res=False
    )

    low_res_latitude_matrix_deg_n, low_res_longitude_matrix_deg_e = (
        _grid_coords_3d_to_4d(
            latitude_matrix_deg_n=low_res_latitude_matrix_deg_n,
            longitude_matrix_deg_e=low_res_longitude_matrix_deg_e
        )
    )

    low_res_latitude_matrix_deg_n = _subset_grid_after_data_aug(
        data_matrix=low_res_latitude_matrix_deg_n,
        num_rows_to_keep=orig_num_rows_low_res,
        num_columns_to_keep=orig_num_columns_low_res,
        for_high_res=False
    )[:, :, 0, :]

    low_res_longitude_matrix_deg_e = _subset_grid_after_data_aug(
        data_matrix=low_res_longitude_matrix_deg_e,
        num_rows_to_keep=orig_num_rows_low_res,
        num_columns_to_keep=orig_num_columns_low_res,
        for_high_res=False
    )[:, 0, :, :]

    if bidirectional_reflectance_matrix is not None:
        bidirectional_reflectance_matrix = _subset_grid_after_data_aug(
            data_matrix=bidirectional_reflectance_matrix,
            num_rows_to_keep=orig_num_rows_low_res * 4,
            num_columns_to_keep=orig_num_columns_low_res * 4,
            for_high_res=True
        )

        high_res_latitude_matrix_deg_n, high_res_longitude_matrix_deg_e = (
            _grid_coords_3d_to_4d(
                latitude_matrix_deg_n=high_res_latitude_matrix_deg_n,
                longitude_matrix_deg_e=high_res_longitude_matrix_deg_e
            )
        )

        high_res_latitude_matrix_deg_n = _subset_grid_after_data_aug(
            data_matrix=high_res_latitude_matrix_deg_n,
            num_rows_to_keep=orig_num_rows_low_res * 4,
            num_columns_to_keep=orig_num_columns_low_res * 4,
            for_high_res=True
        )[:, :, 0, :]

        high_res_longitude_matrix_deg_e = _subset_grid_after_data_aug(
            data_matrix=high_res_longitude_matrix_deg_e,
            num_rows_to_keep=orig_num_rows_low_res * 4,
            num_columns_to_keep=orig_num_columns_low_res * 4,
            for_high_res=True
        )[:, 0, :, :]

    predictor_matrices = [brightness_temp_matrix_kelvins]
    if bidirectional_reflectance_matrix is not None:
        predictor_matrices.insert(0, bidirectional_reflectance_matrix)

    if semantic_segmentation_flag:
        target_matrix = _make_targets_for_semantic_seg(
            row_translations_px=row_translations_low_res_px,
            column_translations_px=column_translations_low_res_px,
            grid_spacings_km=grid_spacings_km,
            cyclone_center_latitudes_deg_n=cyclone_center_latitudes_deg_n,
            gaussian_smoother_stdev_km=target_smoother_stdev_km,
            num_grid_rows=orig_num_rows_low_res,
            num_grid_columns=orig_num_columns_low_res
        )
    else:
        target_matrix = numpy.transpose(numpy.vstack((
            row_translations_low_res_px, column_translations_low_res_px,
            grid_spacings_km, cyclone_center_latitudes_deg_n
        )))

    predictor_matrices = [p.astype('float16') for p in predictor_matrices]

    return {
        PREDICTOR_MATRICES_KEY: predictor_matrices,
        TARGET_MATRIX_KEY: target_matrix,
        TARGET_TIMES_KEY: target_times_unix_sec,
        GRID_SPACINGS_KEY: grid_spacings_km,
        CENTER_LATITUDES_KEY: cyclone_center_latitudes_deg_n,
        HIGH_RES_LATITUDES_KEY: high_res_latitude_matrix_deg_n,
        HIGH_RES_LONGITUDES_KEY: high_res_longitude_matrix_deg_e,
        LOW_RES_LATITUDES_KEY: low_res_latitude_matrix_deg_n,
        LOW_RES_LONGITUDES_KEY: low_res_longitude_matrix_deg_e
    }


def data_generator_cira_ir(option_dict):
    """Generates input data for neural net, using CIRA IR dataset.

    :param option_dict: See doc for `data_generator`.
    :return: predictor_matrices: Same.
    :return: target_matrix: Same.
    """

    option_dict = _check_generator_args(option_dict)

    example_dir_name = option_dict[SATELLITE_DIRECTORY_KEY]
    years = option_dict[YEARS_KEY]
    lag_times_minutes = option_dict[LAG_TIMES_KEY]
    num_examples_per_batch = option_dict[BATCH_SIZE_KEY]
    max_examples_per_cyclone = option_dict[MAX_EXAMPLES_PER_CYCLONE_KEY]
    num_grid_rows = option_dict[NUM_GRID_ROWS_KEY]
    num_grid_columns = option_dict[NUM_GRID_COLUMNS_KEY]
    data_aug_num_translations = option_dict[DATA_AUG_NUM_TRANS_KEY]
    data_aug_mean_translation_low_res_px = option_dict[DATA_AUG_MEAN_TRANS_KEY]
    data_aug_stdev_translation_low_res_px = (
        option_dict[DATA_AUG_STDEV_TRANS_KEY]
    )

    orig_num_grid_rows = num_grid_rows + 0
    orig_num_grid_columns = num_grid_columns + 0

    num_extra_rowcols = 2 * int(numpy.ceil(
        data_aug_mean_translation_low_res_px +
        5 * data_aug_stdev_translation_low_res_px
    ))
    num_grid_rows += num_extra_rowcols
    num_grid_columns += num_extra_rowcols

    cyclone_id_strings = cira_ir_example_io.find_cyclones(
        directory_name=example_dir_name, raise_error_if_all_missing=True
    )
    cyclone_years = numpy.array(
        [misc_utils.parse_cyclone_id(c)[0] for c in cyclone_id_strings],
        dtype=int
    )

    good_flags = numpy.array([y in years for y in cyclone_years], dtype=bool)
    good_indices = numpy.where(good_flags)[0]
    cyclone_id_strings = [cyclone_id_strings[k] for k in good_indices]
    random.shuffle(cyclone_id_strings)

    example_file_names = [
        cira_ir_example_io.find_file(
            directory_name=example_dir_name, cyclone_id_string=c,
            prefer_zipped=False, allow_other_format=False,
            raise_error_if_missing=True
        )
        for c in cyclone_id_strings
    ]

    cyclone_index = 0

    while True:
        brightness_temp_matrix_kelvins = None
        grid_spacings_km = None
        cyclone_center_latitudes_deg_n = None
        num_examples_in_memory = 0

        while num_examples_in_memory < num_examples_per_batch:
            if cyclone_index == len(cyclone_id_strings):
                cyclone_index = 0

            num_examples_to_read = min([
                max_examples_per_cyclone,
                num_examples_per_batch - num_examples_in_memory
            ])

            data_dict = _read_satellite_data_1cyclone_cira_ir(
                input_file_name=example_file_names[cyclone_index],
                lag_times_minutes=lag_times_minutes,
                num_grid_rows=num_grid_rows,
                num_grid_columns=num_grid_columns,
                return_coords=False, num_target_times=num_examples_to_read
            )

            if data_dict is None:
                continue

            this_bt_matrix_kelvins = data_dict[BRIGHTNESS_TEMPS_KEY]
            these_grid_spacings_km = data_dict[GRID_SPACINGS_KEY]
            these_center_latitudes_deg_n = data_dict[CENTER_LATITUDES_KEY]

            if this_bt_matrix_kelvins is None:
                continue
            if this_bt_matrix_kelvins.size == 0:
                continue

            this_bt_matrix_kelvins = combine_lag_times_and_wavelengths(
                this_bt_matrix_kelvins
            )

            if brightness_temp_matrix_kelvins is None:
                these_dim = (
                    (num_examples_per_batch,) + this_bt_matrix_kelvins.shape[1:]
                )
                brightness_temp_matrix_kelvins = numpy.full(
                    these_dim, numpy.nan
                )

                grid_spacings_km = numpy.full(num_examples_per_batch, numpy.nan)
                cyclone_center_latitudes_deg_n = numpy.full(
                    num_examples_per_batch, numpy.nan
                )

            first_index = num_examples_in_memory
            last_index = first_index + this_bt_matrix_kelvins.shape[0]
            brightness_temp_matrix_kelvins[first_index:last_index, ...] = (
                this_bt_matrix_kelvins
            )
            grid_spacings_km[first_index:last_index] = these_grid_spacings_km
            cyclone_center_latitudes_deg_n[first_index:last_index] = (
                these_center_latitudes_deg_n
            )

            cyclone_index += 1
            num_examples_in_memory += this_bt_matrix_kelvins.shape[0]

        (
            _, brightness_temp_matrix_kelvins,
            row_translations_low_res_px, column_translations_low_res_px
        ) = _augment_data(
            bidirectional_reflectance_matrix=None,
            brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
            num_translations_per_example=data_aug_num_translations,
            mean_translation_low_res_px=data_aug_mean_translation_low_res_px,
            stdev_translation_low_res_px=data_aug_stdev_translation_low_res_px,
            sentinel_value=-10.
        )

        brightness_temp_matrix_kelvins = _subset_grid_after_data_aug(
            data_matrix=brightness_temp_matrix_kelvins,
            num_rows_to_keep=orig_num_grid_rows,
            num_columns_to_keep=orig_num_grid_columns,
            for_high_res=False
        )

        grid_spacings_km = numpy.repeat(
            grid_spacings_km, repeats=data_aug_num_translations
        )
        cyclone_center_latitudes_deg_n = numpy.repeat(
            cyclone_center_latitudes_deg_n, repeats=data_aug_num_translations
        )
        predictor_matrices = [brightness_temp_matrix_kelvins]

        target_matrix = numpy.transpose(numpy.vstack((
            row_translations_low_res_px, column_translations_low_res_px,
            grid_spacings_km, cyclone_center_latitudes_deg_n
        )))

        predictor_matrices = [p.astype('float16') for p in predictor_matrices]
        yield predictor_matrices, target_matrix


def data_generator_simple(option_dict):
    """Generates input data for neural net.

    :param option_dict: See doc for `data_generator`.
    :return: predictor_matrices: Same.
    :return: target_matrix: Same.
    """

    option_dict = _check_generator_args(option_dict)

    satellite_dir_name = option_dict[SATELLITE_DIRECTORY_KEY]
    years = option_dict[YEARS_KEY]
    lag_times_minutes = option_dict[LAG_TIMES_KEY]
    num_examples_per_batch = option_dict[BATCH_SIZE_KEY]
    max_examples_per_cyclone = option_dict[MAX_EXAMPLES_PER_CYCLONE_KEY]
    num_rows_low_res = option_dict[NUM_GRID_ROWS_KEY]
    num_columns_low_res = option_dict[NUM_GRID_COLUMNS_KEY]
    data_aug_num_translations = option_dict[DATA_AUG_NUM_TRANS_KEY]
    data_aug_mean_translation_low_res_px = option_dict[DATA_AUG_MEAN_TRANS_KEY]
    data_aug_stdev_translation_low_res_px = (
        option_dict[DATA_AUG_STDEV_TRANS_KEY]
    )

    orig_num_rows_low_res = num_rows_low_res + 0
    orig_num_columns_low_res = num_columns_low_res + 0

    num_extra_rowcols = 2 * int(numpy.ceil(
        data_aug_mean_translation_low_res_px +
        5 * data_aug_stdev_translation_low_res_px
    ))
    num_rows_low_res += num_extra_rowcols
    num_columns_low_res += num_extra_rowcols

    cyclone_id_strings = satellite_io.find_cyclones(
        directory_name=satellite_dir_name, raise_error_if_all_missing=True
    )
    cyclone_years = numpy.array(
        [misc_utils.parse_cyclone_id(c)[0] for c in cyclone_id_strings],
        dtype=int
    )

    good_flags = numpy.array([y in years for y in cyclone_years], dtype=bool)
    good_indices = numpy.where(good_flags)[0]
    cyclone_id_strings = [cyclone_id_strings[k] for k in good_indices]
    random.shuffle(cyclone_id_strings)

    satellite_file_names_by_cyclone = [
        satellite_io.find_files_one_cyclone(
            directory_name=satellite_dir_name, cyclone_id_string=c,
            raise_error_if_all_missing=True
        ) for c in cyclone_id_strings
    ]

    cyclone_index = 0

    while True:
        brightness_temp_matrix_kelvins = None
        grid_spacings_km = None
        cyclone_center_latitudes_deg_n = None
        num_examples_in_memory = 0

        while num_examples_in_memory < num_examples_per_batch:
            if cyclone_index == len(cyclone_id_strings):
                cyclone_index = 0

            num_examples_to_read = min([
                max_examples_per_cyclone,
                num_examples_per_batch - num_examples_in_memory
            ])

            data_dict = _read_satellite_data_1cyclone_simple(
                input_file_names=satellite_file_names_by_cyclone[cyclone_index],
                lag_times_minutes=lag_times_minutes,
                num_rows_low_res=num_rows_low_res,
                num_columns_low_res=num_columns_low_res,
                return_coords=False, num_target_times=num_examples_to_read
            )

            if data_dict is None:
                continue

            this_bt_matrix_kelvins = data_dict[BRIGHTNESS_TEMPS_KEY]
            these_grid_spacings_km = data_dict[GRID_SPACINGS_KEY]
            these_center_latitudes_deg_n = data_dict[CENTER_LATITUDES_KEY]

            if this_bt_matrix_kelvins is None:
                continue
            if this_bt_matrix_kelvins.size == 0:
                continue

            this_bt_matrix_kelvins = combine_lag_times_and_wavelengths(
                this_bt_matrix_kelvins
            )

            if brightness_temp_matrix_kelvins is None:
                these_dim = (
                    (num_examples_per_batch,) + this_bt_matrix_kelvins.shape[1:]
                )
                brightness_temp_matrix_kelvins = numpy.full(
                    these_dim, numpy.nan
                )

                grid_spacings_km = numpy.full(num_examples_per_batch, numpy.nan)
                cyclone_center_latitudes_deg_n = numpy.full(
                    num_examples_per_batch, numpy.nan
                )

            first_index = num_examples_in_memory
            last_index = first_index + this_bt_matrix_kelvins.shape[0]
            brightness_temp_matrix_kelvins[first_index:last_index, ...] = (
                this_bt_matrix_kelvins
            )
            grid_spacings_km[first_index:last_index] = these_grid_spacings_km
            cyclone_center_latitudes_deg_n[first_index:last_index] = (
                these_center_latitudes_deg_n
            )

            cyclone_index += 1
            num_examples_in_memory += this_bt_matrix_kelvins.shape[0]

        (
            _, brightness_temp_matrix_kelvins,
            row_translations_low_res_px, column_translations_low_res_px
        ) = _augment_data(
            bidirectional_reflectance_matrix=None,
            brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
            num_translations_per_example=data_aug_num_translations,
            mean_translation_low_res_px=data_aug_mean_translation_low_res_px,
            stdev_translation_low_res_px=data_aug_stdev_translation_low_res_px,
            sentinel_value=-10.
        )

        brightness_temp_matrix_kelvins = _subset_grid_after_data_aug(
            data_matrix=brightness_temp_matrix_kelvins,
            num_rows_to_keep=orig_num_rows_low_res,
            num_columns_to_keep=orig_num_columns_low_res,
            for_high_res=False
        )

        grid_spacings_km = numpy.repeat(
            grid_spacings_km, repeats=data_aug_num_translations
        )
        cyclone_center_latitudes_deg_n = numpy.repeat(
            cyclone_center_latitudes_deg_n, repeats=data_aug_num_translations
        )
        predictor_matrices = [brightness_temp_matrix_kelvins]

        # TODO(thunderhoser): This could be simplified more.
        target_matrix = numpy.transpose(numpy.vstack((
            row_translations_low_res_px, column_translations_low_res_px,
            grid_spacings_km, cyclone_center_latitudes_deg_n
        )))

        predictor_matrices = [p.astype('float16') for p in predictor_matrices]
        yield predictor_matrices, target_matrix


def data_generator(option_dict):
    """Generates input data for neural net.

    E = batch size = number of examples after data augmentation
    L = number of lag times for predictors
    W = number of high-res wavelengths
    w = number of low-res wavelengths
    M = number of rows in high-res grid
    m = number of rows in low-res grid = M/4
    N = number of columns in high-res grid
    n = number of columns in low-res grid = M/4

    :param option_dict: Dictionary with the following keys.
    option_dict["satellite_dir_name"]: Name of directory with satellite data.
        Files therein will be found by `satellite_io.find_file` and read by
        `satellite_io.read_file`.
    option_dict["years"]: 1-D numpy array of years.  Will generate data only for
        cyclones in these years.
    option_dict["lag_times_minutes"]: length-L numpy array of lag times for
        predictors.
    option_dict["high_res_wavelengths_microns"]: length-W numpy array of
        wavelengths for high-resolution (visible) satellite data.
    option_dict["low_res_wavelengths_microns"]: length-w numpy array of
        wavelengths for low-resolution (infrared) satellite data.
    option_dict["num_examples_per_batch"]: Batch size before data augmentation.
    option_dict["max_examples_per_cyclone"]: Max number of examples per cyclone
        in one batch -- again, before data augmentation.
    option_dict["num_rows_low_res"]: Number of grid rows to retain in low-
        resolution (infrared) satellite data.  This is m in the above
        definitions.
    option_dict["num_columns_low_res"]: Same but for columns.  This is n in the
        above definitions.
    option_dict["data_aug_num_translations"]: Number of translations for each
        cyclone.  Total batch size will be
        num_examples_per_batch * data_aug_num_translations.
    option_dict["data_aug_mean_translation_low_res_px"]: Mean translation
        distance (in units of low-resolution pixels) for data augmentation.
    option_dict["data_aug_stdev_translation_low_res_px"]: Standard deviation of
        translation distance (in units of low-resolution pixels) for data
        augmentation.
    option_dict["lag_time_tolerance_sec"]: Tolerance for lag times.
    option_dict["max_num_missing_lag_times"]: Max number of missing lag times
        for a given data example.
    option_dict["max_interp_gap_sec"]: Max gap (seconds) for interpolation to
        missing lag time.
    option_dict["sentinel_value"]: Sentinel value (will be used to replace NaN).

    :return: predictor_matrices: If both high- and low-resolution data are
        desired, this will be a list with the items
        [bidirectional_reflectance_matrix, brightness_temp_matrix_kelvins].
        If only low-res data are desired, this will be a list with one item,
        [brightness_temp_matrix_kelvins].

        bidirectional_reflectance_matrix: E-by-M-by-N-by-(W * L) numpy array
            of reflectance values (unitless).

        brightness_temp_matrix_kelvins: T-by-m-by-n-by-(w * L) numpy array of
            brightness temperatures.

    :return: target_matrix: If the problem has been cast as semantic
        segmentation...

        E-by-m-by-n numpy array of true TC-center "probabilities," in range
        0...1.

    If the problem has been cast as predicting two scalars (x- and y-coords)...

        E-by-4 numpy array with distances (in low-resolution pixels) between the
        image center and actual cyclone center.  target_matrix[:, 0] contains
        row offsets, and target_matrix[:, 1] contains column offsets.  For
        example, if target_matrix[20, 0] = -2 and target_matrix[20, 1] = 3, this
        means that the true cyclone center for the 21st example is 2 rows above,
        and 3 columns to the right of, the image center.

        target_matrix[:, 2] contains the grid spacing for each data sample in
        km.

        target_matrix[:, 3] contains the true latitude of each cyclone center in
        deg north.
    """

    option_dict = _check_generator_args(option_dict)

    satellite_dir_name = option_dict[SATELLITE_DIRECTORY_KEY]
    years = option_dict[YEARS_KEY]
    lag_times_minutes = option_dict[LAG_TIMES_KEY]
    high_res_wavelengths_microns = option_dict[HIGH_RES_WAVELENGTHS_KEY]  # Not in simple.
    low_res_wavelengths_microns = option_dict[LOW_RES_WAVELENGTHS_KEY]
    num_examples_per_batch = option_dict[BATCH_SIZE_KEY]
    max_examples_per_cyclone = option_dict[MAX_EXAMPLES_PER_CYCLONE_KEY]
    num_rows_low_res = option_dict[NUM_GRID_ROWS_KEY]
    num_columns_low_res = option_dict[NUM_GRID_COLUMNS_KEY]
    data_aug_num_translations = option_dict[DATA_AUG_NUM_TRANS_KEY]
    data_aug_mean_translation_low_res_px = option_dict[DATA_AUG_MEAN_TRANS_KEY]
    data_aug_stdev_translation_low_res_px = (
        option_dict[DATA_AUG_STDEV_TRANS_KEY]
    )
    lag_time_tolerance_sec = option_dict[LAG_TIME_TOLERANCE_KEY]  # Not in simple.
    max_num_missing_lag_times = option_dict[MAX_MISSING_LAG_TIMES_KEY]  # Not in simple.
    max_interp_gap_sec = option_dict[MAX_INTERP_GAP_KEY]  # Not in simple.
    sentinel_value = option_dict[SENTINEL_VALUE_KEY]  # Not in simple.
    semantic_segmentation_flag = option_dict[SEMANTIC_SEG_FLAG_KEY]  # Not in simple.
    target_smoother_stdev_km = option_dict[TARGET_SMOOOTHER_STDEV_KEY]  # Not in simple.

    orig_num_rows_low_res = num_rows_low_res + 0
    orig_num_columns_low_res = num_columns_low_res + 0

    num_extra_rowcols = 2 * int(numpy.ceil(
        data_aug_mean_translation_low_res_px +
        5 * data_aug_stdev_translation_low_res_px
    ))
    num_rows_low_res += num_extra_rowcols
    num_columns_low_res += num_extra_rowcols

    cyclone_id_strings = satellite_io.find_cyclones(
        directory_name=satellite_dir_name, raise_error_if_all_missing=True
    )
    cyclone_years = numpy.array(
        [misc_utils.parse_cyclone_id(c)[0] for c in cyclone_id_strings],
        dtype=int
    )

    good_flags = numpy.array([y in years for y in cyclone_years], dtype=bool)
    good_indices = numpy.where(good_flags)[0]
    cyclone_id_strings = [cyclone_id_strings[k] for k in good_indices]
    random.shuffle(cyclone_id_strings)

    satellite_file_names_by_cyclone = [
        satellite_io.find_files_one_cyclone(
            directory_name=satellite_dir_name, cyclone_id_string=c,
            raise_error_if_all_missing=True
        ) for c in cyclone_id_strings
    ]

    cyclone_index = 0

    while True:
        bidirectional_reflectance_matrix = None
        brightness_temp_matrix_kelvins = None
        grid_spacings_km = None
        cyclone_center_latitudes_deg_n = None
        num_examples_in_memory = 0

        while num_examples_in_memory < num_examples_per_batch:
            if cyclone_index == len(cyclone_id_strings):
                cyclone_index = 0

            num_examples_to_read = min([
                max_examples_per_cyclone,
                num_examples_per_batch - num_examples_in_memory
            ])

            data_dict = _read_satellite_data_one_cyclone(
                input_file_names=satellite_file_names_by_cyclone[cyclone_index],
                num_target_times=num_examples_to_read,
                lag_times_minutes=lag_times_minutes,
                lag_time_tolerance_sec=lag_time_tolerance_sec,
                max_num_missing_lag_times=max_num_missing_lag_times,
                max_interp_gap_sec=max_interp_gap_sec,
                high_res_wavelengths_microns=high_res_wavelengths_microns,
                low_res_wavelengths_microns=low_res_wavelengths_microns,
                num_rows_low_res=num_rows_low_res,
                num_columns_low_res=num_columns_low_res,
                sentinel_value=sentinel_value, return_coords=False
            )

            if data_dict is None:
                continue

            this_reflectance_matrix = data_dict[BIDIRECTIONAL_REFLECTANCES_KEY]
            this_bt_matrix_kelvins = data_dict[BRIGHTNESS_TEMPS_KEY]
            these_grid_spacings_km = data_dict[GRID_SPACINGS_KEY]
            these_center_latitudes_deg_n = data_dict[CENTER_LATITUDES_KEY]

            if this_bt_matrix_kelvins is None:
                continue
            if this_bt_matrix_kelvins.size == 0:
                continue

            this_bt_matrix_kelvins = combine_lag_times_and_wavelengths(
                this_bt_matrix_kelvins
            )
            if this_reflectance_matrix is not None:
                this_reflectance_matrix = combine_lag_times_and_wavelengths(
                    this_reflectance_matrix
                )

            if brightness_temp_matrix_kelvins is None:
                these_dim = (
                    (num_examples_per_batch,) + this_bt_matrix_kelvins.shape[1:]
                )
                brightness_temp_matrix_kelvins = numpy.full(
                    these_dim, numpy.nan
                )

                grid_spacings_km = numpy.full(num_examples_per_batch, numpy.nan)
                cyclone_center_latitudes_deg_n = numpy.full(
                    num_examples_per_batch, numpy.nan
                )

                if this_reflectance_matrix is not None:
                    these_dim = (
                        (num_examples_per_batch,) +
                        this_reflectance_matrix.shape[1:]
                    )
                    bidirectional_reflectance_matrix = numpy.full(
                        these_dim, numpy.nan
                    )

            first_index = num_examples_in_memory
            last_index = first_index + this_bt_matrix_kelvins.shape[0]
            brightness_temp_matrix_kelvins[first_index:last_index, ...] = (
                this_bt_matrix_kelvins
            )
            grid_spacings_km[first_index:last_index] = these_grid_spacings_km
            cyclone_center_latitudes_deg_n[first_index:last_index] = (
                these_center_latitudes_deg_n
            )

            if this_reflectance_matrix is not None:
                bidirectional_reflectance_matrix[
                    first_index:last_index, ...
                ] = this_reflectance_matrix

            cyclone_index += 1
            num_examples_in_memory += this_bt_matrix_kelvins.shape[0]

        (
            bidirectional_reflectance_matrix, brightness_temp_matrix_kelvins,
            row_translations_low_res_px, column_translations_low_res_px
        ) = _augment_data(
            bidirectional_reflectance_matrix=bidirectional_reflectance_matrix,
            brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
            num_translations_per_example=data_aug_num_translations,
            mean_translation_low_res_px=data_aug_mean_translation_low_res_px,
            stdev_translation_low_res_px=data_aug_stdev_translation_low_res_px,
            sentinel_value=sentinel_value
        )

        brightness_temp_matrix_kelvins = _subset_grid_after_data_aug(
            data_matrix=brightness_temp_matrix_kelvins,
            num_rows_to_keep=orig_num_rows_low_res,
            num_columns_to_keep=orig_num_columns_low_res,
            for_high_res=False
        )

        if bidirectional_reflectance_matrix is not None:
            bidirectional_reflectance_matrix = _subset_grid_after_data_aug(
                data_matrix=bidirectional_reflectance_matrix,
                num_rows_to_keep=orig_num_rows_low_res * 4,
                num_columns_to_keep=orig_num_columns_low_res * 4,
                for_high_res=True
            )

        grid_spacings_km = numpy.repeat(
            grid_spacings_km, repeats=data_aug_num_translations
        )
        cyclone_center_latitudes_deg_n = numpy.repeat(
            cyclone_center_latitudes_deg_n, repeats=data_aug_num_translations
        )

        predictor_matrices = [brightness_temp_matrix_kelvins]
        if bidirectional_reflectance_matrix is not None:
            predictor_matrices.insert(0, bidirectional_reflectance_matrix)

        if semantic_segmentation_flag:
            target_matrix = _make_targets_for_semantic_seg(
                row_translations_px=row_translations_low_res_px,
                column_translations_px=column_translations_low_res_px,
                grid_spacings_km=grid_spacings_km,
                cyclone_center_latitudes_deg_n=cyclone_center_latitudes_deg_n,
                gaussian_smoother_stdev_km=target_smoother_stdev_km,
                num_grid_rows=orig_num_rows_low_res,
                num_grid_columns=orig_num_columns_low_res
            )
        else:
            target_matrix = numpy.transpose(numpy.vstack((
                row_translations_low_res_px, column_translations_low_res_px,
                grid_spacings_km, cyclone_center_latitudes_deg_n
            )))

        predictor_matrices = [p.astype('float16') for p in predictor_matrices]
        yield predictor_matrices, target_matrix


def train_model_simple(
        model_object, output_dir_name, num_epochs,
        num_training_batches_per_epoch, training_option_dict,
        num_validation_batches_per_epoch, validation_option_dict,
        loss_function_string, optimizer_function_string,
        plateau_patience_epochs, plateau_learning_rate_multiplier,
        early_stopping_patience_epochs, architecture_dict, is_model_bnn):
    """Trains neural net with generator.

    :param model_object: See doc for `train_model`.
    :param output_dir_name: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validation_option_dict: Same.
    :param loss_function_string: Same.
    :param optimizer_function_string: Same.
    :param plateau_patience_epochs: Same.
    :param plateau_learning_rate_multiplier: Same.
    :param early_stopping_patience_epochs: Same.
    :param architecture_dict: Same.
    :param is_model_bnn: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_geq(num_epochs, 2)
    error_checking.assert_is_integer(num_training_batches_per_epoch)
    error_checking.assert_is_geq(num_training_batches_per_epoch, 2)
    error_checking.assert_is_integer(num_validation_batches_per_epoch)
    error_checking.assert_is_geq(num_validation_batches_per_epoch, 2)
    error_checking.assert_is_integer(plateau_patience_epochs)
    error_checking.assert_is_geq(plateau_patience_epochs, 2)
    error_checking.assert_is_greater(plateau_learning_rate_multiplier, 0.)
    error_checking.assert_is_less_than(plateau_learning_rate_multiplier, 1.)
    error_checking.assert_is_integer(early_stopping_patience_epochs)
    error_checking.assert_is_geq(early_stopping_patience_epochs, 5)
    error_checking.assert_is_boolean(is_model_bnn)

    # TODO(thunderhoser): Maybe I should just max out the last 3 arguments and
    # not let the user set them?
    validation_keys_to_keep = [
        SATELLITE_DIRECTORY_KEY, YEARS_KEY,
        LAG_TIME_TOLERANCE_KEY, MAX_MISSING_LAG_TIMES_KEY, MAX_INTERP_GAP_KEY
    ]
    for this_key in list(training_option_dict.keys()):
        if this_key in validation_keys_to_keep:
            continue

        validation_option_dict[this_key] = training_option_dict[this_key]

    training_option_dict = _check_generator_args(training_option_dict)
    validation_option_dict = _check_generator_args(validation_option_dict)

    model_file_name = '{0:s}/model.h5'.format(output_dir_name)

    history_object = keras.callbacks.CSVLogger(
        filename='{0:s}/history.csv'.format(output_dir_name),
        separator=',', append=False
    )
    checkpoint_object = keras.callbacks.ModelCheckpoint(
        filepath=model_file_name, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, mode='min', period=1
    )
    early_stopping_object = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.,
        patience=early_stopping_patience_epochs, verbose=1, mode='min'
    )
    plateau_object = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=plateau_learning_rate_multiplier,
        patience=plateau_patience_epochs, verbose=1, mode='min',
        min_delta=0., cooldown=0
    )

    list_of_callback_objects = [
        history_object, checkpoint_object, early_stopping_object, plateau_object
    ]

    training_generator = data_generator_simple(training_option_dict)
    validation_generator = data_generator_simple(validation_option_dict)

    metafile_name = find_metafile(
        model_dir_name=output_dir_name, raise_error_if_missing=False
    )
    print('Writing metadata to: "{0:s}"...'.format(metafile_name))

    _write_metafile(
        pickle_file_name=metafile_name, num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_option_dict=training_option_dict,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_option_dict=validation_option_dict,
        loss_function_string=loss_function_string,
        optimizer_function_string=optimizer_function_string,
        plateau_patience_epochs=plateau_patience_epochs,
        plateau_learning_rate_multiplier=plateau_learning_rate_multiplier,
        early_stopping_patience_epochs=early_stopping_patience_epochs,
        architecture_dict=architecture_dict,
        is_model_bnn=is_model_bnn
    )

    model_object.fit_generator(
        generator=training_generator,
        steps_per_epoch=num_training_batches_per_epoch,
        epochs=num_epochs, verbose=1, callbacks=list_of_callback_objects,
        validation_data=validation_generator,
        validation_steps=num_validation_batches_per_epoch
    )


def train_model(
        model_object, output_dir_name, num_epochs,
        num_training_batches_per_epoch, training_option_dict,
        num_validation_batches_per_epoch, validation_option_dict,
        loss_function_string, optimizer_function_string,
        plateau_patience_epochs, plateau_learning_rate_multiplier,
        early_stopping_patience_epochs, architecture_dict, is_model_bnn):
    """Trains neural net with generator.

    :param model_object: Untrained neural net (instance of `keras.models.Model`
        or `keras.models.Sequential`).
    :param output_dir_name: Path to output directory (model and training history
        will be saved here).
    :param num_epochs: Number of training epochs.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param training_option_dict: See doc for `data_generator`.  This dictionary
        will be used to generate training data.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.
    :param validation_option_dict: See doc for `data_generator`.  For validation
        only, the following values will replace corresponding values in
        `training_option_dict`:
    validation_option_dict["satellite_dir_name"]
    validation_option_dict["years"]
    validation_option_dict["lag_time_tolerance_sec"]
    validation_option_dict["max_num_missing_lag_times"]
    validation_option_dict["max_interp_gap_sec"]

    :param loss_function_string: Loss function.  This string should be formatted
        such that `eval(loss_function_string)` returns the actual loss function.
    :param optimizer_function_string: Optimizer.  This string should be
        formatted such that `eval(optimizer_function_string)` returns the actual
        optimizer function.
    :param plateau_patience_epochs: Training will be deemed to have reached
        "plateau" if validation loss has not decreased in the last N epochs,
        where N = plateau_patience_epochs.
    :param plateau_learning_rate_multiplier: If training reaches "plateau,"
        learning rate will be multiplied by this value in range (0, 1).
    :param early_stopping_patience_epochs: Training will be stopped early if
        validation loss has not decreased in the last N epochs, where N =
        early_stopping_patience_epochs.
    :param architecture_dict: Dictionary with architecture options for neural
        network.
    :param is_model_bnn: Boolean flag.  If True, will assume that model is a
        Bayesian neural network.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_geq(num_epochs, 2)
    error_checking.assert_is_integer(num_training_batches_per_epoch)
    error_checking.assert_is_geq(num_training_batches_per_epoch, 2)
    error_checking.assert_is_integer(num_validation_batches_per_epoch)
    error_checking.assert_is_geq(num_validation_batches_per_epoch, 2)
    error_checking.assert_is_integer(plateau_patience_epochs)
    error_checking.assert_is_geq(plateau_patience_epochs, 2)
    error_checking.assert_is_greater(plateau_learning_rate_multiplier, 0.)
    error_checking.assert_is_less_than(plateau_learning_rate_multiplier, 1.)
    error_checking.assert_is_integer(early_stopping_patience_epochs)
    error_checking.assert_is_geq(early_stopping_patience_epochs, 5)
    error_checking.assert_is_boolean(is_model_bnn)

    # TODO(thunderhoser): Maybe I should just max out the last 3 arguments and
    # not let the user set them?
    validation_keys_to_keep = [
        SATELLITE_DIRECTORY_KEY, YEARS_KEY,
        LAG_TIME_TOLERANCE_KEY, MAX_MISSING_LAG_TIMES_KEY, MAX_INTERP_GAP_KEY
    ]
    for this_key in list(training_option_dict.keys()):
        if this_key in validation_keys_to_keep:
            continue

        validation_option_dict[this_key] = training_option_dict[this_key]

    training_option_dict = _check_generator_args(training_option_dict)
    validation_option_dict = _check_generator_args(validation_option_dict)

    model_file_name = '{0:s}/model.h5'.format(output_dir_name)

    history_object = keras.callbacks.CSVLogger(
        filename='{0:s}/history.csv'.format(output_dir_name),
        separator=',', append=False
    )
    checkpoint_object = keras.callbacks.ModelCheckpoint(
        filepath=model_file_name, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, mode='min', period=1
    )
    early_stopping_object = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.,
        patience=early_stopping_patience_epochs, verbose=1, mode='min'
    )
    plateau_object = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=plateau_learning_rate_multiplier,
        patience=plateau_patience_epochs, verbose=1, mode='min',
        min_delta=0., cooldown=0
    )

    list_of_callback_objects = [
        history_object, checkpoint_object, early_stopping_object, plateau_object
    ]

    training_generator = data_generator(training_option_dict)
    validation_generator = data_generator(validation_option_dict)

    metafile_name = find_metafile(
        model_dir_name=output_dir_name, raise_error_if_missing=False
    )
    print('Writing metadata to: "{0:s}"...'.format(metafile_name))

    _write_metafile(
        pickle_file_name=metafile_name, num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_option_dict=training_option_dict,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_option_dict=validation_option_dict,
        loss_function_string=loss_function_string,
        optimizer_function_string=optimizer_function_string,
        plateau_patience_epochs=plateau_patience_epochs,
        plateau_learning_rate_multiplier=plateau_learning_rate_multiplier,
        early_stopping_patience_epochs=early_stopping_patience_epochs,
        architecture_dict=architecture_dict,
        is_model_bnn=is_model_bnn
    )

    model_object.fit_generator(
        generator=training_generator,
        steps_per_epoch=num_training_batches_per_epoch,
        epochs=num_epochs, verbose=1, callbacks=list_of_callback_objects,
        validation_data=validation_generator,
        validation_steps=num_validation_batches_per_epoch
    )


def apply_model(
        model_object, predictor_matrices, num_examples_per_batch, verbose=True):
    """Applies trained neural net -- inference time!

    E = number of examples
    M = number of rows in grid
    N = number of columns grid
    S = ensemble size

    :param model_object: Trained neural net (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param predictor_matrices: See output doc for `data_generator`.
    :param num_examples_per_batch: Batch size.
    :param verbose: Boolean flag.  If True, will print progress messages.
    :return: prediction_matrix: If the model predicts scalar coordinates...

        E-by-2-by-S numpy array.  prediction_tensor[:, 0, :] contains predicted
        row positions of TC centers, and prediction_tensor[:, 1, :] contains
        predicted column positions of TC centers.

    If the model predicts gridded probabilities...

        E-by-M-by-N-by-S numpy array of probabilities.
    """

    # Check input args.
    for this_matrix in predictor_matrices:
        error_checking.assert_is_numpy_array_without_nan(this_matrix)

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 1)
    num_examples = predictor_matrices[0].shape[0]
    num_examples_per_batch = min([num_examples_per_batch, num_examples])

    error_checking.assert_is_boolean(verbose)

    # Do actual stuff.
    prediction_matrix = None

    for i in range(0, num_examples, num_examples_per_batch):
        first_index = i
        last_index = min([i + num_examples_per_batch, num_examples])

        if verbose:
            print('Applying model to examples {0:d}-{1:d} of {2:d}...'.format(
                first_index + 1, last_index, num_examples
            ))

        this_prediction_matrix = model_object.predict_on_batch(
            [a[first_index:last_index, ...] for a in predictor_matrices]
        )

        if prediction_matrix is None:
            dimensions = (num_examples,) + this_prediction_matrix.shape[1:]
            prediction_matrix = numpy.full(dimensions, numpy.nan)

        prediction_matrix[first_index:last_index, ...] = this_prediction_matrix

    if verbose:
        print('Have applied model to all {0:d} examples!'.format(num_examples))

    return prediction_matrix


def find_metafile(model_dir_name, raise_error_if_missing=True):
    """Finds metafile for neural net.

    :param model_dir_name: Name of model directory.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: metafile_name: Path to metafile.
    """

    error_checking.assert_is_string(model_dir_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    metafile_name = '{0:s}/model_metadata.p'.format(model_dir_name)

    if raise_error_if_missing and not os.path.isfile(metafile_name):
        metafile_name = metafile_name.replace(
            '/scratch1/RDARCH',
            '/home/ralager/condo/swatwork/ralager/scratch1/RDARCH'
        )

    if raise_error_if_missing and not os.path.isfile(metafile_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            metafile_name
        )
        raise ValueError(error_string)

    return metafile_name


def read_metafile(pickle_file_name):
    """Reads metadata from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict["num_epochs"]: See doc for `train_model`.
    metadata_dict["num_training_batches_per_epoch"]: Same.
    metadata_dict["training_option_dict"]: Same.
    metadata_dict["num_validation_batches_per_epoch"]: Same.
    metadata_dict["validation_option_dict"]: Same.
    metadata_dict["loss_function_string"]: Same.
    metadata_dict["optimizer_function_string"]: Same.
    metadata_dict["plateau_patience_epochs"]: Same.
    metadata_dict["plateau_learning_rate_multiplier"]: Same.
    metadata_dict["early_stopping_patience_epochs"]: Same.
    metadata_dict["architecture_dict"]: Same.
    metadata_dict["is_model_bnn"]: Same.

    :raises: ValueError: if any expected key is not found in dictionary.
    """

    error_checking.assert_file_exists(pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'rb')
    metadata_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    if OPTIMIZER_FUNCTION_KEY not in metadata_dict:
        metadata_dict[OPTIMIZER_FUNCTION_KEY] = 'keras.optimizers.Adam()'

    if IS_MODEL_BNN_KEY not in metadata_dict:
        metadata_dict[IS_MODEL_BNN_KEY] = False

    if ARCHITECTURE_KEY not in metadata_dict:
        metadata_dict[ARCHITECTURE_KEY] = None

    training_option_dict = metadata_dict[TRAINING_OPTIONS_KEY]
    validation_option_dict = metadata_dict[VALIDATION_OPTIONS_KEY]

    if training_option_dict is None:
        training_option_dict = dict()
        validation_option_dict = dict()

    if SEMANTIC_SEG_FLAG_KEY not in training_option_dict:
        training_option_dict[SEMANTIC_SEG_FLAG_KEY] = False
        validation_option_dict[SEMANTIC_SEG_FLAG_KEY] = False

    if TARGET_SMOOOTHER_STDEV_KEY not in training_option_dict:
        training_option_dict[TARGET_SMOOOTHER_STDEV_KEY] = None
        validation_option_dict[TARGET_SMOOOTHER_STDEV_KEY] = None

    metadata_dict[TRAINING_OPTIONS_KEY] = training_option_dict
    metadata_dict[VALIDATION_OPTIONS_KEY] = validation_option_dict

    missing_keys = list(set(METADATA_KEYS) - set(metadata_dict.keys()))
    if len(missing_keys) == 0:
        return metadata_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)


def read_model(hdf5_file_name):
    """Reads model from HDF5 file.

    :param hdf5_file_name: Path to input file.
    :return: model_object: Instance of `keras.models.Model` or
        `keras.models.Sequential`.
    """

    error_checking.assert_file_exists(hdf5_file_name)

    metafile_name = find_metafile(
        model_dir_name=os.path.split(hdf5_file_name)[0],
        raise_error_if_missing=True
    )
    metadata_dict = read_metafile(metafile_name)
    architecture_dict = metadata_dict[ARCHITECTURE_KEY]
    is_model_bnn = metadata_dict[IS_MODEL_BNN_KEY]
    training_option_dict = metadata_dict[TRAINING_OPTIONS_KEY]
    semantic_segmentation_flag = training_option_dict[SEMANTIC_SEG_FLAG_KEY]

    if architecture_dict is not None:
        if is_model_bnn:
            import cnn_architecture_bayesian

            for this_key in [
                    cnn_architecture_bayesian.LOSS_FUNCTION_KEY,
                    cnn_architecture_bayesian.OPTIMIZER_FUNCTION_KEY
            ]:
                try:
                    architecture_dict[this_key] = eval(
                        architecture_dict[this_key]
                    )
                except NameError:
                    architecture_dict[this_key] = eval(
                        architecture_dict[this_key].replace(
                            'custom_losses', 'custom_losses_scalar'
                        )
                    )

            model_object = cnn_architecture_bayesian.create_model(
                architecture_dict
            )
        else:
            if semantic_segmentation_flag:
                for this_key in [
                        u_net_architecture.LOSS_FUNCTION_KEY,
                        u_net_architecture.OPTIMIZER_FUNCTION_KEY
                ]:
                    try:
                        architecture_dict[this_key] = eval(
                            architecture_dict[this_key]
                        )
                    except NameError:
                        architecture_dict[this_key] = eval(
                            architecture_dict[this_key].replace(
                                'custom_losses', 'custom_losses_scalar'
                            )
                        )

                model_object = u_net_architecture.create_model(
                    architecture_dict
                )
            else:
                for this_key in [
                        cnn_architecture.LOSS_FUNCTION_KEY,
                        cnn_architecture.OPTIMIZER_FUNCTION_KEY
                ]:
                    try:
                        architecture_dict[this_key] = eval(
                            architecture_dict[this_key]
                        )
                    except NameError:
                        architecture_dict[this_key] = eval(
                            architecture_dict[this_key].replace(
                                'custom_losses', 'custom_losses_scalar'
                            )
                        )

                model_object = cnn_architecture.create_model(architecture_dict)

        model_object.load_weights(hdf5_file_name)
        return model_object

    # TODO(thunderhoser): This code should never be reached.
    custom_object_dict = copy.deepcopy(METRIC_FUNCTION_DICT_SCALAR)
    custom_object_dict['loss'] = eval(metadata_dict[LOSS_FUNCTION_KEY])

    return tf_keras.models.load_model(
        hdf5_file_name, custom_objects=custom_object_dict
    )
