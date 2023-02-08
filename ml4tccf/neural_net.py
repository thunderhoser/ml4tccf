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

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import file_system_utils
import error_checking
import satellite_io
import misc_utils
import satellite_utils
import custom_losses

METRIC_FUNCTION_LIST = [
    custom_losses.mean_distance_kilometres,
    custom_losses.mean_squared_distance_kilometres2,
    custom_losses.mean_prediction,
    custom_losses.mean_predictive_stdev,
    custom_losses.mean_predictive_range,
    custom_losses.mean_target,
    custom_losses.mean_grid_spacing_kilometres,
    custom_losses.crps_part1,
    custom_losses.crps_part2,
    custom_losses.crps_part3,
    custom_losses.crps_part4,
    custom_losses.crps_kilometres
]

METRIC_FUNCTION_DICT = {
    'mean_distance_kilometres': custom_losses.mean_distance_kilometres,
    'mean_squared_distance_kilometres2':
        custom_losses.mean_squared_distance_kilometres2,
    'mean_prediction': custom_losses.mean_prediction,
    'mean_predictive_stdev': custom_losses.mean_predictive_stdev,
    'mean_predictive_range': custom_losses.mean_predictive_range,
    'mean_target': custom_losses.mean_target,
    'mean_grid_spacing_kilometres': custom_losses.mean_grid_spacing_kilometres,
    'crps_part1': custom_losses.crps_part1,
    'crps_part2': custom_losses.crps_part2,
    'crps_part3': custom_losses.crps_part3,
    'crps_part4': custom_losses.crps_part4,
    'crps_kilometres': custom_losses.crps_kilometres
}

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

DEFAULT_GENERATOR_OPTION_DICT = {
    HIGH_RES_WAVELENGTHS_KEY: None,
    LOW_RES_WAVELENGTHS_KEY: None,
    BATCH_SIZE_KEY: 8,
    MAX_EXAMPLES_PER_CYCLONE_KEY: 1,
    NUM_GRID_ROWS_KEY: None,
    NUM_GRID_COLUMNS_KEY: None,
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
PLATEAU_PATIENCE_KEY = 'plateau_patience_epochs'
PLATEAU_LR_MUTIPLIER_KEY = 'plateau_learning_rate_multiplier'
EARLY_STOPPING_PATIENCE_KEY = 'early_stopping_patience_epochs'
BNN_ARCHITECTURE_KEY = 'bnn_architecture_dict'

METADATA_KEYS = [
    NUM_EPOCHS_KEY, NUM_TRAINING_BATCHES_KEY, TRAINING_OPTIONS_KEY,
    NUM_VALIDATION_BATCHES_KEY, VALIDATION_OPTIONS_KEY, LOSS_FUNCTION_KEY,
    PLATEAU_PATIENCE_KEY, PLATEAU_LR_MUTIPLIER_KEY, EARLY_STOPPING_PATIENCE_KEY,
    BNN_ARCHITECTURE_KEY
]


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
    error_checking.assert_is_greater(option_dict[BATCH_SIZE_KEY], 1)

    error_checking.assert_is_integer(option_dict[MAX_EXAMPLES_PER_CYCLONE_KEY])
    error_checking.assert_is_geq(option_dict[MAX_EXAMPLES_PER_CYCLONE_KEY], 1)
    error_checking.assert_is_geq(
        option_dict[BATCH_SIZE_KEY],
        option_dict[MAX_EXAMPLES_PER_CYCLONE_KEY]
    )

    if option_dict[NUM_GRID_ROWS_KEY] is not None:
        error_checking.assert_is_integer(option_dict[NUM_GRID_ROWS_KEY])
        error_checking.assert_is_geq(option_dict[NUM_GRID_ROWS_KEY], 100)
        assert numpy.mod(option_dict[NUM_GRID_ROWS_KEY], 2) == 0

    if option_dict[NUM_GRID_COLUMNS_KEY] is not None:
        error_checking.assert_is_integer(option_dict[NUM_GRID_COLUMNS_KEY])
        error_checking.assert_is_geq(option_dict[NUM_GRID_COLUMNS_KEY], 100)
        assert numpy.mod(option_dict[NUM_GRID_COLUMNS_KEY], 2) == 0

    error_checking.assert_is_integer(option_dict[DATA_AUG_NUM_TRANS_KEY])

    if option_dict[DATA_AUG_NUM_TRANS_KEY] > 0:
        error_checking.assert_is_greater(
            option_dict[DATA_AUG_MEAN_TRANS_KEY], 0.
        )
        error_checking.assert_is_greater(
            option_dict[DATA_AUG_STDEV_TRANS_KEY], 0.
        )
    else:
        option_dict[DATA_AUG_NUM_TRANS_KEY] = 0
        option_dict[DATA_AUG_MEAN_TRANS_KEY] = None
        option_dict[DATA_AUG_STDEV_TRANS_KEY] = None

    error_checking.assert_is_integer(option_dict[LAG_TIME_TOLERANCE_KEY])
    error_checking.assert_is_geq(option_dict[LAG_TIME_TOLERANCE_KEY], 0)
    error_checking.assert_is_integer(option_dict[MAX_MISSING_LAG_TIMES_KEY])
    error_checking.assert_is_geq(option_dict[MAX_MISSING_LAG_TIMES_KEY], 0)
    error_checking.assert_is_integer(option_dict[MAX_INTERP_GAP_KEY])
    error_checking.assert_is_geq(option_dict[MAX_INTERP_GAP_KEY], 0)
    error_checking.assert_is_not_nan(option_dict[SENTINEL_VALUE_KEY])

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


def _read_satellite_data_one_cyclone(
        input_file_names, num_target_times, lag_times_minutes,
        lag_time_tolerance_sec, max_num_missing_lag_times, max_interp_gap_sec,
        high_res_wavelengths_microns, low_res_wavelengths_microns,
        num_rows_low_res, num_columns_low_res, sentinel_value):
    """Reads satellite data for one cyclone.

    T = number of target times
    L = number of lag times
    W = number of high-res wavelengths
    w = number of low-res wavelengths
    M = number of rows in high-res grid
    m = number of rows in low-res grid = M/4
    N = number of columns in high-res grid
    n = number of columns in low-res grid = M/4

    :param input_file_names: 1-D list of paths to input files (will be read by
        `satellite_io.read_file`).
    :param num_target_times: T in the above definitions.
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
    :return: bidirectional_reflectance_matrix: T-by-M-by-N-by-L-by-W numpy array
        of reflectance values (unitless).
    :return: brightness_temp_matrix_kelvins: T-by-m-by-n-by-L-by-w numpy array
        of brightness temperatures.
    :return: grid_spacings_low_res_km: length-T numpy array of grid spacings for
        low-resolution data.
    """

    target_times_unix_sec = numpy.concatenate([
        xarray.open_dataset(f).coords[satellite_utils.TIME_DIM].values
        for f in input_file_names
    ])
    target_times_unix_sec = target_times_unix_sec[
        numpy.mod(target_times_unix_sec, INTERVAL_BETWEEN_TARGET_TIMES_SEC) == 0
    ]

    if len(target_times_unix_sec) == 0:
        return None, None, None

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

        if not (
                num_rows_low_res is None
                or num_columns_low_res is None
        ):
            orig_satellite_tables_xarray[i] = satellite_utils.subset_grid(
                satellite_table_xarray=orig_satellite_tables_xarray[i],
                num_rows_to_keep=num_rows_low_res,
                num_columns_to_keep=num_columns_low_res,
                for_high_res=False
            )

            orig_satellite_tables_xarray[i] = satellite_utils.subset_grid(
                satellite_table_xarray=orig_satellite_tables_xarray[i],
                num_rows_to_keep=4 * num_rows_low_res,
                num_columns_to_keep=4 * num_columns_low_res,
                for_high_res=True
            )

        orig_satellite_tables_xarray[i] = satellite_utils.subset_wavelengths(
            satellite_table_xarray=orig_satellite_tables_xarray[i],
            wavelengths_to_keep_microns=low_res_wavelengths_microns,
            for_high_res=False
        )

        orig_satellite_tables_xarray[i] = satellite_utils.subset_wavelengths(
            satellite_table_xarray=orig_satellite_tables_xarray[i],
            wavelengths_to_keep_microns=high_res_wavelengths_microns,
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

    grid_spacings_low_res_km = numpy.full(num_target_times, numpy.nan)

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
    else:
        bidirectional_reflectance_matrix = None

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

        these_x_diffs_metres = numpy.mean(
            numpy.diff(t[satellite_utils.X_COORD_LOW_RES_KEY].values, axis=1),
            axis=1
        )
        these_y_diffs_metres = numpy.mean(
            numpy.diff(t[satellite_utils.Y_COORD_LOW_RES_KEY].values, axis=1),
            axis=1
        )
        grid_spacings_low_res_km[i] = METRES_TO_KM * numpy.mean(
            numpy.concatenate((these_x_diffs_metres, these_y_diffs_metres))
        )

        if len(high_res_wavelengths_microns) == 0:
            continue

        this_refl_matrix_kelvins = numpy.swapaxes(
            t[satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY].values, 0, 1
        )
        bidirectional_reflectance_matrix[i, ...] = numpy.swapaxes(
            this_refl_matrix_kelvins, 1, 2
        )

    good_indices = numpy.where(target_time_success_flags)[0]

    if len(high_res_wavelengths_microns) > 0:
        bidirectional_reflectance_matrix = (
            bidirectional_reflectance_matrix[good_indices, ...]
        )
        bidirectional_reflectance_matrix[
            numpy.isnan(bidirectional_reflectance_matrix)
        ] = sentinel_value

    brightness_temp_matrix_kelvins = (
        brightness_temp_matrix_kelvins[good_indices, ...]
    )
    brightness_temp_matrix_kelvins[
        numpy.isnan(brightness_temp_matrix_kelvins)
    ] = sentinel_value

    grid_spacings_low_res_km = grid_spacings_low_res_km[good_indices]

    return (
        bidirectional_reflectance_matrix, brightness_temp_matrix_kelvins,
        grid_spacings_low_res_km
    )


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

    row_translations_low_res_px = numpy.random.normal(
        loc=mean_translation_low_res_px, scale=stdev_translation_low_res_px,
        size=num_examples_new
    )
    column_translations_low_res_px = numpy.random.normal(
        loc=mean_translation_low_res_px, scale=stdev_translation_low_res_px,
        size=num_examples_new
    )

    row_translations_low_res_px *= _get_random_signs(num_examples_new)
    column_translations_low_res_px *= _get_random_signs(num_examples_new)

    these_flags = numpy.logical_and(
        row_translations_low_res_px < 0, row_translations_low_res_px >= -0.5
    )
    row_translations_low_res_px[these_flags] = -1.

    these_flags = numpy.logical_and(
        row_translations_low_res_px > 0, row_translations_low_res_px <= 0.5
    )
    row_translations_low_res_px[these_flags] = 1.

    these_flags = numpy.logical_and(
        column_translations_low_res_px < 0,
        column_translations_low_res_px >= -0.5
    )
    column_translations_low_res_px[these_flags] = -1.

    these_flags = numpy.logical_and(
        column_translations_low_res_px > 0,
        column_translations_low_res_px <= 0.5
    )
    column_translations_low_res_px[these_flags] = 1.

    row_translations_low_res_px = (
        numpy.round(row_translations_low_res_px).astype(int)
    )
    column_translations_low_res_px = (
        numpy.round(column_translations_low_res_px).astype(int)
    )

    error_checking.assert_is_greater_numpy_array(
        numpy.absolute(row_translations_low_res_px), 0
    )
    error_checking.assert_is_greater_numpy_array(
        numpy.absolute(column_translations_low_res_px), 0
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


def _write_metafile(
        pickle_file_name, num_epochs, num_training_batches_per_epoch,
        training_option_dict, num_validation_batches_per_epoch,
        validation_option_dict, loss_function_string,
        plateau_patience_epochs, plateau_learning_rate_multiplier,
        early_stopping_patience_epochs, bnn_architecture_dict):
    """Writes metadata to Pickle file.

    :param pickle_file_name: Path to output file.
    :param num_epochs: See doc for `train_model`.
    :param num_training_batches_per_epoch: Same.
    :param training_option_dict: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validation_option_dict: Same.
    :param loss_function_string: Same.
    :param plateau_patience_epochs: Same.
    :param plateau_learning_rate_multiplier: Same.
    :param early_stopping_patience_epochs: Same.
    :param bnn_architecture_dict: Same.
    """

    metadata_dict = {
        NUM_EPOCHS_KEY: num_epochs,
        NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        TRAINING_OPTIONS_KEY: training_option_dict,
        NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch,
        VALIDATION_OPTIONS_KEY: validation_option_dict,
        LOSS_FUNCTION_KEY: loss_function_string,
        PLATEAU_PATIENCE_KEY: plateau_patience_epochs,
        PLATEAU_LR_MUTIPLIER_KEY: plateau_learning_rate_multiplier,
        EARLY_STOPPING_PATIENCE_KEY: early_stopping_patience_epochs,
        BNN_ARCHITECTURE_KEY: bnn_architecture_dict
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(metadata_dict, pickle_file_handle)
    pickle_file_handle.close()


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

    :return: target_matrix_low_res_px: E-by-3 numpy array with distances (in
        low-resolution pixels) between the image center and actual cyclone
        center.  target_matrix[:, 0] contains row offsets, and
        target_matrix[:, 1] contains column offsets.  For example, if
        target_matrix[20, 0] = -2 and target_matrix[20, 1] = 3, this means that
        the true cyclone center for the 21st example is 2 rows above, and 3
        columns to the right of, the image center.

        target_matrix[:, 2] contains the grid spacing for each data sample in
        km.
    """

    option_dict = _check_generator_args(option_dict)

    satellite_dir_name = option_dict[SATELLITE_DIRECTORY_KEY]
    years = option_dict[YEARS_KEY]
    lag_times_minutes = option_dict[LAG_TIMES_KEY]
    high_res_wavelengths_microns = option_dict[HIGH_RES_WAVELENGTHS_KEY]
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
    lag_time_tolerance_sec = option_dict[LAG_TIME_TOLERANCE_KEY]
    max_num_missing_lag_times = option_dict[MAX_MISSING_LAG_TIMES_KEY]
    max_interp_gap_sec = option_dict[MAX_INTERP_GAP_KEY]
    sentinel_value = option_dict[SENTINEL_VALUE_KEY]

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
        num_examples_in_memory = 0

        while num_examples_in_memory < num_examples_per_batch:
            if cyclone_index == len(cyclone_id_strings):
                cyclone_index = 0

            num_examples_to_read = min([
                max_examples_per_cyclone,
                num_examples_per_batch - num_examples_in_memory
            ])

            (
                this_reflectance_matrix, this_bt_matrix_kelvins,
                these_grid_spacings_km
            ) = _read_satellite_data_one_cyclone(
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
                sentinel_value=sentinel_value
            )

            if this_bt_matrix_kelvins is None:
                continue
            if this_bt_matrix_kelvins.size == 0:
                continue

            these_dim = this_bt_matrix_kelvins.shape[:-2] + (
                numpy.prod(this_bt_matrix_kelvins.shape[-2:]),
            )
            this_bt_matrix_kelvins = numpy.reshape(
                this_bt_matrix_kelvins, these_dim
            )

            if this_reflectance_matrix is not None:
                these_dim = this_reflectance_matrix.shape[:-2] + (
                    numpy.prod(this_reflectance_matrix.shape[-2:]),
                )
                this_reflectance_matrix = numpy.reshape(
                    this_reflectance_matrix, these_dim
                )

            if brightness_temp_matrix_kelvins is None:
                these_dim = (
                    (num_examples_per_batch,) + this_bt_matrix_kelvins.shape[1:]
                )
                brightness_temp_matrix_kelvins = numpy.full(
                    these_dim, numpy.nan
                )

                grid_spacings_km = numpy.full(num_examples_per_batch, numpy.nan)

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

        grid_spacings_km = numpy.repeat(
            numpy.expand_dims(grid_spacings_km, axis=1),
            axis=1, repeats=data_aug_num_translations
        )
        grid_spacings_km = numpy.ravel(grid_spacings_km)

        predictor_matrices = [brightness_temp_matrix_kelvins]
        if bidirectional_reflectance_matrix is not None:
            predictor_matrices.insert(0, bidirectional_reflectance_matrix)

        target_matrix_low_res_px = numpy.transpose(numpy.vstack((
            row_translations_low_res_px, column_translations_low_res_px,
            grid_spacings_km
        )))

        print('GENERATOR SHAPES')
        print(predictor_matrices[0].shape)
        print(predictor_matrices[1].shape)
        print(target_matrix_low_res_px.shape)

        yield predictor_matrices, target_matrix_low_res_px


def train_model(
        model_object, output_dir_name, num_epochs,
        num_training_batches_per_epoch, training_option_dict,
        num_validation_batches_per_epoch, validation_option_dict,
        loss_function_string, plateau_patience_epochs,
        plateau_learning_rate_multiplier, early_stopping_patience_epochs,
        bnn_architecture_dict):
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
    :param plateau_patience_epochs: Training will be deemed to have reached
        "plateau" if validation loss has not decreased in the last N epochs,
        where N = plateau_patience_epochs.
    :param plateau_learning_rate_multiplier: If training reaches "plateau,"
        learning rate will be multiplied by this value in range (0, 1).
    :param early_stopping_patience_epochs: Training will be stopped early if
        validation loss has not decreased in the last N epochs, where N =
        early_stopping_patience_epochs.
    :param bnn_architecture_dict: Dictionary with architecture options for
        Bayesian neural network (BNN).  If the model being trained is not
        Bayesian, make this None.
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

    training_option_dict = _check_generator_args(training_option_dict)

    # TODO(thunderhoser): Maybe I should just max out the last 3 arguments and
    # not let the user set them.
    validation_keys_to_keep = [
        SATELLITE_DIRECTORY_KEY, YEARS_KEY,
        LAG_TIME_TOLERANCE_KEY, MAX_MISSING_LAG_TIMES_KEY, MAX_INTERP_GAP_KEY
    ]
    for this_key in list(training_option_dict.keys()):
        if this_key in validation_keys_to_keep:
            continue

        validation_option_dict[this_key] = training_option_dict[this_key]

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
        plateau_patience_epochs=plateau_patience_epochs,
        plateau_learning_rate_multiplier=plateau_learning_rate_multiplier,
        early_stopping_patience_epochs=early_stopping_patience_epochs,
        bnn_architecture_dict=bnn_architecture_dict
    )

    model_object.fit_generator(
        generator=training_generator,
        steps_per_epoch=num_training_batches_per_epoch,
        epochs=num_epochs, verbose=1, callbacks=list_of_callback_objects,
        validation_data=validation_generator,
        validation_steps=num_validation_batches_per_epoch
    )


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
    metadata_dict["plateau_patience_epochs"]: Same.
    metadata_dict["plateau_learning_rate_multiplier"]: Same.
    metadata_dict["early_stopping_patience_epochs"]: Same.
    metadata_dict["bnn_architecture_dict"]: Same.

    :raises: ValueError: if any expected key is not found in dictionary.
    """

    error_checking.assert_file_exists(pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'rb')
    metadata_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

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
    bnn_architecture_dict = metadata_dict[BNN_ARCHITECTURE_KEY]

    if bnn_architecture_dict is not None:
        from ml4tccf.machine_learning import cnn_architecture_bayesian

        for this_key in [cnn_architecture_bayesian.LOSS_FUNCTION_KEY]:
            bnn_architecture_dict[this_key] = eval(
                bnn_architecture_dict[this_key]
            )

        model_object = cnn_architecture_bayesian.create_model(
            bnn_architecture_dict
        )
        model_object.load_weights(hdf5_file_name)
        return model_object

    custom_object_dict = copy.deepcopy(METRIC_FUNCTION_DICT)
    custom_object_dict['loss'] = eval(metadata_dict[LOSS_FUNCTION_KEY])

    return tf_keras.models.load_model(
        hdf5_file_name, custom_objects=custom_object_dict
    )
