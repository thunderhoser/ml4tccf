"""Methods for training and applying neural nets."""

import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from ml4tccf.io import satellite_io
from ml4tccf.utils import satellite_utils

MINUTES_TO_SECONDS = 60
DAYS_TO_SECONDS = 86400

DATE_FORMAT = satellite_io.DATE_FORMAT
TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H%M'

PLATEAU_PATIENCE_EPOCHS = 10
DEFAULT_LEARNING_RATE_MULTIPLIER = 0.5
PLATEAU_COOLDOWN_EPOCHS = 0
EARLY_STOPPING_PATIENCE_EPOCHS = 50
LOSS_PATIENCE = 0.

SATELLITE_DIRECTORY_KEY = 'satellite_dir_name'  # Simple
YEARS_KEY = 'years'  # Simple
LAG_TIMES_KEY = 'lag_times_minutes'
HIGH_RES_WAVELENGTHS_KEY = 'high_res_wavelengths_microns'
LOW_RES_WAVELENGTHS_KEY = 'low_res_wavelengths_microns'
BATCH_SIZE_KEY = 'num_examples_per_batch'  # Simple
MAX_EXAMPLES_PER_CYCLONE_KEY = 'max_examples_per_cyclone'  # Simple
NUM_GRID_ROWS_KEY = 'num_grid_rows_low_res'
NUM_GRID_COLUMNS_KEY = 'num_grid_columns_low_res'
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


def _check_generator_args(option_dict):
    """Error-checks input arguments for generator.

    :param option_dict: See doc for `input_generator`.
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
    error_checking.assert_is_greater_numpy_array(option_dict[LAG_TIMES_KEY], 0)
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
    error_checking.assert_is_greater(
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

    # TODO(thunderhoser): Deciding target times a priori is non-trivial.

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
        input_file_names, target_times_unix_sec, lag_times_minutes,
        lag_time_tolerance_sec, max_num_missing_lag_times, max_interp_gap_sec,
        high_res_wavelengths_microns, low_res_wavelengths_microns,
        num_grid_rows_low_res, num_grid_columns_low_res, sentinel_value):
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
    :param target_times_unix_sec: length-T numpy array of desired target times.
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
    :param num_grid_rows_low_res: m in the above discussion.
    :param num_grid_columns_low_res: n in the above discussion.
    :param sentinel_value: NaN's will be replaced with this value.
    :return: bidirectional_reflectance_matrix: T-by-M-by-N-by-L-by-W numpy array
        of reflectance values (unitless).
    :return: brightness_temp_matrix_kelvins: T-by-m-by-n-by-L-by-w numpy array
        of brightness temperatures.
    """

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
                num_grid_rows_low_res is None
                or num_grid_columns_low_res is None
        ):
            orig_satellite_tables_xarray[i] = satellite_utils.subset_grid(
                satellite_table_xarray=orig_satellite_tables_xarray[i],
                num_rows_to_keep=num_grid_rows_low_res,
                num_columns_to_keep=num_grid_columns_low_res,
                for_high_res=False
            )

            orig_satellite_tables_xarray[i] = satellite_utils.subset_grid(
                satellite_table_xarray=orig_satellite_tables_xarray[i],
                num_rows_to_keep=4 * num_grid_rows_low_res,
                num_columns_to_keep=4 * num_grid_columns_low_res,
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

    return bidirectional_reflectance_matrix, brightness_temp_matrix_kelvins
