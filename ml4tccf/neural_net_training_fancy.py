"""NN-training with fancy Robert/Galina satellite data."""

import os
import sys
import random
import numpy
import keras
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import number_rounding
import time_conversion
import file_system_utils
import error_checking
import satellite_io
import misc_utils
import satellite_utils
import neural_net_utils as nn_utils
import data_augmentation

DATE_FORMAT = satellite_io.DATE_FORMAT
TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H%M'

METRES_TO_KM = 0.001
MINUTES_TO_SECONDS = 60
DAYS_TO_SECONDS = 86400

SYNOPTIC_TIME_INTERVAL_SEC = 6 * 3600

BIDIRECTIONAL_REFLECTANCES_KEY = nn_utils.BIDIRECTIONAL_REFLECTANCES_KEY
BRIGHTNESS_TEMPS_KEY = nn_utils.BRIGHTNESS_TEMPS_KEY
GRID_SPACINGS_KEY = nn_utils.GRID_SPACINGS_KEY
CENTER_LATITUDES_KEY = nn_utils.CENTER_LATITUDES_KEY
TARGET_TIMES_KEY = nn_utils.TARGET_TIMES_KEY
LOW_RES_LATITUDES_KEY = nn_utils.LOW_RES_LATITUDES_KEY
LOW_RES_LONGITUDES_KEY = nn_utils.LOW_RES_LONGITUDES_KEY

PREDICTOR_MATRICES_KEY = nn_utils.PREDICTOR_MATRICES_KEY
TARGET_MATRIX_KEY = nn_utils.TARGET_MATRIX_KEY
HIGH_RES_LATITUDES_KEY = nn_utils.HIGH_RES_LATITUDES_KEY
HIGH_RES_LONGITUDES_KEY = nn_utils.HIGH_RES_LONGITUDES_KEY

SATELLITE_DIRECTORY_KEY = nn_utils.SATELLITE_DIRECTORY_KEY
YEARS_KEY = nn_utils.YEARS_KEY
LAG_TIMES_KEY = nn_utils.LAG_TIMES_KEY
HIGH_RES_WAVELENGTHS_KEY = nn_utils.HIGH_RES_WAVELENGTHS_KEY
LOW_RES_WAVELENGTHS_KEY = nn_utils.LOW_RES_WAVELENGTHS_KEY
BATCH_SIZE_KEY = nn_utils.BATCH_SIZE_KEY
MAX_EXAMPLES_PER_CYCLONE_KEY = nn_utils.MAX_EXAMPLES_PER_CYCLONE_KEY
NUM_GRID_ROWS_KEY = nn_utils.NUM_GRID_ROWS_KEY
NUM_GRID_COLUMNS_KEY = nn_utils.NUM_GRID_COLUMNS_KEY
LAG_TIME_TOLERANCE_KEY = nn_utils.LAG_TIME_TOLERANCE_KEY
MAX_MISSING_LAG_TIMES_KEY = nn_utils.MAX_MISSING_LAG_TIMES_KEY
MAX_INTERP_GAP_KEY = nn_utils.MAX_INTERP_GAP_KEY
SENTINEL_VALUE_KEY = nn_utils.SENTINEL_VALUE_KEY
DATA_AUG_NUM_TRANS_KEY = nn_utils.DATA_AUG_NUM_TRANS_KEY
DATA_AUG_MEAN_TRANS_KEY = nn_utils.DATA_AUG_MEAN_TRANS_KEY
DATA_AUG_STDEV_TRANS_KEY = nn_utils.DATA_AUG_STDEV_TRANS_KEY
SYNOPTIC_TIMES_ONLY_KEY = nn_utils.SYNOPTIC_TIMES_ONLY_KEY
A_DECK_FILE_KEY = nn_utils.A_DECK_FILE_KEY
SCALAR_A_DECK_FIELDS_KEY = nn_utils.SCALAR_A_DECK_FIELDS_KEY
REMOVE_NONTROPICAL_KEY = nn_utils.REMOVE_NONTROPICAL_KEY
SEMANTIC_SEG_FLAG_KEY = nn_utils.SEMANTIC_SEG_FLAG_KEY
TARGET_SMOOOTHER_STDEV_KEY = nn_utils.TARGET_SMOOOTHER_STDEV_KEY


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


def _read_satellite_data_1cyclone(
        input_file_names, lag_times_minutes, lag_time_tolerance_sec,
        max_num_missing_lag_times, max_interp_gap_sec,
        high_res_wavelengths_microns, low_res_wavelengths_microns,
        num_rows_low_res, num_columns_low_res, sentinel_value, return_coords,
        target_times_unix_sec):
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
    :param target_times_unix_sec: length-T numpy array of target times.

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
    desired_file_to_times_dict = decide_files_to_read_one_cyclone(
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


def choose_random_target_times(all_target_times_unix_sec, num_times_desired):
    """Chooses random target times from array.

    T = number of times chosen

    :param all_target_times_unix_sec: 1-D numpy array with all target times.
    :param num_times_desired: Number of times desired.
    :return: chosen_target_times_unix_sec: length-T numpy array of chosen target
        times.
    :return: chosen_indices: length-T numpy array of corresponding indices.
    """

    error_checking.assert_is_integer_numpy_array(all_target_times_unix_sec)
    error_checking.assert_is_numpy_array(
        all_target_times_unix_sec, num_dimensions=1
    )
    error_checking.assert_is_integer(num_times_desired)
    error_checking.assert_is_greater(num_times_desired, 0)

    all_target_dates_unix_sec = number_rounding.floor_to_nearest(
        all_target_times_unix_sec, DAYS_TO_SECONDS
    )
    unique_target_dates_unix_sec = numpy.unique(all_target_dates_unix_sec)
    chosen_indices = numpy.array([], dtype=int)

    for i in range(len(unique_target_dates_unix_sec)):
        these_indices = numpy.where(
            all_target_dates_unix_sec == unique_target_dates_unix_sec[i]
        )[0]

        num_times_still_needed = num_times_desired - len(chosen_indices)
        if len(these_indices) > num_times_still_needed:
            these_indices = numpy.random.choice(
                these_indices, size=num_times_still_needed, replace=False
            )

        chosen_indices = numpy.concatenate((chosen_indices, these_indices))
        if len(chosen_indices) == num_times_desired:
            break

    return all_target_times_unix_sec[chosen_indices], chosen_indices


def get_objects_with_desired_lag_times(
        cyclone_id_strings, target_times_unix_sec, predictor_lag_times_sec):
    """Finds cyclone objects for which the desired lag times are available.

    T = original number of cyclone objects
    t = number of cyclone objects for which desired lag times are available

    :param cyclone_id_strings: length-T list of cyclone IDs.
    :param target_times_unix_sec: length-T numpy array of target times.
    :param predictor_lag_times_sec: 1-D numpy array of lag times for predictors.
    :return: cyclone_id_strings: length-t list of cyclone IDs.
    :return: target_times_unix_sec: length-t numpy array of target times.
    """

    # Check input args.
    try:
        error_checking.assert_is_string_list(cyclone_id_strings)
    except:
        error_checking.assert_is_numpy_array(
            cyclone_id_strings, num_dimensions=1
        )

    num_objects = len(cyclone_id_strings)
    expected_dim = numpy.array([num_objects], dtype=int)
    error_checking.assert_is_integer_numpy_array(target_times_unix_sec)
    error_checking.assert_is_numpy_array(
        target_times_unix_sec, exact_dimensions=expected_dim
    )

    error_checking.assert_is_integer_numpy_array(predictor_lag_times_sec)
    error_checking.assert_is_geq_numpy_array(predictor_lag_times_sec, 0)
    error_checking.assert_is_numpy_array(
        predictor_lag_times_sec, num_dimensions=1
    )

    # Do actual stuff.
    cyclone_id_strings = numpy.array(cyclone_id_strings)
    good_indices = numpy.array([], dtype=int)

    for this_unique_id_string in numpy.unique(cyclone_id_strings):
        this_cyclone_indices = numpy.where(
            cyclone_id_strings == this_unique_id_string
        )[0]

        good_flags = numpy.array([
            numpy.all(numpy.isin(
                element=t - predictor_lag_times_sec,
                test_elements=target_times_unix_sec[this_cyclone_indices]
            ))
            for t in target_times_unix_sec[this_cyclone_indices]
        ], dtype=bool)

        good_indices = numpy.concatenate((
            good_indices, this_cyclone_indices[good_flags]
        ))

    good_indices = numpy.sort(good_indices)
    return (
        cyclone_id_strings[good_indices].tolist(),
        target_times_unix_sec[good_indices]
    )


def decide_files_to_read_one_cyclone(
        satellite_file_names, target_times_unix_sec,
        lag_times_minutes, lag_time_tolerance_sec, max_interp_gap_sec):
    """Decides which satellite files to read for one tropical cyclone.

    :param satellite_file_names: 1-D list of paths to input files (will be read
        by `satellite_io.read_file`).
    :param target_times_unix_sec: 1-D numpy array of target times.
    :param lag_times_minutes: 1-D numpy array of lag times for predictors.
    :param lag_time_tolerance_sec: Tolerance for lag times (tolerance for target
        time is always zero).
    :param max_interp_gap_sec: Maximum gap (in seconds) over which to
        interpolate for missing lag times.
    :return: desired_file_to_times_dict: Dictionary, where each key is the path
        to a desired file and the corresponding value is a length-2 list:
        [start_times_unix_sec, end_times_unix_sec].  Each list item is a 1-D
        numpy array; the two arrays have the same length; and they contain
        start/end times to be read.
    """

    error_checking.assert_is_string_list(satellite_file_names)
    error_checking.assert_is_integer_numpy_array(target_times_unix_sec)
    error_checking.assert_is_numpy_array(
        target_times_unix_sec, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(lag_times_minutes)
    error_checking.assert_is_geq_numpy_array(lag_times_minutes, 0)
    error_checking.assert_is_numpy_array(lag_times_minutes, num_dimensions=1)
    error_checking.assert_is_integer(lag_time_tolerance_sec)
    error_checking.assert_is_geq(lag_time_tolerance_sec, 0)
    error_checking.assert_is_integer(max_interp_gap_sec)
    error_checking.assert_is_geq(max_interp_gap_sec, 0)

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


def get_synoptic_target_times(all_target_times_unix_sec):
    """Reduces array of target times to synoptic times only.

    :param all_target_times_unix_sec: 1-D numpy array with all target times.
    :return: synoptic_target_times_unix_sec: 1-D numpy array with only synoptic
        target times.
    """

    error_checking.assert_is_integer_numpy_array(all_target_times_unix_sec)
    error_checking.assert_is_numpy_array(
        all_target_times_unix_sec, num_dimensions=1
    )

    return all_target_times_unix_sec[
        numpy.mod(all_target_times_unix_sec, SYNOPTIC_TIME_INTERVAL_SEC) == 0
    ]


def get_target_times_and_scalar_predictors(
        cyclone_id_strings, synoptic_times_only,
        satellite_file_names_by_cyclone, a_deck_file_name,
        scalar_a_deck_field_names, remove_nontropical_systems,
        predictor_lag_times_sec):
    """Returns target times and scalar predictors for each cyclone.

    C = number of cyclones
    F = number of scalar fields
    T_i = number of target times for [i]th cyclone

    :param cyclone_id_strings: length-C list of cyclone IDs.
    :param synoptic_times_only: Boolean flag.  If True (False), only synoptic
        times (all times) can be target times.
    :param satellite_file_names_by_cyclone: length-C list, where each item is
        a list of paths to satellite files (readable by
        `satellite_io.read_file`).
    :param a_deck_file_name: Path to A-deck file, containing scalar predictors
        (readable by `a_deck_io.read_file`).  If you do not want scalar
        predictors, make this None.
    :param scalar_a_deck_field_names: length-F list of field names.  If
        `a_deck_file_name` is None, make this None.
    :param remove_nontropical_systems:
        [used only if `a_deck_file_name` is not None]
        Boolean flag.  If True, will return only target times corresponding to
        tropical systems (no extratropical, subtropical, etc.).
    :param predictor_lag_times_sec: 1-D numpy array of lag times for predictors.
        Make this None if you do not want to consider predictor lag times.
    :return: target_times_by_cyclone_unix_sec: length-C list, where the [i]th
        item is a numpy array (length T_i) of target times.
    :return: scalar_predictor_matrix_by_cyclone: length-C list, where the [i]th
        item is a numpy array (T_i x F) of scalar predictors.
    """

    # Check input args.
    error_checking.assert_is_string_list(cyclone_id_strings)
    error_checking.assert_is_boolean(synoptic_times_only)
    error_checking.assert_is_list(satellite_file_names_by_cyclone)

    num_cyclones = len(cyclone_id_strings)
    error_checking.assert_equals(
        len(satellite_file_names_by_cyclone), num_cyclones
    )
    for i in range(num_cyclones):
        error_checking.assert_is_string_list(satellite_file_names_by_cyclone[i])

    if predictor_lag_times_sec is not None:
        error_checking.assert_is_integer_numpy_array(predictor_lag_times_sec)
        error_checking.assert_is_numpy_array(
            predictor_lag_times_sec, num_dimensions=1
        )
        error_checking.assert_is_geq_numpy_array(predictor_lag_times_sec, 0)

    # Do actual stuff.
    num_cyclones = len(cyclone_id_strings)
    target_times_by_cyclone_unix_sec = (
        [numpy.array([], dtype=int)] * num_cyclones
    )

    for i in range(num_cyclones):
        target_times_by_cyclone_unix_sec[i] = numpy.concatenate([
            xarray.open_zarr(f).coords[satellite_utils.TIME_DIM].values
            for f in satellite_file_names_by_cyclone[i]
        ])

        if synoptic_times_only:
            target_times_by_cyclone_unix_sec[i] = get_synoptic_target_times(
                all_target_times_unix_sec=target_times_by_cyclone_unix_sec[i]
            )

        if predictor_lag_times_sec is None:
            continue

        this_num_times = len(target_times_by_cyclone_unix_sec[i])

        _, target_times_by_cyclone_unix_sec[i] = (
            get_objects_with_desired_lag_times(
                cyclone_id_strings=[cyclone_id_strings[i]] * this_num_times,
                target_times_unix_sec=target_times_by_cyclone_unix_sec[i],
                predictor_lag_times_sec=predictor_lag_times_sec
            )
        )

    if a_deck_file_name is None:
        scalar_predictor_matrix_by_cyclone = [None] * num_cyclones
        return (
            target_times_by_cyclone_unix_sec, scalar_predictor_matrix_by_cyclone
        )

    scalar_predictor_matrix_by_cyclone = (
        [numpy.array([], dtype=float)] * num_cyclones
    )

    for i in range(num_cyclones):
        this_num_times = len(target_times_by_cyclone_unix_sec[i])

        scalar_predictor_matrix_by_cyclone[i] = nn_utils.read_scalar_data(
            a_deck_file_name=a_deck_file_name,
            field_names=scalar_a_deck_field_names,
            remove_nontropical_systems=remove_nontropical_systems,
            cyclone_id_strings=[cyclone_id_strings[i]] * this_num_times,
            target_times_unix_sec=target_times_by_cyclone_unix_sec[i]
        )

        good_indices = numpy.where(numpy.all(
            numpy.isfinite(scalar_predictor_matrix_by_cyclone[i]), axis=1
        ))[0]
        target_times_by_cyclone_unix_sec[i] = (
            target_times_by_cyclone_unix_sec[i][good_indices]
        )
        scalar_predictor_matrix_by_cyclone[i] = (
            scalar_predictor_matrix_by_cyclone[i][good_indices, :]
        )

    return target_times_by_cyclone_unix_sec, scalar_predictor_matrix_by_cyclone


def create_data(option_dict, cyclone_id_string, num_target_times):
    """Creates, rather than generates, neural-net inputs.

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

    option_dict = nn_utils.check_generator_args(option_dict)
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
    synoptic_times_only = option_dict[SYNOPTIC_TIMES_ONLY_KEY]
    a_deck_file_name = option_dict[A_DECK_FILE_KEY]
    scalar_a_deck_field_names = option_dict[SCALAR_A_DECK_FIELDS_KEY]
    remove_nontropical_systems = option_dict[REMOVE_NONTROPICAL_KEY]

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

    (
        all_target_times_unix_sec, all_scalar_predictor_matrix
    ) = get_target_times_and_scalar_predictors(
        cyclone_id_strings=[cyclone_id_string],
        synoptic_times_only=synoptic_times_only,
        satellite_file_names_by_cyclone=[satellite_file_names],
        a_deck_file_name=a_deck_file_name,
        scalar_a_deck_field_names=scalar_a_deck_field_names,
        remove_nontropical_systems=remove_nontropical_systems,
        predictor_lag_times_sec=None
    )

    all_target_times_unix_sec = all_target_times_unix_sec[0]
    all_scalar_predictor_matrix = all_scalar_predictor_matrix[0]
    conservative_num_target_times = max([
        int(numpy.round(num_target_times * 1.25)),
        num_target_times + 2
    ])

    chosen_target_times_unix_sec = choose_random_target_times(
        all_target_times_unix_sec=all_target_times_unix_sec + 0,
        num_times_desired=conservative_num_target_times
    )[0]

    data_dict = _read_satellite_data_1cyclone(
        input_file_names=satellite_file_names,
        target_times_unix_sec=chosen_target_times_unix_sec,
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

    if (
            data_dict is None
            or data_dict[BRIGHTNESS_TEMPS_KEY] is None
            or data_dict[BRIGHTNESS_TEMPS_KEY].size == 0
    ):
        return None

    if a_deck_file_name is None:
        scalar_predictor_matrix = None
    else:
        row_indices = numpy.array([
            numpy.where(all_target_times_unix_sec == t)[0][0]
            for t in data_dict[TARGET_TIMES_KEY]
        ], dtype=int)

        scalar_predictor_matrix = all_scalar_predictor_matrix[row_indices, :]
        scalar_predictor_matrix = scalar_predictor_matrix[:num_target_times, :]

    brightness_temp_matrix_kelvins = (
        data_dict[BRIGHTNESS_TEMPS_KEY][:num_target_times, ...]
    )
    grid_spacings_km = (
        data_dict[GRID_SPACINGS_KEY][:num_target_times, ...]
    )
    cyclone_center_latitudes_deg_n = (
        data_dict[CENTER_LATITUDES_KEY][:num_target_times, ...]
    )
    target_times_unix_sec = (
        data_dict[TARGET_TIMES_KEY][:num_target_times, ...]
    )
    low_res_latitude_matrix_deg_n = (
        data_dict[LOW_RES_LATITUDES_KEY][:num_target_times, ...]
    )
    low_res_longitude_matrix_deg_e = (
        data_dict[LOW_RES_LONGITUDES_KEY][:num_target_times, ...]
    )

    if data_dict[BIDIRECTIONAL_REFLECTANCES_KEY] is None:
        bidirectional_reflectance_matrix = None
        high_res_latitude_matrix_deg_n = None
        high_res_longitude_matrix_deg_e = None
    else:
        bidirectional_reflectance_matrix = (
            data_dict[BIDIRECTIONAL_REFLECTANCES_KEY][:num_target_times, ...]
        )
        high_res_latitude_matrix_deg_n = (
            data_dict[HIGH_RES_LATITUDES_KEY][:num_target_times, ...]
        )
        high_res_longitude_matrix_deg_e = (
            data_dict[HIGH_RES_LONGITUDES_KEY][:num_target_times, ...]
        )

    brightness_temp_matrix_kelvins = nn_utils.combine_lag_times_and_wavelengths(
        brightness_temp_matrix_kelvins
    )
    if bidirectional_reflectance_matrix is not None:
        bidirectional_reflectance_matrix = (
            nn_utils.combine_lag_times_and_wavelengths(
                bidirectional_reflectance_matrix
            )
        )

    (
        bidirectional_reflectance_matrix, brightness_temp_matrix_kelvins,
        row_translations_low_res_px, column_translations_low_res_px
    ) = data_augmentation.augment_data(
        bidirectional_reflectance_matrix=bidirectional_reflectance_matrix,
        brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
        num_translations_per_example=data_aug_num_translations,
        mean_translation_low_res_px=data_aug_mean_translation_low_res_px,
        stdev_translation_low_res_px=data_aug_stdev_translation_low_res_px,
        sentinel_value=sentinel_value
    )

    brightness_temp_matrix_kelvins = (
        data_augmentation.subset_grid_after_data_aug(
            data_matrix=brightness_temp_matrix_kelvins,
            num_rows_to_keep=orig_num_rows_low_res,
            num_columns_to_keep=orig_num_columns_low_res,
            for_high_res=False
        )
    )

    low_res_latitude_matrix_deg_n, low_res_longitude_matrix_deg_e = (
        nn_utils.grid_coords_3d_to_4d(
            latitude_matrix_deg_n=low_res_latitude_matrix_deg_n,
            longitude_matrix_deg_e=low_res_longitude_matrix_deg_e
        )
    )

    low_res_latitude_matrix_deg_n = (
        data_augmentation.subset_grid_after_data_aug(
            data_matrix=low_res_latitude_matrix_deg_n,
            num_rows_to_keep=orig_num_rows_low_res,
            num_columns_to_keep=orig_num_columns_low_res,
            for_high_res=False
        )[:, :, 0, :]
    )

    low_res_longitude_matrix_deg_e = (
        data_augmentation.subset_grid_after_data_aug(
            data_matrix=low_res_longitude_matrix_deg_e,
            num_rows_to_keep=orig_num_rows_low_res,
            num_columns_to_keep=orig_num_columns_low_res,
            for_high_res=False
        )[:, 0, :, :]
    )

    if bidirectional_reflectance_matrix is not None:
        bidirectional_reflectance_matrix = (
            data_augmentation.subset_grid_after_data_aug(
                data_matrix=bidirectional_reflectance_matrix,
                num_rows_to_keep=orig_num_rows_low_res * 4,
                num_columns_to_keep=orig_num_columns_low_res * 4,
                for_high_res=True
            )
        )

        high_res_latitude_matrix_deg_n, high_res_longitude_matrix_deg_e = (
            nn_utils.grid_coords_3d_to_4d(
                latitude_matrix_deg_n=high_res_latitude_matrix_deg_n,
                longitude_matrix_deg_e=high_res_longitude_matrix_deg_e
            )
        )

        high_res_latitude_matrix_deg_n = (
            data_augmentation.subset_grid_after_data_aug(
                data_matrix=high_res_latitude_matrix_deg_n,
                num_rows_to_keep=orig_num_rows_low_res * 4,
                num_columns_to_keep=orig_num_columns_low_res * 4,
                for_high_res=True
            )[:, :, 0, :]
        )

        high_res_longitude_matrix_deg_e = (
            data_augmentation.subset_grid_after_data_aug(
                data_matrix=high_res_longitude_matrix_deg_e,
                num_rows_to_keep=orig_num_rows_low_res * 4,
                num_columns_to_keep=orig_num_columns_low_res * 4,
                for_high_res=True
            )[:, 0, :, :]
        )

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
    if scalar_predictor_matrix is not None:
        scalar_predictor_matrix = numpy.repeat(
            scalar_predictor_matrix, axis=0,
            repeats=data_aug_num_translations
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
    if scalar_predictor_matrix is not None:
        predictor_matrices.append(scalar_predictor_matrix)

    if semantic_segmentation_flag:
        target_matrix = nn_utils.make_targets_for_semantic_seg(
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
    """Creates data with specific (instead of random) translations.

    E = batch size = number of examples after data augmentation

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
    option_dict = nn_utils.check_generator_args(option_dict)

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
    a_deck_file_name = option_dict[A_DECK_FILE_KEY]
    scalar_a_deck_field_names = option_dict[SCALAR_A_DECK_FIELDS_KEY]
    remove_nontropical_systems = option_dict[REMOVE_NONTROPICAL_KEY]

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

    data_dict = _read_satellite_data_1cyclone(
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
    if data_dict[BRIGHTNESS_TEMPS_KEY] is None:
        return None
    if data_dict[BRIGHTNESS_TEMPS_KEY].size == 0:
        return None

    if a_deck_file_name is None:
        scalar_predictor_matrix = None
    else:
        this_num_times = len(data_dict[TARGET_TIMES_KEY])

        scalar_predictor_matrix = nn_utils.read_scalar_data(
            a_deck_file_name=a_deck_file_name,
            field_names=scalar_a_deck_field_names,
            remove_nontropical_systems=remove_nontropical_systems,
            cyclone_id_strings=[cyclone_id_string] * this_num_times,
            target_times_unix_sec=data_dict[TARGET_TIMES_KEY]
        )

    reconstruction_indices = numpy.array([
        numpy.where(unique_target_times_unix_sec == t)[0][0]
        for t in target_times_unix_sec
    ], dtype=int)

    dd = data_dict
    idxs = reconstruction_indices

    brightness_temp_matrix_kelvins = dd[BRIGHTNESS_TEMPS_KEY][idxs, ...]
    grid_spacings_km = dd[GRID_SPACINGS_KEY][idxs, ...]
    cyclone_center_latitudes_deg_n = dd[CENTER_LATITUDES_KEY][idxs, ...]
    low_res_latitude_matrix_deg_n = dd[LOW_RES_LATITUDES_KEY][idxs, ...]
    low_res_longitude_matrix_deg_e = dd[LOW_RES_LONGITUDES_KEY][idxs, ...]

    if data_dict[BIDIRECTIONAL_REFLECTANCES_KEY] is None:
        bidirectional_reflectance_matrix = None
        high_res_latitude_matrix_deg_n = None
        high_res_longitude_matrix_deg_e = None
    else:
        bidirectional_reflectance_matrix = dd[BIDIRECTIONAL_REFLECTANCES_KEY][
            idxs, ...
        ]
        high_res_latitude_matrix_deg_n = dd[HIGH_RES_LATITUDES_KEY][idxs, ...]
        high_res_longitude_matrix_deg_e = dd[HIGH_RES_LONGITUDES_KEY][idxs, ...]

    if scalar_predictor_matrix is None:
        good_time_indices = numpy.linspace(
            0, len(grid_spacings_km) - 1, num=len(grid_spacings_km), dtype=int
        )
    else:
        scalar_predictor_matrix = scalar_predictor_matrix[idxs, ...]
        good_time_indices = numpy.where(numpy.all(
            numpy.isfinite(scalar_predictor_matrix), axis=1
        ))[0]
        scalar_predictor_matrix = scalar_predictor_matrix[
            good_time_indices, ...
        ]

    idxs = good_time_indices
    brightness_temp_matrix_kelvins = brightness_temp_matrix_kelvins[idxs, ...]
    grid_spacings_km = grid_spacings_km[idxs, ...]
    cyclone_center_latitudes_deg_n = cyclone_center_latitudes_deg_n[idxs, ...]
    target_times_unix_sec = target_times_unix_sec[idxs, ...]
    low_res_latitude_matrix_deg_n = low_res_latitude_matrix_deg_n[idxs, ...]
    low_res_longitude_matrix_deg_e = low_res_longitude_matrix_deg_e[idxs, ...]

    if bidirectional_reflectance_matrix is not None:
        bidirectional_reflectance_matrix = bidirectional_reflectance_matrix[
            idxs, ...
        ]
        high_res_latitude_matrix_deg_n = high_res_latitude_matrix_deg_n[
            idxs, ...
        ]
        high_res_longitude_matrix_deg_e = high_res_longitude_matrix_deg_e[
            idxs, ...
        ]

    brightness_temp_matrix_kelvins = nn_utils.combine_lag_times_and_wavelengths(
        brightness_temp_matrix_kelvins
    )
    if bidirectional_reflectance_matrix is not None:
        bidirectional_reflectance_matrix = (
            nn_utils.combine_lag_times_and_wavelengths(
                bidirectional_reflectance_matrix
            )
        )

    (
        bidirectional_reflectance_matrix, brightness_temp_matrix_kelvins
    ) = data_augmentation.augment_data_specific_trans(
        bidirectional_reflectance_matrix=bidirectional_reflectance_matrix,
        brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
        row_translations_low_res_px=row_translations_low_res_px,
        column_translations_low_res_px=column_translations_low_res_px,
        sentinel_value=sentinel_value
    )

    brightness_temp_matrix_kelvins = (
        data_augmentation.subset_grid_after_data_aug(
            data_matrix=brightness_temp_matrix_kelvins,
            num_rows_to_keep=orig_num_rows_low_res,
            num_columns_to_keep=orig_num_columns_low_res,
            for_high_res=False
        )
    )

    low_res_latitude_matrix_deg_n, low_res_longitude_matrix_deg_e = (
        nn_utils.grid_coords_3d_to_4d(
            latitude_matrix_deg_n=low_res_latitude_matrix_deg_n,
            longitude_matrix_deg_e=low_res_longitude_matrix_deg_e
        )
    )

    low_res_latitude_matrix_deg_n = (
        data_augmentation.subset_grid_after_data_aug(
            data_matrix=low_res_latitude_matrix_deg_n,
            num_rows_to_keep=orig_num_rows_low_res,
            num_columns_to_keep=orig_num_columns_low_res,
            for_high_res=False
        )[:, :, 0, :]
    )

    low_res_longitude_matrix_deg_e = (
        data_augmentation.subset_grid_after_data_aug(
            data_matrix=low_res_longitude_matrix_deg_e,
            num_rows_to_keep=orig_num_rows_low_res,
            num_columns_to_keep=orig_num_columns_low_res,
            for_high_res=False
        )[:, 0, :, :]
    )

    if bidirectional_reflectance_matrix is not None:
        bidirectional_reflectance_matrix = (
            data_augmentation.subset_grid_after_data_aug(
                data_matrix=bidirectional_reflectance_matrix,
                num_rows_to_keep=orig_num_rows_low_res * 4,
                num_columns_to_keep=orig_num_columns_low_res * 4,
                for_high_res=True
            )
        )

        high_res_latitude_matrix_deg_n, high_res_longitude_matrix_deg_e = (
            nn_utils.grid_coords_3d_to_4d(
                latitude_matrix_deg_n=high_res_latitude_matrix_deg_n,
                longitude_matrix_deg_e=high_res_longitude_matrix_deg_e
            )
        )

        high_res_latitude_matrix_deg_n = (
            data_augmentation.subset_grid_after_data_aug(
                data_matrix=high_res_latitude_matrix_deg_n,
                num_rows_to_keep=orig_num_rows_low_res * 4,
                num_columns_to_keep=orig_num_columns_low_res * 4,
                for_high_res=True
            )[:, :, 0, :]
        )

        high_res_longitude_matrix_deg_e = (
            data_augmentation.subset_grid_after_data_aug(
                data_matrix=high_res_longitude_matrix_deg_e,
                num_rows_to_keep=orig_num_rows_low_res * 4,
                num_columns_to_keep=orig_num_columns_low_res * 4,
                for_high_res=True
            )[:, 0, :, :]
        )

    predictor_matrices = [brightness_temp_matrix_kelvins]
    if bidirectional_reflectance_matrix is not None:
        predictor_matrices.insert(0, bidirectional_reflectance_matrix)
    if scalar_predictor_matrix is not None:
        predictor_matrices.append(scalar_predictor_matrix)

    if semantic_segmentation_flag:
        target_matrix = nn_utils.make_targets_for_semantic_seg(
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
    F = number of scalar fields

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
        example.  Total batch size will be
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
    option_dict["synoptic_times_only"]: Boolean flag.  If True, only synoptic
        times (0000 UTC, 0600 UTC, 1200 UTC, 1800 UTC) can be used as target
        times.  If False, any time can be a target time.
    option_dict["scalar_a_deck_field_names"]: length-F list of scalar fields.
    option_dict["remove_nontropical_systems"]: Boolean flag.  If True, only
        tropical systems will be used for training.  If False, all systems
        (including subtropical, post-tropical, etc.) will be used.
    option_dict["a_deck_file_name"]: Path to A-deck file, which is needed if
        `len(scalar_a_deck_field_names) > 0 or remove_nontropical_systems`.
        If A-deck file is not needed, you can make this None.

    :return: predictor_matrices: If both high- and low-resolution data are
        used, this will be a list with the items
        [bidirectional_reflectance_matrix, brightness_temp_matrix_kelvins].

        If only low-res data are used, this will be a list with one item,
        [brightness_temp_matrix_kelvins].

        If scalar data are also used, append scalar_predictor_matrix to the
        above list.

        bidirectional_reflectance_matrix: E-by-M-by-N-by-(W * L) numpy array
            of reflectance values (unitless).

        brightness_temp_matrix_kelvins: E-by-m-by-n-by-(w * L) numpy array of
            brightness temperatures.

        scalar_predictor_matrix: E-by-F numpy array of scalar predictors.

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

    option_dict = nn_utils.check_generator_args(option_dict)

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
    semantic_segmentation_flag = option_dict[SEMANTIC_SEG_FLAG_KEY]
    target_smoother_stdev_km = option_dict[TARGET_SMOOOTHER_STDEV_KEY]
    synoptic_times_only = option_dict[SYNOPTIC_TIMES_ONLY_KEY]
    a_deck_file_name = option_dict[A_DECK_FILE_KEY]
    scalar_a_deck_field_names = option_dict[SCALAR_A_DECK_FIELDS_KEY]
    remove_nontropical_systems = option_dict[REMOVE_NONTROPICAL_KEY]

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

    (
        target_times_by_cyclone_unix_sec, scalar_predictor_matrix_by_cyclone
    ) = get_target_times_and_scalar_predictors(
        cyclone_id_strings=cyclone_id_strings,
        synoptic_times_only=synoptic_times_only,
        satellite_file_names_by_cyclone=satellite_file_names_by_cyclone,
        a_deck_file_name=a_deck_file_name,
        scalar_a_deck_field_names=scalar_a_deck_field_names,
        remove_nontropical_systems=remove_nontropical_systems,
        predictor_lag_times_sec=None
    )

    cyclone_index = 0

    while True:
        bidirectional_reflectance_matrix = None
        brightness_temp_matrix_kelvins = None
        scalar_predictor_matrix = None
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

            new_target_times_unix_sec = choose_random_target_times(
                all_target_times_unix_sec=
                target_times_by_cyclone_unix_sec[cyclone_index] + 0,
                num_times_desired=num_examples_to_read
            )[0]

            if len(new_target_times_unix_sec) == 0:
                cyclone_index += 1
                continue

            data_dict = _read_satellite_data_1cyclone(
                input_file_names=satellite_file_names_by_cyclone[cyclone_index],
                target_times_unix_sec=new_target_times_unix_sec,
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

            cyclone_index += 1

            if (
                    data_dict is None
                    or data_dict[BRIGHTNESS_TEMPS_KEY] is None
                    or data_dict[BRIGHTNESS_TEMPS_KEY].size == 0
            ):
                continue

            this_reflectance_matrix = data_dict[BIDIRECTIONAL_REFLECTANCES_KEY]
            this_bt_matrix_kelvins = data_dict[BRIGHTNESS_TEMPS_KEY]
            these_grid_spacings_km = data_dict[GRID_SPACINGS_KEY]
            these_center_latitudes_deg_n = data_dict[CENTER_LATITUDES_KEY]

            if a_deck_file_name is None:
                this_scalar_predictor_matrix = None
            else:
                row_indices = numpy.array([
                    numpy.where(
                        target_times_by_cyclone_unix_sec[cyclone_index - 1] == t
                    )[0][0]
                    for t in data_dict[TARGET_TIMES_KEY]
                ], dtype=int)

                this_scalar_predictor_matrix = (
                    scalar_predictor_matrix_by_cyclone[cyclone_index - 1][
                        row_indices, :
                    ]
                )

            this_bt_matrix_kelvins = nn_utils.combine_lag_times_and_wavelengths(
                this_bt_matrix_kelvins
            )
            if this_reflectance_matrix is not None:
                this_reflectance_matrix = (
                    nn_utils.combine_lag_times_and_wavelengths(
                        this_reflectance_matrix
                    )
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

                if this_scalar_predictor_matrix is not None:
                    these_dim = (
                        (num_examples_per_batch,) +
                        this_scalar_predictor_matrix.shape[1:]
                    )
                    scalar_predictor_matrix = numpy.full(these_dim, numpy.nan)

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

            if this_scalar_predictor_matrix is not None:
                scalar_predictor_matrix[first_index:last_index, ...] = (
                    this_scalar_predictor_matrix
                )

            num_examples_in_memory += this_bt_matrix_kelvins.shape[0]

        (
            bidirectional_reflectance_matrix, brightness_temp_matrix_kelvins,
            row_translations_low_res_px, column_translations_low_res_px
        ) = data_augmentation.augment_data(
            bidirectional_reflectance_matrix=bidirectional_reflectance_matrix,
            brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
            num_translations_per_example=data_aug_num_translations,
            mean_translation_low_res_px=data_aug_mean_translation_low_res_px,
            stdev_translation_low_res_px=data_aug_stdev_translation_low_res_px,
            sentinel_value=sentinel_value
        )

        brightness_temp_matrix_kelvins = (
            data_augmentation.subset_grid_after_data_aug(
                data_matrix=brightness_temp_matrix_kelvins,
                num_rows_to_keep=orig_num_rows_low_res,
                num_columns_to_keep=orig_num_columns_low_res,
                for_high_res=False
            )
        )

        if bidirectional_reflectance_matrix is not None:
            bidirectional_reflectance_matrix = (
                data_augmentation.subset_grid_after_data_aug(
                    data_matrix=bidirectional_reflectance_matrix,
                    num_rows_to_keep=orig_num_rows_low_res * 4,
                    num_columns_to_keep=orig_num_columns_low_res * 4,
                    for_high_res=True
                )
            )

        grid_spacings_km = numpy.repeat(
            grid_spacings_km, repeats=data_aug_num_translations
        )
        cyclone_center_latitudes_deg_n = numpy.repeat(
            cyclone_center_latitudes_deg_n, repeats=data_aug_num_translations
        )
        if scalar_predictor_matrix is not None:
            scalar_predictor_matrix = numpy.repeat(
                scalar_predictor_matrix, axis=0,
                repeats=data_aug_num_translations
            )

        predictor_matrices = [brightness_temp_matrix_kelvins]
        if bidirectional_reflectance_matrix is not None:
            predictor_matrices.insert(0, bidirectional_reflectance_matrix)
        if scalar_predictor_matrix is not None:
            predictor_matrices.append(scalar_predictor_matrix)

        if semantic_segmentation_flag:
            target_matrix = nn_utils.make_targets_for_semantic_seg(
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

    training_option_dict = nn_utils.check_generator_args(training_option_dict)
    validation_option_dict = nn_utils.check_generator_args(
        validation_option_dict
    )

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

    metafile_name = nn_utils.find_metafile(
        model_dir_name=output_dir_name, raise_error_if_missing=False
    )
    print('Writing metadata to: "{0:s}"...'.format(metafile_name))

    nn_utils.write_metafile(
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
        is_model_bnn=is_model_bnn,
        data_type_string=nn_utils.RG_FANCY_DATA_TYPE_STRING,
        train_with_shuffled_data=False
    )

    model_object.fit_generator(
        generator=training_generator,
        steps_per_epoch=num_training_batches_per_epoch,
        epochs=num_epochs, verbose=1, callbacks=list_of_callback_objects,
        validation_data=validation_generator,
        validation_steps=num_validation_batches_per_epoch
    )
