"""NN-training with simplified Robert/Galina satellite data."""

import random
import warnings
import numpy
import keras
import xarray
import pandas
from scipy.interpolate import interp1d
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4tccf.io import a_deck_io
from ml4tccf.io import satellite_io
from ml4tccf.io import short_track_io
from ml4tccf.utils import misc_utils
from ml4tccf.utils import satellite_utils
from ml4tccf.machine_learning import neural_net_utils as nn_utils
from ml4tccf.machine_learning import \
    neural_net_training_fancy as nn_training_fancy
from ml4tccf.machine_learning import data_augmentation

DATE_FORMAT = '%Y%m%d'
TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H%M'

METRES_TO_KM = 0.001
MINUTES_TO_SECONDS = 60
DAYS_TO_SECONDS = 86400

SENTINEL_VALUE = -1000.
SYNOPTIC_TIME_INTERVAL_SEC = 6 * 3600

BRIGHTNESS_TEMPS_KEY = nn_utils.BRIGHTNESS_TEMPS_KEY
GRID_SPACINGS_KEY = nn_utils.GRID_SPACINGS_KEY
CENTER_LATITUDES_KEY = nn_utils.CENTER_LATITUDES_KEY
TARGET_TIMES_KEY = nn_utils.TARGET_TIMES_KEY
CYCLONE_IDS_KEY = 'cyclone_id_strings'
LOW_RES_LATITUDES_KEY = nn_utils.LOW_RES_LATITUDES_KEY
LOW_RES_LONGITUDES_KEY = nn_utils.LOW_RES_LONGITUDES_KEY

PREDICTOR_MATRICES_KEY = nn_utils.PREDICTOR_MATRICES_KEY
TARGET_MATRIX_KEY = nn_utils.TARGET_MATRIX_KEY
HIGH_RES_LATITUDES_KEY = nn_utils.HIGH_RES_LATITUDES_KEY
HIGH_RES_LONGITUDES_KEY = nn_utils.HIGH_RES_LONGITUDES_KEY

SATELLITE_DIRECTORY_KEY = nn_utils.SATELLITE_DIRECTORY_KEY
YEARS_KEY = nn_utils.YEARS_KEY
LAG_TIMES_KEY = nn_utils.LAG_TIMES_KEY
LOW_RES_WAVELENGTHS_KEY = nn_utils.LOW_RES_WAVELENGTHS_KEY
BATCH_SIZE_KEY = nn_utils.BATCH_SIZE_KEY
MAX_EXAMPLES_PER_CYCLONE_KEY = nn_utils.MAX_EXAMPLES_PER_CYCLONE_KEY
NUM_GRID_ROWS_KEY = nn_utils.NUM_GRID_ROWS_KEY
NUM_GRID_COLUMNS_KEY = nn_utils.NUM_GRID_COLUMNS_KEY
SHORT_TRACK_DIR_KEY = 'short_track_dir_name'
SHORT_TRACK_MAX_LEAD_KEY = 'short_track_max_lead_minutes'
SHORT_TRACK_DIFF_CENTERS_KEY = 'short_track_center_each_lag_diffly'
DATA_AUG_NUM_TRANS_KEY = nn_utils.DATA_AUG_NUM_TRANS_KEY
DATA_AUG_MEAN_TRANS_KEY = nn_utils.DATA_AUG_MEAN_TRANS_KEY
DATA_AUG_STDEV_TRANS_KEY = nn_utils.DATA_AUG_STDEV_TRANS_KEY
DATA_AUG_WITHIN_MEAN_TRANS_KEY = 'data_aug_within_mean_trans_px'
DATA_AUG_WITHIN_STDEV_TRANS_KEY = 'data_aug_within_stdev_trans_px'
SYNOPTIC_TIMES_ONLY_KEY = nn_utils.SYNOPTIC_TIMES_ONLY_KEY
A_DECK_FILE_KEY = nn_utils.A_DECK_FILE_KEY
SCALAR_A_DECK_FIELDS_KEY = nn_utils.SCALAR_A_DECK_FIELDS_KEY
A_DECKS_AT_LEAST_6H_OLD_KEY = 'a_decks_at_least_6h_old'
REMOVE_NONTROPICAL_KEY = nn_utils.REMOVE_NONTROPICAL_KEY
REMOVE_TROPICAL_KEY = nn_utils.REMOVE_TROPICAL_KEY
SEMANTIC_SEG_FLAG_KEY = nn_utils.SEMANTIC_SEG_FLAG_KEY
TARGET_SMOOOTHER_STDEV_KEY = nn_utils.TARGET_SMOOOTHER_STDEV_KEY
USE_XY_COORDS_KEY = nn_utils.USE_XY_COORDS_KEY


def __are_satellite_data_empty(satellite_data_dict):
    """Determines whether satellite data are empty.

    :param satellite_data_dict: Dictionary returned by
        `_read_satellite_data_1cyclone` or
        `_read_satellite_data_1shuffled_file`.
    :return: empty_flag: Boolean flag.
    """

    return (
        satellite_data_dict is None
        or satellite_data_dict[BRIGHTNESS_TEMPS_KEY] is None
        or satellite_data_dict[BRIGHTNESS_TEMPS_KEY].size == 0
    )


def _check_generator_args(option_dict):
    """Error-checks input arguments for generator.

    :param option_dict: See documentation for `data_generator`.
    :return: option_dict: Same as input, except defaults may have been added.
    """

    option_dict[nn_utils.HIGH_RES_WAVELENGTHS_KEY] = numpy.array([])
    option_dict[nn_utils.LAG_TIME_TOLERANCE_KEY] = 0
    option_dict[nn_utils.MAX_MISSING_LAG_TIMES_KEY] = 0
    option_dict[nn_utils.MAX_INTERP_GAP_KEY] = 0
    option_dict[nn_utils.SENTINEL_VALUE_KEY] = SENTINEL_VALUE
    option_dict[USE_XY_COORDS_KEY] = False

    option_dict = nn_utils.check_generator_args(option_dict)

    if option_dict[SHORT_TRACK_DIR_KEY] is None:
        option_dict[SHORT_TRACK_MAX_LEAD_KEY] = -1
        option_dict[SHORT_TRACK_DIFF_CENTERS_KEY] = False
    else:
        error_checking.assert_is_string(option_dict[SHORT_TRACK_DIR_KEY])
        error_checking.assert_is_integer(option_dict[SHORT_TRACK_MAX_LEAD_KEY])
        error_checking.assert_is_greater(
            option_dict[SHORT_TRACK_MAX_LEAD_KEY], 0
        )
        error_checking.assert_is_boolean(
            option_dict[SHORT_TRACK_DIFF_CENTERS_KEY]
        )

    error_checking.assert_is_greater(
        option_dict[DATA_AUG_WITHIN_MEAN_TRANS_KEY], 0.
    )
    error_checking.assert_is_greater(
        option_dict[DATA_AUG_WITHIN_STDEV_TRANS_KEY], 0.
    )
    error_checking.assert_is_boolean(option_dict[A_DECKS_AT_LEAST_6H_OLD_KEY])

    return option_dict


def _recenter_1example_on_short_track(
        satellite_data_dict, example_index,
        satellite_lag_times_minutes, short_track_dir_name,
        max_short_track_lead_time_minutes, center_each_lag_time_differently):
    """Recenters one example (TC object) on short-track center.

    :param satellite_data_dict: See documentation for
        `_recenter_satellite_on_short_track`.
    :param example_index: Will recenter the [i]th example, where
        i = example_index.
    :param satellite_lag_times_minutes: Same.
    :param short_track_dir_name: Same.
    :param max_short_track_lead_time_minutes: Same.
    :param center_each_lag_time_differently: Same.
    :return: satellite_data_dict: Same as input but with recentered imagery for
        the [i]th example.
    :return: row_trans_label_low_res_px: Row translation (in units of low-res
        pixels) needed to recenter image on best track, which was the original
        image center.  This is the first half of the label vector for training.
    :return: column_trans_label_low_res_px: Same as above but for columns.
        This is the second half of the label vector for training.
    """

    i = example_index
    cyclone_id_string = satellite_data_dict[CYCLONE_IDS_KEY][i]
    target_time_unix_sec = satellite_data_dict[TARGET_TIMES_KEY][i]

    satellite_valid_times_unix_sec = (
        target_time_unix_sec - satellite_lag_times_minutes * MINUTES_TO_SECONDS
    )
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(satellite_valid_times_unix_sec), 0
    )
    error_checking.assert_equals(satellite_lag_times_minutes[-1], 0)

    short_track_file_name = short_track_io.find_file(
        directory_name=short_track_dir_name,
        cyclone_id_string=cyclone_id_string,
        raise_error_if_missing=True
    )

    print('Reading short-track forecasts from: "{0:s}"...'.format(
        short_track_file_name
    ))
    short_track_table_xarray = short_track_io.read_file(
        short_track_file_name
    )
    sttx = short_track_table_xarray

    short_track_init_time_unix_sec = target_time_unix_sec + 0
    min_allowed_init_time_unix_sec = (
        target_time_unix_sec -
        max_short_track_lead_time_minutes * MINUTES_TO_SECONDS
    )

    init_time_idx = -1
    lead_time_idxs = []

    while short_track_init_time_unix_sec >= min_allowed_init_time_unix_sec:
        these_indices = numpy.where(
            sttx.coords[short_track_io.INIT_TIME_DIM].values ==
            short_track_init_time_unix_sec
        )[0]

        if len(these_indices) == 0:
            short_track_init_time_unix_sec -= 10 * MINUTES_TO_SECONDS
            continue

        init_time_idx = these_indices[0]
        lead_time_idxs = []

        desired_lead_times_sec = (
            satellite_valid_times_unix_sec - short_track_init_time_unix_sec
        )

        if not center_each_lag_time_differently:
            desired_lead_times_sec[:] = (
                satellite_valid_times_unix_sec[-1] -
                short_track_init_time_unix_sec
            )

        for this_lead_time in desired_lead_times_sec:
            first_flags = (
                sttx.coords[short_track_io.LEAD_TIME_DIM].values ==
                this_lead_time
            )

            second_flags = numpy.invert(numpy.logical_or(
                numpy.isnan(
                    sttx[short_track_io.LATITUDE_KEY].values[init_time_idx, :]
                ),
                numpy.isnan(
                    sttx[short_track_io.LONGITUDE_KEY].values[init_time_idx, :]
                )
            ))

            these_indices = numpy.where(
                numpy.logical_and(first_flags, second_flags)
            )[0]

            if len(these_indices) == 0:
                init_time_idx = -1
                break

            lead_time_idxs.append(these_indices[0])

        if init_time_idx == -1:
            short_track_init_time_unix_sec -= 10 * MINUTES_TO_SECONDS
            continue

        lead_time_idxs = numpy.array(lead_time_idxs, dtype=int)
        break

    if init_time_idx == -1:
        warning_string = (
            'POTENTIAL ERROR: Cannot find short-track forecasts for {0:s} '
            'initialized between {1:s} and {2:s}.'
        ).format(
            cyclone_id_string,
            time_conversion.unix_sec_to_string(
                min_allowed_init_time_unix_sec, TIME_FORMAT_FOR_LOG_MESSAGES
            ),
            time_conversion.unix_sec_to_string(
                target_time_unix_sec, TIME_FORMAT_FOR_LOG_MESSAGES
            )
        )

        warnings.warn(warning_string)
        return satellite_data_dict, None, None

    info_string = (
        'WANTED short-track forecasts for {0:s} init at {1:s}.  FOUND '
        'forecasts init at {2:s}.'
    ).format(
        cyclone_id_string,
        time_conversion.unix_sec_to_string(
            target_time_unix_sec, TIME_FORMAT_FOR_LOG_MESSAGES
        ),
        time_conversion.unix_sec_to_string(
            sttx.coords[short_track_io.INIT_TIME_DIM].values[init_time_idx],
            TIME_FORMAT_FOR_LOG_MESSAGES
        )
    )

    print(info_string)

    short_track_latitudes_deg_n = sttx[short_track_io.LATITUDE_KEY].values[
        init_time_idx, lead_time_idxs
    ]
    short_track_longitudes_deg_e = sttx[short_track_io.LONGITUDE_KEY].values[
        init_time_idx, lead_time_idxs
    ]

    sdd = satellite_data_dict
    half_num_grid_rows = sdd[BRIGHTNESS_TEMPS_KEY].shape[1] // 2
    half_num_grid_columns = sdd[BRIGHTNESS_TEMPS_KEY].shape[2] // 2

    row_translation_px = None
    column_translation_px = None
    num_lag_times = len(satellite_valid_times_unix_sec)

    for j in range(num_lag_times):
        grid_latitudes_deg_n = sdd[LOW_RES_LATITUDES_KEY][i, :, j]
        grid_longitudes_deg_e = sdd[LOW_RES_LONGITUDES_KEY][i, :, j]

        hr = half_num_grid_rows
        hc = half_num_grid_columns
        orig_latitude_deg_n = numpy.mean(
            grid_latitudes_deg_n[(hr - 1):(hr + 1)]
        )
        orig_longitude_deg_e = numpy.mean(
            grid_longitudes_deg_e[(hc - 1):(hc + 1)]
        )

        print((
            'Recentering ({0:s}, target {1:s}, {2:d}-min lag) from '
            '({3:.4f} deg N, {4:.4f} deg E) to '
            '({5:.4f} deg N, {6:.4f} deg E)...'
        ).format(
            cyclone_id_string,
            time_conversion.unix_sec_to_string(
                target_time_unix_sec, TIME_FORMAT_FOR_LOG_MESSAGES
            ),
            satellite_lag_times_minutes[j],
            orig_latitude_deg_n,
            orig_longitude_deg_e,
            short_track_latitudes_deg_n[j],
            short_track_longitudes_deg_e[j]
        ))

        latitude_diff_deg = numpy.absolute(
            short_track_latitudes_deg_n[j] - orig_latitude_deg_n
        )
        if latitude_diff_deg > 5:
            warning_string = (
                'POTENTIAL ERROR: Difference between original ({0:.4f} deg N) '
                'and short-track ({1:.4f} deg N) latitudes is too big.'
            ).format(
                orig_latitude_deg_n,
                short_track_latitudes_deg_n[j]
            )

            warnings.warn(warning_string)
            return satellite_data_dict, None, None

        longitude_diff_deg = numpy.absolute(
            short_track_longitudes_deg_e[j] - orig_longitude_deg_e
        )
        if longitude_diff_deg > 5:
            warning_string = (
                'POTENTIAL ERROR: Difference between original ({0:.4f} deg E) '
                'and short-track ({1:.4f} deg E) longitudes is too big.'
            ).format(
                orig_longitude_deg_e,
                short_track_longitudes_deg_e[j]
            )

            warnings.warn(warning_string)
            return satellite_data_dict, None, None

        latitude_diffs_deg = numpy.absolute(
            grid_latitudes_deg_n - short_track_latitudes_deg_n[j]
        )
        assert numpy.min(latitude_diffs_deg) < 0.03
        new_center_row = numpy.argmin(latitude_diffs_deg)

        longitude_diffs_deg = numpy.absolute(
            grid_longitudes_deg_e - short_track_longitudes_deg_e[j]
        )
        assert numpy.min(longitude_diffs_deg) < 0.05
        new_center_column = numpy.argmin(longitude_diffs_deg)

        row_translation_px = half_num_grid_rows - new_center_row
        column_translation_px = half_num_grid_columns - new_center_column

        (
            _, satellite_data_dict[BRIGHTNESS_TEMPS_KEY][[0], :, :, j, ...]
        ) = data_augmentation.augment_data_specific_trans(
            bidirectional_reflectance_matrix=None,
            brightness_temp_matrix_kelvins=
            satellite_data_dict[BRIGHTNESS_TEMPS_KEY][[0], :, :, j, ...],
            row_translations_low_res_px=
            numpy.array([row_translation_px], dtype=int),
            column_translations_low_res_px=
            numpy.array([column_translation_px], dtype=int),
            sentinel_value=SENTINEL_VALUE
        )

    return satellite_data_dict, row_translation_px, column_translation_px


def _recenter_satellite_on_short_track(
        satellite_data_dict, satellite_lag_times_minutes, short_track_dir_name,
        max_short_track_lead_time_minutes, center_each_lag_time_differently):
    """Centers satellite data on short-track, instead of best-track, TC center.

    E = number of examples (TC objects)
    L = number of lag times

    :param satellite_data_dict: Dictionary returned by
        `_read_satellite_data_1cyclone` or
        `_read_satellite_data_1shuffled_file`.
    :param satellite_lag_times_minutes: length-L numpy array of lag times.
    :param short_track_dir_name: Path to directory with short-track data.
        Relevant file therein will be found by `short_track_io.find_file` and
        read by `short_track_io.read_file`.
    :param max_short_track_lead_time_minutes: Max lead time for short-track
        forecasts.
    :param center_each_lag_time_differently: Boolean flag.  If True, the
        satellite imagery for each lag time will be centered at a different
        location.  If False, the satellite imagery for every lag time will be
        centered at the short-track center for the last lag time.
    :return: satellite_data_dict: Same as input but with different
        image center(s).
    :return: row_trans_labels_low_res_px: length-E numpy array of row
        translations (units of low-res pixels) needed to recenter image on
        best track, which was the original image center.  This is the first
        half of the label matrix for training.
    :return: column_trans_labels_low_res_px: Same as above but for columns.
        This is the second half of the label matrix for training.
    """

    # Check input args.
    brightness_temp_matrix_kelvins = satellite_data_dict[BRIGHTNESS_TEMPS_KEY]
    num_lag_times = len(satellite_lag_times_minutes)
    error_checking.assert_equals(
        num_lag_times, brightness_temp_matrix_kelvins.shape[3]
    )

    # Do actual stuff.
    num_cyclone_objects = len(satellite_data_dict[CYCLONE_IDS_KEY])
    row_trans_labels_low_res_px = numpy.full(num_cyclone_objects, 0)
    column_trans_labels_low_res_px = numpy.full(num_cyclone_objects, 0)
    good_indices = []

    for i in range(num_cyclone_objects):
        (
            satellite_data_dict, this_row_trans, this_column_trans
        ) = _recenter_1example_on_short_track(
            satellite_data_dict=satellite_data_dict,
            example_index=i,
            satellite_lag_times_minutes=satellite_lag_times_minutes,
            short_track_dir_name=short_track_dir_name,
            max_short_track_lead_time_minutes=max_short_track_lead_time_minutes,
            center_each_lag_time_differently=center_each_lag_time_differently
        )

        if this_row_trans is None:
            continue

        good_indices.append(i)
        row_trans_labels_low_res_px[i] = this_row_trans + 0
        column_trans_labels_low_res_px[i] = this_column_trans + 0

    good_indices = numpy.array(good_indices, dtype=int)
    row_trans_labels_low_res_px = row_trans_labels_low_res_px[good_indices]
    column_trans_labels_low_res_px = column_trans_labels_low_res_px[
        good_indices
    ]

    for this_key in [
            BRIGHTNESS_TEMPS_KEY, GRID_SPACINGS_KEY,
            CENTER_LATITUDES_KEY, TARGET_TIMES_KEY
    ]:
        satellite_data_dict[this_key] = satellite_data_dict[this_key][
            good_indices, ...
        ]

    for this_key in [CYCLONE_IDS_KEY]:
        satellite_data_dict[this_key] = [
            satellite_data_dict[this_key][k] for k in good_indices
        ]

    for this_key in [LOW_RES_LATITUDES_KEY, LOW_RES_LONGITUDES_KEY]:
        if satellite_data_dict[this_key] is None:
            continue

        satellite_data_dict[this_key] = satellite_data_dict[this_key][
            good_indices, ...
        ]

    return (
        satellite_data_dict,
        row_trans_labels_low_res_px,
        column_trans_labels_low_res_px
    )


def _read_satellite_data_1shuffled_file(
        input_file_name, lag_times_minutes, low_res_wavelengths_microns,
        num_rows_low_res, num_columns_low_res, return_coords,
        cyclone_id_strings, target_times_unix_sec):
    """Reads satellite data from one shuffled file.

    T = number of cyclone objects
    L = number of lag times
    w = number of wavelengths
    m = number of rows in grid
    n = number of columns in grid

    :param input_file_name: Path to input file (will be read by
        `satellite_io.read_file`).
    :param lag_times_minutes: length-L numpy array of lag times for predictors.
    :param low_res_wavelengths_microns: length-w numpy array of desired
        wavelengths.
    :param num_rows_low_res: m in the above discussion.  If None, this method
        will not subset the grid.
    :param num_columns_low_res: n in the above discussion.  If None, this method
        will not subset the grid.
    :param return_coords: Boolean flag.  If True, will return coordinates.
    :param cyclone_id_strings: length-T list of cyclone IDs.
    :param target_times_unix_sec: length-T numpy array of target times.

    :return: data_dict: Dictionary with the following keys.  If
        `return_coords == False`, then keys "low_res_latitude_matrix_deg_n" and
        "low_res_longitude_matrix_deg_e" will be None.

    data_dict["brightness_temp_matrix_kelvins"]: T-by-m-by-n-by-L-by-w numpy
        array of brightness temperatures.
    data_dict["grid_spacings_low_res_km"]: length-T numpy array of grid spacings
        for low-resolution data.
    data_dict["cyclone_center_latitudes_deg_n"]: length-T numpy array of
        center latitudes (deg north).
    data_dict["cyclone_id_strings"]: length-T list of cyclone IDs.
    data_dict["target_times_unix_sec"]: length-T numpy array of target times.
    data_dict["low_res_latitude_matrix_deg_n"]: T-by-m-by-L numpy array of
        latitudes (deg north).
    data_dict["low_res_longitude_matrix_deg_e"]: T-by-n-by-L numpy array of
        longitudes (deg east).
    """

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    satellite_table_xarray = satellite_io.read_file(input_file_name)
    stx = satellite_table_xarray

    num_cyclone_objects = len(target_times_unix_sec)
    num_lag_times = len(lag_times_minutes)
    lag_times_sec = MINUTES_TO_SECONDS * lag_times_minutes

    cyclone_object_success_flags = numpy.full(
        num_cyclone_objects, 0, dtype=bool
    )

    brightness_temp_matrix_kelvins = None
    low_res_latitude_matrix_deg_n = None
    low_res_longitude_matrix_deg_e = None
    grid_spacings_low_res_km = numpy.full(num_cyclone_objects, numpy.nan)
    cyclone_center_latitudes_deg_n = numpy.full(num_cyclone_objects, numpy.nan)

    for i in range(num_cyclone_objects):
        print((
            'Finding satellite data for cyclone {0:s} and target time {1:s}...'
        ).format(
            cyclone_id_strings[i],
            time_conversion.unix_sec_to_string(
                target_times_unix_sec[i], TIME_FORMAT_FOR_LOG_MESSAGES
            )
        ))

        these_desired_times_unix_sec = numpy.sort(
            target_times_unix_sec[i] - lag_times_sec
        )

        good_indices = numpy.where(numpy.logical_and(
            numpy.isin(
                element=stx.coords[satellite_utils.TIME_DIM].values,
                test_elements=these_desired_times_unix_sec
            ),
            stx[satellite_utils.CYCLONE_ID_KEY].values == cyclone_id_strings[i]
        ))[0]

        if len(good_indices) != num_lag_times:
            these_desired_time_strings = [
                time_conversion.unix_sec_to_string(
                    t, TIME_FORMAT_FOR_LOG_MESSAGES
                )
                for t in these_desired_times_unix_sec
            ]

            these_found_time_strings = [
                time_conversion.unix_sec_to_string(
                    t, TIME_FORMAT_FOR_LOG_MESSAGES
                )
                for t in
                stx.coords[satellite_utils.TIME_DIM].values[good_indices]
            ]

            warning_string = (
                'POTENTIAL ERROR: Could not find all desired predictor times.  '
                'Wanted:\n{0:s}\n\nFound:\n{1:s}'
            ).format(
                str(these_desired_time_strings),
                str(these_found_time_strings)
            )

            warnings.warn(warning_string)
            continue

        cyclone_object_success_flags[i] = True

        subindices = numpy.argsort(
            stx.coords[satellite_utils.TIME_DIM].values[good_indices]
        )
        good_indices = good_indices[subindices]

        # TODO(thunderhoser): With simplified data, shouldn't have to worry
        # about all-NaN maps (already removed).
        new_stx = stx.isel(indexers={satellite_utils.TIME_DIM: good_indices})
        new_stx = satellite_utils.subset_wavelengths(
            satellite_table_xarray=new_stx,
            wavelengths_to_keep_microns=low_res_wavelengths_microns,
            for_high_res=False
        )

        if not (num_rows_low_res is None or num_columns_low_res is None):
            new_stx = satellite_utils.subset_grid(
                satellite_table_xarray=new_stx,
                num_rows_to_keep=num_rows_low_res,
                num_columns_to_keep=num_columns_low_res,
                for_high_res=False
            )

        if brightness_temp_matrix_kelvins is None:
            num_rows = len(
                new_stx.coords[satellite_utils.LOW_RES_ROW_DIM].values
            )
            num_columns = len(
                new_stx.coords[satellite_utils.LOW_RES_COLUMN_DIM].values
            )
            num_wavelengths = len(
                new_stx.coords[satellite_utils.LOW_RES_WAVELENGTH_DIM].values
            )

            these_dim = (
                num_cyclone_objects, num_rows, num_columns, num_lag_times,
                num_wavelengths
            )
            brightness_temp_matrix_kelvins = numpy.full(these_dim, numpy.nan)

            if return_coords:
                low_res_latitude_matrix_deg_n = numpy.full(
                    (num_cyclone_objects, num_rows, num_lag_times), numpy.nan
                )
                low_res_longitude_matrix_deg_e = numpy.full(
                    (num_cyclone_objects, num_columns, num_lag_times), numpy.nan
                )

        this_bt_matrix_kelvins = numpy.swapaxes(
            new_stx[satellite_utils.BRIGHTNESS_TEMPERATURE_KEY].values, 0, 1
        )
        brightness_temp_matrix_kelvins[i, ...] = numpy.swapaxes(
            this_bt_matrix_kelvins, 1, 2
        )

        these_x_diffs_metres = numpy.diff(
            new_stx[satellite_utils.X_COORD_LOW_RES_KEY].values[-1, :]
        )
        these_y_diffs_metres = numpy.diff(
            new_stx[satellite_utils.Y_COORD_LOW_RES_KEY].values[-1, :]
        )
        grid_spacings_low_res_km[i] = METRES_TO_KM * numpy.mean(
            numpy.concatenate((these_x_diffs_metres, these_y_diffs_metres))
        )

        cyclone_center_latitudes_deg_n[i] = numpy.median(
            new_stx[satellite_utils.LATITUDE_LOW_RES_KEY].values[-1, :]
        )

        if return_coords:
            low_res_latitude_matrix_deg_n[i, ...] = numpy.swapaxes(
                new_stx[satellite_utils.LATITUDE_LOW_RES_KEY].values, 0, 1
            )
            low_res_longitude_matrix_deg_e[i, ...] = numpy.swapaxes(
                new_stx[satellite_utils.LONGITUDE_LOW_RES_KEY].values, 0, 1
            )

    good_indices = numpy.where(cyclone_object_success_flags)[0]
    if len(good_indices) == 0:
        return None

    cyclone_id_strings = [cyclone_id_strings[k] for k in good_indices]
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
        BRIGHTNESS_TEMPS_KEY: brightness_temp_matrix_kelvins,
        GRID_SPACINGS_KEY: grid_spacings_low_res_km,
        CENTER_LATITUDES_KEY: cyclone_center_latitudes_deg_n,
        CYCLONE_IDS_KEY: cyclone_id_strings,
        TARGET_TIMES_KEY: target_times_unix_sec,
        LOW_RES_LATITUDES_KEY: low_res_latitude_matrix_deg_n,
        LOW_RES_LONGITUDES_KEY: low_res_longitude_matrix_deg_e
    }


def _read_satellite_data_1shuffled_file_old(
        input_file_name, lag_times_minutes, low_res_wavelengths_microns,
        num_rows_low_res, num_columns_low_res, return_coords,
        cyclone_id_strings, target_times_unix_sec):
    """Reads satellite data from one shuffled file.

    T = number of cyclone objects
    L = number of lag times
    w = number of wavelengths
    m = number of rows in grid
    n = number of columns in grid

    :param input_file_name: Path to input file (will be read by
        `satellite_io.read_file`).
    :param lag_times_minutes: length-L numpy array of lag times for predictors.
    :param low_res_wavelengths_microns: length-w numpy array of desired
        wavelengths.
    :param num_rows_low_res: m in the above discussion.  If None, this method
        will not subset the grid.
    :param num_columns_low_res: n in the above discussion.  If None, this method
        will not subset the grid.
    :param return_coords: Boolean flag.  If True, will return coordinates.
    :param cyclone_id_strings: length-T list of cyclone IDs.
    :param target_times_unix_sec: length-T numpy array of target times.

    :return: data_dict: Dictionary with the following keys.  If
        `return_coords == False`, then keys "low_res_latitude_matrix_deg_n" and
        "low_res_longitude_matrix_deg_e" will be None.

    data_dict["brightness_temp_matrix_kelvins"]: T-by-m-by-n-by-L-by-w numpy
        array of brightness temperatures.
    data_dict["grid_spacings_low_res_km"]: length-T numpy array of grid spacings
        for low-resolution data.
    data_dict["cyclone_center_latitudes_deg_n"]: length-T numpy array of
        center latitudes (deg north).
    data_dict["cyclone_id_strings"]: length-T list of cyclone IDs.
    data_dict["target_times_unix_sec"]: length-T numpy array of target times.
    data_dict["low_res_latitude_matrix_deg_n"]: T-by-m-by-L numpy array of
        latitudes (deg north).
    data_dict["low_res_longitude_matrix_deg_e"]: T-by-n-by-L numpy array of
        longitudes (deg east).
    """

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    satellite_table_xarray = satellite_io.read_file(input_file_name)

    satellite_table_xarray = satellite_utils.subset_wavelengths(
        satellite_table_xarray=satellite_table_xarray,
        wavelengths_to_keep_microns=low_res_wavelengths_microns,
        for_high_res=False
    )

    if not (num_rows_low_res is None or num_columns_low_res is None):
        satellite_table_xarray = satellite_utils.subset_grid(
            satellite_table_xarray=satellite_table_xarray,
            num_rows_to_keep=num_rows_low_res,
            num_columns_to_keep=num_columns_low_res,
            for_high_res=False
        )

    num_cyclone_objects = len(target_times_unix_sec)
    num_lag_times = len(lag_times_minutes)

    stx = satellite_table_xarray
    num_rows = len(stx.coords[satellite_utils.LOW_RES_ROW_DIM].values)
    num_columns = len(stx.coords[satellite_utils.LOW_RES_COLUMN_DIM].values)
    num_wavelengths = len(
        stx.coords[satellite_utils.LOW_RES_WAVELENGTH_DIM].values
    )

    these_dim = (
        num_cyclone_objects, num_rows, num_columns, num_lag_times,
        num_wavelengths
    )
    brightness_temp_matrix_kelvins = numpy.full(these_dim, numpy.nan)

    if return_coords:
        low_res_latitude_matrix_deg_n = numpy.full(
            (num_cyclone_objects, num_rows, num_lag_times), numpy.nan
        )
        low_res_longitude_matrix_deg_e = numpy.full(
            (num_cyclone_objects, num_columns, num_lag_times), numpy.nan
        )
    else:
        low_res_latitude_matrix_deg_n = None
        low_res_longitude_matrix_deg_e = None

    grid_spacings_low_res_km = numpy.full(num_cyclone_objects, numpy.nan)
    cyclone_center_latitudes_deg_n = numpy.full(num_cyclone_objects, numpy.nan)

    lag_times_sec = MINUTES_TO_SECONDS * lag_times_minutes
    cyclone_object_success_flags = numpy.full(
        num_cyclone_objects, 0, dtype=bool
    )

    for i in range(num_cyclone_objects):
        print((
            'Finding satellite data for cyclone {0:s} and target time {1:s}...'
        ).format(
            cyclone_id_strings[i],
            time_conversion.unix_sec_to_string(
                target_times_unix_sec[i], TIME_FORMAT_FOR_LOG_MESSAGES
            )
        ))

        these_desired_times_unix_sec = numpy.sort(
            target_times_unix_sec[i] - lag_times_sec
        )

        good_indices = numpy.where(numpy.logical_and(
            numpy.isin(
                element=stx.coords[satellite_utils.TIME_DIM].values,
                test_elements=these_desired_times_unix_sec
            ),
            stx[satellite_utils.CYCLONE_ID_KEY].values == cyclone_id_strings[i]
        ))[0]

        if len(good_indices) != num_lag_times:
            these_desired_time_strings = [
                time_conversion.unix_sec_to_string(
                    t, TIME_FORMAT_FOR_LOG_MESSAGES
                )
                for t in these_desired_times_unix_sec
            ]

            these_found_time_strings = [
                time_conversion.unix_sec_to_string(
                    t, TIME_FORMAT_FOR_LOG_MESSAGES
                )
                for t in
                stx.coords[satellite_utils.TIME_DIM].values[good_indices]
            ]

            warning_string = (
                'POTENTIAL ERROR: Could not find all desired predictor times.  '
                'Wanted:\n{0:s}\n\nFound:\n{1:s}'
            ).format(
                str(these_desired_time_strings),
                str(these_found_time_strings)
            )

            warnings.warn(warning_string)
            continue

        cyclone_object_success_flags[i] = True

        subindices = numpy.argsort(
            stx.coords[satellite_utils.TIME_DIM].values[good_indices]
        )
        good_indices = good_indices[subindices]

        # TODO(thunderhoser): With simplified data, shouldn't have to worry
        # about all-NaN maps (already removed).
        new_stx = stx.isel(indexers={satellite_utils.TIME_DIM: good_indices})
        this_bt_matrix_kelvins = numpy.swapaxes(
            new_stx[satellite_utils.BRIGHTNESS_TEMPERATURE_KEY].values, 0, 1
        )
        brightness_temp_matrix_kelvins[i, ...] = numpy.swapaxes(
            this_bt_matrix_kelvins, 1, 2
        )

        these_x_diffs_metres = numpy.diff(
            new_stx[satellite_utils.X_COORD_LOW_RES_KEY].values[-1, :]
        )
        these_y_diffs_metres = numpy.diff(
            new_stx[satellite_utils.Y_COORD_LOW_RES_KEY].values[-1, :]
        )
        grid_spacings_low_res_km[i] = METRES_TO_KM * numpy.mean(
            numpy.concatenate((these_x_diffs_metres, these_y_diffs_metres))
        )

        cyclone_center_latitudes_deg_n[i] = numpy.median(
            new_stx[satellite_utils.LATITUDE_LOW_RES_KEY].values[-1, :]
        )

        if return_coords:
            low_res_latitude_matrix_deg_n[i, ...] = numpy.swapaxes(
                new_stx[satellite_utils.LATITUDE_LOW_RES_KEY].values, 0, 1
            )
            low_res_longitude_matrix_deg_e[i, ...] = numpy.swapaxes(
                new_stx[satellite_utils.LONGITUDE_LOW_RES_KEY].values, 0, 1
            )

    good_indices = numpy.where(cyclone_object_success_flags)[0]
    if len(good_indices) == 0:
        return None

    cyclone_id_strings = [cyclone_id_strings[k] for k in good_indices]
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
        BRIGHTNESS_TEMPS_KEY: brightness_temp_matrix_kelvins,
        GRID_SPACINGS_KEY: grid_spacings_low_res_km,
        CENTER_LATITUDES_KEY: cyclone_center_latitudes_deg_n,
        CYCLONE_IDS_KEY: cyclone_id_strings,
        TARGET_TIMES_KEY: target_times_unix_sec,
        LOW_RES_LATITUDES_KEY: low_res_latitude_matrix_deg_n,
        LOW_RES_LONGITUDES_KEY: low_res_longitude_matrix_deg_e
    }


def _extrap_based_forecasts_to_rowcol(
        scalar_predictor_matrix, scalar_a_deck_field_names,
        satellite_data_dict):
    """Converts extrapolation-based forecasts from lat-long to row-column space.

    E = number of examples
    F = number of fields

    :param scalar_predictor_matrix: E-by-F numpy array of scalar predictors.
    :param scalar_a_deck_field_names: length-F list with names of scalar
        predictors.
    :param satellite_data_dict: Dictionary returned by
        `_read_satellite_data_1shuffled_file`, also containing data for E
        examples.
    :return: scalar_predictor_matrix: Same as input, but extrapolation-based
        forecasts are now row-column offsets.
    """

    latitude_index = scalar_a_deck_field_names.index(
        a_deck_io.UNNORM_EXTRAP_LATITUDE_KEY
    )
    longitude_index = scalar_a_deck_field_names.index(
        a_deck_io.UNNORM_EXTRAP_LONGITUDE_KEY
    )

    extrap_latitudes_deg_n = scalar_predictor_matrix[:, latitude_index]
    extrap_longitudes_deg_e = scalar_predictor_matrix[:, longitude_index]
    extrap_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        extrap_longitudes_deg_e
    )

    grid_latitude_matrix_deg_n = (
        satellite_data_dict[LOW_RES_LATITUDES_KEY][..., 0]
    )
    grid_longitude_matrix_deg_e = (
        satellite_data_dict[LOW_RES_LONGITUDES_KEY][..., 0]
    )

    num_grid_rows = grid_latitude_matrix_deg_n.shape[1]
    num_grid_columns = grid_longitude_matrix_deg_e.shape[1]
    num_examples = len(extrap_latitudes_deg_n)
    extrap_row_offsets_px = numpy.full(num_examples, numpy.nan)
    extrap_column_offsets_px = numpy.full(num_examples, numpy.nan)

    for i in range(num_examples):
        this_grid_row_indices = numpy.linspace(
            0, num_grid_rows - 1, num=num_grid_rows, dtype=float
        )
        this_grid_row_offsets_px = (
            this_grid_row_indices - 0.5 * num_grid_rows + 0.5
        )
        interp_object = interp1d(
            x=grid_latitude_matrix_deg_n[i, :], y=this_grid_row_offsets_px,
            kind='linear', bounds_error=False, fill_value='extrapolate',
            assume_sorted=True
        )
        extrap_row_offsets_px[i] = interp_object(extrap_latitudes_deg_n[i])

        grid_longitude_matrix_deg_e[i, :] = (
            lng_conversion.convert_lng_positive_in_west(
                grid_longitude_matrix_deg_e[i, :]
            )
        )

        if numpy.any(numpy.diff(grid_longitude_matrix_deg_e[i, :]) < 0):
            grid_longitude_matrix_deg_e[i, :] = (
                lng_conversion.convert_lng_negative_in_west(
                    grid_longitude_matrix_deg_e[i, :]
                )
            )
            extrap_longitudes_deg_e[i] = (
                lng_conversion.convert_lng_negative_in_west(
                    extrap_longitudes_deg_e[i]
                )
            )

        assert not numpy.any(numpy.diff(grid_longitude_matrix_deg_e[i, :]) < 0)

        this_grid_column_indices = numpy.linspace(
            0, num_grid_columns - 1, num=num_grid_columns, dtype=float
        )
        this_grid_column_offsets_px = (
            this_grid_column_indices - 0.5 * num_grid_columns + 0.5
        )
        interp_object = interp1d(
            x=grid_longitude_matrix_deg_e[i, :], y=this_grid_column_offsets_px,
            kind='linear', bounds_error=False, fill_value='extrapolate',
            assume_sorted=True
        )
        extrap_column_offsets_px[i] = interp_object(extrap_longitudes_deg_e[i])

    scalar_predictor_matrix[:, latitude_index] = extrap_row_offsets_px
    scalar_predictor_matrix[:, longitude_index] = extrap_column_offsets_px
    return scalar_predictor_matrix


def _read_satellite_data_1cyclone(
        input_file_names, lag_times_minutes, low_res_wavelengths_microns,
        num_rows_low_res, num_columns_low_res, return_coords,
        target_times_unix_sec):
    """Reads satellite data for one cyclone.

    T = number of target times
    L = number of lag times
    w = number of wavelengths
    m = number of rows in grid
    n = number of columns in grid

    :param input_file_names: 1-D list of paths to input files (will be read by
        `satellite_io.read_file`).
    :param lag_times_minutes: length-L numpy array of lag times for predictors.
    :param low_res_wavelengths_microns: length-w numpy array of desired
        wavelengths.
    :param num_rows_low_res: m in the above discussion.  If None, this method
        will not subset the grid.
    :param num_columns_low_res: n in the above discussion.  If None, this method
        will not subset the grid.
    :param return_coords: Boolean flag.  If True, will return coordinates.
    :param target_times_unix_sec: length-T numpy array of target times.

    :return: data_dict: Dictionary with the following keys.  If
        `return_coords == False`, then keys "low_res_latitude_matrix_deg_n" and
        "low_res_longitude_matrix_deg_e" will be None.

    data_dict["brightness_temp_matrix_kelvins"]: T-by-m-by-n-by-L-by-w numpy
        array of brightness temperatures.
    data_dict["grid_spacings_low_res_km"]: length-T numpy array of grid spacings
        for low-resolution data.
    data_dict["cyclone_center_latitudes_deg_n"]: length-T numpy array of
        center latitudes (deg north).
    data_dict["cyclone_id_strings"]: length-T list of cyclone IDs.
    data_dict["target_times_unix_sec"]: length-T numpy array of target times.
    data_dict["low_res_latitude_matrix_deg_n"]: T-by-m-by-L numpy array of
        latitudes (deg north).
    data_dict["low_res_longitude_matrix_deg_e"]: T-by-n-by-L numpy array of
        longitudes (deg east).
    """

    # TODO(thunderhoser): This could be simplified more.
    desired_file_to_times_dict = (
        nn_training_fancy.decide_files_to_read_one_cyclone(
            satellite_file_names=input_file_names,
            target_times_unix_sec=target_times_unix_sec,
            lag_times_minutes=lag_times_minutes,
            lag_time_tolerance_sec=0, max_interp_gap_sec=0
        )
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

        orig_satellite_tables_xarray[i] = satellite_utils.subset_wavelengths(
            satellite_table_xarray=orig_satellite_tables_xarray[i],
            wavelengths_to_keep_microns=low_res_wavelengths_microns,
            for_high_res=False
        )

        if not (num_rows_low_res is None or num_columns_low_res is None):
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
    if len(good_indices) == 0:
        return None

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

    t = satellite_table_xarray
    cyclone_id_strings = (
        [t[satellite_utils.CYCLONE_ID_KEY].values[0]] * len(good_indices)
    )

    return {
        BRIGHTNESS_TEMPS_KEY: brightness_temp_matrix_kelvins,
        GRID_SPACINGS_KEY: grid_spacings_low_res_km,
        CENTER_LATITUDES_KEY: cyclone_center_latitudes_deg_n,
        CYCLONE_IDS_KEY: cyclone_id_strings,
        TARGET_TIMES_KEY: target_times_unix_sec,
        LOW_RES_LATITUDES_KEY: low_res_latitude_matrix_deg_n,
        LOW_RES_LONGITUDES_KEY: low_res_longitude_matrix_deg_e
    }


def choose_random_cyclone_objects(
        all_cyclone_id_strings, all_target_times_unix_sec, num_objects_desired):
    """Chooses random cyclone objects from array.

    A = number of objects available
    T = number of objects chosen

    :param all_cyclone_id_strings: length-A list with all cyclone IDs.
    :param all_target_times_unix_sec: length-A numpy array with all target
        times.
    :param num_objects_desired: Number of cyclone objects desired.
    :return: chosen_cyclone_id_strings: length-T list of chosen cyclone IDs.
    :return: chosen_target_times_unix_sec: length-T numpy array of chosen target
        times.
    :return: chosen_indices: length-T numpy array of corresponding indices.
    """

    error_checking.assert_is_string_list(all_cyclone_id_strings)
    error_checking.assert_is_integer_numpy_array(all_target_times_unix_sec)

    num_objects_available = len(all_cyclone_id_strings)
    expected_dim = numpy.array([num_objects_available], dtype=int)
    error_checking.assert_is_numpy_array(
        all_target_times_unix_sec, exact_dimensions=expected_dim
    )

    error_checking.assert_is_integer(num_objects_desired)
    error_checking.assert_is_greater(num_objects_desired, 0)

    all_cyclone_id_strings = numpy.array(all_cyclone_id_strings)
    unique_cyclone_id_strings = numpy.unique(all_cyclone_id_strings)
    chosen_indices = numpy.array([], dtype=int)

    for _ in range(10):
        for i in range(len(unique_cyclone_id_strings)):
            these_indices = numpy.where(
                all_cyclone_id_strings == unique_cyclone_id_strings[i]
            )[0]

            these_indices = these_indices[
                numpy.invert(numpy.isin(
                    element=these_indices, test_elements=chosen_indices
                ))
            ]

            if len(these_indices) == 0:
                continue

            these_indices = numpy.random.choice(
                these_indices, size=1, replace=False
            )
            chosen_indices = numpy.concatenate((chosen_indices, these_indices))
            if len(chosen_indices) == num_objects_desired:
                break

        if len(chosen_indices) == num_objects_desired:
            break

    return (
        [all_cyclone_id_strings[k] for k in chosen_indices],
        all_target_times_unix_sec[chosen_indices],
        chosen_indices
    )


def get_times_and_scalar_preds_shuffled(
        satellite_file_names, a_deck_file_name,
        scalar_a_deck_field_names, remove_nontropical_systems,
        remove_tropical_systems, desired_years, predictor_lag_times_sec,
        a_decks_at_least_6h_old=False):
    """Returns cyclone objects and scalar predictors for each shuffled file.

    One cyclone object = one tropical cyclone and one target time

    S = number of shuffled files
    F = number of scalar fields
    O_i = number of cyclone objects for [i]th file

    :param satellite_file_names: length-S list of paths to input files (readable
        by `satellite_io.read_file`).
    :param a_deck_file_name: See doc for
        `nn_training_fancy.get_target_times_and_scalar_predictors`.
    :param scalar_a_deck_field_names: Same.
    :param remove_nontropical_systems: Same.
    :param remove_tropical_systems: Same.
    :param desired_years: 1-D numpy array of desired years.
    :param predictor_lag_times_sec: 1-D numpy array of lag times for predictors.
    :param a_decks_at_least_6h_old: Boolean flag.
    :return: cyclone_id_strings_by_file: length-S list, where the [i]th
        item is a list (length O_i) of cyclone IDs.
    :return: target_times_by_file_unix_sec: length-S list, where the [i]th
        item is a numpy array (length O_i) of target times.
    :return: scalar_predictor_matrix_by_file: length-S list, where the [i]th
        item is a numpy array (O_i x F) of scalar predictors.
    """

    error_checking.assert_is_string_list(satellite_file_names)
    error_checking.assert_is_integer_numpy_array(desired_years)
    error_checking.assert_is_numpy_array(desired_years, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(predictor_lag_times_sec)
    error_checking.assert_is_numpy_array(
        predictor_lag_times_sec, num_dimensions=1
    )
    error_checking.assert_is_geq_numpy_array(predictor_lag_times_sec, 0)
    error_checking.assert_is_boolean(a_decks_at_least_6h_old)

    num_files = len(satellite_file_names)
    cyclone_id_strings_by_file = [[]] * num_files
    target_times_by_file_unix_sec = [numpy.array([], dtype=int)] * num_files

    for i in range(num_files):
        cyclone_id_strings_by_file[i] = xarray.open_zarr(
            satellite_file_names[i]
        )[satellite_utils.CYCLONE_ID_KEY].values

        target_times_by_file_unix_sec[i] = xarray.open_zarr(
            satellite_file_names[i]
        ).coords[satellite_utils.TIME_DIM].values

        these_years = numpy.array([
            int(time_conversion.unix_sec_to_string(t, '%Y'))
            for t in target_times_by_file_unix_sec[i]
        ], dtype=int)

        good_indices = numpy.where(
            numpy.isin(element=these_years, test_elements=desired_years)
        )[0]

        cyclone_id_strings_by_file[i] = cyclone_id_strings_by_file[i][
            good_indices
        ]
        target_times_by_file_unix_sec[i] = target_times_by_file_unix_sec[i][
            good_indices
        ]

        if len(cyclone_id_strings_by_file[i]) == 0:
            continue

        (
            cyclone_id_strings_by_file[i], target_times_by_file_unix_sec[i]
        ) = nn_training_fancy.get_objects_with_desired_lag_times(
            cyclone_id_strings=cyclone_id_strings_by_file[i],
            target_times_unix_sec=target_times_by_file_unix_sec[i],
            predictor_lag_times_sec=predictor_lag_times_sec
        )

    if a_deck_file_name is None:
        scalar_predictor_matrix_by_file = [None] * num_files

        return (
            cyclone_id_strings_by_file,
            target_times_by_file_unix_sec,
            scalar_predictor_matrix_by_file
        )

    scalar_predictor_matrix_by_file = [numpy.array([], dtype=float)] * num_files

    for i in range(num_files):
        if len(cyclone_id_strings_by_file[i]) == 0:
            scalar_predictor_matrix_by_file[i] = None
            continue

        scalar_predictor_matrix_by_file[i] = nn_utils.read_scalar_data(
            a_deck_file_name=a_deck_file_name,
            field_names=scalar_a_deck_field_names,
            remove_nontropical_systems=remove_nontropical_systems,
            remove_tropical_systems=remove_tropical_systems,
            cyclone_id_strings=cyclone_id_strings_by_file[i],
            target_times_unix_sec=target_times_by_file_unix_sec[i],
            a_decks_at_least_6h_old=a_decks_at_least_6h_old
        )

        good_indices = numpy.where(numpy.all(
            numpy.isfinite(scalar_predictor_matrix_by_file[i]), axis=1
        ))[0]
        cyclone_id_strings_by_file[i] = [
            cyclone_id_strings_by_file[i][k] for k in good_indices
        ]
        target_times_by_file_unix_sec[i] = (
            target_times_by_file_unix_sec[i][good_indices]
        )
        scalar_predictor_matrix_by_file[i] = (
            scalar_predictor_matrix_by_file[i][good_indices, :]
        )

    return (
        cyclone_id_strings_by_file,
        target_times_by_file_unix_sec,
        scalar_predictor_matrix_by_file
    )


def data_generator_shuffled_old(option_dict):
    """Generates input data for neural net from shuffled files.

    :param option_dict: See doc for `data_generator`.
    :return: predictor_matrices: Same.
    :return: target_matrix: Same.
    """

    option_dict[nn_utils.HIGH_RES_WAVELENGTHS_KEY] = numpy.array([])
    option_dict[nn_utils.LAG_TIME_TOLERANCE_KEY] = 0
    option_dict[nn_utils.MAX_MISSING_LAG_TIMES_KEY] = 0
    option_dict[nn_utils.MAX_INTERP_GAP_KEY] = 0
    option_dict[nn_utils.SENTINEL_VALUE_KEY] = -10.
    option_dict[nn_utils.SEMANTIC_SEG_FLAG_KEY] = False
    option_dict[nn_utils.TARGET_SMOOOTHER_STDEV_KEY] = None
    option_dict[MAX_EXAMPLES_PER_CYCLONE_KEY] = option_dict[BATCH_SIZE_KEY]
    option_dict[SYNOPTIC_TIMES_ONLY_KEY] = False
    option_dict[USE_XY_COORDS_KEY] = False

    option_dict = nn_utils.check_generator_args(option_dict)

    satellite_dir_name = option_dict[SATELLITE_DIRECTORY_KEY]
    years = option_dict[YEARS_KEY]
    lag_times_minutes = option_dict[LAG_TIMES_KEY]
    low_res_wavelengths_microns = option_dict[LOW_RES_WAVELENGTHS_KEY]
    num_examples_per_batch = option_dict[BATCH_SIZE_KEY]
    num_rows_low_res = option_dict[NUM_GRID_ROWS_KEY]
    num_columns_low_res = option_dict[NUM_GRID_COLUMNS_KEY]
    data_aug_num_translations = option_dict[DATA_AUG_NUM_TRANS_KEY]
    data_aug_mean_translation_low_res_px = option_dict[DATA_AUG_MEAN_TRANS_KEY]
    data_aug_stdev_translation_low_res_px = (
        option_dict[DATA_AUG_STDEV_TRANS_KEY]
    )
    a_deck_file_name = option_dict[A_DECK_FILE_KEY]
    scalar_a_deck_field_names = option_dict[SCALAR_A_DECK_FIELDS_KEY]
    remove_nontropical_systems = option_dict[REMOVE_NONTROPICAL_KEY]
    remove_tropical_systems = option_dict[REMOVE_TROPICAL_KEY]

    orig_num_rows_low_res = num_rows_low_res + 0
    orig_num_columns_low_res = num_columns_low_res + 0

    num_extra_rowcols = 2 * int(numpy.ceil(
        data_aug_mean_translation_low_res_px +
        5 * data_aug_stdev_translation_low_res_px
    ))
    num_rows_low_res += num_extra_rowcols
    num_columns_low_res += num_extra_rowcols

    satellite_file_names = satellite_io.find_shuffled_files(
        directory_name=satellite_dir_name, raise_error_if_all_missing=True
    )
    random.shuffle(satellite_file_names)

    (
        cyclone_id_strings_by_file,
        target_times_by_file_unix_sec,
        scalar_predictor_matrix_by_file
    ) = get_times_and_scalar_preds_shuffled(
        satellite_file_names=satellite_file_names,
        a_deck_file_name=a_deck_file_name,
        scalar_a_deck_field_names=scalar_a_deck_field_names,
        remove_nontropical_systems=remove_nontropical_systems,
        remove_tropical_systems=remove_tropical_systems,
        desired_years=years,
        predictor_lag_times_sec=MINUTES_TO_SECONDS * lag_times_minutes
    )

    use_extrap_based_forecasts = (
        scalar_a_deck_field_names is not None
        and a_deck_io.UNNORM_EXTRAP_LATITUDE_KEY in scalar_a_deck_field_names
        and a_deck_io.UNNORM_EXTRAP_LONGITUDE_KEY in scalar_a_deck_field_names
    )

    file_index = 0

    while True:
        vector_predictor_matrix = None
        scalar_predictor_matrix = None
        grid_spacings_km = None
        cyclone_center_latitudes_deg_n = None
        num_examples_in_memory = 0

        while num_examples_in_memory < num_examples_per_batch:
            if file_index == len(satellite_file_names):
                file_index = 0

            if len(cyclone_id_strings_by_file[file_index]) == 0:
                file_index += 1
                continue

            (
                these_cyclone_id_strings, these_target_times_unix_sec, _
            ) = choose_random_cyclone_objects(
                all_cyclone_id_strings=cyclone_id_strings_by_file[file_index],
                all_target_times_unix_sec=
                target_times_by_file_unix_sec[file_index],
                num_objects_desired=
                num_examples_per_batch - num_examples_in_memory
            )

            if len(these_cyclone_id_strings) == 0:
                file_index += 1
                continue

            data_dict = _read_satellite_data_1shuffled_file(
                input_file_name=satellite_file_names[file_index],
                lag_times_minutes=lag_times_minutes,
                low_res_wavelengths_microns=low_res_wavelengths_microns,
                num_rows_low_res=num_rows_low_res,
                num_columns_low_res=num_columns_low_res,
                return_coords=use_extrap_based_forecasts,
                cyclone_id_strings=these_cyclone_id_strings,
                target_times_unix_sec=these_target_times_unix_sec
            )
            file_index += 1

            if (
                    data_dict is None
                    or data_dict[BRIGHTNESS_TEMPS_KEY] is None
                    or data_dict[BRIGHTNESS_TEMPS_KEY].size == 0
            ):
                continue

            this_bt_matrix_kelvins = data_dict[BRIGHTNESS_TEMPS_KEY]
            these_grid_spacings_km = data_dict[GRID_SPACINGS_KEY]
            these_center_latitudes_deg_n = data_dict[CENTER_LATITUDES_KEY]

            if a_deck_file_name is None:
                this_scalar_predictor_matrix = None
            else:
                prev_idx = file_index - 1

                row_indices = numpy.array([
                    numpy.where(numpy.logical_and(
                        numpy.array(cyclone_id_strings_by_file[prev_idx]) == c,
                        target_times_by_file_unix_sec[prev_idx] == t
                    ))[0][0]
                    for c, t in zip(
                        data_dict[CYCLONE_IDS_KEY], data_dict[TARGET_TIMES_KEY]
                    )
                ], dtype=int)

                this_scalar_predictor_matrix = (
                    scalar_predictor_matrix_by_file[prev_idx][row_indices, :]
                )

                if use_extrap_based_forecasts:
                    this_scalar_predictor_matrix = (
                        _extrap_based_forecasts_to_rowcol(
                            scalar_predictor_matrix=
                            this_scalar_predictor_matrix,
                            scalar_a_deck_field_names=scalar_a_deck_field_names,
                            satellite_data_dict=data_dict
                        )
                    )

            this_bt_matrix_kelvins = nn_utils.combine_lag_times_and_wavelengths(
                this_bt_matrix_kelvins
            )
            this_vector_predictor_matrix = this_bt_matrix_kelvins

            if vector_predictor_matrix is None:
                these_dim = (
                    (num_examples_per_batch,) +
                    this_vector_predictor_matrix.shape[1:]
                )
                vector_predictor_matrix = numpy.full(these_dim, numpy.nan)

                grid_spacings_km = numpy.full(num_examples_per_batch, numpy.nan)
                cyclone_center_latitudes_deg_n = numpy.full(
                    num_examples_per_batch, numpy.nan
                )

                if this_scalar_predictor_matrix is not None:
                    these_dim = (
                        (num_examples_per_batch,) +
                        this_scalar_predictor_matrix.shape[1:]
                    )
                    scalar_predictor_matrix = numpy.full(these_dim, numpy.nan)

            first_index = num_examples_in_memory
            last_index = first_index + this_vector_predictor_matrix.shape[0]
            vector_predictor_matrix[first_index:last_index, ...] = (
                this_vector_predictor_matrix
            )
            grid_spacings_km[first_index:last_index] = these_grid_spacings_km
            cyclone_center_latitudes_deg_n[first_index:last_index] = (
                these_center_latitudes_deg_n
            )

            if this_scalar_predictor_matrix is not None:
                scalar_predictor_matrix[first_index:last_index, ...] = (
                    this_scalar_predictor_matrix
                )

            num_examples_in_memory += this_vector_predictor_matrix.shape[0]

        translation_dict = data_augmentation.augment_data(
            bidirectional_reflectance_matrix=None,
            brightness_temp_matrix_kelvins=vector_predictor_matrix,
            num_translations_per_example=data_aug_num_translations,
            mean_translation_low_res_px=data_aug_mean_translation_low_res_px,
            stdev_translation_low_res_px=data_aug_stdev_translation_low_res_px,
            sentinel_value=-10.
        )
        vector_predictor_matrix = translation_dict[
            data_augmentation.BRIGHTNESS_TEMPS_KEY
        ]
        row_translations_low_res_px = translation_dict[
            data_augmentation.ROW_TRANSLATIONS_KEY
        ]
        column_translations_low_res_px = translation_dict[
            data_augmentation.COLUMN_TRANSLATIONS_KEY
        ]

        vector_predictor_matrix = data_augmentation.subset_grid_after_data_aug(
            data_matrix=vector_predictor_matrix,
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
        if scalar_predictor_matrix is not None:
            scalar_predictor_matrix = numpy.repeat(
                scalar_predictor_matrix, axis=0,
                repeats=data_aug_num_translations
            )

        if use_extrap_based_forecasts:
            row_index = scalar_a_deck_field_names.index(
                a_deck_io.UNNORM_EXTRAP_LATITUDE_KEY
            )
            column_index = scalar_a_deck_field_names.index(
                a_deck_io.UNNORM_EXTRAP_LONGITUDE_KEY
            )

            scalar_predictor_matrix[:, row_index] = (
                scalar_predictor_matrix[:, row_index] +
                row_translations_low_res_px
            )
            scalar_predictor_matrix[:, column_index] = (
                scalar_predictor_matrix[:, column_index] +
                column_translations_low_res_px
            )

        # TODO(thunderhoser): This is a HACK.  Should be controlled by an input
        # arg.
        new_dimensions = (
            vector_predictor_matrix.shape[:3] +
            (len(lag_times_minutes), len(low_res_wavelengths_microns))
        )
        vector_predictor_matrix = numpy.reshape(
            vector_predictor_matrix, new_dimensions
        )

        predictor_matrices = [vector_predictor_matrix]
        if scalar_predictor_matrix is not None:
            predictor_matrices.append(scalar_predictor_matrix)

        # TODO(thunderhoser): This could be simplified more.
        target_matrix = numpy.transpose(numpy.vstack((
            row_translations_low_res_px, column_translations_low_res_px,
            grid_spacings_km, cyclone_center_latitudes_deg_n
        )))

        predictor_matrices = [p.astype('float16') for p in predictor_matrices]
        yield tuple(predictor_matrices), target_matrix


def data_generator_shuffled(option_dict):
    """Generates input data for neural net from shuffled files.

    :param option_dict: See doc for `data_generator`.
    :return: predictor_matrices: Same.
    :return: target_matrix: Same.
    """

    option_dict[nn_utils.SEMANTIC_SEG_FLAG_KEY] = False
    option_dict[nn_utils.TARGET_SMOOOTHER_STDEV_KEY] = None
    option_dict[MAX_EXAMPLES_PER_CYCLONE_KEY] = option_dict[BATCH_SIZE_KEY]
    option_dict[SYNOPTIC_TIMES_ONLY_KEY] = False
    option_dict = _check_generator_args(option_dict)

    satellite_dir_name = option_dict[SATELLITE_DIRECTORY_KEY]
    years = option_dict[YEARS_KEY]
    lag_times_minutes = option_dict[LAG_TIMES_KEY]
    low_res_wavelengths_microns = option_dict[LOW_RES_WAVELENGTHS_KEY]
    num_examples_per_batch = option_dict[BATCH_SIZE_KEY]
    num_rows_low_res = option_dict[NUM_GRID_ROWS_KEY]
    num_columns_low_res = option_dict[NUM_GRID_COLUMNS_KEY]
    short_track_dir_name = option_dict[SHORT_TRACK_DIR_KEY]
    short_track_max_lead_minutes = option_dict[SHORT_TRACK_MAX_LEAD_KEY]
    short_track_center_each_lag_diffly = option_dict[
        SHORT_TRACK_DIFF_CENTERS_KEY
    ]
    data_aug_num_translations = option_dict[DATA_AUG_NUM_TRANS_KEY]
    data_aug_mean_translation_low_res_px = option_dict[DATA_AUG_MEAN_TRANS_KEY]
    data_aug_stdev_translation_low_res_px = option_dict[
        DATA_AUG_STDEV_TRANS_KEY
    ]
    data_aug_within_mean_trans_px = option_dict[DATA_AUG_WITHIN_MEAN_TRANS_KEY]
    data_aug_within_stdev_trans_px = option_dict[
        DATA_AUG_WITHIN_STDEV_TRANS_KEY
    ]
    a_deck_file_name = option_dict[A_DECK_FILE_KEY]
    scalar_a_deck_field_names = option_dict[SCALAR_A_DECK_FIELDS_KEY]
    a_decks_at_least_6h_old = option_dict[A_DECKS_AT_LEAST_6H_OLD_KEY]
    remove_nontropical_systems = option_dict[REMOVE_NONTROPICAL_KEY]
    remove_tropical_systems = option_dict[REMOVE_TROPICAL_KEY]

    satellite_file_names = satellite_io.find_shuffled_files(
        directory_name=satellite_dir_name, raise_error_if_all_missing=True
    )
    random.shuffle(satellite_file_names)

    (
        cyclone_id_strings_by_file,
        target_times_by_file_unix_sec,
        scalar_predictor_matrix_by_file
    ) = get_times_and_scalar_preds_shuffled(
        satellite_file_names=satellite_file_names,
        a_deck_file_name=a_deck_file_name,
        scalar_a_deck_field_names=scalar_a_deck_field_names,
        remove_nontropical_systems=remove_nontropical_systems,
        remove_tropical_systems=remove_tropical_systems,
        desired_years=years,
        predictor_lag_times_sec=MINUTES_TO_SECONDS * lag_times_minutes,
        a_decks_at_least_6h_old=a_decks_at_least_6h_old
    )

    use_extrap_based_forecasts = (
        scalar_a_deck_field_names is not None
        and a_deck_io.UNNORM_EXTRAP_LATITUDE_KEY in scalar_a_deck_field_names
        and a_deck_io.UNNORM_EXTRAP_LONGITUDE_KEY in scalar_a_deck_field_names
    )

    file_index = 0

    while True:
        vector_predictor_matrix = None
        scalar_predictor_matrix = None
        grid_spacings_km = None
        cyclone_center_latitudes_deg_n = None
        row_translations_low_res_px = None
        column_translations_low_res_px = None
        num_examples_in_memory = 0

        while num_examples_in_memory < num_examples_per_batch:
            if file_index == len(satellite_file_names):
                file_index = 0

            if len(cyclone_id_strings_by_file[file_index]) == 0:
                file_index += 1
                continue

            (
                these_cyclone_id_strings, these_target_times_unix_sec, _
            ) = choose_random_cyclone_objects(
                all_cyclone_id_strings=cyclone_id_strings_by_file[file_index],
                all_target_times_unix_sec=
                target_times_by_file_unix_sec[file_index],
                num_objects_desired=
                num_examples_per_batch - num_examples_in_memory
            )

            if len(these_cyclone_id_strings) == 0:
                file_index += 1
                continue

            data_dict = _read_satellite_data_1shuffled_file(
                input_file_name=satellite_file_names[file_index],
                lag_times_minutes=lag_times_minutes,
                low_res_wavelengths_microns=low_res_wavelengths_microns,
                num_rows_low_res=None,
                num_columns_low_res=None,
                return_coords=(
                    use_extrap_based_forecasts or
                    short_track_dir_name is not None
                ),
                cyclone_id_strings=these_cyclone_id_strings,
                target_times_unix_sec=these_target_times_unix_sec
            )
            file_index += 1

            if __are_satellite_data_empty(data_dict):
                continue

            if short_track_dir_name is None:
                this_num_examples = data_dict[BRIGHTNESS_TEMPS_KEY].shape[0]
                these_row_trans_px = numpy.full(this_num_examples, 0, dtype=int)
                these_column_trans_px = numpy.full(
                    this_num_examples, 0, dtype=int
                )
            else:
                (
                    data_dict, these_row_trans_px, these_column_trans_px
                ) = _recenter_satellite_on_short_track(
                    satellite_data_dict=data_dict,
                    satellite_lag_times_minutes=lag_times_minutes,
                    short_track_dir_name=short_track_dir_name,
                    max_short_track_lead_time_minutes=
                    short_track_max_lead_minutes,
                    center_each_lag_time_differently=
                    short_track_center_each_lag_diffly
                )

            if __are_satellite_data_empty(data_dict):
                continue

            this_bt_matrix_kelvins = data_dict[BRIGHTNESS_TEMPS_KEY]
            these_grid_spacings_km = data_dict[GRID_SPACINGS_KEY]
            these_center_latitudes_deg_n = data_dict[CENTER_LATITUDES_KEY]

            if a_deck_file_name is None:
                this_scalar_predictor_matrix = None
            else:
                prev_idx = file_index - 1

                row_indices = numpy.array([
                    numpy.where(numpy.logical_and(
                        numpy.array(cyclone_id_strings_by_file[prev_idx]) == c,
                        target_times_by_file_unix_sec[prev_idx] == t
                    ))[0][0]
                    for c, t in zip(
                        data_dict[CYCLONE_IDS_KEY], data_dict[TARGET_TIMES_KEY]
                    )
                ], dtype=int)

                this_scalar_predictor_matrix = (
                    scalar_predictor_matrix_by_file[prev_idx][row_indices, :]
                )

                if use_extrap_based_forecasts:
                    this_scalar_predictor_matrix = (
                        _extrap_based_forecasts_to_rowcol(
                            scalar_predictor_matrix=
                            this_scalar_predictor_matrix,
                            scalar_a_deck_field_names=scalar_a_deck_field_names,
                            satellite_data_dict=data_dict
                        )
                    )

            this_vector_predictor_matrix = (
                nn_utils.combine_lag_times_and_wavelengths(
                    this_bt_matrix_kelvins
                )
            )

            if vector_predictor_matrix is None:
                these_dim = (
                    (num_examples_per_batch,) +
                    this_vector_predictor_matrix.shape[1:]
                )
                vector_predictor_matrix = numpy.full(these_dim, numpy.nan)

                grid_spacings_km = numpy.full(num_examples_per_batch, numpy.nan)
                cyclone_center_latitudes_deg_n = numpy.full(
                    num_examples_per_batch, numpy.nan
                )
                row_translations_low_res_px = numpy.full(
                    num_examples_per_batch, 0, dtype=int
                )
                column_translations_low_res_px = numpy.full(
                    num_examples_per_batch, 0, dtype=int
                )

                if this_scalar_predictor_matrix is not None:
                    these_dim = (
                        (num_examples_per_batch,) +
                        this_scalar_predictor_matrix.shape[1:]
                    )
                    scalar_predictor_matrix = numpy.full(these_dim, numpy.nan)

            first_index = num_examples_in_memory
            last_index = first_index + this_vector_predictor_matrix.shape[0]
            vector_predictor_matrix[first_index:last_index, ...] = (
                this_vector_predictor_matrix
            )
            grid_spacings_km[first_index:last_index] = these_grid_spacings_km
            cyclone_center_latitudes_deg_n[first_index:last_index] = (
                these_center_latitudes_deg_n
            )
            row_translations_low_res_px[first_index:last_index] = (
                these_row_trans_px
            )
            column_translations_low_res_px[first_index:last_index] = (
                these_column_trans_px
            )

            if this_scalar_predictor_matrix is not None:
                scalar_predictor_matrix[first_index:last_index, ...] = (
                    this_scalar_predictor_matrix
                )

            num_examples_in_memory += this_vector_predictor_matrix.shape[0]

        translation_dict = data_augmentation.augment_data(
            bidirectional_reflectance_matrix=None,
            brightness_temp_matrix_kelvins=vector_predictor_matrix,
            num_translations_per_example=data_aug_num_translations,
            mean_translation_low_res_px=data_aug_mean_translation_low_res_px,
            stdev_translation_low_res_px=data_aug_stdev_translation_low_res_px,
            sentinel_value=SENTINEL_VALUE
        )
        vector_predictor_matrix = translation_dict[
            data_augmentation.BRIGHTNESS_TEMPS_KEY
        ]
        new_row_trans_px = translation_dict[
            data_augmentation.ROW_TRANSLATIONS_KEY
        ]
        new_column_trans_px = translation_dict[
            data_augmentation.COLUMN_TRANSLATIONS_KEY
        ]
        orig_idxs = translation_dict[
            data_augmentation.ORIG_EXAMPLE_INDICES_KEY
        ]
        del translation_dict

        row_translations_low_res_px = (
            row_translations_low_res_px[orig_idxs] + new_row_trans_px
        )
        column_translations_low_res_px = (
            column_translations_low_res_px[orig_idxs] + new_column_trans_px
        )

        vector_predictor_matrix = nn_utils.separate_lag_times_and_wavelengths(
            satellite_data_matrix=vector_predictor_matrix,
            num_lag_times=len(lag_times_minutes)
        )

        for j in range(len(lag_times_minutes)):
            translation_dict = data_augmentation.augment_data(
                bidirectional_reflectance_matrix=None,
                brightness_temp_matrix_kelvins=
                vector_predictor_matrix[..., j, :],
                num_translations_per_example=1,
                mean_translation_low_res_px=data_aug_within_mean_trans_px,
                stdev_translation_low_res_px=data_aug_within_stdev_trans_px,
                sentinel_value=SENTINEL_VALUE
            )
            vector_predictor_matrix[..., j, :] = translation_dict[
                data_augmentation.BRIGHTNESS_TEMPS_KEY
            ]

            if j != len(lag_times_minutes) - 1:
                del translation_dict
                break

            new_row_trans_px = translation_dict[
                data_augmentation.ROW_TRANSLATIONS_KEY
            ]
            new_column_trans_px = translation_dict[
                data_augmentation.COLUMN_TRANSLATIONS_KEY
            ]
            del translation_dict

            row_translations_low_res_px += new_row_trans_px
            column_translations_low_res_px += new_column_trans_px

        vector_predictor_matrix = nn_utils.combine_lag_times_and_wavelengths(
            vector_predictor_matrix
        )
        vector_predictor_matrix = data_augmentation.subset_grid_after_data_aug(
            data_matrix=vector_predictor_matrix,
            num_rows_to_keep=num_rows_low_res,
            num_columns_to_keep=num_columns_low_res,
            for_high_res=False
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

        if use_extrap_based_forecasts:
            row_index = scalar_a_deck_field_names.index(
                a_deck_io.UNNORM_EXTRAP_LATITUDE_KEY
            )
            column_index = scalar_a_deck_field_names.index(
                a_deck_io.UNNORM_EXTRAP_LONGITUDE_KEY
            )

            scalar_predictor_matrix[:, row_index] = (
                scalar_predictor_matrix[:, row_index] +
                row_translations_low_res_px
            )
            scalar_predictor_matrix[:, column_index] = (
                scalar_predictor_matrix[:, column_index] +
                column_translations_low_res_px
            )

        vector_predictor_matrix = nn_utils.separate_lag_times_and_wavelengths(
            satellite_data_matrix=vector_predictor_matrix,
            num_lag_times=len(lag_times_minutes)
        )

        predictor_matrices = [vector_predictor_matrix]
        if scalar_predictor_matrix is not None:
            predictor_matrices.append(scalar_predictor_matrix)

        target_matrix = numpy.transpose(numpy.vstack((
            row_translations_low_res_px, column_translations_low_res_px,
            grid_spacings_km, cyclone_center_latitudes_deg_n
        )))

        bad_example_flags = numpy.any(
            vector_predictor_matrix < SENTINEL_VALUE + 1,
            axis=(1, 2, 3, 4)
        )

        if numpy.any(bad_example_flags):
            warning_string = (
                'POTENTIAL ERROR: {0:d} of {1:d} examples contain sentinel '
                'value, which means that translation was too large.'
            ).format(
                numpy.sum(bad_example_flags),
                len(bad_example_flags)
            )

            warnings.warn(warning_string)

        if numpy.all(bad_example_flags):
            continue

        good_indices = numpy.where(numpy.invert(bad_example_flags))[0]
        predictor_matrices = [
            p.astype('float16')[good_indices, ...] for p in predictor_matrices
        ]
        target_matrix = target_matrix[good_indices, ...]

        yield tuple(predictor_matrices), target_matrix


def data_generator_old(option_dict):
    """Generates input data for neural net.

    E = batch size = number of examples after data augmentation
    L = number of lag times for predictors
    w = number of wavelengths
    m = number of rows in grid
    n = number of columns in grid
    F = number of scalar fields

    :param option_dict: Dictionary with the following keys.
    option_dict["satellite_dir_name"]: Name of directory with satellite data.
        Files therein will be found by `satellite_io.find_file` and read
        by `satellite_io.read_file`.
    option_dict["years"]: 1-D numpy array of years.  Will generate data only for
        cyclones in these years.
    option_dict["lag_times_minutes"]: length-L numpy array of lag times for
        predictors.
    option_dict["low_res_wavelengths_microns"]: length-w numpy array of
        wavelengths.
    option_dict["num_examples_per_batch"]: Batch size before data augmentation.
    option_dict["max_examples_per_cyclone"]: Max number of examples per cyclone
        in one batch -- again, before data augmentation.
    option_dict["num_rows_low_res"]: Number of grid rows to keep.  This is m in
        the above definitions.
    option_dict["num_columns_low_res"]: Number of grid columns to keep.  This is
        n in the above definitions.
    option_dict["data_aug_num_translations"]: Number of translations for each
        example.  Total batch size will be
        num_examples_per_batch * data_aug_num_translations.
    option_dict["data_aug_mean_translation_low_res_px"]: Mean translation
        distance (in units of low-resolution pixels) for data augmentation.
    option_dict["data_aug_stdev_translation_low_res_px"]: Standard deviation of
        translation distance (in units of low-resolution pixels) for data
        augmentation.
    option_dict["synoptic_times_only"]: Boolean flag.  If True, only synoptic
        times (0000 UTC, 0600 UTC, 1200 UTC, 1800 UTC) can be used as target
        times.  If False, any time can be a target time.
    option_dict["scalar_a_deck_field_names"]: length-F list of scalar fields.
    option_dict["remove_nontropical_systems"]: Boolean flag.  If True, only
        tropical systems will be used for training.  If False, all systems
        (including subtropical, post-tropical, etc.) will be used.
    option_dict["remove_tropical_systems"]: Boolean flag.  If True, only
        non-tropical systems will be used for training.  If False, all systems
        (including subtropical, post-tropical, etc.) will be used.
    option_dict["use_xy_coords_as_predictors"]: Boolean flag.
    option_dict["a_deck_file_name"]: Path to A-deck file, which is needed if
        `len(scalar_a_deck_field_names) > 0 or remove_nontropical_systems or
        remove_tropical_systems`.  If A-deck file is not needed, you can make
        this None.

    :return: predictor_matrices: If predictors include scalars, this will be a
        list with [vector_predictor_matrix, scalar_predictor_matrix].
        Otherwise, this will be a list with only the first item.

        vector_predictor_matrix: If `use_xy_coords_as_predictors == False`, this
            is a E-by-m-by-n-by-(w * L) numpy array of brightness temperatures.
            If `use_xy_coords_as_predictors == True`, this
            is a E-by-m-by-n-by-([w + 2] * L) numpy array with both brightness
            temperatures and xy-coords.

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

    option_dict[nn_utils.HIGH_RES_WAVELENGTHS_KEY] = numpy.array([])
    option_dict[nn_utils.LAG_TIME_TOLERANCE_KEY] = 0
    option_dict[nn_utils.MAX_MISSING_LAG_TIMES_KEY] = 0
    option_dict[nn_utils.MAX_INTERP_GAP_KEY] = 0
    option_dict[nn_utils.SENTINEL_VALUE_KEY] = -10.
    option_dict[nn_utils.SEMANTIC_SEG_FLAG_KEY] = False
    option_dict[nn_utils.TARGET_SMOOOTHER_STDEV_KEY] = None
    option_dict[nn_utils.USE_XY_COORDS_KEY] = False

    option_dict = nn_utils.check_generator_args(option_dict)

    satellite_dir_name = option_dict[SATELLITE_DIRECTORY_KEY]
    years = option_dict[YEARS_KEY]
    lag_times_minutes = option_dict[LAG_TIMES_KEY]
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
    synoptic_times_only = option_dict[SYNOPTIC_TIMES_ONLY_KEY]
    a_deck_file_name = option_dict[A_DECK_FILE_KEY]
    scalar_a_deck_field_names = option_dict[SCALAR_A_DECK_FIELDS_KEY]
    remove_nontropical_systems = option_dict[REMOVE_NONTROPICAL_KEY]
    remove_tropical_systems = option_dict[REMOVE_TROPICAL_KEY]

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
    ) = nn_training_fancy.get_target_times_and_scalar_predictors(
        cyclone_id_strings=cyclone_id_strings,
        synoptic_times_only=synoptic_times_only,
        satellite_file_names_by_cyclone=satellite_file_names_by_cyclone,
        a_deck_file_name=a_deck_file_name,
        scalar_a_deck_field_names=scalar_a_deck_field_names,
        remove_nontropical_systems=remove_nontropical_systems,
        remove_tropical_systems=remove_tropical_systems,
        predictor_lag_times_sec=MINUTES_TO_SECONDS * lag_times_minutes
    )

    use_extrap_based_forecasts = (
        scalar_a_deck_field_names is not None
        and a_deck_io.UNNORM_EXTRAP_LATITUDE_KEY in scalar_a_deck_field_names
        and a_deck_io.UNNORM_EXTRAP_LONGITUDE_KEY in scalar_a_deck_field_names
    )

    cyclone_index = 0

    while True:
        vector_predictor_matrix = None
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

            new_target_times_unix_sec = (
                nn_training_fancy.choose_random_target_times(
                    all_target_times_unix_sec=
                    target_times_by_cyclone_unix_sec[cyclone_index] + 0,
                    num_times_desired=num_examples_to_read
                )[0]
            )

            if len(new_target_times_unix_sec) == 0:
                cyclone_index += 1
                continue

            data_dict = _read_satellite_data_1cyclone(
                input_file_names=satellite_file_names_by_cyclone[cyclone_index],
                lag_times_minutes=lag_times_minutes,
                low_res_wavelengths_microns=low_res_wavelengths_microns,
                num_rows_low_res=num_rows_low_res,
                num_columns_low_res=num_columns_low_res,
                return_coords=use_extrap_based_forecasts,
                target_times_unix_sec=new_target_times_unix_sec
            )
            cyclone_index += 1

            if (
                    data_dict is None
                    or data_dict[BRIGHTNESS_TEMPS_KEY] is None
                    or data_dict[BRIGHTNESS_TEMPS_KEY].size == 0
            ):
                continue

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

                if use_extrap_based_forecasts:
                    this_scalar_predictor_matrix = (
                        _extrap_based_forecasts_to_rowcol(
                            scalar_predictor_matrix=
                            this_scalar_predictor_matrix,
                            scalar_a_deck_field_names=scalar_a_deck_field_names,
                            satellite_data_dict=data_dict
                        )
                    )

            this_bt_matrix_kelvins = nn_utils.combine_lag_times_and_wavelengths(
                this_bt_matrix_kelvins
            )
            this_vector_predictor_matrix = this_bt_matrix_kelvins

            if vector_predictor_matrix is None:
                these_dim = (
                    (num_examples_per_batch,) +
                    this_vector_predictor_matrix.shape[1:]
                )
                vector_predictor_matrix = numpy.full(these_dim, numpy.nan)

                grid_spacings_km = numpy.full(num_examples_per_batch, numpy.nan)
                cyclone_center_latitudes_deg_n = numpy.full(
                    num_examples_per_batch, numpy.nan
                )

                if this_scalar_predictor_matrix is not None:
                    these_dim = (
                        (num_examples_per_batch,) +
                        this_scalar_predictor_matrix.shape[1:]
                    )
                    scalar_predictor_matrix = numpy.full(these_dim, numpy.nan)

            first_index = num_examples_in_memory
            last_index = first_index + this_vector_predictor_matrix.shape[0]
            vector_predictor_matrix[first_index:last_index, ...] = (
                this_vector_predictor_matrix
            )
            grid_spacings_km[first_index:last_index] = these_grid_spacings_km
            cyclone_center_latitudes_deg_n[first_index:last_index] = (
                these_center_latitudes_deg_n
            )

            if this_scalar_predictor_matrix is not None:
                scalar_predictor_matrix[first_index:last_index, ...] = (
                    this_scalar_predictor_matrix
                )

            num_examples_in_memory += this_vector_predictor_matrix.shape[0]

        translation_dict = data_augmentation.augment_data(
            bidirectional_reflectance_matrix=None,
            brightness_temp_matrix_kelvins=vector_predictor_matrix,
            num_translations_per_example=data_aug_num_translations,
            mean_translation_low_res_px=data_aug_mean_translation_low_res_px,
            stdev_translation_low_res_px=data_aug_stdev_translation_low_res_px,
            sentinel_value=-10.
        )
        vector_predictor_matrix = translation_dict[
            data_augmentation.BRIGHTNESS_TEMPS_KEY
        ]
        row_translations_low_res_px = translation_dict[
            data_augmentation.ROW_TRANSLATIONS_KEY
        ]
        column_translations_low_res_px = translation_dict[
            data_augmentation.COLUMN_TRANSLATIONS_KEY
        ]

        vector_predictor_matrix = data_augmentation.subset_grid_after_data_aug(
            data_matrix=vector_predictor_matrix,
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
        if scalar_predictor_matrix is not None:
            scalar_predictor_matrix = numpy.repeat(
                scalar_predictor_matrix, axis=0,
                repeats=data_aug_num_translations
            )

        if use_extrap_based_forecasts:
            row_index = scalar_a_deck_field_names.index(
                a_deck_io.UNNORM_EXTRAP_LATITUDE_KEY
            )
            column_index = scalar_a_deck_field_names.index(
                a_deck_io.UNNORM_EXTRAP_LONGITUDE_KEY
            )

            scalar_predictor_matrix[:, row_index] = (
                scalar_predictor_matrix[:, row_index] +
                row_translations_low_res_px
            )
            scalar_predictor_matrix[:, column_index] = (
                scalar_predictor_matrix[:, column_index] +
                column_translations_low_res_px
            )

        new_dimensions = (
            vector_predictor_matrix.shape[:3] +
            (len(lag_times_minutes), len(low_res_wavelengths_microns))
        )
        vector_predictor_matrix = numpy.reshape(
            vector_predictor_matrix, new_dimensions
        )

        predictor_matrices = [vector_predictor_matrix]
        if scalar_predictor_matrix is not None:
            predictor_matrices.append(scalar_predictor_matrix)

        # TODO(thunderhoser): This could be simplified more.
        target_matrix = numpy.transpose(numpy.vstack((
            row_translations_low_res_px, column_translations_low_res_px,
            grid_spacings_km, cyclone_center_latitudes_deg_n
        )))

        predictor_matrices = [p.astype('float16') for p in predictor_matrices]
        yield tuple(predictor_matrices), target_matrix


def data_generator(option_dict, for_plotting=False):
    """Generates input data for neural net.

    E = batch size = number of examples after data augmentation
    L = number of lag times for predictors
    w = number of wavelengths
    m = number of rows in grid
    n = number of columns in grid
    F = number of scalar fields

    :param option_dict: Dictionary with the following keys.
    option_dict["satellite_dir_name"]: Name of directory with satellite data.
        Files therein will be found by `satellite_io.find_file` and read
        by `satellite_io.read_file`.
    option_dict["years"]: 1-D numpy array of years.  Will generate data only for
        cyclones in these years.
    option_dict["lag_times_minutes"]: length-L numpy array of lag times for
        predictors.
    option_dict["low_res_wavelengths_microns"]: length-w numpy array of
        wavelengths.
    option_dict["num_examples_per_batch"]: Batch size before data augmentation.
    option_dict["max_examples_per_cyclone"]: Max number of examples per cyclone
        in one batch -- again, before data augmentation.
    option_dict["num_rows_low_res"]: Number of grid rows to keep.  This is m in
        the above definitions.
    option_dict["num_columns_low_res"]: Number of grid columns to keep.  This is
        n in the above definitions.
    option_dict["short_track_dir_name"]: Path to directory with short-track data
        (files therein will be found by `short_track_io.find_file` and read by
        `short_track_io.read_file`).  If you do not want to use short track as a
        first guess -- i.e., if you want to use data augmentation only -- make
        this argument None.
    option_dict["short_track_max_lead_minutes"]:
        (used only if `short_track_dir_name is not None`)
        Max lead time for short-track forecasts.  Any forecast with a longer
        lead time will not be used for the first guess.
    option_dict["short_track_center_each_lag_diffly"]:
        (used only if `short_track_dir_name is not None`)
        Boolean flag.  If True, the first guess will involve a different
        lat/long center for each lag time, based on the short-track forecast
        valid at that lag time.  If False, the first guess will involve one
        center for all lag times, based on the short-track forecast for lag time
        = 0.
    option_dict["data_aug_num_translations"]: Number of translations for each
        example.  Total batch size will be
        num_examples_per_batch * data_aug_num_translations.
    option_dict["data_aug_mean_translation_low_res_px"]: Mean translation
        distance (in units of low-resolution pixels) for data augmentation.
    option_dict["data_aug_stdev_translation_low_res_px"]: Standard deviation of
        translation distance (in units of low-resolution pixels) for data
        augmentation.
    option_dict["data_aug_within_mean_trans_px"]: Mean translation distance for
        within-track data augmentation.
    option_dict["data_aug_within_stdev_trans_px"]: Standard deviation of
        translation distance for within-track data augmentation.
    option_dict["synoptic_times_only"]: Boolean flag.  If True, only synoptic
        times (0000 UTC, 0600 UTC, 1200 UTC, 1800 UTC) can be used as target
        times.  If False, any time can be a target time.
    option_dict["scalar_a_deck_field_names"]: length-F list of scalar fields.
    option_dict["a_decks_at_least_6h_old"]: Boolean flag.  If True, when A-deck
        scalars are used in predictors, they will be 6-12 h old, rather than
        0-6 h old.
    option_dict["remove_nontropical_systems"]: Boolean flag.  If True, only
        tropical systems will be used for training.  If False, all systems
        (including subtropical, post-tropical, etc.) will be used.
    option_dict["remove_tropical_systems"]: Boolean flag.  If True, only
        non-tropical systems will be used for training.  If False, all systems
        (including subtropical, post-tropical, etc.) will be used.
    option_dict["a_deck_file_name"]: Path to A-deck file, which is needed if
        `len(scalar_a_deck_field_names) > 0 or remove_nontropical_systems or
        remove_tropical_systems`.  If A-deck file is not needed, you can make
        this None.

    :param for_plotting: Leave this alone.

    :return: predictor_matrices: If predictors include scalars, this will be a
        list with [vector_predictor_matrix, scalar_predictor_matrix].
        Otherwise, this will be a list with only the first item.

        vector_predictor_matrix: E-by-m-by-n-by-(w * L) numpy array of
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

    error_checking.assert_is_boolean(for_plotting)

    option_dict[nn_utils.SEMANTIC_SEG_FLAG_KEY] = False
    option_dict[nn_utils.TARGET_SMOOOTHER_STDEV_KEY] = None
    option_dict = _check_generator_args(option_dict)

    satellite_dir_name = option_dict[SATELLITE_DIRECTORY_KEY]
    years = option_dict[YEARS_KEY]
    lag_times_minutes = option_dict[LAG_TIMES_KEY]
    low_res_wavelengths_microns = option_dict[LOW_RES_WAVELENGTHS_KEY]
    num_examples_per_batch = option_dict[BATCH_SIZE_KEY]
    max_examples_per_cyclone = option_dict[MAX_EXAMPLES_PER_CYCLONE_KEY]
    num_rows_low_res = option_dict[NUM_GRID_ROWS_KEY]
    num_columns_low_res = option_dict[NUM_GRID_COLUMNS_KEY]
    short_track_dir_name = option_dict[SHORT_TRACK_DIR_KEY]
    short_track_max_lead_minutes = option_dict[SHORT_TRACK_MAX_LEAD_KEY]
    short_track_center_each_lag_diffly = option_dict[
        SHORT_TRACK_DIFF_CENTERS_KEY
    ]
    data_aug_num_translations = option_dict[DATA_AUG_NUM_TRANS_KEY]
    data_aug_mean_translation_low_res_px = option_dict[DATA_AUG_MEAN_TRANS_KEY]
    data_aug_stdev_translation_low_res_px = option_dict[
        DATA_AUG_STDEV_TRANS_KEY
    ]
    data_aug_within_mean_trans_px = option_dict[DATA_AUG_WITHIN_MEAN_TRANS_KEY]
    data_aug_within_stdev_trans_px = option_dict[
        DATA_AUG_WITHIN_STDEV_TRANS_KEY
    ]
    synoptic_times_only = option_dict[SYNOPTIC_TIMES_ONLY_KEY]
    a_deck_file_name = option_dict[A_DECK_FILE_KEY]
    scalar_a_deck_field_names = option_dict[SCALAR_A_DECK_FIELDS_KEY]
    a_decks_at_least_6h_old = option_dict[A_DECKS_AT_LEAST_6H_OLD_KEY]
    remove_nontropical_systems = option_dict[REMOVE_NONTROPICAL_KEY]
    remove_tropical_systems = option_dict[REMOVE_TROPICAL_KEY]

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
    ) = nn_training_fancy.get_target_times_and_scalar_predictors(
        cyclone_id_strings=cyclone_id_strings,
        synoptic_times_only=synoptic_times_only,
        satellite_file_names_by_cyclone=satellite_file_names_by_cyclone,
        a_deck_file_name=a_deck_file_name,
        scalar_a_deck_field_names=scalar_a_deck_field_names,
        remove_nontropical_systems=remove_nontropical_systems,
        remove_tropical_systems=remove_tropical_systems,
        predictor_lag_times_sec=MINUTES_TO_SECONDS * lag_times_minutes,
        a_decks_at_least_6h_old=a_decks_at_least_6h_old
    )

    use_extrap_based_forecasts = (
        scalar_a_deck_field_names is not None
        and a_deck_io.UNNORM_EXTRAP_LATITUDE_KEY in scalar_a_deck_field_names
        and a_deck_io.UNNORM_EXTRAP_LONGITUDE_KEY in scalar_a_deck_field_names
    )

    cyclone_index = 0

    while True:
        vector_predictor_matrix = None
        scalar_predictor_matrix = None
        grid_spacings_km = None
        cyclone_center_latitudes_deg_n = None
        row_translations_low_res_px = None
        column_translations_low_res_px = None
        num_examples_in_memory = 0

        return_cyclone_id_strings = None
        target_times_unix_sec = None

        while num_examples_in_memory < num_examples_per_batch:
            if cyclone_index == len(cyclone_id_strings):
                cyclone_index = 0

            num_examples_to_read = min([
                max_examples_per_cyclone,
                num_examples_per_batch - num_examples_in_memory
            ])

            new_target_times_unix_sec = (
                nn_training_fancy.choose_random_target_times(
                    all_target_times_unix_sec=
                    target_times_by_cyclone_unix_sec[cyclone_index] + 0,
                    num_times_desired=num_examples_to_read
                )[0]
            )

            if len(new_target_times_unix_sec) == 0:
                cyclone_index += 1
                continue

            data_dict = _read_satellite_data_1cyclone(
                input_file_names=satellite_file_names_by_cyclone[cyclone_index],
                lag_times_minutes=lag_times_minutes,
                low_res_wavelengths_microns=low_res_wavelengths_microns,
                num_rows_low_res=None,
                num_columns_low_res=None,
                return_coords=(
                    use_extrap_based_forecasts or
                    short_track_dir_name is not None
                ),
                target_times_unix_sec=new_target_times_unix_sec
            )
            cyclone_index += 1

            if __are_satellite_data_empty(data_dict):
                continue

            if short_track_dir_name is None:
                this_num_examples = data_dict[BRIGHTNESS_TEMPS_KEY].shape[0]
                these_row_trans_px = numpy.full(this_num_examples, 0, dtype=int)
                these_column_trans_px = numpy.full(
                    this_num_examples, 0, dtype=int
                )
            else:
                (
                    data_dict, these_row_trans_px, these_column_trans_px
                ) = _recenter_satellite_on_short_track(
                    satellite_data_dict=data_dict,
                    satellite_lag_times_minutes=lag_times_minutes,
                    short_track_dir_name=short_track_dir_name,
                    max_short_track_lead_time_minutes=
                    short_track_max_lead_minutes,
                    center_each_lag_time_differently=
                    short_track_center_each_lag_diffly
                )

            if __are_satellite_data_empty(data_dict):
                continue

            this_bt_matrix_kelvins = data_dict[BRIGHTNESS_TEMPS_KEY]
            these_grid_spacings_km = data_dict[GRID_SPACINGS_KEY]
            these_center_latitudes_deg_n = data_dict[CENTER_LATITUDES_KEY]
            these_id_strings = data_dict[CYCLONE_IDS_KEY]
            these_target_times_unix_sec = data_dict[TARGET_TIMES_KEY]

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

                if use_extrap_based_forecasts:
                    this_scalar_predictor_matrix = (
                        _extrap_based_forecasts_to_rowcol(
                            scalar_predictor_matrix=
                            this_scalar_predictor_matrix,
                            scalar_a_deck_field_names=scalar_a_deck_field_names,
                            satellite_data_dict=data_dict
                        )
                    )

            this_vector_predictor_matrix = (
                nn_utils.combine_lag_times_and_wavelengths(
                    this_bt_matrix_kelvins
                )
            )

            if vector_predictor_matrix is None:
                these_dim = (
                    (num_examples_per_batch,) +
                    this_vector_predictor_matrix.shape[1:]
                )
                vector_predictor_matrix = numpy.full(these_dim, numpy.nan)

                grid_spacings_km = numpy.full(num_examples_per_batch, numpy.nan)
                cyclone_center_latitudes_deg_n = numpy.full(
                    num_examples_per_batch, numpy.nan
                )
                row_translations_low_res_px = numpy.full(
                    num_examples_per_batch, 0, dtype=int
                )
                column_translations_low_res_px = numpy.full(
                    num_examples_per_batch, 0, dtype=int
                )

                return_cyclone_id_strings = numpy.full(
                    num_examples_per_batch, '', dtype=object
                )
                target_times_unix_sec = numpy.full(
                    num_examples_per_batch, -1, dtype=int
                )

                if this_scalar_predictor_matrix is not None:
                    these_dim = (
                        (num_examples_per_batch,) +
                        this_scalar_predictor_matrix.shape[1:]
                    )
                    scalar_predictor_matrix = numpy.full(these_dim, numpy.nan)

            first_index = num_examples_in_memory
            last_index = first_index + this_vector_predictor_matrix.shape[0]
            vector_predictor_matrix[first_index:last_index, ...] = (
                this_vector_predictor_matrix
            )
            grid_spacings_km[first_index:last_index] = these_grid_spacings_km
            cyclone_center_latitudes_deg_n[first_index:last_index] = (
                these_center_latitudes_deg_n
            )
            return_cyclone_id_strings[first_index:last_index] = numpy.array(
                these_id_strings
            )
            target_times_unix_sec[first_index:last_index] = (
                these_target_times_unix_sec
            )
            row_translations_low_res_px[first_index:last_index] = (
                these_row_trans_px
            )
            column_translations_low_res_px[first_index:last_index] = (
                these_column_trans_px
            )

            if this_scalar_predictor_matrix is not None:
                scalar_predictor_matrix[first_index:last_index, ...] = (
                    this_scalar_predictor_matrix
                )

            num_examples_in_memory += this_vector_predictor_matrix.shape[0]

        translation_dict = data_augmentation.augment_data(
            bidirectional_reflectance_matrix=None,
            brightness_temp_matrix_kelvins=vector_predictor_matrix,
            num_translations_per_example=data_aug_num_translations,
            mean_translation_low_res_px=data_aug_mean_translation_low_res_px,
            stdev_translation_low_res_px=data_aug_stdev_translation_low_res_px,
            sentinel_value=SENTINEL_VALUE
        )
        vector_predictor_matrix = translation_dict[
            data_augmentation.BRIGHTNESS_TEMPS_KEY
        ]
        new_row_trans_px = translation_dict[
            data_augmentation.ROW_TRANSLATIONS_KEY
        ]
        new_column_trans_px = translation_dict[
            data_augmentation.COLUMN_TRANSLATIONS_KEY
        ]
        orig_idxs = translation_dict[
            data_augmentation.ORIG_EXAMPLE_INDICES_KEY
        ]
        del translation_dict

        row_translations_low_res_px = (
            row_translations_low_res_px[orig_idxs] + new_row_trans_px
        )
        column_translations_low_res_px = (
            column_translations_low_res_px[orig_idxs] + new_column_trans_px
        )

        vector_predictor_matrix = nn_utils.separate_lag_times_and_wavelengths(
            satellite_data_matrix=vector_predictor_matrix,
            num_lag_times=len(lag_times_minutes)
        )

        for j in range(len(lag_times_minutes)):
            translation_dict = data_augmentation.augment_data(
                bidirectional_reflectance_matrix=None,
                brightness_temp_matrix_kelvins=
                vector_predictor_matrix[..., j, :],
                num_translations_per_example=1,
                mean_translation_low_res_px=data_aug_within_mean_trans_px,
                stdev_translation_low_res_px=data_aug_within_stdev_trans_px,
                sentinel_value=SENTINEL_VALUE
            )
            vector_predictor_matrix[..., j, :] = translation_dict[
                data_augmentation.BRIGHTNESS_TEMPS_KEY
            ]

            if j != len(lag_times_minutes) - 1:
                del translation_dict
                break

            new_row_trans_px = translation_dict[
                data_augmentation.ROW_TRANSLATIONS_KEY
            ]
            new_column_trans_px = translation_dict[
                data_augmentation.COLUMN_TRANSLATIONS_KEY
            ]
            del translation_dict

            row_translations_low_res_px += new_row_trans_px
            column_translations_low_res_px += new_column_trans_px

        vector_predictor_matrix = nn_utils.combine_lag_times_and_wavelengths(
            vector_predictor_matrix
        )
        vector_predictor_matrix = data_augmentation.subset_grid_after_data_aug(
            data_matrix=vector_predictor_matrix,
            num_rows_to_keep=num_rows_low_res,
            num_columns_to_keep=num_columns_low_res,
            for_high_res=False
        )

        grid_spacings_km = numpy.repeat(
            grid_spacings_km, repeats=data_aug_num_translations
        )
        cyclone_center_latitudes_deg_n = numpy.repeat(
            cyclone_center_latitudes_deg_n, repeats=data_aug_num_translations
        )
        return_cyclone_id_strings = numpy.repeat(
            return_cyclone_id_strings, repeats=data_aug_num_translations
        )
        target_times_unix_sec = numpy.repeat(
            target_times_unix_sec, repeats=data_aug_num_translations
        )
        if scalar_predictor_matrix is not None:
            scalar_predictor_matrix = numpy.repeat(
                scalar_predictor_matrix, axis=0,
                repeats=data_aug_num_translations
            )

        if use_extrap_based_forecasts:
            row_index = scalar_a_deck_field_names.index(
                a_deck_io.UNNORM_EXTRAP_LATITUDE_KEY
            )
            column_index = scalar_a_deck_field_names.index(
                a_deck_io.UNNORM_EXTRAP_LONGITUDE_KEY
            )

            scalar_predictor_matrix[:, row_index] = (
                scalar_predictor_matrix[:, row_index] +
                row_translations_low_res_px
            )
            scalar_predictor_matrix[:, column_index] = (
                scalar_predictor_matrix[:, column_index] +
                column_translations_low_res_px
            )

        vector_predictor_matrix = nn_utils.separate_lag_times_and_wavelengths(
            satellite_data_matrix=vector_predictor_matrix,
            num_lag_times=len(lag_times_minutes)
        )

        predictor_matrices = [vector_predictor_matrix]
        if scalar_predictor_matrix is not None:
            predictor_matrices.append(scalar_predictor_matrix)

        target_matrix = numpy.transpose(numpy.vstack((
            row_translations_low_res_px, column_translations_low_res_px,
            grid_spacings_km, cyclone_center_latitudes_deg_n
        )))

        bad_example_flags = numpy.any(
            vector_predictor_matrix < SENTINEL_VALUE + 1,
            axis=(1, 2, 3, 4)
        )

        if numpy.any(bad_example_flags):
            warning_string = (
                'POTENTIAL ERROR: {0:d} of {1:d} examples contain sentinel '
                'value, which means that translation was too large.'
            ).format(
                numpy.sum(bad_example_flags),
                len(bad_example_flags)
            )

            warnings.warn(warning_string)

        if numpy.all(bad_example_flags):
            continue

        good_indices = numpy.where(numpy.invert(bad_example_flags))[0]
        predictor_matrices = [
            p.astype('float16')[good_indices, ...] for p in predictor_matrices
        ]
        target_matrix = target_matrix[good_indices, ...]

        predictor_matrices = [p.astype('float16') for p in predictor_matrices]

        if for_plotting:
            yield tuple(predictor_matrices), target_matrix, return_cyclone_id_strings, target_times_unix_sec
        else:
            yield tuple(predictor_matrices), target_matrix


def train_model(
        model_object, output_dir_name, num_epochs,
        num_training_batches_per_epoch, training_option_dict,
        num_validation_batches_per_epoch, validation_option_dict,
        loss_function_string, optimizer_function_string,
        plateau_patience_epochs, plateau_learning_rate_multiplier,
        early_stopping_patience_epochs, use_shuffled_data,
        cnn_architecture_dict, temporal_cnn_architecture_dict,
        u_net_architecture_dict, use_old_shuffled_generator=False):
    """Trains neural net.

    :param model_object: See doc for `neural_net_training_fancy.train_model`.
    :param output_dir_name: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param training_option_dict: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validation_option_dict: Same.
    :param loss_function_string: Same.
    :param optimizer_function_string: Same.
    :param plateau_patience_epochs: Same.
    :param plateau_learning_rate_multiplier: Same.
    :param early_stopping_patience_epochs: Same.
    :param use_shuffled_data: Boolean flag.  If True, will read training data
        from shuffled input files, each containing multiple cyclones.  If False,
        will read training data from input files with one cyclone-day each.
    :param cnn_architecture_dict: See doc for
        `neural_net_training_fancy.train_model`.
    :param temporal_cnn_architecture_dict: Same.
    :param u_net_architecture_dict: Same.
    :param use_old_shuffled_generator: Boolean flag.  If True, will use
        `data_generator_shuffled_old`.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    backup_dir_name = '{0:s}/backup_and_restore'.format(output_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=backup_dir_name
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
    error_checking.assert_is_boolean(use_shuffled_data)

    validation_keys_to_keep = [SATELLITE_DIRECTORY_KEY, YEARS_KEY]
    for this_key in list(training_option_dict.keys()):
        if this_key in validation_keys_to_keep:
            continue

        validation_option_dict[this_key] = training_option_dict[this_key]

    model_file_name = '{0:s}/model.weights.h5'.format(output_dir_name)
    history_file_name = '{0:s}/history.csv'.format(output_dir_name)

    try:
        history_table_pandas = pandas.read_csv(history_file_name)
        initial_epoch = history_table_pandas['epoch'].max() + 1
        best_validation_loss = history_table_pandas['val_loss'].min()
    except:
        initial_epoch = 0
        best_validation_loss = numpy.inf

    history_object = keras.callbacks.CSVLogger(
        filename=history_file_name, separator=',', append=True
    )
    checkpoint_object = keras.callbacks.ModelCheckpoint(
        filepath=model_file_name, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=True, mode='min',
        save_freq='epoch'
    )
    checkpoint_object.best = best_validation_loss

    early_stopping_object = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.,
        patience=early_stopping_patience_epochs, verbose=1, mode='min'
    )
    plateau_object = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=plateau_learning_rate_multiplier,
        patience=plateau_patience_epochs, verbose=1, mode='min',
        min_delta=0., cooldown=0
    )
    backup_object = keras.callbacks.BackupAndRestore(
        backup_dir_name, save_freq='epoch', delete_checkpoint=False
    )

    list_of_callback_objects = [
        history_object, checkpoint_object,
        early_stopping_object, plateau_object,
        backup_object
    ]

    if use_shuffled_data:
        if use_old_shuffled_generator:
            print('USING OLD GENERATOR')
            training_generator = data_generator_shuffled_old(
                training_option_dict
            )
            validation_generator = data_generator_shuffled_old(
                validation_option_dict
            )
        else:
            training_generator = data_generator_shuffled(training_option_dict)
            validation_generator = data_generator_shuffled(
                validation_option_dict
            )
    else:
        if use_old_shuffled_generator:
            print('USING OLD GENERATOR')
            training_generator = data_generator_old(training_option_dict)
            validation_generator = data_generator_old(validation_option_dict)
        else:
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
        cnn_architecture_dict=cnn_architecture_dict,
        temporal_cnn_architecture_dict=temporal_cnn_architecture_dict,
        u_net_architecture_dict=u_net_architecture_dict,
        structure_cnn_architecture_dict=None,
        data_type_string=nn_utils.RG_SIMPLE_DATA_TYPE_STRING,
        train_with_shuffled_data=use_shuffled_data
    )

    model_object.fit(
        x=training_generator,
        steps_per_epoch=num_training_batches_per_epoch,
        epochs=num_epochs,
        initial_epoch=initial_epoch,
        verbose=1,
        callbacks=list_of_callback_objects,
        validation_data=validation_generator,
        validation_steps=num_validation_batches_per_epoch
    )


def create_data(option_dict, cyclone_id_string, num_target_times,
                valid_date_string=None):
    """Creates, rather than generates, neural-net inputs.

    E = batch size = number of examples after data augmentation
    L = number of lag times for predictors
    w = number of wavelengths
    m = number of rows in grid
    n = number of columns in grid

    :param option_dict: See doc for `data_generator`.
    :param cyclone_id_string: Will create data for this cyclone.
    :param num_target_times: Will create data for this number of target times
        for the given cyclone.
    :param valid_date_string: Valid date (format "yyyymmdd").  If you want to
        create data for every time step in the cyclone, regardless of date,
        leave this argument alone.

    :return: data_dict: Dictionary with the following keys.
    data_dict["predictor_matrices"]: See doc for `data_generator`.
    data_dict["target_matrix"]: Same.
    data_dict["target_times_unix_sec"]: length-E numpy array of target times.
    data_dict["grid_spacings_low_res_km"]: length-E numpy array of grid
        spacings.
    data_dict["cyclone_center_latitudes_deg_n"]: length-E numpy array of true
        TC-center latitudes (deg north).
    data_dict["high_res_latitude_matrix_deg_n"]: None.
    data_dict["high_res_longitude_matrix_deg_e"]: None.
    data_dict["low_res_latitude_matrix_deg_n"]: E-by-m-by-L numpy array of
        latitudes (deg north).
    data_dict["low_res_longitude_matrix_deg_e"]: E-by-n-by-L numpy array of
        longitudes (deg east).
    """

    error_checking.assert_is_integer(num_target_times)
    error_checking.assert_is_greater(num_target_times, 0)
    option_dict = _check_generator_args(option_dict)

    satellite_dir_name = option_dict[SATELLITE_DIRECTORY_KEY]
    lag_times_minutes = option_dict[LAG_TIMES_KEY]
    low_res_wavelengths_microns = option_dict[LOW_RES_WAVELENGTHS_KEY]
    num_rows_low_res = option_dict[NUM_GRID_ROWS_KEY]
    num_columns_low_res = option_dict[NUM_GRID_COLUMNS_KEY]
    short_track_dir_name = option_dict[SHORT_TRACK_DIR_KEY]
    short_track_max_lead_minutes = option_dict[SHORT_TRACK_MAX_LEAD_KEY]
    short_track_center_each_lag_diffly = option_dict[
        SHORT_TRACK_DIFF_CENTERS_KEY
    ]
    data_aug_num_translations = option_dict[DATA_AUG_NUM_TRANS_KEY]
    data_aug_mean_translation_low_res_px = option_dict[DATA_AUG_MEAN_TRANS_KEY]
    data_aug_stdev_translation_low_res_px = option_dict[
        DATA_AUG_STDEV_TRANS_KEY
    ]
    data_aug_within_mean_trans_px = option_dict[DATA_AUG_WITHIN_MEAN_TRANS_KEY]
    data_aug_within_stdev_trans_px = option_dict[
        DATA_AUG_WITHIN_STDEV_TRANS_KEY
    ]
    semantic_segmentation_flag = option_dict[SEMANTIC_SEG_FLAG_KEY]
    target_smoother_stdev_km = option_dict[TARGET_SMOOOTHER_STDEV_KEY]
    synoptic_times_only = option_dict[SYNOPTIC_TIMES_ONLY_KEY]
    a_deck_file_name = option_dict[A_DECK_FILE_KEY]
    scalar_a_deck_field_names = option_dict[SCALAR_A_DECK_FIELDS_KEY]
    a_decks_at_least_6h_old = option_dict[A_DECKS_AT_LEAST_6H_OLD_KEY]
    remove_nontropical_systems = option_dict[REMOVE_NONTROPICAL_KEY]
    remove_tropical_systems = option_dict[REMOVE_TROPICAL_KEY]

    satellite_file_names = satellite_io.find_files_one_cyclone(
        directory_name=satellite_dir_name,
        cyclone_id_string=cyclone_id_string,
        raise_error_if_all_missing=True
    )

    (
        all_target_times_unix_sec, all_scalar_predictor_matrix
    ) = nn_training_fancy.get_target_times_and_scalar_predictors(
        cyclone_id_strings=[cyclone_id_string],
        synoptic_times_only=synoptic_times_only,
        satellite_file_names_by_cyclone=[satellite_file_names],
        a_deck_file_name=a_deck_file_name,
        scalar_a_deck_field_names=scalar_a_deck_field_names,
        remove_nontropical_systems=remove_nontropical_systems,
        remove_tropical_systems=remove_tropical_systems,
        predictor_lag_times_sec=MINUTES_TO_SECONDS * lag_times_minutes,
        a_decks_at_least_6h_old=a_decks_at_least_6h_old
    )

    all_target_times_unix_sec = all_target_times_unix_sec[0]
    all_scalar_predictor_matrix = all_scalar_predictor_matrix[0]

    if valid_date_string is not None:
        all_target_dates_unix_sec = number_rounding.floor_to_nearest(
            all_target_times_unix_sec, DAYS_TO_SECONDS
        )
        all_target_dates_unix_sec = numpy.round(
            all_target_dates_unix_sec
        ).astype(int)

        valid_date_unix_sec = time_conversion.string_to_unix_sec(
            valid_date_string, DATE_FORMAT
        )
        good_indices = numpy.where(
            all_target_dates_unix_sec == valid_date_unix_sec
        )[0]

        all_target_times_unix_sec = all_target_times_unix_sec[good_indices]
        all_scalar_predictor_matrix = (
            all_scalar_predictor_matrix[good_indices, :]
        )

    conservative_num_target_times = max([
        int(numpy.round(num_target_times * 1.25)),
        num_target_times + 2
    ])

    chosen_target_times_unix_sec = (
        nn_training_fancy.choose_random_target_times(
            all_target_times_unix_sec=all_target_times_unix_sec + 0,
            num_times_desired=conservative_num_target_times
        )[0]
    )

    data_dict = _read_satellite_data_1cyclone(
        input_file_names=satellite_file_names,
        target_times_unix_sec=chosen_target_times_unix_sec,
        lag_times_minutes=lag_times_minutes,
        low_res_wavelengths_microns=low_res_wavelengths_microns,
        num_rows_low_res=None,
        num_columns_low_res=None,
        return_coords=True
    )

    if __are_satellite_data_empty(data_dict):
        return None

    if short_track_dir_name is None:
        this_num_examples = data_dict[BRIGHTNESS_TEMPS_KEY].shape[0]
        row_translations_low_res_px = numpy.full(
            this_num_examples, 0, dtype=int
        )
        column_translations_low_res_px = numpy.full(
            this_num_examples, 0, dtype=int
        )
    else:
        (
            data_dict,
            row_translations_low_res_px,
            column_translations_low_res_px
        ) = _recenter_satellite_on_short_track(
            satellite_data_dict=data_dict,
            satellite_lag_times_minutes=lag_times_minutes,
            short_track_dir_name=short_track_dir_name,
            max_short_track_lead_time_minutes=short_track_max_lead_minutes,
            center_each_lag_time_differently=short_track_center_each_lag_diffly
        )

    if __are_satellite_data_empty(data_dict):
        return None

    use_extrap_based_forecasts = (
        scalar_a_deck_field_names is not None
        and a_deck_io.UNNORM_EXTRAP_LATITUDE_KEY in scalar_a_deck_field_names
        and a_deck_io.UNNORM_EXTRAP_LONGITUDE_KEY in scalar_a_deck_field_names
    )

    if a_deck_file_name is None:
        scalar_predictor_matrix = None
    else:
        row_indices = numpy.array([
            numpy.where(all_target_times_unix_sec == t)[0][0]
            for t in data_dict[TARGET_TIMES_KEY]
        ], dtype=int)

        scalar_predictor_matrix = all_scalar_predictor_matrix[row_indices, :]

        if use_extrap_based_forecasts:
            scalar_predictor_matrix = _extrap_based_forecasts_to_rowcol(
                scalar_predictor_matrix=scalar_predictor_matrix,
                scalar_a_deck_field_names=scalar_a_deck_field_names,
                satellite_data_dict=data_dict
            )

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

    vector_predictor_matrix = nn_utils.combine_lag_times_and_wavelengths(
        brightness_temp_matrix_kelvins
    )

    translation_dict = data_augmentation.augment_data(
        bidirectional_reflectance_matrix=None,
        brightness_temp_matrix_kelvins=vector_predictor_matrix,
        num_translations_per_example=data_aug_num_translations,
        mean_translation_low_res_px=data_aug_mean_translation_low_res_px,
        stdev_translation_low_res_px=data_aug_stdev_translation_low_res_px,
        sentinel_value=SENTINEL_VALUE
    )
    vector_predictor_matrix = translation_dict[
        data_augmentation.BRIGHTNESS_TEMPS_KEY
    ]
    new_row_trans_px = translation_dict[
        data_augmentation.ROW_TRANSLATIONS_KEY
    ]
    new_column_trans_px = translation_dict[
        data_augmentation.COLUMN_TRANSLATIONS_KEY
    ]
    orig_idxs = translation_dict[
        data_augmentation.ORIG_EXAMPLE_INDICES_KEY
    ]
    del translation_dict

    row_translations_low_res_px = (
        row_translations_low_res_px[orig_idxs] + new_row_trans_px
    )
    column_translations_low_res_px = (
        column_translations_low_res_px[orig_idxs] + new_column_trans_px
    )

    vector_predictor_matrix = nn_utils.separate_lag_times_and_wavelengths(
        satellite_data_matrix=vector_predictor_matrix,
        num_lag_times=len(lag_times_minutes)
    )

    for j in range(len(lag_times_minutes)):
        translation_dict = data_augmentation.augment_data(
            bidirectional_reflectance_matrix=None,
            brightness_temp_matrix_kelvins=
            vector_predictor_matrix[..., j, :],
            num_translations_per_example=1,
            mean_translation_low_res_px=data_aug_within_mean_trans_px,
            stdev_translation_low_res_px=data_aug_within_stdev_trans_px,
            sentinel_value=SENTINEL_VALUE
        )
        vector_predictor_matrix[..., j, :] = translation_dict[
            data_augmentation.BRIGHTNESS_TEMPS_KEY
        ]

        if j != len(lag_times_minutes) - 1:
            del translation_dict
            break

        new_row_trans_px = translation_dict[
            data_augmentation.ROW_TRANSLATIONS_KEY
        ]
        new_column_trans_px = translation_dict[
            data_augmentation.COLUMN_TRANSLATIONS_KEY
        ]
        del translation_dict

        row_translations_low_res_px += new_row_trans_px
        column_translations_low_res_px += new_column_trans_px

    vector_predictor_matrix = nn_utils.combine_lag_times_and_wavelengths(
        vector_predictor_matrix
    )
    vector_predictor_matrix = data_augmentation.subset_grid_after_data_aug(
        data_matrix=vector_predictor_matrix,
        num_rows_to_keep=num_rows_low_res,
        num_columns_to_keep=num_columns_low_res,
        for_high_res=False
    )

    # TODO(thunderhoser): This should work, but I'm not 100% sure yet.
    low_res_latitude_matrix_deg_n = numpy.repeat(
        low_res_latitude_matrix_deg_n, repeats=data_aug_num_translations,
        axis=0
    )
    low_res_longitude_matrix_deg_e = numpy.repeat(
        low_res_longitude_matrix_deg_e, repeats=data_aug_num_translations,
        axis=0
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
            num_rows_to_keep=num_rows_low_res,
            num_columns_to_keep=num_columns_low_res,
            for_high_res=False
        )[:, :, 0, :]
    )
    low_res_longitude_matrix_deg_e = (
        data_augmentation.subset_grid_after_data_aug(
            data_matrix=low_res_longitude_matrix_deg_e,
            num_rows_to_keep=num_rows_low_res,
            num_columns_to_keep=num_columns_low_res,
            for_high_res=False
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

    if scalar_predictor_matrix is not None:
        scalar_predictor_matrix = numpy.repeat(
            scalar_predictor_matrix, axis=0,
            repeats=data_aug_num_translations
        )

    if use_extrap_based_forecasts:
        row_index = scalar_a_deck_field_names.index(
            a_deck_io.UNNORM_EXTRAP_LATITUDE_KEY
        )
        column_index = scalar_a_deck_field_names.index(
            a_deck_io.UNNORM_EXTRAP_LONGITUDE_KEY
        )

        scalar_predictor_matrix[:, row_index] = (
            scalar_predictor_matrix[:, row_index] +
            row_translations_low_res_px
        )
        scalar_predictor_matrix[:, column_index] = (
            scalar_predictor_matrix[:, column_index] +
            column_translations_low_res_px
        )

    vector_predictor_matrix = nn_utils.separate_lag_times_and_wavelengths(
        satellite_data_matrix=vector_predictor_matrix,
        num_lag_times=len(lag_times_minutes)
    )

    predictor_matrices = [vector_predictor_matrix]
    if scalar_predictor_matrix is not None:
        predictor_matrices.append(scalar_predictor_matrix)

    if semantic_segmentation_flag:
        target_matrix = nn_utils.make_targets_for_semantic_seg(
            row_translations_px=row_translations_low_res_px,
            column_translations_px=column_translations_low_res_px,
            grid_spacings_km=grid_spacings_km,
            cyclone_center_latitudes_deg_n=cyclone_center_latitudes_deg_n,
            gaussian_smoother_stdev_km=target_smoother_stdev_km,
            num_grid_rows=num_rows_low_res,
            num_grid_columns=num_columns_low_res
        )
    else:
        target_matrix = numpy.transpose(numpy.vstack((
            row_translations_low_res_px, column_translations_low_res_px,
            grid_spacings_km, cyclone_center_latitudes_deg_n
        )))

    bad_example_flags = numpy.any(
        vector_predictor_matrix < SENTINEL_VALUE + 1,
        axis=(1, 2, 3, 4)
    )

    if numpy.any(bad_example_flags):
        warning_string = (
            'POTENTIAL ERROR: {0:d} of {1:d} examples contain sentinel '
            'value, which means that translation was too large.'
        ).format(
            numpy.sum(bad_example_flags),
            len(bad_example_flags)
        )

        warnings.warn(warning_string)

    if numpy.all(bad_example_flags):
        return None

    good_indices = numpy.where(numpy.invert(bad_example_flags))[0]
    predictor_matrices = [p.astype('float16') for p in predictor_matrices]

    return {
        PREDICTOR_MATRICES_KEY:
            [p[good_indices, ...] for p in predictor_matrices],
        TARGET_MATRIX_KEY: target_matrix[good_indices, ...],
        TARGET_TIMES_KEY: target_times_unix_sec[good_indices, ...],
        GRID_SPACINGS_KEY: grid_spacings_km[good_indices, ...],
        CENTER_LATITUDES_KEY: cyclone_center_latitudes_deg_n[good_indices, ...],
        HIGH_RES_LATITUDES_KEY: None,
        HIGH_RES_LONGITUDES_KEY: None,
        LOW_RES_LATITUDES_KEY: low_res_latitude_matrix_deg_n[good_indices, ...],
        LOW_RES_LONGITUDES_KEY:
            low_res_longitude_matrix_deg_e[good_indices, ...]
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

    # TODO(thunderhoser): The input args `row_translations_low_res_px`
    # and `column_translations_low_res_px` should have dimensions E x L,
    # allowing for a different translation at each lag time.  However, this
    # method is currently used only for plotting purposes, so I won't bother for
    # now.

    # Check input args.
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

    option_dict[nn_utils.DATA_AUG_NUM_TRANS_KEY] = 5
    option_dict[nn_utils.DATA_AUG_MEAN_TRANS_KEY] = 10.
    option_dict[nn_utils.DATA_AUG_STDEV_TRANS_KEY] = 10.
    option_dict = _check_generator_args(option_dict)

    # Do actual stuff.
    satellite_dir_name = option_dict[SATELLITE_DIRECTORY_KEY]
    lag_times_minutes = option_dict[LAG_TIMES_KEY]
    low_res_wavelengths_microns = option_dict[LOW_RES_WAVELENGTHS_KEY]
    num_rows_low_res = option_dict[NUM_GRID_ROWS_KEY]
    num_columns_low_res = option_dict[NUM_GRID_COLUMNS_KEY]
    semantic_segmentation_flag = option_dict[SEMANTIC_SEG_FLAG_KEY]
    target_smoother_stdev_km = option_dict[TARGET_SMOOOTHER_STDEV_KEY]
    a_deck_file_name = option_dict[A_DECK_FILE_KEY]
    scalar_a_deck_field_names = option_dict[SCALAR_A_DECK_FIELDS_KEY]
    a_decks_at_least_6h_old = option_dict[A_DECKS_AT_LEAST_6H_OLD_KEY]
    remove_nontropical_systems = option_dict[REMOVE_NONTROPICAL_KEY]
    remove_tropical_systems = option_dict[REMOVE_TROPICAL_KEY]

    satellite_file_names = satellite_io.find_files_one_cyclone(
        directory_name=satellite_dir_name, cyclone_id_string=cyclone_id_string,
        raise_error_if_all_missing=True
    )

    unique_target_times_unix_sec = numpy.unique(target_times_unix_sec)

    data_dict = _read_satellite_data_1cyclone(
        input_file_names=satellite_file_names,
        target_times_unix_sec=unique_target_times_unix_sec,
        lag_times_minutes=lag_times_minutes,
        low_res_wavelengths_microns=low_res_wavelengths_microns,
        num_rows_low_res=None,
        num_columns_low_res=None,
        return_coords=True
    )

    if __are_satellite_data_empty(data_dict):
        return None

    use_extrap_based_forecasts = (
        scalar_a_deck_field_names is not None
        and a_deck_io.UNNORM_EXTRAP_LATITUDE_KEY in scalar_a_deck_field_names
        and a_deck_io.UNNORM_EXTRAP_LONGITUDE_KEY in scalar_a_deck_field_names
    )

    if a_deck_file_name is None:
        scalar_predictor_matrix = None
    else:
        this_num_times = len(data_dict[TARGET_TIMES_KEY])

        scalar_predictor_matrix = nn_utils.read_scalar_data(
            a_deck_file_name=a_deck_file_name,
            field_names=scalar_a_deck_field_names,
            remove_nontropical_systems=remove_nontropical_systems,
            remove_tropical_systems=remove_tropical_systems,
            cyclone_id_strings=[cyclone_id_string] * this_num_times,
            target_times_unix_sec=data_dict[TARGET_TIMES_KEY],
            a_decks_at_least_6h_old=a_decks_at_least_6h_old
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

    if use_extrap_based_forecasts:
        scalar_predictor_matrix = _extrap_based_forecasts_to_rowcol(
            scalar_predictor_matrix=scalar_predictor_matrix,
            scalar_a_deck_field_names=scalar_a_deck_field_names,
            satellite_data_dict={
                LOW_RES_LATITUDES_KEY: low_res_latitude_matrix_deg_n,
                LOW_RES_LONGITUDES_KEY: low_res_longitude_matrix_deg_e
            }
        )

        row_index = scalar_a_deck_field_names.index(
            a_deck_io.UNNORM_EXTRAP_LATITUDE_KEY
        )
        column_index = scalar_a_deck_field_names.index(
            a_deck_io.UNNORM_EXTRAP_LONGITUDE_KEY
        )

        scalar_predictor_matrix[:, row_index] = (
            scalar_predictor_matrix[:, row_index] +
            row_translations_low_res_px
        )
        scalar_predictor_matrix[:, column_index] = (
            scalar_predictor_matrix[:, column_index] +
            column_translations_low_res_px
        )

    vector_predictor_matrix = nn_utils.combine_lag_times_and_wavelengths(
        brightness_temp_matrix_kelvins
    )

    _, vector_predictor_matrix = data_augmentation.augment_data_specific_trans(
        bidirectional_reflectance_matrix=None,
        brightness_temp_matrix_kelvins=vector_predictor_matrix,
        row_translations_low_res_px=row_translations_low_res_px,
        column_translations_low_res_px=column_translations_low_res_px,
        sentinel_value=SENTINEL_VALUE
    )

    vector_predictor_matrix = data_augmentation.subset_grid_after_data_aug(
        data_matrix=vector_predictor_matrix,
        num_rows_to_keep=num_rows_low_res,
        num_columns_to_keep=num_columns_low_res,
        for_high_res=False
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
            num_rows_to_keep=num_rows_low_res,
            num_columns_to_keep=num_columns_low_res,
            for_high_res=False
        )[:, :, 0, :]
    )
    low_res_longitude_matrix_deg_e = (
        data_augmentation.subset_grid_after_data_aug(
            data_matrix=low_res_longitude_matrix_deg_e,
            num_rows_to_keep=num_rows_low_res,
            num_columns_to_keep=num_columns_low_res,
            for_high_res=False
        )[:, 0, :, :]
    )

    vector_predictor_matrix = nn_utils.separate_lag_times_and_wavelengths(
        satellite_data_matrix=vector_predictor_matrix,
        num_lag_times=len(lag_times_minutes)
    )

    predictor_matrices = [vector_predictor_matrix]
    if scalar_predictor_matrix is not None:
        predictor_matrices.append(scalar_predictor_matrix)

    if semantic_segmentation_flag:
        target_matrix = nn_utils.make_targets_for_semantic_seg(
            row_translations_px=row_translations_low_res_px,
            column_translations_px=column_translations_low_res_px,
            grid_spacings_km=grid_spacings_km,
            cyclone_center_latitudes_deg_n=cyclone_center_latitudes_deg_n,
            gaussian_smoother_stdev_km=target_smoother_stdev_km,
            num_grid_rows=num_rows_low_res,
            num_grid_columns=num_columns_low_res
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
        HIGH_RES_LATITUDES_KEY: None,
        HIGH_RES_LONGITUDES_KEY: None,
        LOW_RES_LATITUDES_KEY: low_res_latitude_matrix_deg_n,
        LOW_RES_LONGITUDES_KEY: low_res_longitude_matrix_deg_e
    }
