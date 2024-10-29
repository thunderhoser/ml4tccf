"""NN-training for TC-structure parameters with simple Robert/Galina data."""

import os
import sys
import copy
import time
import random
import numpy
import keras
import pandas
from scipy.interpolate import interp1d

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import number_rounding
import file_system_utils
import error_checking
import a_deck_io
import satellite_io
import extended_best_track_io as ebtrk_io
import extended_best_track_utils as ebtrk_utils
import neural_net_training_simple as nn_training_simple
import neural_net_training_fancy as nn_training_fancy
import neural_net_utils as nn_utils

DATE_FORMAT = '%Y%m%d'

DAYS_TO_SECONDS = 86400
HOURS_TO_SECONDS = 3600
MINUTES_TO_SECONDS = 60
SYNOPTIC_TIME_INTERVAL_SEC = 6 * 3600

R34_FIELD_NAME = 'radius_of_34kt_wind_km'
R50_FIELD_NAME = 'radius_of_50kt_wind_km'
R64_FIELD_NAME = 'radius_of_64kt_wind_km'
RMW_FIELD_NAME = 'radius_of_max_wind_km'
INTENSITY_FIELD_NAME = 'intensity_kt'
ALL_TARGET_FIELD_NAMES = [
    R34_FIELD_NAME, R50_FIELD_NAME, R64_FIELD_NAME, RMW_FIELD_NAME,
    INTENSITY_FIELD_NAME
]

TARGET_NAME_TO_EBTRK_NAME = {
    R34_FIELD_NAME: ebtrk_utils.RADII_OF_34KT_WIND_KEY,
    R50_FIELD_NAME: ebtrk_utils.RADII_OF_50KT_WIND_KEY,
    R64_FIELD_NAME: ebtrk_utils.RADII_OF_64KT_WIND_KEY,
    RMW_FIELD_NAME: ebtrk_utils.MAX_WIND_RADIUS_KEY,
    INTENSITY_FIELD_NAME: ebtrk_utils.MAX_SUSTAINED_WIND_KEY
}

METRES_TO_KM = 0.001
METRES_PER_SECOND_TO_KT = 3.6 / 1.852

TARGET_NAME_TO_CONV_FACTOR = {
    R34_FIELD_NAME: METRES_TO_KM,
    R50_FIELD_NAME: METRES_TO_KM,
    R64_FIELD_NAME: METRES_TO_KM,
    RMW_FIELD_NAME: METRES_TO_KM,
    INTENSITY_FIELD_NAME: METRES_PER_SECOND_TO_KT
}

BRIGHTNESS_TEMPS_KEY = nn_utils.BRIGHTNESS_TEMPS_KEY
TARGET_TIMES_KEY = nn_utils.TARGET_TIMES_KEY
CYCLONE_IDS_KEY = 'cyclone_id_strings'
PREDICTOR_MATRICES_KEY = nn_utils.PREDICTOR_MATRICES_KEY
TARGET_MATRIX_KEY = nn_utils.TARGET_MATRIX_KEY
LATITUDES_KEY = 'latitude_matrix_deg_n'
LONGITUDES_KEY = 'longitude_matrix_deg_e'

SATELLITE_DIRECTORY_KEY = 'satellite_dir_name'
YEARS_KEY = 'years'
LAG_TIMES_KEY = 'lag_times_minutes'
LOW_RES_WAVELENGTHS_KEY = 'low_res_wavelengths_microns'
BATCH_SIZE_KEY = 'num_examples_per_batch'
NUM_GRID_ROWS_KEY = 'num_rows_low_res'
NUM_GRID_COLUMNS_KEY = 'num_columns_low_res'
# DATA_AUG_NUM_TRANS_KEY = 'data_aug_num_translations'
# DATA_AUG_MEAN_TRANS_KEY = 'data_aug_mean_translation_low_res_px'
# DATA_AUG_STDEV_TRANS_KEY = 'data_aug_stdev_translation_low_res_px'
SENTINEL_VALUE_KEY = 'sentinel_value'
SYNOPTIC_TIMES_ONLY_KEY = 'synoptic_times_only'
A_DECK_FILE_KEY = 'a_deck_file_name'
SCALAR_A_DECK_FIELDS_KEY = 'scalar_a_deck_field_names'
REMOVE_NONTROPICAL_KEY = 'remove_nontropical_systems'
REMOVE_TROPICAL_KEY = 'remove_tropical_systems'
TARGET_FIELDS_KEY = 'target_field_names'
TARGET_FILE_KEY = 'target_file_name'
DO_RESIDUAL_PREDICTION_KEY = 'do_residual_prediction'

DEFAULT_GENERATOR_OPTION_DICT = {
    SENTINEL_VALUE_KEY: -10.,
    SYNOPTIC_TIMES_ONLY_KEY: True,
    A_DECK_FILE_KEY: None,
    SCALAR_A_DECK_FIELDS_KEY: [],
    REMOVE_NONTROPICAL_KEY: False,
    REMOVE_TROPICAL_KEY: False
}


def _get_target_variables(ebtrk_table_xarray, target_field_names,
                          cyclone_id_string, target_time_unix_sec):
    """Returns target variables for the given TC sample.

    T = number of target fields

    :param ebtrk_table_xarray: xarray table in format returned by
        `extended_best_track_io.read_file`.
    :param target_field_names: 1-D list with names of target fields.
    :param cyclone_id_string: Cyclone ID.
    :param target_time_unix_sec: Target time.
    :return: target_values: length-T numpy array of target values, with all wind
        radii in km and intensity in kt.
    """

    earlier_synoptic_time_unix_sec = int(number_rounding.floor_to_nearest(
        target_time_unix_sec, SYNOPTIC_TIME_INTERVAL_SEC
    ))
    later_synoptic_time_unix_sec = int(number_rounding.ceiling_to_nearest(
        target_time_unix_sec, SYNOPTIC_TIME_INTERVAL_SEC
    ))
    all_valid_times_unix_sec = (
        ebtrk_table_xarray[ebtrk_utils.VALID_TIME_KEY].values * HOURS_TO_SECONDS
    )

    earlier_indices = numpy.where(numpy.logical_and(
        ebtrk_table_xarray[ebtrk_utils.STORM_ID_KEY].values ==
        cyclone_id_string,
        all_valid_times_unix_sec == earlier_synoptic_time_unix_sec
    ))[0]
    if len(earlier_indices) == 0:
        return None

    later_indices = numpy.where(numpy.logical_and(
        ebtrk_table_xarray[ebtrk_utils.STORM_ID_KEY].values ==
        cyclone_id_string,
        all_valid_times_unix_sec == later_synoptic_time_unix_sec
    ))[0]
    if len(later_indices) == 0:
        return None

    earlier_index = earlier_indices[0]
    later_index = later_indices[0]
    num_target_fields = len(target_field_names)
    target_values = numpy.full(num_target_fields, numpy.nan)

    for f in range(num_target_fields):
        this_name = TARGET_NAME_TO_EBTRK_NAME[target_field_names[f]]

        if target_field_names[f] in [
                R34_FIELD_NAME, R50_FIELD_NAME, R64_FIELD_NAME
        ]:
            earlier_value = numpy.mean(
                ebtrk_table_xarray[this_name].values[earlier_index, :]
            )
            later_value = numpy.mean(
                ebtrk_table_xarray[this_name].values[later_index, :]
            )
        else:
            earlier_value = ebtrk_table_xarray[this_name].values[earlier_index]
            later_value = ebtrk_table_xarray[this_name].values[later_index]

        if numpy.isnan(earlier_value):
            return None
        if numpy.isnan(later_value):
            return None

        earlier_value *= TARGET_NAME_TO_CONV_FACTOR[target_field_names[f]]
        later_value *= TARGET_NAME_TO_CONV_FACTOR[target_field_names[f]]

        if earlier_synoptic_time_unix_sec == later_synoptic_time_unix_sec:
            target_values[f] = earlier_value
            continue

        interp_object = interp1d(
            x=numpy.array(
                [earlier_synoptic_time_unix_sec, later_synoptic_time_unix_sec]
            ),
            y=numpy.array([earlier_value, later_value]),
            kind='linear',
            bounds_error=True,
            assume_sorted=True
        )
        target_values[f] = interp_object(target_time_unix_sec)

    if (
            R64_FIELD_NAME in target_field_names and
            R50_FIELD_NAME in target_field_names
    ):
        r64_index = target_field_names.index(R64_FIELD_NAME)
        r50_index = target_field_names.index(R50_FIELD_NAME)
        target_values[r50_index] = max([
            target_values[r50_index], target_values[r64_index]
        ])

    if (
            R50_FIELD_NAME in target_field_names and
            R34_FIELD_NAME in target_field_names
    ):
        r50_index = target_field_names.index(R50_FIELD_NAME)
        r34_index = target_field_names.index(R34_FIELD_NAME)
        target_values[r34_index] = max([
            target_values[r34_index], target_values[r50_index]
        ])

    if (
            INTENSITY_FIELD_NAME in target_field_names and
            R34_FIELD_NAME in target_field_names
    ):
        intensity_index = target_field_names.index(INTENSITY_FIELD_NAME)
        r34_index = target_field_names.index(R34_FIELD_NAME)
        if target_values[intensity_index] < 34:
            target_values[r34_index] = 0.

    if (
            INTENSITY_FIELD_NAME in target_field_names and
            R50_FIELD_NAME in target_field_names
    ):
        intensity_index = target_field_names.index(INTENSITY_FIELD_NAME)
        r50_index = target_field_names.index(R50_FIELD_NAME)
        if target_values[intensity_index] < 50:
            target_values[r50_index] = 0.

    if (
            INTENSITY_FIELD_NAME in target_field_names and
            R64_FIELD_NAME in target_field_names
    ):
        intensity_index = target_field_names.index(INTENSITY_FIELD_NAME)
        r64_index = target_field_names.index(R64_FIELD_NAME)
        if target_values[intensity_index] < 64:
            target_values[r64_index] = 0.

    return target_values


def check_generator_args(option_dict):
    """Error-checks input arguments for generator.

    :param option_dict: See documentation for
        `neural_net_training_fancy.data_generator`.
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

    error_checking.assert_is_integer_numpy_array(option_dict[LAG_TIMES_KEY])
    error_checking.assert_is_geq_numpy_array(option_dict[LAG_TIMES_KEY], 0)
    assert numpy.all(numpy.mod(option_dict[LAG_TIMES_KEY], 30) == 0)
    option_dict[LAG_TIMES_KEY] = numpy.sort(option_dict[LAG_TIMES_KEY])[::-1]

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

    error_checking.assert_is_integer(option_dict[NUM_GRID_ROWS_KEY])
    error_checking.assert_is_geq(option_dict[NUM_GRID_ROWS_KEY], 10)
    assert numpy.mod(option_dict[NUM_GRID_ROWS_KEY], 2) == 0

    error_checking.assert_is_integer(option_dict[NUM_GRID_COLUMNS_KEY])
    error_checking.assert_is_geq(option_dict[NUM_GRID_COLUMNS_KEY], 10)
    assert numpy.mod(option_dict[NUM_GRID_COLUMNS_KEY], 2) == 0

    error_checking.assert_is_not_nan(option_dict[SENTINEL_VALUE_KEY])
    error_checking.assert_is_boolean(option_dict[SYNOPTIC_TIMES_ONLY_KEY])
    error_checking.assert_is_list(option_dict[SCALAR_A_DECK_FIELDS_KEY])
    error_checking.assert_is_boolean(option_dict[REMOVE_NONTROPICAL_KEY])
    error_checking.assert_is_boolean(option_dict[REMOVE_TROPICAL_KEY])
    assert not (
        option_dict[REMOVE_NONTROPICAL_KEY] and option_dict[REMOVE_TROPICAL_KEY]
    )

    if option_dict[REMOVE_NONTROPICAL_KEY] or option_dict[REMOVE_TROPICAL_KEY]:
        assert option_dict[A_DECK_FILE_KEY] is not None
    elif len(option_dict[SCALAR_A_DECK_FIELDS_KEY]) > 0:
        assert option_dict[A_DECK_FILE_KEY] is not None
    else:
        option_dict[A_DECK_FILE_KEY] = None

    if option_dict[A_DECK_FILE_KEY] is not None:
        error_checking.assert_is_string(option_dict[A_DECK_FILE_KEY])
    if len(option_dict[SCALAR_A_DECK_FIELDS_KEY]) > 0:
        error_checking.assert_is_string_list(
            option_dict[SCALAR_A_DECK_FIELDS_KEY]
        )

    error_checking.assert_is_boolean(option_dict[DO_RESIDUAL_PREDICTION_KEY])
    if option_dict[DO_RESIDUAL_PREDICTION_KEY]:
        assert option_dict[A_DECK_FILE_KEY] is not None

    return option_dict


def create_data(option_dict, cyclone_id_string, num_target_times,
                target_date_string=None):
    """Creates validation or testing data from non-shuffled files.

    E = number of examples (or number of "TC samples")
    L = number of lag times for predictors
    M = number of rows in grid
    N = number of columns in grid

    :param option_dict: See documentation for `data_generator_shuffled`.
    :param cyclone_id_string: Cyclone ID (must be accepted by
        `misc_utils.parse_cyclone_id`).  Data will be created only for this
        tropical cyclone.
    :param num_target_times: Number of target times.  Data will be created for
        this many TC samples.
    :param target_date_string: Target date (format "yyyymmdd").  Data will be
        created only for the given TC and this date.  If you want to create data
        for the given TC regardless of date, leave this argument alone.
    :return: data_dict: Dictionary with the following keys.
    data_dict["predictor_matrices"]: See documentation for
        `data_generator_shuffled`.
    data_dict["target_matrix"]: Same.
    data_dict["target_times_unix_sec"]: length-E numpy array of target times.
    data_dict["latitude_matrix_deg_n"]: E-by-M-by-L numpy array of latitudes
        (deg north).
    data_dict["longitude_matrix_deg_e"]: E-by-M-by-L numpy array of longitudes
        (deg east).
    """

    error_checking.assert_is_integer(num_target_times)
    error_checking.assert_is_greater(num_target_times, 0)

    option_dict[BATCH_SIZE_KEY] = 10
    option_dict[YEARS_KEY] = numpy.array([1990], dtype=int)
    option_dict = check_generator_args(option_dict)

    satellite_dir_name = option_dict[SATELLITE_DIRECTORY_KEY]
    lag_times_minutes = option_dict[LAG_TIMES_KEY]
    low_res_wavelengths_microns = option_dict[LOW_RES_WAVELENGTHS_KEY]
    num_rows_low_res = option_dict[NUM_GRID_ROWS_KEY]
    num_columns_low_res = option_dict[NUM_GRID_COLUMNS_KEY]
    synoptic_times_only = option_dict[SYNOPTIC_TIMES_ONLY_KEY]
    scalar_a_deck_field_names = copy.deepcopy(
        option_dict[SCALAR_A_DECK_FIELDS_KEY]
    )
    remove_nontropical_systems = option_dict[REMOVE_NONTROPICAL_KEY]
    remove_tropical_systems = option_dict[REMOVE_TROPICAL_KEY]
    a_deck_file_name = option_dict[A_DECK_FILE_KEY]
    target_field_names = option_dict[TARGET_FIELDS_KEY]
    target_file_name = option_dict[TARGET_FILE_KEY]
    do_residual_prediction = option_dict[DO_RESIDUAL_PREDICTION_KEY]

    assert not a_deck_io.UNNORM_EXTRAP_LATITUDE_KEY in scalar_a_deck_field_names
    assert not a_deck_io.UNNORM_EXTRAP_LONGITUDE_KEY in scalar_a_deck_field_names

    satellite_file_names = satellite_io.find_files_one_cyclone(
        directory_name=satellite_dir_name,
        cyclone_id_string=cyclone_id_string,
        raise_error_if_all_missing=True
    )

    first_scalar_a_deck_field_names = copy.deepcopy(scalar_a_deck_field_names)

    if do_residual_prediction:
        for this_target_field_name in target_field_names:
            if this_target_field_name == INTENSITY_FIELD_NAME:
                scalar_a_deck_field_names.append(
                    a_deck_io.UNNORM_INTENSITY_KEY
                )
            elif this_target_field_name == R34_FIELD_NAME:
                scalar_a_deck_field_names.append(
                    a_deck_io.UNNORM_WIND_RADIUS_34KT_KEY
                )
            elif this_target_field_name == R50_FIELD_NAME:
                scalar_a_deck_field_names.append(
                    a_deck_io.UNNORM_WIND_RADIUS_50KT_KEY
                )
            elif this_target_field_name == R64_FIELD_NAME:
                scalar_a_deck_field_names.append(
                    a_deck_io.UNNORM_WIND_RADIUS_64KT_KEY
                )
            elif this_target_field_name == RMW_FIELD_NAME:
                scalar_a_deck_field_names.append(
                    a_deck_io.UNNORM_MAX_WIND_RADIUS_KEY
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
        a_decks_at_least_6h_old=True
    )

    all_target_times_unix_sec = all_target_times_unix_sec[0]
    all_scalar_predictor_matrix = all_scalar_predictor_matrix[0]

    if target_date_string is not None:
        all_target_dates_unix_sec = number_rounding.floor_to_nearest(
            all_target_times_unix_sec, DAYS_TO_SECONDS
        )
        all_target_dates_unix_sec = numpy.round(
            all_target_dates_unix_sec
        ).astype(int)

        target_date_unix_sec = time_conversion.string_to_unix_sec(
            target_date_string, DATE_FORMAT
        )
        good_indices = numpy.where(
            all_target_dates_unix_sec == target_date_unix_sec
        )[0]

        all_target_times_unix_sec = all_target_times_unix_sec[good_indices]
        all_scalar_predictor_matrix = (
            all_scalar_predictor_matrix[good_indices, :]
        )

    print('Reading data from: "{0:s}"...'.format(target_file_name))
    ebtrk_table_xarray = ebtrk_io.read_file(target_file_name)

    all_target_values_by_sample = [
        _get_target_variables(
            ebtrk_table_xarray=ebtrk_table_xarray,
            target_field_names=target_field_names,
            cyclone_id_string=cyclone_id_string,
            target_time_unix_sec=t
        )
        for t in all_target_times_unix_sec
    ]

    good_flags = numpy.array(
        [tv is not None for tv in all_target_values_by_sample], dtype=int
    )
    good_indices = numpy.where(good_flags)[0]
    if len(good_indices) == 0:
        return None

    all_scalar_predictor_matrix = all_scalar_predictor_matrix[good_indices, :]
    all_target_times_unix_sec = all_target_times_unix_sec[good_indices]
    all_target_values_by_sample = [
        all_target_values_by_sample[k] for k in good_indices
    ]
    all_target_matrix = numpy.vstack(all_target_values_by_sample)

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

    data_dict = nn_training_simple._read_satellite_data_1cyclone(
        input_file_names=satellite_file_names,
        target_times_unix_sec=chosen_target_times_unix_sec,
        lag_times_minutes=lag_times_minutes,
        low_res_wavelengths_microns=low_res_wavelengths_microns,
        num_rows_low_res=num_rows_low_res,
        num_columns_low_res=num_columns_low_res,
        return_coords=True,
        return_xy_coords=False
    )

    if (
            data_dict is None
            or data_dict[BRIGHTNESS_TEMPS_KEY] is None
            or data_dict[BRIGHTNESS_TEMPS_KEY].size == 0
    ):
        return None

    row_indices = numpy.array([
        numpy.where(all_target_times_unix_sec == t)[0][0]
        for t in data_dict[TARGET_TIMES_KEY]
    ], dtype=int)

    target_matrix = all_target_matrix[row_indices, :]
    target_matrix = target_matrix[:num_target_times, :]

    if a_deck_file_name is None:
        scalar_predictor_matrix = None
    else:
        scalar_predictor_matrix = all_scalar_predictor_matrix[row_indices, :]
        scalar_predictor_matrix = scalar_predictor_matrix[:num_target_times, :]

    vector_predictor_matrix = (
        data_dict[BRIGHTNESS_TEMPS_KEY][:num_target_times, ...]
    )
    target_times_unix_sec = data_dict[TARGET_TIMES_KEY][:num_target_times, ...]
    latitude_matrix_deg_n = data_dict[
        nn_training_simple.LOW_RES_LATITUDES_KEY
    ][:num_target_times, ...]
    longitude_matrix_deg_e = data_dict[
        nn_training_simple.LOW_RES_LONGITUDES_KEY
    ][:num_target_times, ...]

    vector_predictor_matrix = nn_utils.combine_lag_times_and_wavelengths(
        vector_predictor_matrix
    )

    if do_residual_prediction:
        residual_baseline_matrix = (
            scalar_predictor_matrix[:, len(first_scalar_a_deck_field_names):]
        )
        scalar_predictor_matrix = (
            scalar_predictor_matrix[:, :len(first_scalar_a_deck_field_names)]
        )

        for f in range(len(target_field_names)):
            residual_baseline_matrix[:, f] *= (
                TARGET_NAME_TO_CONV_FACTOR[target_field_names[f]]
            )
    else:
        residual_baseline_matrix = None

    # TODO(thunderhoser): This is a HACK.  Should be controlled by an input
    # arg.
    final_axis_length = len(low_res_wavelengths_microns)
    new_dimensions = (
        vector_predictor_matrix.shape[:3] +
        (len(lag_times_minutes), final_axis_length)
    )
    vector_predictor_matrix = numpy.reshape(
        vector_predictor_matrix, new_dimensions
    )

    predictor_matrices = [vector_predictor_matrix]
    if scalar_predictor_matrix is not None:
        predictor_matrices.append(scalar_predictor_matrix)
    if residual_baseline_matrix is not None:
        predictor_matrices.append(residual_baseline_matrix)

    predictor_matrices = [p.astype('float32') for p in predictor_matrices]

    for i in range(len(predictor_matrices)):
        print('Shape of {0:d}th predictor matrix = {1:s}'.format(
            i + 1, str(predictor_matrices[i].shape)
        ))
        print((
            'NaN fraction and min/max in {0:d}th predictor matrix = '
            '{1:.4f}, {2:.4f}, {3:.4f}'
        ).format(
            i + 1,
            numpy.mean(numpy.isnan(predictor_matrices[i])),
            numpy.nanmin(predictor_matrices[i]),
            numpy.nanmax(predictor_matrices[i])
        ))

    if residual_baseline_matrix is not None:
        print((
            'Min/max of every channel in residual baseline matrix:'
            '\n{0:s}\n{1:s}'
        ).format(
            str(numpy.min(residual_baseline_matrix, axis=0)),
            str(numpy.max(residual_baseline_matrix, axis=0))
        ))

    print('Shape of target matrix = {0:s}'.format(
        str(target_matrix.shape)
    ))
    print((
        'NaN fraction and min/max in target matrix = '
        '{0:.4f}, {1:.4f}, {2:.4f}'
    ).format(
        numpy.mean(numpy.isnan(target_matrix)),
        numpy.nanmin(target_matrix),
        numpy.nanmax(target_matrix)
    ))

    return {
        PREDICTOR_MATRICES_KEY: predictor_matrices,
        TARGET_MATRIX_KEY: target_matrix,
        TARGET_TIMES_KEY: target_times_unix_sec,
        LATITUDES_KEY: latitude_matrix_deg_n,
        LONGITUDES_KEY: longitude_matrix_deg_e
    }


def data_generator_shuffled(option_dict, return_cyclone_ids=False):
    """Generates training data from shuffled files.

    E = batch size = number of examples
    L = number of lag times for predictors
    W = number of wavelengths
    M = number of rows in grid
    N = number of columns in grid
    F = number of scalar predictor fields
    T = number of target fields

    :param option_dict: Dictionary with the following keys.
    option_dict["satellite_dir_name"]: Name of directory with satellite data.
        Files therein will be found by `satellite_io.find_file` and read
        by `satellite_io.read_file`.
    option_dict["years"]: 1-D numpy array of years.  Will generate data only for
        cyclones in these years.
    option_dict["lag_times_minutes"]: length-L numpy array of lag times for
        predictors.
    option_dict["low_res_wavelengths_microns"]: length-W numpy array of
        wavelengths.
    option_dict["num_examples_per_batch"]: Batch size.
    option_dict["num_rows_low_res"]: Number of grid rows to keep.  This is M in
        the above definitions.
    option_dict["num_columns_low_res"]: Number of grid columns to keep.  This is
        N in the above definitions.
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
    option_dict["a_deck_file_name"]: Path to A-deck file, which is needed if
        `len(scalar_a_deck_field_names) > 0 or remove_nontropical_systems or
        remove_tropical_systems`.  If A-deck file is not needed, you can make
        this None.
    option_dict["target_field_names"]: length-T list with names of target
        fields.  Each name must belong to the list `ALL_TARGET_FIELD_NAMES`
        defined at the top of this module.
    option_dict["target_file_name"]: Path to target file (will be read by
        `extended_best_track_io.read_file`).
    option_dict["do_residual_prediction"]: Boolean flag.

    :param return_cyclone_ids: Leave this alone.

    :return: predictor_matrices: Tuple with the following items:
        (vector_predictor_matrix, scalar_predictor_matrix,
         resid_baseline_predictor_matrix).  Either of the last two items might
        be missing.

        vector_predictor_matrix: E-by-M-by-N-by-W-by-L numpy array of
            brightness temperatures.
        scalar_predictor_matrix: E-by-F numpy array of scalar predictors.
        resid_baseline_predictor_matrix: E-by-T numpy array of baseline
            predictions.

    :return: target_matrix: E-by-T numpy array of target values.
    """

    option_dict = check_generator_args(option_dict)
    error_checking.assert_is_boolean(return_cyclone_ids)

    satellite_dir_name = option_dict[SATELLITE_DIRECTORY_KEY]
    years = option_dict[YEARS_KEY]
    lag_times_minutes = option_dict[LAG_TIMES_KEY]
    low_res_wavelengths_microns = option_dict[LOW_RES_WAVELENGTHS_KEY]
    num_examples_per_batch = option_dict[BATCH_SIZE_KEY]
    num_rows_low_res = option_dict[NUM_GRID_ROWS_KEY]
    num_columns_low_res = option_dict[NUM_GRID_COLUMNS_KEY]
    # data_aug_num_translations = option_dict[DATA_AUG_NUM_TRANS_KEY]
    # data_aug_mean_translation_low_res_px = option_dict[DATA_AUG_MEAN_TRANS_KEY]
    # data_aug_stdev_translation_low_res_px = (
    #     option_dict[DATA_AUG_STDEV_TRANS_KEY]
    # )
    # sentinel_value = option_dict[SENTINEL_VALUE_KEY]
    synoptic_times_only = option_dict[SYNOPTIC_TIMES_ONLY_KEY]
    scalar_a_deck_field_names = copy.deepcopy(
        option_dict[SCALAR_A_DECK_FIELDS_KEY]
    )
    remove_nontropical_systems = option_dict[REMOVE_NONTROPICAL_KEY]
    remove_tropical_systems = option_dict[REMOVE_TROPICAL_KEY]
    a_deck_file_name = option_dict[A_DECK_FILE_KEY]
    target_field_names = option_dict[TARGET_FIELDS_KEY]
    target_file_name = option_dict[TARGET_FILE_KEY]
    do_residual_prediction = option_dict[DO_RESIDUAL_PREDICTION_KEY]

    assert not a_deck_io.UNNORM_EXTRAP_LATITUDE_KEY in scalar_a_deck_field_names
    assert not a_deck_io.UNNORM_EXTRAP_LONGITUDE_KEY in scalar_a_deck_field_names

    satellite_file_names = satellite_io.find_shuffled_files(
        directory_name=satellite_dir_name, raise_error_if_all_missing=True
    )
    random.shuffle(satellite_file_names)

    first_scalar_a_deck_field_names = copy.deepcopy(scalar_a_deck_field_names)

    if do_residual_prediction:
        for this_target_field_name in target_field_names:
            if this_target_field_name == INTENSITY_FIELD_NAME:
                scalar_a_deck_field_names.append(
                    a_deck_io.UNNORM_INTENSITY_KEY
                )
            elif this_target_field_name == R34_FIELD_NAME:
                scalar_a_deck_field_names.append(
                    a_deck_io.UNNORM_WIND_RADIUS_34KT_KEY
                )
            elif this_target_field_name == R50_FIELD_NAME:
                scalar_a_deck_field_names.append(
                    a_deck_io.UNNORM_WIND_RADIUS_50KT_KEY
                )
            elif this_target_field_name == R64_FIELD_NAME:
                scalar_a_deck_field_names.append(
                    a_deck_io.UNNORM_WIND_RADIUS_64KT_KEY
                )
            elif this_target_field_name == RMW_FIELD_NAME:
                scalar_a_deck_field_names.append(
                    a_deck_io.UNNORM_MAX_WIND_RADIUS_KEY
                )

    (
        cyclone_id_strings_by_file,
        target_times_by_file_unix_sec,
        scalar_predictor_matrix_by_file
    ) = nn_training_simple.get_times_and_scalar_preds_shuffled(
        satellite_file_names=satellite_file_names,
        a_deck_file_name=a_deck_file_name,
        scalar_a_deck_field_names=scalar_a_deck_field_names,
        remove_nontropical_systems=remove_nontropical_systems,
        remove_tropical_systems=remove_tropical_systems,
        desired_years=years,
        predictor_lag_times_sec=MINUTES_TO_SECONDS * lag_times_minutes,
        a_decks_at_least_6h_old=True
    )

    if synoptic_times_only:
        num_files = len(cyclone_id_strings_by_file)
        good_file_indices = []

        for i in range(num_files):
            good_indices = numpy.where(
                numpy.mod(
                    target_times_by_file_unix_sec[i], SYNOPTIC_TIME_INTERVAL_SEC
                ) == 0
            )[0]
            if len(good_indices) == 0:
                continue

            cyclone_id_strings_by_file[i] = [
                cyclone_id_strings_by_file[i][k] for k in good_indices
            ]
            target_times_by_file_unix_sec[i] = (
                target_times_by_file_unix_sec[i][good_indices]
            )
            scalar_predictor_matrix_by_file[i] = (
                scalar_predictor_matrix_by_file[i][good_indices, :]
            )

            good_file_indices.append(i)

        cyclone_id_strings_by_file = [
            cyclone_id_strings_by_file[i] for i in good_file_indices
        ]
        target_times_by_file_unix_sec = [
            target_times_by_file_unix_sec[i] for i in good_file_indices
        ]
        scalar_predictor_matrix_by_file = [
            scalar_predictor_matrix_by_file[i] for i in good_file_indices
        ]

    print('Reading data from: "{0:s}"...'.format(target_file_name))
    ebtrk_table_xarray = ebtrk_io.read_file(target_file_name)

    file_index = 0

    while True:
        vector_predictor_matrix = None
        scalar_predictor_matrix = None
        residual_baseline_matrix = None
        target_matrix = None
        cyclone_id_strings = []
        target_times_unix_sec = []
        num_examples_in_memory = 0

        while num_examples_in_memory < num_examples_per_batch:
            if file_index == len(satellite_file_names):
                file_index = 0

            if len(cyclone_id_strings_by_file[file_index]) == 0:
                file_index += 1
                continue

            (
                these_cyclone_id_strings, these_target_times_unix_sec, _
            ) = nn_training_simple.choose_random_cyclone_objects(
                all_cyclone_id_strings=cyclone_id_strings_by_file[file_index],
                all_target_times_unix_sec=
                target_times_by_file_unix_sec[file_index],
                num_objects_desired=min([
                    num_examples_per_batch - num_examples_in_memory, 15
                ])
            )

            if len(these_cyclone_id_strings) == 0:
                file_index += 1
                continue

            print(these_cyclone_id_strings)
            these_target_time_strings = [
                time_conversion.unix_sec_to_string(t, '%Y-%m-%d-%H%M')
                for t in these_target_times_unix_sec
            ]
            print(these_target_time_strings)

            target_values_by_sample = [
                _get_target_variables(
                    ebtrk_table_xarray=ebtrk_table_xarray,
                    target_field_names=target_field_names,
                    cyclone_id_string=c,
                    target_time_unix_sec=t
                )
                for c, t in zip(
                    these_cyclone_id_strings, these_target_times_unix_sec
                )
            ]

            print(target_field_names)
            print(target_values_by_sample)

            good_flags = numpy.array(
                [tv is not None for tv in target_values_by_sample], dtype=int
            )
            good_indices = numpy.where(good_flags)[0]
            if len(good_indices) == 0:
                file_index += 1
                continue

            these_cyclone_id_strings = [
                these_cyclone_id_strings[k] for k in good_indices
            ]
            these_target_times_unix_sec = these_target_times_unix_sec[
                good_indices
            ]
            del target_values_by_sample

            data_dict = nn_training_simple._read_satellite_data_1shuffled_file(
                input_file_name=satellite_file_names[file_index],
                lag_times_minutes=lag_times_minutes,
                low_res_wavelengths_microns=low_res_wavelengths_microns,
                num_rows_low_res=num_rows_low_res,
                num_columns_low_res=num_columns_low_res,
                return_coords=False,
                cyclone_id_strings=these_cyclone_id_strings,
                target_times_unix_sec=these_target_times_unix_sec,
                return_xy_coords=False
            )

            del these_cyclone_id_strings
            del these_target_times_unix_sec
            file_index += 1

            if (
                    data_dict is None
                    or data_dict[BRIGHTNESS_TEMPS_KEY] is None
                    or data_dict[BRIGHTNESS_TEMPS_KEY].size == 0
            ):
                continue

            target_values_by_sample = [
                _get_target_variables(
                    ebtrk_table_xarray=ebtrk_table_xarray,
                    target_field_names=target_field_names,
                    cyclone_id_string=c,
                    target_time_unix_sec=t
                )
                for c, t in zip(
                    data_dict[CYCLONE_IDS_KEY], data_dict[TARGET_TIMES_KEY]
                )
            ]

            this_target_matrix = numpy.vstack(target_values_by_sample)

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

            if do_residual_prediction:
                this_resid_baseline_matrix = this_scalar_predictor_matrix[
                    :, len(first_scalar_a_deck_field_names):
                ]
                this_scalar_predictor_matrix = this_scalar_predictor_matrix[
                    :, :len(first_scalar_a_deck_field_names)
                ]

                for f in range(len(target_field_names)):
                    this_resid_baseline_matrix[:, f] *= (
                        TARGET_NAME_TO_CONV_FACTOR[target_field_names[f]]
                    )
            else:
                this_resid_baseline_matrix = None

            this_vector_predictor_matrix = (
                nn_utils.combine_lag_times_and_wavelengths(
                    data_dict[BRIGHTNESS_TEMPS_KEY]
                )
            )

            if vector_predictor_matrix is None:
                these_dim = (
                    (num_examples_per_batch,) +
                    this_vector_predictor_matrix.shape[1:]
                )
                vector_predictor_matrix = numpy.full(these_dim, numpy.nan)

                these_dim = (
                    (num_examples_per_batch,) + this_target_matrix.shape[1:]
                )
                target_matrix = numpy.full(these_dim, numpy.nan)

                if this_scalar_predictor_matrix is not None:
                    these_dim = (
                        (num_examples_per_batch,) +
                        this_scalar_predictor_matrix.shape[1:]
                    )
                    scalar_predictor_matrix = numpy.full(these_dim, numpy.nan)

                if this_resid_baseline_matrix is not None:
                    these_dim = (
                        (num_examples_per_batch,) +
                        this_resid_baseline_matrix.shape[1:]
                    )
                    residual_baseline_matrix = numpy.full(these_dim, numpy.nan)

            first_index = num_examples_in_memory
            last_index = first_index + this_vector_predictor_matrix.shape[0]
            vector_predictor_matrix[first_index:last_index, ...] = (
                this_vector_predictor_matrix
            )
            target_matrix[first_index:last_index, ...] = this_target_matrix

            if this_scalar_predictor_matrix is not None:
                scalar_predictor_matrix[first_index:last_index, ...] = (
                    this_scalar_predictor_matrix
                )

            if this_resid_baseline_matrix is not None:
                residual_baseline_matrix[first_index:last_index, ...] = (
                    this_resid_baseline_matrix
                )

            cyclone_id_strings += data_dict[CYCLONE_IDS_KEY]
            target_times_unix_sec += data_dict[TARGET_TIMES_KEY].tolist()

            num_examples_in_memory += this_vector_predictor_matrix.shape[0]

        # TODO(thunderhoser): This is a HACK.  Should be controlled by an input
        # arg.
        final_axis_length = len(low_res_wavelengths_microns)
        new_dimensions = (
            vector_predictor_matrix.shape[:3] +
            (len(lag_times_minutes), final_axis_length)
        )
        vector_predictor_matrix = numpy.reshape(
            vector_predictor_matrix, new_dimensions
        )

        predictor_matrices = [vector_predictor_matrix]
        if scalar_predictor_matrix is not None:
            predictor_matrices.append(scalar_predictor_matrix)
        if residual_baseline_matrix is not None:
            predictor_matrices.append(residual_baseline_matrix)

        predictor_matrices = [p.astype('float32') for p in predictor_matrices]

        for i in range(len(predictor_matrices)):
            print('Shape of {0:d}th predictor matrix = {1:s}'.format(
                i + 1, str(predictor_matrices[i].shape)
            ))
            print((
                'NaN fraction and min/max in {0:d}th predictor matrix = '
                '{1:.4f}, {2:.4f}, {3:.4f}'
            ).format(
                i + 1,
                numpy.mean(numpy.isnan(predictor_matrices[i])),
                numpy.nanmin(predictor_matrices[i]),
                numpy.nanmax(predictor_matrices[i])
            ))

        if residual_baseline_matrix is not None:
            print((
                'Min/max of every channel in residual baseline matrix:'
                '\n{0:s}\n{1:s}'
            ).format(
                str(numpy.min(residual_baseline_matrix, axis=0)),
                str(numpy.max(residual_baseline_matrix, axis=0))
            ))

        print('Shape of target matrix = {0:s}'.format(
            str(target_matrix.shape)
        ))
        print((
            'NaN fraction and min/max in target matrix = '
            '{0:.4f}, {1:.4f}, {2:.4f}'
        ).format(
            numpy.mean(numpy.isnan(target_matrix)),
            numpy.nanmin(target_matrix),
            numpy.nanmax(target_matrix)
        ))

        for i in range(len(cyclone_id_strings)):
            print((
                '{0:s} at {1:s} ... mean BT in inner 400 x 400 km = {2:s} ... '
                'targets = {3:s}'
            ).format(
                cyclone_id_strings[i],
                time_conversion.unix_sec_to_string(target_times_unix_sec[i], '%Y-%m-%d-%H%M'),
                str(numpy.mean(
                    predictor_matrices[0][i, 300:500, 300:500, -1, :],
                    axis=(0, 1)
                )),
                str(target_matrix[i, :])
            ))

        if return_cyclone_ids:
            yield (
                tuple(predictor_matrices),
                target_matrix,
                cyclone_id_strings,
                numpy.array(target_times_unix_sec, dtype=int)
            )
        else:
            yield tuple(predictor_matrices), target_matrix


def train_model(
        model_object, output_dir_name, num_epochs,
        num_training_batches_per_epoch, training_option_dict,
        num_validation_batches_per_epoch, validation_option_dict,
        loss_function_string, optimizer_function_string,
        plateau_patience_epochs, plateau_learning_rate_multiplier,
        early_stopping_patience_epochs, architecture_dict):
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
    :param architecture_dict: Same.
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

    validation_keys_to_keep = [SATELLITE_DIRECTORY_KEY, YEARS_KEY]
    for this_key in list(training_option_dict.keys()):
        if this_key in validation_keys_to_keep:
            continue

        validation_option_dict[this_key] = training_option_dict[this_key]

    training_option_dict = check_generator_args(training_option_dict)
    validation_option_dict = check_generator_args(validation_option_dict)

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

    training_generator = data_generator_shuffled(training_option_dict)
    validation_generator = data_generator_shuffled(validation_option_dict)

    metafile_name = nn_utils.find_metafile(
        model_dir_name=output_dir_name, raise_error_if_missing=False
    )
    print('Writing metadata to: "{0:s}"...'.format(metafile_name))

    nn_utils.write_metafile(
        pickle_file_name=metafile_name,
        num_epochs=num_epochs,
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
        is_model_bnn=False,
        data_type_string=nn_utils.RG_SIMPLE_DATA_TYPE_STRING,
        train_with_shuffled_data=True
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


def apply_model(
        model_object, predictor_matrices, num_examples_per_batch, verbose=True):
    """Applies trained neural net -- inference time!

    E = number of examples (or number of "TC samples")
    F = number of target fields
    S = ensemble size

    :param model_object: Trained neural net (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param predictor_matrices: See output documentation for `create_data`.
    :param num_examples_per_batch: Batch size.
    :param verbose: Boolean flag.  If True, will print progress messages.
    :return: prediction_matrix: E-by-F-by-S numpy array of predictions.
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
    exec_start_time_unix_sec = time.time()

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

    elapsed_time_sec = time.time() - exec_start_time_unix_sec

    if verbose:
        print((
            'Have applied model to all {0:d} examples!  It took {1:.4f} '
            'seconds.'
        ).format(
            num_examples, elapsed_time_sec
        ))

    return prediction_matrix
