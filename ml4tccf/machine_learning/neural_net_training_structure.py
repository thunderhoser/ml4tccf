"""NN-training for TC-structure parameters with simple Robert/Galina data."""

import random
import numpy
from scipy.interpolate import interp1d
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import error_checking
from ml4tccf.io import a_deck_io
from ml4tccf.io import satellite_io
from ml4tccf.io import extended_best_track_io as ebtrk_io
from ml4tccf.utils import extended_best_track_utils as ebtrk_utils
from ml4tccf.machine_learning import \
    neural_net_training_simple as nn_training_simple
from ml4tccf.machine_learning import neural_net_utils as nn_utils

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

    earlier_indices = numpy.where(numpy.logical_and(
        ebtrk_table_xarray[ebtrk_utils.STORM_ID_KEY].values ==
        cyclone_id_string,
        ebtrk_table_xarray[ebtrk_utils.VALID_TIME_KEY].values ==
        earlier_synoptic_time_unix_sec
    ))[0]
    if len(earlier_indices) == 0:
        return None

    later_indices = numpy.where(numpy.logical_and(
        ebtrk_table_xarray[ebtrk_utils.STORM_ID_KEY].values ==
        cyclone_id_string,
        ebtrk_table_xarray[ebtrk_utils.VALID_TIME_KEY].values ==
        later_synoptic_time_unix_sec
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

    return option_dict


def data_generator_shuffled(option_dict):
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

    :return: predictor_matrices: If predictors include scalars, this will be a
        list with [vector_predictor_matrix, scalar_predictor_matrix].
        Otherwise, this will be a list with only the first item.

        vector_predictor_matrix: E-by-M-by-N-by-W-by-L numpy array of
            brightness temperatures.
        scalar_predictor_matrix: E-by-F numpy array of scalar predictors.

    :return: target_matrix: E-by-T numpy array of target values.
    """

    option_dict = check_generator_args(option_dict)

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
    scalar_a_deck_field_names = option_dict[SCALAR_A_DECK_FIELDS_KEY]
    remove_nontropical_systems = option_dict[REMOVE_NONTROPICAL_KEY]
    remove_tropical_systems = option_dict[REMOVE_TROPICAL_KEY]
    a_deck_file_name = option_dict[A_DECK_FILE_KEY]
    target_field_names = option_dict[TARGET_FIELDS_KEY]
    target_file_name = option_dict[TARGET_FILE_KEY]

    assert not a_deck_io.UNNORM_EXTRAP_LATITUDE_KEY in scalar_a_deck_field_names
    assert not a_deck_io.UNNORM_EXTRAP_LONGITUDE_KEY in scalar_a_deck_field_names

    satellite_file_names = satellite_io.find_shuffled_files(
        directory_name=satellite_dir_name, raise_error_if_all_missing=True
    )
    random.shuffle(satellite_file_names)

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
        target_matrix = None
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
                num_objects_desired=
                num_examples_per_batch - num_examples_in_memory
            )

            if len(these_cyclone_id_strings) == 0:
                file_index += 1
                continue

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

            good_indices = numpy.array(
                [tv is not None for tv in target_values_by_sample], dtype=int
            )
            if len(good_indices) == 0:
                file_index += 1
                continue

            these_cyclone_id_strings = [
                these_cyclone_id_strings[k] for k in good_indices
            ]
            these_target_times_unix_sec = these_target_times_unix_sec[
                good_indices
            ]
            target_values_by_sample = [
                target_values_by_sample[k] for k in good_indices
            ]
            this_target_matrix = numpy.vstack(target_values_by_sample)

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
            file_index += 1

            if (
                    data_dict is None
                    or data_dict[BRIGHTNESS_TEMPS_KEY] is None
                    or data_dict[BRIGHTNESS_TEMPS_KEY].size == 0
            ):
                continue

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

        yield predictor_matrices, target_matrix