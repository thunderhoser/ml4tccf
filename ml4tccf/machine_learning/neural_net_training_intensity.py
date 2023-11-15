"""NN-training with simplified Robert/Galina data to estimate intensity."""

import random
import warnings
import numpy
import keras
import netCDF4
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4tccf.io import a_deck_io
from ml4tccf.io import satellite_io
from ml4tccf.io import extended_best_track_io as ebtrk_io
from ml4tccf.utils import misc_utils
from ml4tccf.utils import extended_best_track_utils as ebtrk_utils
from ml4tccf.machine_learning import neural_net_utils as nn_utils
from ml4tccf.machine_learning import \
    neural_net_training_fancy as nn_training_fancy
from ml4tccf.machine_learning import \
    neural_net_training_simple as nn_training_simple
from ml4tccf.machine_learning import data_augmentation

MODEL_FILE_KEY = 'model_file_name'
EXAMPLE_DIM_KEY = 'example'
CYCLONE_ID_CHAR_DIM_KEY = 'cyclone_id_char'
PREDICTED_INTENSITY_KEY = 'predicted_intensity_m_s01'
TARGET_INTENSITY_KEY = 'target_intensity_m_s01'
TARGET_TIME_KEY = 'target_time_unix_sec'
CYCLONE_ID_KEY = 'cyclone_id_string'

TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H%M'

MINUTES_TO_SECONDS = 60
HOURS_TO_SECONDS = 3600
METRES_PER_SECOND_TO_KT = 3.6 / 1.852

A_DECK_FIELD_NAMES_FOR_CENTER_FIXING_ONLY = [
    a_deck_io.UNNORM_EXTRAP_LATITUDE_KEY,
    a_deck_io.UNNORM_EXTRAP_LONGITUDE_KEY,
    a_deck_io.INTENSITY_KEY,
    a_deck_io.SEA_LEVEL_PRESSURE_KEY
]

BRIGHTNESS_TEMPS_KEY = nn_utils.BRIGHTNESS_TEMPS_KEY
TARGET_TIMES_KEY = nn_utils.TARGET_TIMES_KEY
CYCLONE_IDS_KEY = 'cyclone_id_strings'
LOW_RES_LATITUDES_KEY = nn_utils.LOW_RES_LATITUDES_KEY
LOW_RES_LONGITUDES_KEY = nn_utils.LOW_RES_LONGITUDES_KEY

PREDICTOR_MATRICES_KEY = nn_utils.PREDICTOR_MATRICES_KEY
TARGET_INTENSITIES_KEY = 'target_intensities_m_s01'
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
DATA_AUG_NUM_TRANS_KEY = nn_utils.DATA_AUG_NUM_TRANS_KEY
DATA_AUG_MEAN_TRANS_KEY = nn_utils.DATA_AUG_MEAN_TRANS_KEY
DATA_AUG_STDEV_TRANS_KEY = nn_utils.DATA_AUG_STDEV_TRANS_KEY
SYNOPTIC_TIMES_ONLY_KEY = nn_utils.SYNOPTIC_TIMES_ONLY_KEY
A_DECK_FILE_KEY = nn_utils.A_DECK_FILE_KEY
SCALAR_A_DECK_FIELDS_KEY = nn_utils.SCALAR_A_DECK_FIELDS_KEY
REMOVE_NONTROPICAL_KEY = nn_utils.REMOVE_NONTROPICAL_KEY
SEMANTIC_SEG_FLAG_KEY = nn_utils.SEMANTIC_SEG_FLAG_KEY
TARGET_SMOOOTHER_STDEV_KEY = nn_utils.TARGET_SMOOOTHER_STDEV_KEY


def data_generator_shuffled(option_dict, ebtrk_file_name,
                            center_fixing_model_object):
    """Generates input data for neural net from shuffled files.

    :param option_dict: See doc for `data_generator`.
    :param ebtrk_file_name: Path to EBTRK (extended best-track) file.  Will be
        read by `extended_best_track_io.read_file`.
    :param center_fixing_model_object: Trained NN for center-fixing.
    :return: predictor_matrices: BLAH.
    :return: target_intensities_m_s01: BLAH.
    """

    # TODO(thunderhoser): Fix output documentation.

    option_dict[nn_utils.HIGH_RES_WAVELENGTHS_KEY] = numpy.array([])
    option_dict[nn_utils.LAG_TIME_TOLERANCE_KEY] = 0
    option_dict[nn_utils.MAX_MISSING_LAG_TIMES_KEY] = 0
    option_dict[nn_utils.MAX_INTERP_GAP_KEY] = 0
    option_dict[nn_utils.SENTINEL_VALUE_KEY] = -10.
    option_dict[nn_utils.SEMANTIC_SEG_FLAG_KEY] = False
    option_dict[nn_utils.TARGET_SMOOOTHER_STDEV_KEY] = None
    option_dict[MAX_EXAMPLES_PER_CYCLONE_KEY] = option_dict[BATCH_SIZE_KEY]
    option_dict[SYNOPTIC_TIMES_ONLY_KEY] = True

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
    ) = nn_training_simple.get_times_and_scalar_preds_shuffled(
        satellite_file_names=satellite_file_names,
        a_deck_file_name=a_deck_file_name,
        scalar_a_deck_field_names=scalar_a_deck_field_names,
        remove_nontropical_systems=remove_nontropical_systems,
        remove_tropical_systems=False,
        desired_years=years,
        predictor_lag_times_sec=MINUTES_TO_SECONDS * lag_times_minutes
    )

    # TODO(thunderhoser): Maybe modularize all the EBTRK matching shit below?
    print('Reading data from: "{0:s}"...'.format(ebtrk_file_name))
    ebtrk_table_xarray = ebtrk_io.read_file(ebtrk_file_name)
    ebtrk_cyclone_id_strings = (
        ebtrk_table_xarray[ebtrk_utils.STORM_ID_KEY].values
    )
    ebtrk_valid_times_unix_sec = (
        HOURS_TO_SECONDS * ebtrk_table_xarray[ebtrk_utils.VALID_TIME_KEY].values
    )
    ebtrk_intensities_m_s01 = (
        ebtrk_table_xarray[ebtrk_utils.MAX_SUSTAINED_WIND_KEY].values
    )

    num_files = len(cyclone_id_strings_by_file)
    intensities_by_file_m_s01 = [numpy.array([], dtype=float)] * num_files

    for i in range(num_files):
        good_indices_this_file = []

        for j in range(len(cyclone_id_strings_by_file[i])):
            good_indices_this_example = numpy.where(numpy.logical_and(
                ebtrk_cyclone_id_strings == cyclone_id_strings_by_file[i][j],
                ebtrk_valid_times_unix_sec ==
                target_times_by_file_unix_sec[i][j]
            ))[0]

            if len(good_indices_this_example) > 0:
                good_indices_this_file.append(j)

                k = good_indices_this_example[0]
                intensities_by_file_m_s01[i] = numpy.concatenate((
                    intensities_by_file_m_s01[i],
                    numpy.array([ebtrk_intensities_m_s01[k]])
                ))

                continue

            warning_string = (
                'POTENTIAL ERROR: cannot find cyclone {0:s} at time {1:s} in '
                'extended best-track data.'
            ).format(
                cyclone_id_strings_by_file[i][j],
                time_conversion.unix_sec_to_string(
                    target_times_by_file_unix_sec[i][j],
                    TIME_FORMAT_FOR_LOG_MESSAGES
                )
            )

            warnings.warn(warning_string)

        good_indices_this_file = numpy.array(good_indices_this_file, dtype=int)
        cyclone_id_strings_by_file[i] = [
            cyclone_id_strings_by_file[i][k] for k in good_indices_this_file
        ]
        target_times_by_file_unix_sec[i] = (
            target_times_by_file_unix_sec[i][good_indices_this_file]
        )
        if scalar_predictor_matrix_by_file[i] is not None:
            scalar_predictor_matrix_by_file[i] = (
                scalar_predictor_matrix_by_file[i][good_indices_this_file, ...]
            )

        for j in range(len(cyclone_id_strings_by_file[i])):
            print('{0:s} at {1:s} ... intensity = {2:.0f} kt'.format(
                cyclone_id_strings_by_file[i][j],
                time_conversion.unix_sec_to_string(
                    target_times_by_file_unix_sec[i][j],
                    TIME_FORMAT_FOR_LOG_MESSAGES
                ),
                METRES_PER_SECOND_TO_KT * intensities_by_file_m_s01[i][j]
            ))

    use_extrap_based_forecasts = (
        scalar_a_deck_field_names is not None
        and a_deck_io.UNNORM_EXTRAP_LATITUDE_KEY in scalar_a_deck_field_names
        and a_deck_io.UNNORM_EXTRAP_LONGITUDE_KEY in scalar_a_deck_field_names
    )

    file_index = 0

    while True:
        brightness_temp_matrix_kelvins = None
        scalar_predictor_matrix = None
        target_intensities_m_s01 = None
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

            data_dict = nn_training_simple._read_satellite_data_1shuffled_file(
                input_file_name=satellite_file_names[file_index],
                lag_times_minutes=lag_times_minutes,
                low_res_wavelengths_microns=low_res_wavelengths_microns,
                num_rows_low_res=num_rows_low_res,
                num_columns_low_res=num_columns_low_res,
                return_coords=use_extrap_based_forecasts,
                return_xy_coords=False,
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
            prev_idx = file_index - 1

            if a_deck_file_name is None:
                this_scalar_predictor_matrix = None
            else:
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
                        nn_training_simple._extrap_based_forecasts_to_rowcol(
                            scalar_predictor_matrix=
                            this_scalar_predictor_matrix,
                            scalar_a_deck_field_names=scalar_a_deck_field_names,
                            satellite_data_dict=data_dict
                        )
                    )

            row_indices = numpy.array([
                numpy.where(numpy.logical_and(
                    numpy.array(cyclone_id_strings_by_file[prev_idx]) == c,
                    target_times_by_file_unix_sec[prev_idx] == t
                ))[0][0]
                for c, t in zip(
                    data_dict[CYCLONE_IDS_KEY], data_dict[TARGET_TIMES_KEY]
                )
            ], dtype=int)

            these_target_intensities_m_s01 = (
                intensities_by_file_m_s01[prev_idx][row_indices]
            )

            this_bt_matrix_kelvins = nn_utils.combine_lag_times_and_wavelengths(
                this_bt_matrix_kelvins
            )

            if brightness_temp_matrix_kelvins is None:
                these_dim = (
                    (num_examples_per_batch,) + this_bt_matrix_kelvins.shape[1:]
                )
                brightness_temp_matrix_kelvins = numpy.full(
                    these_dim, numpy.nan
                )
                target_intensities_m_s01 = numpy.full(
                    num_examples_per_batch, numpy.nan
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
            target_intensities_m_s01[first_index:last_index] = (
                these_target_intensities_m_s01
            )

            if this_scalar_predictor_matrix is not None:
                scalar_predictor_matrix[first_index:last_index, ...] = (
                    this_scalar_predictor_matrix
                )

            num_examples_in_memory += this_bt_matrix_kelvins.shape[0]

        (
            _, brightness_temp_matrix_kelvins,
            row_translations_low_res_px, column_translations_low_res_px
        ) = data_augmentation.augment_data(
            bidirectional_reflectance_matrix=None,
            brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
            num_translations_per_example=data_aug_num_translations,
            mean_translation_low_res_px=data_aug_mean_translation_low_res_px,
            stdev_translation_low_res_px=data_aug_stdev_translation_low_res_px,
            sentinel_value=-10.
        )

        brightness_temp_matrix_kelvins = (
            data_augmentation.subset_grid_after_data_aug(
                data_matrix=brightness_temp_matrix_kelvins,
                num_rows_to_keep=orig_num_rows_low_res,
                num_columns_to_keep=orig_num_columns_low_res,
                for_high_res=False
            )
        )
        target_intensities_m_s01 = numpy.repeat(
            target_intensities_m_s01, repeats=data_aug_num_translations
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

        predictor_matrices = [brightness_temp_matrix_kelvins]
        if scalar_predictor_matrix is not None:
            predictor_matrices.append(scalar_predictor_matrix)

        # TODO(thunderhoser): Maybe I should modularize shit with the CF-NN?
        if center_fixing_model_object is not None:
            nn_translation_matrix = nn_utils.apply_model(
                model_object=center_fixing_model_object,
                predictor_matrices=predictor_matrices,
                num_examples_per_batch=16,
                verbose=True
            )
            nn_translation_matrix = numpy.mean(nn_translation_matrix, axis=-1)
            row_translations_low_res_px = numpy.round(
                nn_translation_matrix[:, 0]
            ).astype(int)
            column_translations_low_res_px = numpy.round(
                nn_translation_matrix[:, 1]
            ).astype(int)

            (
                _, brightness_temp_matrix_kelvins
            ) = data_augmentation.augment_data_specific_trans(
                bidirectional_reflectance_matrix=None,
                brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
                row_translations_low_res_px=row_translations_low_res_px,
                column_translations_low_res_px=column_translations_low_res_px,
                sentinel_value=-10.
            )

        if scalar_predictor_matrix is not None:
            good_scalar_predictor_indices = []

            for j in range(len(scalar_a_deck_field_names)):
                if (
                        scalar_a_deck_field_names[j] in
                        A_DECK_FIELD_NAMES_FOR_CENTER_FIXING_ONLY
                ):
                    continue

                good_scalar_predictor_indices.append(j)

            good_scalar_predictor_indices = numpy.array(
                good_scalar_predictor_indices, dtype=int
            )
            scalar_predictor_matrix = (
                scalar_predictor_matrix[:, good_scalar_predictor_indices]
            )

        predictor_matrices = [brightness_temp_matrix_kelvins]
        if scalar_predictor_matrix is not None:
            predictor_matrices.append(scalar_predictor_matrix)

        predictor_matrices = [p.astype('float16') for p in predictor_matrices]
        yield predictor_matrices, target_intensities_m_s01


def data_generator_shuffled_no_trans(option_dict, ebtrk_file_name):
    """Generates input data for neural net from shuffled files.

    This generator uses untranslated images, where the image center is the best-
    track TC center.

    :param option_dict: See doc for `data_generator`.
    :param ebtrk_file_name: Path to EBTRK (extended best-track) file.  Will be
        read by `extended_best_track_io.read_file`.
    :return: predictor_matrices: BLAH.
    :return: target_intensities_m_s01: BLAH.
    """

    # TODO(thunderhoser): Fix output documentation.

    option_dict[nn_utils.HIGH_RES_WAVELENGTHS_KEY] = numpy.array([])
    option_dict[nn_utils.LAG_TIME_TOLERANCE_KEY] = 0
    option_dict[nn_utils.MAX_MISSING_LAG_TIMES_KEY] = 0
    option_dict[nn_utils.MAX_INTERP_GAP_KEY] = 0
    option_dict[nn_utils.SENTINEL_VALUE_KEY] = -10.
    option_dict[nn_utils.SEMANTIC_SEG_FLAG_KEY] = False
    option_dict[nn_utils.TARGET_SMOOOTHER_STDEV_KEY] = None
    option_dict[MAX_EXAMPLES_PER_CYCLONE_KEY] = option_dict[BATCH_SIZE_KEY]
    option_dict[SYNOPTIC_TIMES_ONLY_KEY] = True
    option_dict[DATA_AUG_NUM_TRANS_KEY] = 8
    option_dict[DATA_AUG_MEAN_TRANS_KEY] = 12.
    option_dict[DATA_AUG_STDEV_TRANS_KEY] = 6.

    option_dict = nn_utils.check_generator_args(option_dict)

    satellite_dir_name = option_dict[SATELLITE_DIRECTORY_KEY]
    years = option_dict[YEARS_KEY]
    lag_times_minutes = option_dict[LAG_TIMES_KEY]
    low_res_wavelengths_microns = option_dict[LOW_RES_WAVELENGTHS_KEY]
    num_examples_per_batch = option_dict[BATCH_SIZE_KEY]
    num_rows_low_res = option_dict[NUM_GRID_ROWS_KEY]
    num_columns_low_res = option_dict[NUM_GRID_COLUMNS_KEY]
    a_deck_file_name = option_dict[A_DECK_FILE_KEY]
    scalar_a_deck_field_names = option_dict[SCALAR_A_DECK_FIELDS_KEY]
    remove_nontropical_systems = option_dict[REMOVE_NONTROPICAL_KEY]

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
        remove_tropical_systems=False,
        desired_years=years,
        predictor_lag_times_sec=MINUTES_TO_SECONDS * lag_times_minutes
    )

    # TODO(thunderhoser): Maybe modularize all the EBTRK matching shit below?
    print('Reading data from: "{0:s}"...'.format(ebtrk_file_name))
    ebtrk_table_xarray = ebtrk_io.read_file(ebtrk_file_name)
    ebtrk_cyclone_id_strings = (
        ebtrk_table_xarray[ebtrk_utils.STORM_ID_KEY].values
    )
    ebtrk_valid_times_unix_sec = (
        HOURS_TO_SECONDS * ebtrk_table_xarray[ebtrk_utils.VALID_TIME_KEY].values
    )
    ebtrk_intensities_m_s01 = (
        ebtrk_table_xarray[ebtrk_utils.MAX_SUSTAINED_WIND_KEY].values
    )

    num_files = len(cyclone_id_strings_by_file)
    intensities_by_file_m_s01 = [numpy.array([], dtype=float)] * num_files

    for i in range(num_files):
        good_indices_this_file = []

        for j in range(len(cyclone_id_strings_by_file[i])):
            good_indices_this_example = numpy.where(numpy.logical_and(
                ebtrk_cyclone_id_strings == cyclone_id_strings_by_file[i][j],
                ebtrk_valid_times_unix_sec ==
                target_times_by_file_unix_sec[i][j]
            ))[0]

            if len(good_indices_this_example) > 0:
                good_indices_this_file.append(j)

                k = good_indices_this_example[0]
                intensities_by_file_m_s01[i] = numpy.concatenate((
                    intensities_by_file_m_s01[i],
                    numpy.array([ebtrk_intensities_m_s01[k]])
                ))

                continue

            warning_string = (
                'POTENTIAL ERROR: cannot find cyclone {0:s} at time {1:s} in '
                'extended best-track data.'
            ).format(
                cyclone_id_strings_by_file[i][j],
                time_conversion.unix_sec_to_string(
                    target_times_by_file_unix_sec[i][j],
                    TIME_FORMAT_FOR_LOG_MESSAGES
                )
            )

            warnings.warn(warning_string)

        good_indices_this_file = numpy.array(good_indices_this_file, dtype=int)
        cyclone_id_strings_by_file[i] = [
            cyclone_id_strings_by_file[i][k] for k in good_indices_this_file
        ]
        target_times_by_file_unix_sec[i] = (
            target_times_by_file_unix_sec[i][good_indices_this_file]
        )
        if scalar_predictor_matrix_by_file[i] is not None:
            scalar_predictor_matrix_by_file[i] = (
                scalar_predictor_matrix_by_file[i][good_indices_this_file, ...]
            )

        for j in range(len(cyclone_id_strings_by_file[i])):
            print('{0:s} at {1:s} ... intensity = {2:.0f} kt'.format(
                cyclone_id_strings_by_file[i][j],
                time_conversion.unix_sec_to_string(
                    target_times_by_file_unix_sec[i][j],
                    TIME_FORMAT_FOR_LOG_MESSAGES
                ),
                METRES_PER_SECOND_TO_KT * intensities_by_file_m_s01[i][j]
            ))

    file_index = 0

    while True:
        brightness_temp_matrix_kelvins = None
        scalar_predictor_matrix = None
        target_intensities_m_s01 = None
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

            data_dict = nn_training_simple._read_satellite_data_1shuffled_file(
                input_file_name=satellite_file_names[file_index],
                lag_times_minutes=lag_times_minutes,
                low_res_wavelengths_microns=low_res_wavelengths_microns,
                num_rows_low_res=num_rows_low_res,
                num_columns_low_res=num_columns_low_res,
                return_coords=False,
                return_xy_coords=False,
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
            prev_idx = file_index - 1

            if a_deck_file_name is None:
                this_scalar_predictor_matrix = None
            else:
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

            row_indices = numpy.array([
                numpy.where(numpy.logical_and(
                    numpy.array(cyclone_id_strings_by_file[prev_idx]) == c,
                    target_times_by_file_unix_sec[prev_idx] == t
                ))[0][0]
                for c, t in zip(
                    data_dict[CYCLONE_IDS_KEY], data_dict[TARGET_TIMES_KEY]
                )
            ], dtype=int)

            these_target_intensities_m_s01 = (
                intensities_by_file_m_s01[prev_idx][row_indices]
            )

            this_bt_matrix_kelvins = nn_utils.combine_lag_times_and_wavelengths(
                this_bt_matrix_kelvins
            )

            if brightness_temp_matrix_kelvins is None:
                these_dim = (
                    (num_examples_per_batch,) + this_bt_matrix_kelvins.shape[1:]
                )
                brightness_temp_matrix_kelvins = numpy.full(
                    these_dim, numpy.nan
                )
                target_intensities_m_s01 = numpy.full(
                    num_examples_per_batch, numpy.nan
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
            target_intensities_m_s01[first_index:last_index] = (
                these_target_intensities_m_s01
            )

            if this_scalar_predictor_matrix is not None:
                scalar_predictor_matrix[first_index:last_index, ...] = (
                    this_scalar_predictor_matrix
                )

            num_examples_in_memory += this_bt_matrix_kelvins.shape[0]

        if scalar_predictor_matrix is not None:
            good_scalar_predictor_indices = []

            for j in range(len(scalar_a_deck_field_names)):
                if (
                        scalar_a_deck_field_names[j] in
                        A_DECK_FIELD_NAMES_FOR_CENTER_FIXING_ONLY
                ):
                    continue

                good_scalar_predictor_indices.append(j)

            good_scalar_predictor_indices = numpy.array(
                good_scalar_predictor_indices, dtype=int
            )
            scalar_predictor_matrix = (
                scalar_predictor_matrix[:, good_scalar_predictor_indices]
            )

        predictor_matrices = [brightness_temp_matrix_kelvins]
        if scalar_predictor_matrix is not None:
            predictor_matrices.append(scalar_predictor_matrix)

        predictor_matrices = [p.astype('float16') for p in predictor_matrices]
        yield predictor_matrices, target_intensities_m_s01


def train_model(
        model_object, center_fixing_model_object,
        ebtrk_file_name_for_training, ebtrk_file_name_for_validation,
        output_dir_name, num_epochs,
        num_training_batches_per_epoch, training_option_dict,
        num_validation_batches_per_epoch, validation_option_dict,
        loss_function_string, optimizer_function_string,
        plateau_patience_epochs, plateau_learning_rate_multiplier,
        early_stopping_patience_epochs, architecture_dict, is_model_bnn,
        use_shuffled_data):
    """Trains neural net.

    :param model_object: See doc for `neural_net_training_fancy.train_model`.
    :param center_fixing_model_object: See doc for `data_generator_shuffled`.
    :param ebtrk_file_name_for_training: Same.
    :param ebtrk_file_name_for_validation: Same.
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
    :param is_model_bnn: Same.
    :param use_shuffled_data: Boolean flag.  If True, will read training data
        from shuffled input files, each containing multiple cyclones.  If False,
        will read training data from input files with one cyclone-day each.
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
    error_checking.assert_is_boolean(use_shuffled_data)

    validation_keys_to_keep = [SATELLITE_DIRECTORY_KEY, YEARS_KEY]
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

    if use_shuffled_data:
        training_generator = data_generator_shuffled(
            option_dict=training_option_dict,
            ebtrk_file_name=ebtrk_file_name_for_training,
            center_fixing_model_object=center_fixing_model_object
        )
        validation_generator = data_generator_shuffled(
            option_dict=validation_option_dict,
            ebtrk_file_name=ebtrk_file_name_for_validation,
            center_fixing_model_object=center_fixing_model_object
        )
    else:
        return

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
        data_type_string=nn_utils.RG_SIMPLE_DATA_TYPE_STRING,
        train_with_shuffled_data=use_shuffled_data
    )

    model_object.fit_generator(
        generator=training_generator,
        steps_per_epoch=num_training_batches_per_epoch,
        epochs=num_epochs, verbose=1, callbacks=list_of_callback_objects,
        validation_data=validation_generator,
        validation_steps=num_validation_batches_per_epoch
    )


def train_model_no_trans(
        model_object, ebtrk_file_name_for_training,
        ebtrk_file_name_for_validation, output_dir_name, num_epochs,
        num_training_batches_per_epoch, training_option_dict,
        num_validation_batches_per_epoch, validation_option_dict,
        loss_function_string, optimizer_function_string,
        plateau_patience_epochs, plateau_learning_rate_multiplier,
        early_stopping_patience_epochs, architecture_dict, is_model_bnn,
        use_shuffled_data):
    """Trains neural net with untranslated images (no data augmentation).

    :param model_object: See doc for `neural_net_training_fancy.train_model`.
    :param ebtrk_file_name_for_training: See doc for
        `data_generator_shuffled_no_trans`.
    :param ebtrk_file_name_for_validation: Same.
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
    :param is_model_bnn: Same.
    :param use_shuffled_data: Boolean flag.  If True, will read training data
        from shuffled input files, each containing multiple cyclones.  If False,
        will read training data from input files with one cyclone-day each.
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
    error_checking.assert_is_boolean(use_shuffled_data)

    validation_keys_to_keep = [SATELLITE_DIRECTORY_KEY, YEARS_KEY]
    for this_key in list(training_option_dict.keys()):
        if this_key in validation_keys_to_keep:
            continue

        validation_option_dict[this_key] = training_option_dict[this_key]

    training_option_dict[DATA_AUG_NUM_TRANS_KEY] = 8
    training_option_dict[DATA_AUG_MEAN_TRANS_KEY] = 12.
    training_option_dict[DATA_AUG_STDEV_TRANS_KEY] = 6.
    validation_option_dict[DATA_AUG_NUM_TRANS_KEY] = 8
    validation_option_dict[DATA_AUG_MEAN_TRANS_KEY] = 12.
    validation_option_dict[DATA_AUG_STDEV_TRANS_KEY] = 6.

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

    if use_shuffled_data:
        training_generator = data_generator_shuffled_no_trans(
            option_dict=training_option_dict,
            ebtrk_file_name=ebtrk_file_name_for_training
        )
        validation_generator = data_generator_shuffled_no_trans(
            option_dict=validation_option_dict,
            ebtrk_file_name=ebtrk_file_name_for_validation
        )
    else:
        return

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
        data_type_string=nn_utils.RG_SIMPLE_DATA_TYPE_STRING,
        train_with_shuffled_data=use_shuffled_data
    )

    model_object.fit_generator(
        generator=training_generator,
        steps_per_epoch=num_training_batches_per_epoch,
        epochs=num_epochs, verbose=1, callbacks=list_of_callback_objects,
        validation_data=validation_generator,
        validation_steps=num_validation_batches_per_epoch
    )


def create_data(option_dict, ebtrk_file_name, center_fixing_model_object,
                cyclone_id_string, num_target_times):
    """Creates, rather than generates, neural-net inputs.

    E = batch size = number of examples after data augmentation
    L = number of lag times for predictors
    w = number of wavelengths
    m = number of rows in grid
    n = number of columns in grid

    :param option_dict: See doc for `data_generator`.
    :param ebtrk_file_name: Same.
    :param center_fixing_model_object: Same.
    :param cyclone_id_string: Will create data for this cyclone.
    :param num_target_times: Will create data for this number of target times
        for the given cyclone.

    :return: data_dict: Dictionary with the following keys.
    data_dict["predictor_matrices"]: See doc for `data_generator`.
    data_dict["target_intensities_m_s01"]: Same.
    data_dict["target_times_unix_sec"]: length-E numpy array of target times.
    data_dict["high_res_latitude_matrix_deg_n"]: None.
    data_dict["high_res_longitude_matrix_deg_e"]: None.
    data_dict["low_res_latitude_matrix_deg_n"]: E-by-m-by-L numpy array of
        latitudes (deg north).
    data_dict["low_res_longitude_matrix_deg_e"]: E-by-n-by-L numpy array of
        longitudes (deg east).
    """

    error_checking.assert_is_integer(num_target_times)
    error_checking.assert_is_greater(num_target_times, 0)

    option_dict[nn_utils.HIGH_RES_WAVELENGTHS_KEY] = numpy.array([])
    option_dict[nn_utils.LAG_TIME_TOLERANCE_KEY] = 0
    option_dict[nn_utils.MAX_MISSING_LAG_TIMES_KEY] = 0
    option_dict[nn_utils.MAX_INTERP_GAP_KEY] = 0
    option_dict[nn_utils.SENTINEL_VALUE_KEY] = -10.
    option_dict[nn_utils.SYNOPTIC_TIMES_ONLY_KEY] = True

    option_dict = nn_utils.check_generator_args(option_dict)

    satellite_dir_name = option_dict[SATELLITE_DIRECTORY_KEY]
    lag_times_minutes = option_dict[LAG_TIMES_KEY]
    low_res_wavelengths_microns = option_dict[LOW_RES_WAVELENGTHS_KEY]
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
    ) = nn_training_fancy.get_target_times_and_scalar_predictors(
        cyclone_id_strings=[cyclone_id_string],
        synoptic_times_only=True,
        satellite_file_names_by_cyclone=[satellite_file_names],
        a_deck_file_name=a_deck_file_name,
        scalar_a_deck_field_names=scalar_a_deck_field_names,
        remove_nontropical_systems=remove_nontropical_systems,
        remove_tropical_systems=False,
        predictor_lag_times_sec=MINUTES_TO_SECONDS * lag_times_minutes
    )

    all_target_times_unix_sec = all_target_times_unix_sec[0]
    all_scalar_predictor_matrix = all_scalar_predictor_matrix[0]

    # TODO(thunderhoser): Maybe modularize all the EBTRK matching shit below?
    print('Reading data from: "{0:s}"...'.format(ebtrk_file_name))
    ebtrk_table_xarray = ebtrk_io.read_file(ebtrk_file_name)
    ebtrk_cyclone_id_strings = (
        ebtrk_table_xarray[ebtrk_utils.STORM_ID_KEY].values
    )
    ebtrk_valid_times_unix_sec = (
        HOURS_TO_SECONDS * ebtrk_table_xarray[ebtrk_utils.VALID_TIME_KEY].values
    )
    ebtrk_intensities_m_s01 = (
        ebtrk_table_xarray[ebtrk_utils.MAX_SUSTAINED_WIND_KEY].values
    )

    all_target_intensities_m_s01 = []
    good_indices = []

    for i in range(len(all_target_times_unix_sec)):
        good_indices_this_example = numpy.where(numpy.logical_and(
            ebtrk_cyclone_id_strings == cyclone_id_string,
            ebtrk_valid_times_unix_sec == all_target_times_unix_sec[i]
        ))[0]

        if len(good_indices_this_example) > 0:
            good_indices.append(i)

            k = good_indices_this_example[0]
            all_target_intensities_m_s01.append(ebtrk_intensities_m_s01[k])

            continue

        warning_string = (
            'POTENTIAL ERROR: cannot find cyclone {0:s} at time {1:s} in '
            'extended best-track data.'
        ).format(
            cyclone_id_string,
            time_conversion.unix_sec_to_string(
                all_target_times_unix_sec[i], TIME_FORMAT_FOR_LOG_MESSAGES
            )
        )

        warnings.warn(warning_string)

    all_target_intensities_m_s01 = numpy.array(
        all_target_intensities_m_s01, dtype=int
    )
    good_indices = numpy.array(good_indices, dtype=int)
    all_target_times_unix_sec = all_target_times_unix_sec[good_indices]
    all_scalar_predictor_matrix = all_scalar_predictor_matrix[good_indices, :]

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
        return_coords=True, return_xy_coords=False
    )

    if (
            data_dict is None
            or data_dict[BRIGHTNESS_TEMPS_KEY] is None
            or data_dict[BRIGHTNESS_TEMPS_KEY].size == 0
    ):
        return None

    use_extrap_based_forecasts = (
        scalar_a_deck_field_names is not None
        and a_deck_io.UNNORM_EXTRAP_LATITUDE_KEY in scalar_a_deck_field_names
        and a_deck_io.UNNORM_EXTRAP_LONGITUDE_KEY in scalar_a_deck_field_names
    )

    row_indices = numpy.array([
        numpy.where(all_target_times_unix_sec == t)[0][0]
        for t in data_dict[TARGET_TIMES_KEY]
    ], dtype=int)

    target_intensities_m_s01 = all_target_intensities_m_s01[row_indices]
    target_intensities_m_s01 = target_intensities_m_s01[:num_target_times]

    if a_deck_file_name is None:
        scalar_predictor_matrix = None
    else:
        scalar_predictor_matrix = all_scalar_predictor_matrix[row_indices, :]

        if use_extrap_based_forecasts:
            scalar_predictor_matrix = (
                nn_training_simple._extrap_based_forecasts_to_rowcol(
                    scalar_predictor_matrix=scalar_predictor_matrix,
                    scalar_a_deck_field_names=scalar_a_deck_field_names,
                    satellite_data_dict=data_dict
                )
            )

        scalar_predictor_matrix = scalar_predictor_matrix[:num_target_times, :]

    brightness_temp_matrix_kelvins = (
        data_dict[BRIGHTNESS_TEMPS_KEY][:num_target_times, ...]
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

    brightness_temp_matrix_kelvins = nn_utils.combine_lag_times_and_wavelengths(
        brightness_temp_matrix_kelvins
    )

    (
        _, brightness_temp_matrix_kelvins,
        row_translations_low_res_px, column_translations_low_res_px
    ) = data_augmentation.augment_data(
        bidirectional_reflectance_matrix=None,
        brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
        num_translations_per_example=data_aug_num_translations,
        mean_translation_low_res_px=data_aug_mean_translation_low_res_px,
        stdev_translation_low_res_px=data_aug_stdev_translation_low_res_px,
        sentinel_value=-10.
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

    target_intensities_m_s01 = numpy.repeat(
        target_intensities_m_s01, repeats=data_aug_num_translations
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

    predictor_matrices = [brightness_temp_matrix_kelvins]
    if scalar_predictor_matrix is not None:
        predictor_matrices.append(scalar_predictor_matrix)

    # TODO(thunderhoser): Maybe I should modularize shit with the CF-NN?
    if center_fixing_model_object is not None:
        nn_translation_matrix = nn_utils.apply_model(
            model_object=center_fixing_model_object,
            predictor_matrices=predictor_matrices,
            num_examples_per_batch=16,
            verbose=True
        )
        nn_translation_matrix = numpy.mean(nn_translation_matrix, axis=-1)
        row_translations_low_res_px = numpy.round(
            nn_translation_matrix[:, 0]
        ).astype(int)
        column_translations_low_res_px = numpy.round(
            nn_translation_matrix[:, 1]
        ).astype(int)

        (
            _, brightness_temp_matrix_kelvins
        ) = data_augmentation.augment_data_specific_trans(
            bidirectional_reflectance_matrix=None,
            brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
            row_translations_low_res_px=row_translations_low_res_px,
            column_translations_low_res_px=column_translations_low_res_px,
            sentinel_value=-10.
        )

    if scalar_predictor_matrix is not None:
        good_scalar_predictor_indices = []

        for j in range(len(scalar_a_deck_field_names)):
            if (
                    scalar_a_deck_field_names[j] in
                    A_DECK_FIELD_NAMES_FOR_CENTER_FIXING_ONLY
            ):
                continue

            good_scalar_predictor_indices.append(j)

        good_scalar_predictor_indices = numpy.array(
            good_scalar_predictor_indices, dtype=int
        )
        scalar_predictor_matrix = (
            scalar_predictor_matrix[:, good_scalar_predictor_indices]
        )

    predictor_matrices = [brightness_temp_matrix_kelvins]
    if scalar_predictor_matrix is not None:
        predictor_matrices.append(scalar_predictor_matrix)

    predictor_matrices = [p.astype('float16') for p in predictor_matrices]

    return {
        PREDICTOR_MATRICES_KEY: predictor_matrices,
        TARGET_INTENSITIES_KEY: target_intensities_m_s01,
        TARGET_TIMES_KEY: target_times_unix_sec,
        HIGH_RES_LATITUDES_KEY: None,
        HIGH_RES_LONGITUDES_KEY: None,
        LOW_RES_LATITUDES_KEY: low_res_latitude_matrix_deg_n,
        LOW_RES_LONGITUDES_KEY: low_res_longitude_matrix_deg_e
    }


def create_data_no_trans(
        option_dict, ebtrk_file_name, cyclone_id_string, num_target_times):
    """Creates, rather than generates, neural-net inputs.

    This method returns untranslated images, where the image center is the best-
    track TC center.

    E = batch size = number of examples after data augmentation
    L = number of lag times for predictors
    w = number of wavelengths
    m = number of rows in grid
    n = number of columns in grid

    :param option_dict: See doc for `data_generator`.
    :param ebtrk_file_name: Same.
    :param cyclone_id_string: Will create data for this cyclone.
    :param num_target_times: Will create data for this number of target times
        for the given cyclone.

    :return: data_dict: Dictionary with the following keys.
    data_dict["predictor_matrices"]: See doc for `data_generator`.
    data_dict["target_intensities_m_s01"]: Same.
    data_dict["target_times_unix_sec"]: length-E numpy array of target times.
    data_dict["high_res_latitude_matrix_deg_n"]: None.
    data_dict["high_res_longitude_matrix_deg_e"]: None.
    data_dict["low_res_latitude_matrix_deg_n"]: E-by-m-by-L numpy array of
        latitudes (deg north).
    data_dict["low_res_longitude_matrix_deg_e"]: E-by-n-by-L numpy array of
        longitudes (deg east).
    """

    error_checking.assert_is_integer(num_target_times)
    error_checking.assert_is_greater(num_target_times, 0)

    option_dict[nn_utils.HIGH_RES_WAVELENGTHS_KEY] = numpy.array([])
    option_dict[nn_utils.LAG_TIME_TOLERANCE_KEY] = 0
    option_dict[nn_utils.MAX_MISSING_LAG_TIMES_KEY] = 0
    option_dict[nn_utils.MAX_INTERP_GAP_KEY] = 0
    option_dict[nn_utils.SENTINEL_VALUE_KEY] = -10.
    option_dict[nn_utils.SYNOPTIC_TIMES_ONLY_KEY] = True
    option_dict[DATA_AUG_NUM_TRANS_KEY] = 8
    option_dict[DATA_AUG_MEAN_TRANS_KEY] = 12.
    option_dict[DATA_AUG_STDEV_TRANS_KEY] = 6.

    option_dict = nn_utils.check_generator_args(option_dict)

    satellite_dir_name = option_dict[SATELLITE_DIRECTORY_KEY]
    lag_times_minutes = option_dict[LAG_TIMES_KEY]
    low_res_wavelengths_microns = option_dict[LOW_RES_WAVELENGTHS_KEY]
    num_rows_low_res = option_dict[NUM_GRID_ROWS_KEY]
    num_columns_low_res = option_dict[NUM_GRID_COLUMNS_KEY]
    a_deck_file_name = option_dict[A_DECK_FILE_KEY]
    scalar_a_deck_field_names = option_dict[SCALAR_A_DECK_FIELDS_KEY]
    remove_nontropical_systems = option_dict[REMOVE_NONTROPICAL_KEY]

    satellite_file_names = satellite_io.find_files_one_cyclone(
        directory_name=satellite_dir_name, cyclone_id_string=cyclone_id_string,
        raise_error_if_all_missing=True
    )

    (
        all_target_times_unix_sec, all_scalar_predictor_matrix
    ) = nn_training_fancy.get_target_times_and_scalar_predictors(
        cyclone_id_strings=[cyclone_id_string],
        synoptic_times_only=True,
        satellite_file_names_by_cyclone=[satellite_file_names],
        a_deck_file_name=a_deck_file_name,
        scalar_a_deck_field_names=scalar_a_deck_field_names,
        remove_nontropical_systems=remove_nontropical_systems,
        remove_tropical_systems=False,
        predictor_lag_times_sec=MINUTES_TO_SECONDS * lag_times_minutes
    )

    all_target_times_unix_sec = all_target_times_unix_sec[0]
    all_scalar_predictor_matrix = all_scalar_predictor_matrix[0]

    # TODO(thunderhoser): Maybe modularize all the EBTRK matching shit below?
    print('Reading data from: "{0:s}"...'.format(ebtrk_file_name))
    ebtrk_table_xarray = ebtrk_io.read_file(ebtrk_file_name)
    ebtrk_cyclone_id_strings = (
        ebtrk_table_xarray[ebtrk_utils.STORM_ID_KEY].values
    )
    ebtrk_valid_times_unix_sec = (
        HOURS_TO_SECONDS * ebtrk_table_xarray[ebtrk_utils.VALID_TIME_KEY].values
    )
    ebtrk_intensities_m_s01 = (
        ebtrk_table_xarray[ebtrk_utils.MAX_SUSTAINED_WIND_KEY].values
    )

    all_target_intensities_m_s01 = []
    good_indices = []

    for i in range(len(all_target_times_unix_sec)):
        good_indices_this_example = numpy.where(numpy.logical_and(
            ebtrk_cyclone_id_strings == cyclone_id_string,
            ebtrk_valid_times_unix_sec == all_target_times_unix_sec[i]
        ))[0]

        if len(good_indices_this_example) > 0:
            good_indices.append(i)

            k = good_indices_this_example[0]
            all_target_intensities_m_s01.append(ebtrk_intensities_m_s01[k])

            continue

        warning_string = (
            'POTENTIAL ERROR: cannot find cyclone {0:s} at time {1:s} in '
            'extended best-track data.'
        ).format(
            cyclone_id_string,
            time_conversion.unix_sec_to_string(
                all_target_times_unix_sec[i], TIME_FORMAT_FOR_LOG_MESSAGES
            )
        )

        warnings.warn(warning_string)

    all_target_intensities_m_s01 = numpy.array(
        all_target_intensities_m_s01, dtype=int
    )
    good_indices = numpy.array(good_indices, dtype=int)
    all_target_times_unix_sec = all_target_times_unix_sec[good_indices]
    all_scalar_predictor_matrix = all_scalar_predictor_matrix[good_indices, :]

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
        return_coords=True, return_xy_coords=False
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

    target_intensities_m_s01 = all_target_intensities_m_s01[row_indices]
    target_intensities_m_s01 = target_intensities_m_s01[:num_target_times]

    if a_deck_file_name is None:
        scalar_predictor_matrix = None
    else:
        scalar_predictor_matrix = all_scalar_predictor_matrix[row_indices, :]
        scalar_predictor_matrix = scalar_predictor_matrix[:num_target_times, :]

    brightness_temp_matrix_kelvins = (
        data_dict[BRIGHTNESS_TEMPS_KEY][:num_target_times, ...]
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

    brightness_temp_matrix_kelvins = nn_utils.combine_lag_times_and_wavelengths(
        brightness_temp_matrix_kelvins
    )

    if scalar_predictor_matrix is not None:
        good_scalar_predictor_indices = []

        for j in range(len(scalar_a_deck_field_names)):
            if (
                    scalar_a_deck_field_names[j] in
                    A_DECK_FIELD_NAMES_FOR_CENTER_FIXING_ONLY
            ):
                continue

            good_scalar_predictor_indices.append(j)

        good_scalar_predictor_indices = numpy.array(
            good_scalar_predictor_indices, dtype=int
        )
        scalar_predictor_matrix = (
            scalar_predictor_matrix[:, good_scalar_predictor_indices]
        )

    predictor_matrices = [brightness_temp_matrix_kelvins]
    if scalar_predictor_matrix is not None:
        predictor_matrices.append(scalar_predictor_matrix)

    predictor_matrices = [p.astype('float16') for p in predictor_matrices]

    return {
        PREDICTOR_MATRICES_KEY: predictor_matrices,
        TARGET_INTENSITIES_KEY: target_intensities_m_s01,
        TARGET_TIMES_KEY: target_times_unix_sec,
        HIGH_RES_LATITUDES_KEY: None,
        HIGH_RES_LONGITUDES_KEY: None,
        LOW_RES_LATITUDES_KEY: low_res_latitude_matrix_deg_n,
        LOW_RES_LONGITUDES_KEY: low_res_longitude_matrix_deg_e
    }


def apply_model(
        model_object, predictor_matrices, num_examples_per_batch, verbose=True):
    """Applies trained neural net -- inference time!

    E = number of examples

    :param model_object: Trained neural net (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param predictor_matrices: See output doc for `data_generator_shuffled`.
    :param num_examples_per_batch: Batch size.
    :param verbose: Boolean flag.  If True, will print progress messages.
    :return: predicted_intensities_m_s01: length-E numpy array of predictions.
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
    predicted_intensities_m_s01 = numpy.full(num_examples, numpy.nan)

    for i in range(0, num_examples, num_examples_per_batch):
        first_index = i
        last_index = min([i + num_examples_per_batch, num_examples])

        if verbose:
            print('Applying model to examples {0:d}-{1:d} of {2:d}...'.format(
                first_index + 1, last_index, num_examples
            ))

        predicted_intensities_m_s01[first_index:last_index] = (
            numpy.squeeze(model_object.predict_on_batch(
                [a[first_index:last_index, ...] for a in predictor_matrices]
            ))
        )

    if verbose:
        print('Have applied model to all {0:d} examples!'.format(num_examples))

    return predicted_intensities_m_s01


def write_prediction_file(
        netcdf_file_name, target_intensities_m_s01, predicted_intensities_m_s01,
        cyclone_id_string, target_times_unix_sec, model_file_name):
    """Writes predictions to NetCDF file.

    E = number of examples

    :param netcdf_file_name: Path to output file.
    :param target_intensities_m_s01: length-E numpy array of true intensities.
    :param predicted_intensities_m_s01: length-E numpy array of predicted
        intensities.
    :param cyclone_id_string: Cyclone ID.
    :param target_times_unix_sec: length-E numpy array of target times.
    :param model_file_name: Path to trained model (readable by
        `neural_net_utils.read_model`).
    """

    # Check input args.
    error_checking.assert_is_numpy_array(
        target_intensities_m_s01, num_dimensions=1
    )
    error_checking.assert_is_greater_numpy_array(target_intensities_m_s01, 0.)

    num_examples = len(target_intensities_m_s01)
    error_checking.assert_is_numpy_array(
        predicted_intensities_m_s01,
        exact_dimensions=numpy.array([num_examples], dtype=int)
    )
    error_checking.assert_is_geq_numpy_array(predicted_intensities_m_s01, 0.)

    _ = misc_utils.parse_cyclone_id(cyclone_id_string)

    error_checking.assert_is_integer_numpy_array(target_times_unix_sec)
    error_checking.assert_is_numpy_array(
        target_times_unix_sec,
        exact_dimensions=numpy.array([num_examples], dtype=int)
    )

    error_checking.assert_is_string(model_file_name)

    # Do actual stuff.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    num_cyclone_id_chars = len(cyclone_id_string)

    dataset_object.setncattr(MODEL_FILE_KEY, model_file_name)
    dataset_object.createDimension(EXAMPLE_DIM_KEY, num_examples)
    dataset_object.createDimension(
        CYCLONE_ID_CHAR_DIM_KEY, num_cyclone_id_chars
    )

    dataset_object.createVariable(
        PREDICTED_INTENSITY_KEY, datatype=numpy.float32,
        dimensions=EXAMPLE_DIM_KEY
    )
    dataset_object.variables[PREDICTED_INTENSITY_KEY][:] = (
        predicted_intensities_m_s01
    )

    dataset_object.createVariable(
        TARGET_INTENSITY_KEY, datatype=numpy.float32, dimensions=EXAMPLE_DIM_KEY
    )
    dataset_object.variables[TARGET_INTENSITY_KEY][:] = target_intensities_m_s01

    dataset_object.createVariable(
        TARGET_TIME_KEY, datatype=numpy.int32, dimensions=EXAMPLE_DIM_KEY
    )
    dataset_object.variables[TARGET_TIME_KEY][:] = target_times_unix_sec

    this_string_format = 'S{0:d}'.format(num_cyclone_id_chars)
    cyclone_ids_char_array = netCDF4.stringtochar(numpy.array(
        [cyclone_id_string] * num_examples, dtype=this_string_format
    ))

    dataset_object.createVariable(
        CYCLONE_ID_KEY, datatype='S1',
        dimensions=(EXAMPLE_DIM_KEY, CYCLONE_ID_CHAR_DIM_KEY)
    )
    dataset_object.variables[CYCLONE_ID_KEY][:] = numpy.array(
        cyclone_ids_char_array
    )

    dataset_object.close()
