"""Helper methods for neural networks (training and inference)."""

import os
import time
import pickle
import numpy
import keras
import keras.layers as layers
import tensorflow
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4tccf.io import a_deck_io
from ml4tccf.utils import image_filtering
from ml4tccf.machine_learning import custom_losses_scalar
from ml4tccf.machine_learning import custom_metrics_scalar
from ml4tccf.machine_learning import custom_losses_gridded
from ml4tccf.machine_learning import custom_metrics_gridded
from ml4tccf.machine_learning import custom_losses_structure
from ml4tccf.machine_learning import custom_metrics_structure

try:
    input_layer_object_low_res = layers.Input(shape=(3, 4, 5))
except:
    import tensorflow.keras as keras
    # import tensorflow.keras.layers as layers

TOLERANCE = 1e-6
SYNOPTIC_TIME_INTERVAL_SEC = 6 * 3600

KT_TO_METRES_PER_SECOND = 1.852 / 3.6
MIN_TROPICAL_INTENSITY_M_S01 = 25 * KT_TO_METRES_PER_SECOND

CIRA_IR_DATA_TYPE_STRING = 'cira_ir'
RG_FANCY_DATA_TYPE_STRING = 'robert_galina_fancy'
RG_SIMPLE_DATA_TYPE_STRING = 'robert_galina_simple'
VALID_DATA_TYPE_STRINGS = [
    CIRA_IR_DATA_TYPE_STRING, RG_FANCY_DATA_TYPE_STRING,
    RG_SIMPLE_DATA_TYPE_STRING
]

TROPICAL_SYSTEM_TYPE_STRINGS = [
    a_deck_io.TROPICAL_DEPRESSION_TYPE_STRING,
    a_deck_io.TROPICAL_STORM_TYPE_STRING,
    a_deck_io.TROPICAL_TYPHOON_TYPE_STRING,
    a_deck_io.TROPICAL_SUPER_TYPHOON_TYPE_STRING,
    a_deck_io.TROPICAL_CYCLONE_TYPE_STRING,
    a_deck_io.TROPICAL_HURRICANE_TYPE_STRING
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
SYNOPTIC_TIMES_ONLY_KEY = 'synoptic_times_only'
A_DECK_FILE_KEY = 'a_deck_file_name'
SCALAR_A_DECK_FIELDS_KEY = 'scalar_a_deck_field_names'
REMOVE_NONTROPICAL_KEY = 'remove_nontropical_systems'
REMOVE_TROPICAL_KEY = 'remove_tropical_systems'
USE_XY_COORDS_KEY = 'use_xy_coords_as_predictors'

DEFAULT_GENERATOR_OPTION_DICT = {
    HIGH_RES_WAVELENGTHS_KEY: None,
    LOW_RES_WAVELENGTHS_KEY: None,
    BATCH_SIZE_KEY: 8,
    MAX_EXAMPLES_PER_CYCLONE_KEY: 1,
    DATA_AUG_NUM_TRANS_KEY: 8,
    DATA_AUG_MEAN_TRANS_KEY: 15.,
    DATA_AUG_STDEV_TRANS_KEY: 7.5,
    MAX_INTERP_GAP_KEY: 0,
    SENTINEL_VALUE_KEY: -10.,
    SYNOPTIC_TIMES_ONLY_KEY: True,
    A_DECK_FILE_KEY: None,
    SCALAR_A_DECK_FIELDS_KEY: [],
    REMOVE_NONTROPICAL_KEY: False,
    REMOVE_TROPICAL_KEY: False
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
DATA_TYPE_KEY = 'data_type_string'
TRAIN_WITH_SHUFFLED_DATA_KEY = 'train_with_shuffled_data'
CNN_ARCHITECTURE_KEY = 'cnn_architecture_dict'
TEMPORAL_CNN_ARCHITECTURE_KEY = 'temporal_cnn_architecture_dict'
TEMPORAL_CONVNEXT_ARCHITECTURE_KEY = 'temporal_convnext_architecture_dict'
STRUCTURE_CNN_ARCHITECTURE_KEY = 'structure_cnn_architecture_dict'
U_NET_ARCHITECTURE_KEY = 'u_net_architecture_dict'

METADATA_KEYS = [
    NUM_EPOCHS_KEY, NUM_TRAINING_BATCHES_KEY, TRAINING_OPTIONS_KEY,
    NUM_VALIDATION_BATCHES_KEY, VALIDATION_OPTIONS_KEY,
    LOSS_FUNCTION_KEY, OPTIMIZER_FUNCTION_KEY,
    PLATEAU_PATIENCE_KEY, PLATEAU_LR_MUTIPLIER_KEY, EARLY_STOPPING_PATIENCE_KEY,
    CNN_ARCHITECTURE_KEY, TEMPORAL_CNN_ARCHITECTURE_KEY,
    TEMPORAL_CONVNEXT_ARCHITECTURE_KEY,
    STRUCTURE_CNN_ARCHITECTURE_KEY, U_NET_ARCHITECTURE_KEY,
    DATA_TYPE_KEY, TRAIN_WITH_SHUFFLED_DATA_KEY
]

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
    custom_metrics_scalar.discretized_crps_kilometres,
    custom_metrics_scalar.correlation
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
        custom_metrics_scalar.discretized_crps_kilometres,
    'correlation': custom_metrics_scalar.correlation
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


def check_data_type(data_type_string):
    """Ensures valid data type.

    :param data_type_string: Data type.
    :raises: ValueError: if `data_type_string not in VALID_DATA_TYPE_STRINGS`.
    """

    error_checking.assert_is_string(data_type_string)
    if data_type_string in VALID_DATA_TYPE_STRINGS:
        return

    error_string = (
        'Data type "{0:s}" is not one of the valid data types (listed below):'
        '\n{1:s}'
    ).format(data_type_string, str(VALID_DATA_TYPE_STRINGS))

    raise ValueError(error_string)


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
    error_checking.assert_is_boolean(option_dict[USE_XY_COORDS_KEY])
    error_checking.assert_is_boolean(option_dict[SEMANTIC_SEG_FLAG_KEY])
    error_checking.assert_is_boolean(option_dict[SYNOPTIC_TIMES_ONLY_KEY])

    if option_dict[SEMANTIC_SEG_FLAG_KEY]:
        error_checking.assert_is_greater(
            option_dict[TARGET_SMOOOTHER_STDEV_KEY], 0.
        )
    else:
        option_dict[TARGET_SMOOOTHER_STDEV_KEY] = None

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

    # error_checking.assert_is_numpy_array_without_nan(satellite_data_matrix)
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

    # error_checking.assert_is_numpy_array_without_nan(satellite_data_matrix)
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


def get_low_res_data_from_predictors(predictor_matrices):
    """Fetches low-resolution satellite data from list of predictor matrices.

    :param predictor_matrices: Same as output from `create_data`.
    :return: brightness_temp_matrix: numpy array of brightness temperatures.
    :raises: ValueError: if low-resolution satellite data cannot be found.
    """

    error_checking.assert_is_list(predictor_matrices)
    num_matrices = len(predictor_matrices)

    for k in range(num_matrices)[::-1]:
        error_checking.assert_is_numpy_array(predictor_matrices[k])
        if len(predictor_matrices[k].shape) > 2:
            return predictor_matrices[k]

    error_string = (
        'Cannot find low-resolution satellite data in predictor matrices with '
        'the following shapes:\n'
    )
    for k in range(num_matrices):
        error_string += '{0:s}\n'.format(str(predictor_matrices[k].shape))

    raise ValueError(error_string)


def get_high_res_data_from_predictors(predictor_matrices):
    """Fetches high-resolution satellite data from list of predictor matrices.

    :param predictor_matrices: Same as output from `create_data`.
    :return: bidirectional_reflectance_matrix: numpy array of bidirectional
        reflectances.
    :raises: ValueError: if high-resolution satellite data cannot be found.
    """

    error_checking.assert_is_list(predictor_matrices)
    num_matrices = len(predictor_matrices)

    if len(predictor_matrices[0].shape) > 2:
        return predictor_matrices[0]

    error_string = (
        'Cannot find high-resolution satellite data in predictor matrices with '
        'the following shapes:\n'
    )
    for k in range(num_matrices):
        error_string += '{0:s}\n'.format(str(predictor_matrices[k].shape))

    raise ValueError(error_string)


def read_scalar_data(
        a_deck_file_name, field_names, remove_nontropical_systems,
        remove_tropical_systems, cyclone_id_strings, target_times_unix_sec,
        a_decks_at_least_6h_old=False):
    """Reads scalar data for the given cyclone objects.

    One cyclone object = one cyclone and one target time

    F = number of fields
    T = number of cyclone objects

    :param a_deck_file_name: Path to A-deck file.  Will be read by
        `a_deck_io.read_file`.
    :param field_names: length-F list of fields to read.  Each field name must
        be in the constant list `VALID_SCALAR_FIELD_NAMES`.
    :param remove_nontropical_systems: Boolean flag.  If True, will return only
        NaN for non-tropical systems.
    :param remove_tropical_systems: Boolean flag.  If True, will return only
        NaN for tropical systems.
    :param cyclone_id_strings: length-T list of cyclone IDs.
    :param target_times_unix_sec: length-T numpy array of target times.
    :param a_decks_at_least_6h_old: Boolean flag.
    :return: scalar_predictor_matrix: T-by-F numpy array.
    """

    # Check input args.
    error_checking.assert_file_exists(a_deck_file_name)
    error_checking.assert_is_string_list(field_names)
    error_checking.assert_is_boolean(remove_nontropical_systems)
    error_checking.assert_is_boolean(remove_tropical_systems)
    assert not (remove_nontropical_systems and remove_tropical_systems)
    error_checking.assert_is_string_list(cyclone_id_strings)

    num_cyclone_objects = len(cyclone_id_strings)
    expected_dim = numpy.array([num_cyclone_objects], dtype=int)
    error_checking.assert_is_integer_numpy_array(target_times_unix_sec)
    error_checking.assert_is_numpy_array(
        target_times_unix_sec, exact_dimensions=expected_dim
    )

    error_checking.assert_is_boolean(a_decks_at_least_6h_old)

    # Do actual stuff.
    print('Reading data from: "{0:s}"...'.format(a_deck_file_name))
    a_deck_table_xarray = a_deck_io.read_file(a_deck_file_name)

    predictor_times_unix_sec = number_rounding.floor_to_nearest(
        target_times_unix_sec, SYNOPTIC_TIME_INTERVAL_SEC
    )
    if a_decks_at_least_6h_old:
        predictor_times_unix_sec -= SYNOPTIC_TIME_INTERVAL_SEC

    id_predictor_time_matrix = numpy.transpose(numpy.vstack((
        numpy.array(cyclone_id_strings), predictor_times_unix_sec
    )))
    unique_id_predictor_time_matrix, orig_to_unique_indices = numpy.unique(
        id_predictor_time_matrix, return_inverse=True, axis=0
    )

    num_fields = len(field_names)
    scalar_predictor_matrix = numpy.full(
        (num_cyclone_objects, num_fields), numpy.nan
    )
    adt = a_deck_table_xarray

    for i in range(unique_id_predictor_time_matrix.shape[0]):
        a_deck_indices = numpy.where(numpy.logical_and(
            adt[a_deck_io.VALID_TIME_KEY].values ==
            int(numpy.round(float(unique_id_predictor_time_matrix[i, 1]))),
            adt[a_deck_io.CYCLONE_ID_KEY].values ==
            unique_id_predictor_time_matrix[i, 0]
        ))[0]

        if len(a_deck_indices) == 0:
            continue

        keep_this_time = True
        a_deck_index = a_deck_indices[0]

        if remove_nontropical_systems:
            keep_this_time = (
                adt[a_deck_io.STORM_TYPE_KEY].values[a_deck_index]
                in TROPICAL_SYSTEM_TYPE_STRINGS
            )
        elif remove_tropical_systems:
            keep_this_time = (
                adt[a_deck_io.STORM_TYPE_KEY].values[a_deck_index]
                not in TROPICAL_SYSTEM_TYPE_STRINGS
            )

        keep_this_time = (
            keep_this_time and
            adt[a_deck_io.UNNORM_INTENSITY_KEY].values[a_deck_index]
            >= MIN_TROPICAL_INTENSITY_M_S01
        )

        if not keep_this_time:
            continue

        matrix_row_indices = numpy.where(orig_to_unique_indices == i)[0]
        for j in range(num_fields):
            scalar_predictor_matrix[matrix_row_indices, j] = (
                a_deck_table_xarray[field_names[j]].values[a_deck_index]
            )

    return scalar_predictor_matrix


def grid_coords_3d_to_4d(latitude_matrix_deg_n, longitude_matrix_deg_e):
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

    # Check input args.
    error_checking.assert_is_valid_lat_numpy_array(
        latitudes_deg=latitude_matrix_deg_n, allow_nan=False
    )
    error_checking.assert_is_numpy_array(
        latitude_matrix_deg_n, num_dimensions=3
    )

    error_checking.assert_is_valid_lng_numpy_array(
        longitudes_deg=longitude_matrix_deg_e, allow_nan=False
    )
    error_checking.assert_is_numpy_array(
        longitude_matrix_deg_e, num_dimensions=3
    )

    num_target_times = latitude_matrix_deg_n.shape[0]
    num_lag_times = latitude_matrix_deg_n.shape[2]
    expected_dim = numpy.array(
        [num_target_times, longitude_matrix_deg_e.shape[1], num_lag_times],
        dtype=int
    )
    error_checking.assert_is_numpy_array(
        longitude_matrix_deg_e, exact_dimensions=expected_dim
    )

    # Do actual stuff.
    latitude_matrix_deg_n_4d = numpy.stack(
        [
            numpy.stack(
                [
                    grids.latlng_vectors_to_matrices(
                        unique_latitudes_deg=latitude_matrix_deg_n[i, :, j],
                        unique_longitudes_deg=longitude_matrix_deg_e[i, :, j]
                    )[0]
                    for j in range(num_lag_times)
                ],
                axis=-1)
            for i in range(num_target_times)
        ],
        axis=0
    )

    longitude_matrix_deg_e_4d = numpy.stack(
        [
            numpy.stack(
                [
                    grids.latlng_vectors_to_matrices(
                        unique_latitudes_deg=latitude_matrix_deg_n[i, :, j],
                        unique_longitudes_deg=longitude_matrix_deg_e[i, :, j]
                    )[1]
                    for j in range(num_lag_times)
                ],
                axis=-1
            )
            for i in range(num_target_times)
        ],
        axis=0
    )

    return latitude_matrix_deg_n_4d, longitude_matrix_deg_e_4d


def make_targets_for_semantic_seg(
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

    # Check input args.
    error_checking.assert_is_integer_numpy_array(row_translations_px)
    error_checking.assert_is_numpy_array(row_translations_px, num_dimensions=1)

    num_examples = len(row_translations_px)
    expected_dim = numpy.array([num_examples], dtype=int)

    error_checking.assert_is_integer_numpy_array(column_translations_px)
    error_checking.assert_is_numpy_array(
        column_translations_px, exact_dimensions=expected_dim
    )

    error_checking.assert_is_greater_numpy_array(grid_spacings_km, 0.)
    error_checking.assert_is_numpy_array(
        grid_spacings_km, exact_dimensions=expected_dim
    )

    error_checking.assert_is_valid_lat_numpy_array(
        latitudes_deg=cyclone_center_latitudes_deg_n, allow_nan=False
    )
    error_checking.assert_is_numpy_array(
        cyclone_center_latitudes_deg_n, exact_dimensions=expected_dim
    )

    error_checking.assert_is_greater(gaussian_smoother_stdev_km, 0.)

    error_checking.assert_is_integer(num_grid_rows)
    error_checking.assert_is_greater(num_grid_rows, 0)
    error_checking.assert_equals(
        numpy.mod(num_grid_rows, 2),
        0
    )

    error_checking.assert_is_integer(num_grid_columns)
    error_checking.assert_is_greater(num_grid_columns, 0)
    error_checking.assert_equals(
        numpy.mod(num_grid_columns, 2),
        0
    )

    # Do actual stuff.
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


def find_metafile(model_dir_name, raise_error_if_missing=True):
    """Finds metafile for neural net.

    :param model_dir_name: Name of model directory.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: metafile_name: Path to metafile.
    """

    # TODO(thunderhoser): HACK to deal with ensembles.
    model_dir_name = model_dir_name.split(' ')[-1]
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


def write_metafile(
        pickle_file_name, num_epochs, num_training_batches_per_epoch,
        training_option_dict, num_validation_batches_per_epoch,
        validation_option_dict, loss_function_string, optimizer_function_string,
        plateau_patience_epochs, plateau_learning_rate_multiplier,
        early_stopping_patience_epochs,
        cnn_architecture_dict, temporal_cnn_architecture_dict,
        temporal_convnext_architecture_dict,
        structure_cnn_architecture_dict, u_net_architecture_dict,
        data_type_string, train_with_shuffled_data):
    """Writes metadata to Pickle file.

    :param pickle_file_name: Path to output file.
    :param num_epochs: See doc for `neural_net_training_fancy.train_model`.
    :param num_training_batches_per_epoch: Same.
    :param training_option_dict: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validation_option_dict: Same.
    :param loss_function_string: Same.
    :param optimizer_function_string: Same.
    :param plateau_patience_epochs: Same.
    :param plateau_learning_rate_multiplier: Same.
    :param early_stopping_patience_epochs: Same.
    :param cnn_architecture_dict: Same.
    :param temporal_cnn_architecture_dict: Same.
    :param temporal_convnext_architecture_dict: Same.
    :param structure_cnn_architecture_dict: Same.
    :param u_net_architecture_dict: Same.
    :param data_type_string: Data type (must be accepted by `check_data_type`).
    :param train_with_shuffled_data: Boolean flag.  If True, the model is
        trained with shuffled data files.  If False, the model is trained with
        sorted files (one file per cyclone or per cyclone-day).
    """

    check_data_type(data_type_string)

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
        CNN_ARCHITECTURE_KEY: cnn_architecture_dict,
        TEMPORAL_CNN_ARCHITECTURE_KEY: temporal_cnn_architecture_dict,
        TEMPORAL_CONVNEXT_ARCHITECTURE_KEY: temporal_convnext_architecture_dict,
        STRUCTURE_CNN_ARCHITECTURE_KEY: structure_cnn_architecture_dict,
        U_NET_ARCHITECTURE_KEY: u_net_architecture_dict,
        DATA_TYPE_KEY: data_type_string,
        TRAIN_WITH_SHUFFLED_DATA_KEY: train_with_shuffled_data
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(metadata_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_metafile(pickle_file_name):
    """Reads metadata from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict["num_epochs"]: See doc for
        `neural_net_training_fancy.train_model`.
    metadata_dict["num_training_batches_per_epoch"]: Same.
    metadata_dict["training_option_dict"]: Same.
    metadata_dict["num_validation_batches_per_epoch"]: Same.
    metadata_dict["validation_option_dict"]: Same.
    metadata_dict["loss_function_string"]: Same.
    metadata_dict["optimizer_function_string"]: Same.
    metadata_dict["plateau_patience_epochs"]: Same.
    metadata_dict["plateau_learning_rate_multiplier"]: Same.
    metadata_dict["early_stopping_patience_epochs"]: Same.
    metadata_dict["cnn_architecture_dict"]: Same.
    metadata_dict["temporal_cnn_architecture_dict"]: Same.
    metadata_dict["temporal_convnext_architecture_dict"]: Same.
    metadata_dict["structure_cnn_architecture_dict"]: Same.
    metadata_dict["u_net_architecture_dict"]: Same.
    metadata_dict["data_type_string"]: Data type (must be accepted by
        `check_data_type`).
    metadata_dict["train_with_shuffled_data"]: Boolean flag.  If True, the model
        is trained with shuffled data files.  If False, the model is trained
        with sorted files (one file per cyclone or per cyclone-day).

    :raises: ValueError: if any expected key is not found in dictionary.
    """

    error_checking.assert_file_exists(pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'rb')
    metadata_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    training_option_dict = metadata_dict[TRAINING_OPTIONS_KEY]
    validation_option_dict = metadata_dict[VALIDATION_OPTIONS_KEY]

    if USE_XY_COORDS_KEY not in training_option_dict:
        training_option_dict[USE_XY_COORDS_KEY] = False
        validation_option_dict[USE_XY_COORDS_KEY] = False
    if REMOVE_TROPICAL_KEY not in training_option_dict:
        training_option_dict[REMOVE_TROPICAL_KEY] = False
        validation_option_dict[REMOVE_TROPICAL_KEY] = False

    if CNN_ARCHITECTURE_KEY not in metadata_dict:
        metadata_dict[TEMPORAL_CNN_ARCHITECTURE_KEY] = metadata_dict[
            'architecture_dict'
        ]
        metadata_dict[CNN_ARCHITECTURE_KEY] = None
        metadata_dict[STRUCTURE_CNN_ARCHITECTURE_KEY] = None
        metadata_dict[U_NET_ARCHITECTURE_KEY] = None

    if TEMPORAL_CONVNEXT_ARCHITECTURE_KEY not in metadata_dict:
        metadata_dict[TEMPORAL_CONVNEXT_ARCHITECTURE_KEY] = None

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
    # for this_matrix in predictor_matrices:
    #     error_checking.assert_is_numpy_array_without_nan(this_matrix)

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


def read_model(keras_file_name):
    """Reads model from .keras file.

    :param keras_file_name: Path to input file.
    :return: model_object: Instance of `keras.models.Model` or
        `keras.models.Sequential`.
    """

    error_checking.assert_file_exists(keras_file_name)

    metafile_name = find_metafile(
        model_dir_name=os.path.split(keras_file_name)[0],
        raise_error_if_missing=True
    )
    metadata_dict = read_metafile(metafile_name)
    cnn_architecture_dict = metadata_dict[CNN_ARCHITECTURE_KEY]
    temporal_cnn_architecture_dict = metadata_dict[
        TEMPORAL_CNN_ARCHITECTURE_KEY
    ]
    temporal_convnext_architecture_dict = metadata_dict[
        TEMPORAL_CONVNEXT_ARCHITECTURE_KEY
    ]
    structure_cnn_architecture_dict = metadata_dict[
        STRUCTURE_CNN_ARCHITECTURE_KEY
    ]
    u_net_architecture_dict = metadata_dict[U_NET_ARCHITECTURE_KEY]

    if u_net_architecture_dict is not None:
        from ml4tccf.machine_learning import u_net_architecture

        for this_key in [
                u_net_architecture.LOSS_FUNCTION_KEY,
                u_net_architecture.OPTIMIZER_FUNCTION_KEY
        ]:
            u_net_architecture_dict[this_key] = eval(
                u_net_architecture_dict[this_key]
            )

        model_object = u_net_architecture.create_model(u_net_architecture_dict)
        model_object.load_weights(keras_file_name)
        return model_object

    if cnn_architecture_dict is not None:
        from ml4tccf.machine_learning import cnn_architecture

        for this_key in [
                cnn_architecture.LOSS_FUNCTION_KEY,
                cnn_architecture.OPTIMIZER_FUNCTION_KEY
        ]:
            cnn_architecture_dict[this_key] = eval(
                cnn_architecture_dict[this_key]
            )

        model_object = cnn_architecture.create_model(cnn_architecture_dict)
        model_object.load_weights(keras_file_name)
        return model_object

    if temporal_cnn_architecture_dict is not None:
        from ml4tccf.machine_learning import \
            temporal_cnn_architecture as tcnn_architecture

        for this_key in [
                tcnn_architecture.LOSS_FUNCTION_KEY,
                tcnn_architecture.OPTIMIZER_FUNCTION_KEY
        ]:
            temporal_cnn_architecture_dict[this_key] = eval(
                temporal_cnn_architecture_dict[this_key]
            )

        model_object = tcnn_architecture.create_model(
            temporal_cnn_architecture_dict
        )
        model_object.load_weights(keras_file_name)
        return model_object

    if temporal_convnext_architecture_dict is not None:
        from ml4tccf.machine_learning import \
            temporal_convnext_architecture as convnext_arch

        for this_key in [
            convnext_arch.LOSS_FUNCTION_KEY,
            convnext_arch.OPTIMIZER_FUNCTION_KEY
        ]:
            temporal_convnext_architecture_dict[this_key] = eval(
                temporal_convnext_architecture_dict[this_key]
            )

        model_object = convnext_arch.create_model(
            temporal_convnext_architecture_dict
        )
        model_object.load_weights(keras_file_name)
        return model_object

    from ml4tccf.machine_learning import \
        temporal_cnn_arch_for_structure as structure_architecture

    for this_key in [
            structure_architecture.LOSS_FUNCTION_KEY,
            structure_architecture.OPTIMIZER_FUNCTION_KEY
    ]:
        structure_cnn_architecture_dict[this_key] = eval(
            structure_cnn_architecture_dict[this_key]
        )

    metric_functions = structure_cnn_architecture_dict[
        structure_architecture.METRIC_FUNCTIONS_KEY
    ]
    for i in range(len(metric_functions)):
        metric_functions[i] = eval(metric_functions[i])

    structure_cnn_architecture_dict[
        structure_architecture.METRIC_FUNCTIONS_KEY
    ] = metric_functions

    model_object = structure_architecture.create_model(
        structure_cnn_architecture_dict
    )
    model_object.load_weights(keras_file_name)
    return model_object
