"""NN-training with CIRA IR satellite data."""

import random
import numpy
import keras
from pyproj import Geod
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4tc.io import example_io as cira_ir_example_io
from ml4tccf.utils import misc_utils
from ml4tccf.machine_learning import neural_net_utils
from ml4tccf.machine_learning import data_augmentation

TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H%M'

METRES_TO_KM = 0.001
MINUTES_TO_SECONDS = 60

SYNOPTIC_TIME_INTERVAL_SEC = 6 * 3600
SYNOPTIC_TIME_TOLERANCE_SEC = 900

BRIGHTNESS_TEMPS_KEY = neural_net_utils.BRIGHTNESS_TEMPS_KEY
GRID_SPACINGS_KEY = neural_net_utils.GRID_SPACINGS_KEY
CENTER_LATITUDES_KEY = neural_net_utils.CENTER_LATITUDES_KEY
TARGET_TIMES_KEY = neural_net_utils.TARGET_TIMES_KEY
LOW_RES_LATITUDES_KEY = neural_net_utils.LOW_RES_LATITUDES_KEY
LOW_RES_LONGITUDES_KEY = neural_net_utils.LOW_RES_LONGITUDES_KEY

PREDICTOR_MATRICES_KEY = neural_net_utils.PREDICTOR_MATRICES_KEY
TARGET_MATRIX_KEY = neural_net_utils.TARGET_MATRIX_KEY
HIGH_RES_LATITUDES_KEY = neural_net_utils.HIGH_RES_LATITUDES_KEY
HIGH_RES_LONGITUDES_KEY = neural_net_utils.HIGH_RES_LONGITUDES_KEY

CIRA_IR_TIME_DIM = 'satellite_valid_time_unix_sec'
CIRA_IR_BRIGHTNESS_TEMP_KEY = 'satellite_predictors_gridded'
CIRA_IR_GRID_LATITUDE_KEY = 'satellite_grid_latitude_deg_n'
CIRA_IR_GRID_LONGITUDE_KEY = 'satellite_grid_longitude_deg_e'

SATELLITE_DIRECTORY_KEY = neural_net_utils.SATELLITE_DIRECTORY_KEY
YEARS_KEY = neural_net_utils.YEARS_KEY
LAG_TIMES_KEY = neural_net_utils.LAG_TIMES_KEY
BATCH_SIZE_KEY = neural_net_utils.BATCH_SIZE_KEY
MAX_EXAMPLES_PER_CYCLONE_KEY = neural_net_utils.MAX_EXAMPLES_PER_CYCLONE_KEY
NUM_GRID_ROWS_KEY = neural_net_utils.NUM_GRID_ROWS_KEY
NUM_GRID_COLUMNS_KEY = neural_net_utils.NUM_GRID_COLUMNS_KEY
DATA_AUG_NUM_TRANS_KEY = neural_net_utils.DATA_AUG_NUM_TRANS_KEY
DATA_AUG_MEAN_TRANS_KEY = neural_net_utils.DATA_AUG_MEAN_TRANS_KEY
DATA_AUG_STDEV_TRANS_KEY = neural_net_utils.DATA_AUG_STDEV_TRANS_KEY
SYNOPTIC_TIMES_ONLY_KEY = neural_net_utils.SYNOPTIC_TIMES_ONLY_KEY
A_DECK_FILE_KEY = neural_net_utils.A_DECK_FILE_KEY
SCALAR_A_DECK_FIELDS_KEY = neural_net_utils.SCALAR_A_DECK_FIELDS_KEY
REMOVE_NONTROPICAL_KEY = neural_net_utils.REMOVE_NONTROPICAL_KEY
SEMANTIC_SEG_FLAG_KEY = neural_net_utils.SEMANTIC_SEG_FLAG_KEY
TARGET_SMOOOTHER_STDEV_KEY = neural_net_utils.TARGET_SMOOOTHER_STDEV_KEY


def _get_synoptic_target_times(all_target_times_unix_sec):
    """Reduces array of target times to synoptic times only.

    :param all_target_times_unix_sec: 1-D numpy array with all target times.
    :return: synoptic_target_times_unix_sec: 1-D numpy array with only synoptic
        target times.
    """

    first_synoptic_times_unix_sec = number_rounding.floor_to_nearest(
        all_target_times_unix_sec, SYNOPTIC_TIME_INTERVAL_SEC
    )
    second_synoptic_times_unix_sec = number_rounding.ceiling_to_nearest(
        all_target_times_unix_sec, SYNOPTIC_TIME_INTERVAL_SEC
    )
    synoptic_times_unix_sec = numpy.concatenate((
        first_synoptic_times_unix_sec, second_synoptic_times_unix_sec
    ))
    synoptic_times_unix_sec = numpy.unique(synoptic_times_unix_sec)

    good_indices = numpy.array([
        numpy.argmin(numpy.absolute(st - all_target_times_unix_sec))
        for st in synoptic_times_unix_sec
    ], dtype=int)

    time_diffs_sec = numpy.absolute(
        all_target_times_unix_sec[good_indices] - synoptic_times_unix_sec
    )
    good_subindices = numpy.where(
        time_diffs_sec <= SYNOPTIC_TIME_TOLERANCE_SEC
    )[0]
    good_indices = good_indices[good_subindices]
    return all_target_times_unix_sec[good_indices]


def _choose_random_target_times(all_target_times_unix_sec, num_times_desired):
    """Chooses random target times from array.

    T = number of times chosen

    :param all_target_times_unix_sec: 1-D numpy array with all target times.
    :param num_times_desired: Number of times desired.
    :return: chosen_target_times_unix_sec: length-T numpy array of chosen target
        times.
    :return: chosen_indices: length-T numpy array of corresponding indices.
    """

    all_indices = numpy.linspace(
        0, len(all_target_times_unix_sec) - 1,
        num=len(all_target_times_unix_sec), dtype=int
    )

    if len(all_target_times_unix_sec) > num_times_desired:
        chosen_indices = numpy.random.choice(
            all_indices, size=num_times_desired, replace=False
        )
    else:
        chosen_indices = all_indices + 0

    return all_target_times_unix_sec[chosen_indices], chosen_indices


def _get_target_times_and_scalar_predictors(
        cyclone_id_strings, synoptic_times_only, cira_ir_file_name_by_cyclone,
        a_deck_file_name, scalar_a_deck_field_names,
        remove_nontropical_systems):
    """Returns target times and scalar predictors for each cyclone.

    C = number of cyclones
    F = number of scalar fields
    T_i = number of target times for [i]th cyclone

    :param cyclone_id_strings: length-C list of cyclone IDs.
    :param synoptic_times_only: Boolean flag.  If True (False), only synoptic
        times (all times) can be target times.
    :param cira_ir_file_name_by_cyclone: length-C list of paths to CIRA IR
        files.
    :param a_deck_file_name: Path to A-deck file, containing scalar predictors
        (readable by `a_deck_io.read_file`).  If you do not want scalar
        predictors, make this None.
    :param scalar_a_deck_field_names: length-F list of field names.  If
        `a_deck_file_name` is None, make this None.
    :param remove_nontropical_systems:
        [used only if `a_deck_file_name` is not None]
        Boolean flag.  If True, will return only target times corresponding to
        tropical systems (no extratropical, subtropical, etc.).
    :return: target_times_by_cyclone_unix_sec: length-C list, where the [i]th
        item is a numpy array (length T_i) of target times.
    :return: scalar_predictor_matrix_by_cyclone: length-C list, where the [i]th
        item is a numpy array (T_i x F) of scalar predictors.
    """

    num_cyclones = len(cyclone_id_strings)
    target_times_by_cyclone_unix_sec = (
        [numpy.array([], dtype=int)] * num_cyclones
    )

    for i in range(num_cyclones):
        target_times_by_cyclone_unix_sec[i] = cira_ir_example_io.read_file(
            cira_ir_file_name_by_cyclone[i]
        ).coords[CIRA_IR_TIME_DIM].values

        if synoptic_times_only:
            target_times_by_cyclone_unix_sec[i] = _get_synoptic_target_times(
                all_target_times_unix_sec=target_times_by_cyclone_unix_sec[i]
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

        scalar_predictor_matrix_by_cyclone[i] = (
            neural_net_utils.read_scalar_data(
                a_deck_file_name=a_deck_file_name,
                field_names=scalar_a_deck_field_names,
                remove_nontropical_systems=remove_nontropical_systems,
                remove_tropical_systems=False,
                cyclone_id_strings=[cyclone_id_strings[i]] * this_num_times,
                target_times_unix_sec=target_times_by_cyclone_unix_sec[i]
            )
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


def _read_brightness_temp_1cyclone(
        example_table_xarray, table_rows_by_example, lag_times_sec,
        num_grid_rows=None, num_grid_columns=None):
    """Reads brightness temperatures for one cyclone.

    T = number of target times
    L = number of lag times
    m = number of rows in grid
    n = number of columns in grid

    :param example_table_xarray: xarray table.  Variable names and metadata
        therein should make the table self-explanatory.
    :param table_rows_by_example: length-T list, where each element is a 1-D
        numpy array of indices to satellite times needed for the given target
        time.  These are row indices into `example_table_xarray`.
    :param lag_times_sec: length-L numpy array of lag times.
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
        if not regular_grids:
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

        if regular_grids:
            grid_longitude_matrix_deg_e = (
                grid_longitude_matrix_deg_e[:, first_index:last_index, ...]
            )
        else:
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


def _read_satellite_data_1cyclone(
        input_file_name, lag_times_minutes, num_grid_rows, num_grid_columns,
        return_coords, target_times_unix_sec):
    """Reads satellite data for one cyclone.

    T = number of target times
    L = number of lag times
    m = number of rows in grid
    n = number of columns in grid

    :param input_file_name: Path to input file (will be read by
        `cira_ir_example_io.read_file`).
    :param lag_times_minutes: length-L numpy array of lag times for predictors.
    :param num_grid_rows: m in the above discussion.
    :param num_grid_columns: n in the above discussion.
    :param return_coords: Boolean flag.  If True, will return coordinates.
    :param target_times_unix_sec: length-T numpy array of target times.

    :return: data_dict: Dictionary with the following keys.  If
        `return_coords == False`, the last 2 keys will be None.

    data_dict["brightness_temp_matrix_kelvins"]: T-by-m-by-n-by-L-by-1 numpy
        array of brightness temperatures.
    data_dict["grid_spacings_low_res_km"]: length-T numpy array of grid
        spacings.
    data_dict["cyclone_center_latitudes_deg_n"]: length-T numpy array of
        center latitudes (deg north).
    data_dict["target_times_unix_sec"]: length-T numpy array of target times.
    data_dict["low_res_latitude_matrix_deg_n"]: T-by-m-by-L numpy array of
        latitudes (deg north).
    data_dict["low_res_longitude_matrix_deg_e"]: T-by-n-by-L numpy array of
        longitudes (deg east).

    :raises: ValueError: if any desired target times are missing from the input
        file.
    """

    lag_times_sec = lag_times_minutes * MINUTES_TO_SECONDS
    all_times_unix_sec = cira_ir_example_io.read_file(input_file_name).coords[
        CIRA_IR_TIME_DIM
    ].values

    table_rows_by_target_time = []

    for i in range(len(target_times_unix_sec)):
        these_predictor_times_unix_sec = (
            target_times_unix_sec[i] - lag_times_sec
        )

        if not numpy.all(numpy.isin(
                element=these_predictor_times_unix_sec,
                test_elements=all_times_unix_sec
        )):
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

        table_rows_by_target_time.append(these_rows)

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    example_table_xarray = cira_ir_example_io.read_file(input_file_name)

    (
        brightness_temp_matrix_kelvins,
        grid_latitude_matrix_deg_n,
        grid_longitude_matrix_deg_e
    ) = _read_brightness_temp_1cyclone(
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
        else:
            start_latitude_deg_n = grid_latitude_matrix_deg_n[
                i, i_start, j_start, -1
            ]
            start_longitude_deg_e = grid_longitude_matrix_deg_e[
                i, i_start, j_start, -1
            ]
            end_latitude_deg_n = grid_latitude_matrix_deg_n[
                i, i_start, j_end, -1
            ]
            end_longitude_deg_e = grid_longitude_matrix_deg_e[
                i, i_start, j_end, -1
            ]
            first_distance_metres = geodesic_object.inv(
                lons1=start_longitude_deg_e, lats1=start_latitude_deg_n,
                lons2=end_longitude_deg_e, lats2=end_latitude_deg_n
            )[2]

            end_latitude_deg_n = grid_latitude_matrix_deg_n[
                i, i_end, j_start, -1
            ]
            end_longitude_deg_e = grid_longitude_matrix_deg_e[
                i, i_end, j_start, -1
            ]
            second_distance_metres = geodesic_object.inv(
                lons1=start_longitude_deg_e, lats1=start_latitude_deg_n,
                lons2=end_longitude_deg_e, lats2=end_latitude_deg_n
            )[2]

            start_latitude_deg_n = grid_latitude_matrix_deg_n[
                i, i_end, j_end, -1
            ]
            start_longitude_deg_e = grid_longitude_matrix_deg_e[
                i, i_end, j_end, -1
            ]
            third_distance_metres = geodesic_object.inv(
                lons1=start_longitude_deg_e, lats1=start_latitude_deg_n,
                lons2=end_longitude_deg_e, lats2=end_latitude_deg_n
            )[2]

            end_latitude_deg_n = grid_latitude_matrix_deg_n[
                i, i_start, j_end, -1
            ]
            end_longitude_deg_e = grid_longitude_matrix_deg_e[
                i, i_start, j_end, -1
            ]
            fourth_distance_metres = geodesic_object.inv(
                lons1=start_longitude_deg_e, lats1=start_latitude_deg_n,
                lons2=end_longitude_deg_e, lats2=end_latitude_deg_n
            )[2]

            these_distances_km = METRES_TO_KM * numpy.array([
                first_distance_metres, second_distance_metres,
                third_distance_metres, fourth_distance_metres
            ])
            grid_spacings_km[i] = numpy.mean(these_distances_km)

        cyclone_center_latitudes_deg_n[i] = (
            0.5 * (start_latitude_deg_n + end_latitude_deg_n)
        )

    return {
        BRIGHTNESS_TEMPS_KEY: brightness_temp_matrix_kelvins,
        GRID_SPACINGS_KEY: grid_spacings_km,
        CENTER_LATITUDES_KEY: cyclone_center_latitudes_deg_n,
        TARGET_TIMES_KEY: target_times_unix_sec,
        LOW_RES_LATITUDES_KEY:
            grid_latitude_matrix_deg_n if return_coords else None,
        LOW_RES_LONGITUDES_KEY:
            grid_longitude_matrix_deg_e if return_coords else None
    }


def data_generator(option_dict):
    """Generates input data for neural net.

    E = batch size = number of examples after data augmentation
    L = number of lag times for predictors
    w = number of wavelengths
    m = number of rows in grid
    n = number of columns in grid
    F = number of scalar fields

    :param option_dict: Dictionary with the following keys.
    option_dict["satellite_dir_name"]: Name of directory with satellite data.
        Files therein will be found by `cira_ir_example_io.find_file` and read
        by `cira_ir_example_io.read_file`.
    option_dict["years"]: 1-D numpy array of years.  Will generate data only for
        cyclones in these years.
    option_dict["lag_times_minutes"]: length-L numpy array of lag times for
        predictors.
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
    option_dict["a_deck_file_name"]: Path to A-deck file, which is needed if
        `len(scalar_a_deck_field_names) > 0 or remove_nontropical_systems`.
        If A-deck file is not needed, you can make this None.

    :return: predictor_matrices: If predictors include scalars, this will be a
        list with [brightness_temp_matrix_kelvins, scalar_predictor_matrix].
        Otherwise, this will be a list with only the first item.

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

    option_dict[neural_net_utils.HIGH_RES_WAVELENGTHS_KEY] = numpy.array([])
    option_dict[neural_net_utils.LOW_RES_WAVELENGTHS_KEY] = numpy.array([11.2])
    option_dict[neural_net_utils.LAG_TIME_TOLERANCE_KEY] = 0
    option_dict[neural_net_utils.MAX_MISSING_LAG_TIMES_KEY] = 0
    option_dict[neural_net_utils.MAX_INTERP_GAP_KEY] = 0
    option_dict[neural_net_utils.SENTINEL_VALUE_KEY] = -10.
    option_dict[neural_net_utils.SEMANTIC_SEG_FLAG_KEY] = False
    option_dict[neural_net_utils.TARGET_SMOOOTHER_STDEV_KEY] = None

    option_dict = neural_net_utils.check_generator_args(option_dict)

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
    synoptic_times_only = option_dict[SYNOPTIC_TIMES_ONLY_KEY]
    scalar_a_deck_field_names = option_dict[SCALAR_A_DECK_FIELDS_KEY]
    remove_nontropical_systems = option_dict[REMOVE_NONTROPICAL_KEY]
    a_deck_file_name = option_dict[A_DECK_FILE_KEY]

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

    example_file_name_by_cyclone = [
        cira_ir_example_io.find_file(
            directory_name=example_dir_name, cyclone_id_string=c,
            prefer_zipped=False, allow_other_format=False,
            raise_error_if_missing=True
        )
        for c in cyclone_id_strings
    ]

    (
        target_times_by_cyclone_unix_sec, scalar_predictor_matrix_by_cyclone
    ) = _get_target_times_and_scalar_predictors(
        cyclone_id_strings=cyclone_id_strings,
        synoptic_times_only=synoptic_times_only,
        cira_ir_file_name_by_cyclone=example_file_name_by_cyclone,
        a_deck_file_name=a_deck_file_name,
        scalar_a_deck_field_names=scalar_a_deck_field_names,
        remove_nontropical_systems=remove_nontropical_systems
    )

    cyclone_index = 0

    while True:
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

            new_target_times_unix_sec = _choose_random_target_times(
                all_target_times_unix_sec=
                target_times_by_cyclone_unix_sec[cyclone_index] + 0,
                num_times_desired=num_examples_to_read
            )[0]

            if len(new_target_times_unix_sec) == 0:
                cyclone_index += 1
                continue

            data_dict = _read_satellite_data_1cyclone(
                input_file_name=example_file_name_by_cyclone[cyclone_index],
                lag_times_minutes=lag_times_minutes,
                num_grid_rows=num_grid_rows,
                num_grid_columns=num_grid_columns,
                return_coords=False,
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

            this_bt_matrix_kelvins = neural_net_utils.combine_lag_times_and_wavelengths(
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
                num_rows_to_keep=orig_num_grid_rows,
                num_columns_to_keep=orig_num_grid_columns,
                for_high_res=False
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
        if scalar_predictor_matrix is not None:
            predictor_matrices.append(scalar_predictor_matrix)

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
    """Trains neural net.

    :param model_object: See doc for `neural_net_training_fancy.train_model`.
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

    validation_keys_to_keep = [SATELLITE_DIRECTORY_KEY, YEARS_KEY]
    for this_key in list(training_option_dict.keys()):
        if this_key in validation_keys_to_keep:
            continue

        validation_option_dict[this_key] = training_option_dict[this_key]

    training_option_dict = neural_net_utils.check_generator_args(
        training_option_dict
    )
    validation_option_dict = neural_net_utils.check_generator_args(
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

    metafile_name = neural_net_utils.find_metafile(
        model_dir_name=output_dir_name, raise_error_if_missing=False
    )
    print('Writing metadata to: "{0:s}"...'.format(metafile_name))

    neural_net_utils.write_metafile(
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
        data_type_string=neural_net_utils.CIRA_IR_DATA_TYPE_STRING,
        train_with_shuffled_data=False
    )

    model_object.fit_generator(
        generator=training_generator,
        steps_per_epoch=num_training_batches_per_epoch,
        epochs=num_epochs, verbose=1, callbacks=list_of_callback_objects,
        validation_data=validation_generator,
        validation_steps=num_validation_batches_per_epoch
    )


def create_data(option_dict, cyclone_id_string, num_target_times):
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
    data_dict["low_res_latitude_matrix_deg_n"]: numpy array of latitudes (deg
        north).  If regular grids, this array has shape E x m x L.  If irregular
        grids, this array has shape E x m x n x L.
    data_dict["low_res_longitude_matrix_deg_e"]: numpy array of longitudes (deg
        east).  If regular grids, this array has shape E x n x L.  If irregular
        grids, this array has shape E x m x n x L.
    """

    error_checking.assert_is_integer(num_target_times)
    error_checking.assert_is_greater(num_target_times, 0)

    option_dict[neural_net_utils.HIGH_RES_WAVELENGTHS_KEY] = numpy.array([])
    option_dict[neural_net_utils.LOW_RES_WAVELENGTHS_KEY] = numpy.array([11.2])
    option_dict[neural_net_utils.LAG_TIME_TOLERANCE_KEY] = 0
    option_dict[neural_net_utils.MAX_MISSING_LAG_TIMES_KEY] = 0
    option_dict[neural_net_utils.MAX_INTERP_GAP_KEY] = 0
    option_dict[neural_net_utils.SENTINEL_VALUE_KEY] = -10.

    option_dict = neural_net_utils.check_generator_args(option_dict)

    example_dir_name = option_dict[SATELLITE_DIRECTORY_KEY]
    lag_times_minutes = option_dict[LAG_TIMES_KEY]
    num_grid_rows = option_dict[NUM_GRID_ROWS_KEY]
    num_grid_columns = option_dict[NUM_GRID_COLUMNS_KEY]
    data_aug_num_translations = option_dict[DATA_AUG_NUM_TRANS_KEY]
    data_aug_mean_translation_low_res_px = option_dict[DATA_AUG_MEAN_TRANS_KEY]
    data_aug_stdev_translation_low_res_px = (
        option_dict[DATA_AUG_STDEV_TRANS_KEY]
    )
    semantic_segmentation_flag = option_dict[SEMANTIC_SEG_FLAG_KEY]
    target_smoother_stdev_km = option_dict[TARGET_SMOOOTHER_STDEV_KEY]
    synoptic_times_only = option_dict[SYNOPTIC_TIMES_ONLY_KEY]
    scalar_a_deck_field_names = option_dict[SCALAR_A_DECK_FIELDS_KEY]
    remove_nontropical_systems = option_dict[REMOVE_NONTROPICAL_KEY]
    a_deck_file_name = option_dict[A_DECK_FILE_KEY]

    orig_num_grid_rows = num_grid_rows + 0
    orig_num_grid_columns = num_grid_columns + 0

    num_extra_rowcols = 2 * int(numpy.ceil(
        data_aug_mean_translation_low_res_px +
        5 * data_aug_stdev_translation_low_res_px
    ))
    num_grid_rows += num_extra_rowcols
    num_grid_columns += num_extra_rowcols

    example_file_name = cira_ir_example_io.find_file(
        directory_name=example_dir_name, cyclone_id_string=cyclone_id_string,
        prefer_zipped=False, allow_other_format=False,
        raise_error_if_missing=True
    )

    (
        all_target_times_unix_sec, all_scalar_predictor_matrix
    ) = _get_target_times_and_scalar_predictors(
        cyclone_id_strings=[cyclone_id_string],
        synoptic_times_only=synoptic_times_only,
        cira_ir_file_name_by_cyclone=[example_file_name],
        a_deck_file_name=a_deck_file_name,
        scalar_a_deck_field_names=scalar_a_deck_field_names,
        remove_nontropical_systems=remove_nontropical_systems
    )

    all_target_times_unix_sec = all_target_times_unix_sec[0]
    all_scalar_predictor_matrix = all_scalar_predictor_matrix[0]
    conservative_num_target_times = max([
        int(numpy.round(num_target_times * 1.25)),
        num_target_times + 2
    ])

    chosen_target_times_unix_sec = _choose_random_target_times(
        all_target_times_unix_sec=all_target_times_unix_sec + 0,
        num_times_desired=conservative_num_target_times
    )[0]

    data_dict = _read_satellite_data_1cyclone(
        input_file_name=example_file_name,
        lag_times_minutes=lag_times_minutes,
        num_grid_rows=num_grid_rows,
        num_grid_columns=num_grid_columns,
        return_coords=True,
        target_times_unix_sec=chosen_target_times_unix_sec
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
    grid_latitude_matrix_deg_n = (
        data_dict[LOW_RES_LATITUDES_KEY][:num_target_times, ...]
    )
    grid_longitude_matrix_deg_e = (
        data_dict[LOW_RES_LONGITUDES_KEY][:num_target_times, ...]
    )

    brightness_temp_matrix_kelvins = (
        neural_net_utils.combine_lag_times_and_wavelengths(
            brightness_temp_matrix_kelvins
        )
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
            num_rows_to_keep=orig_num_grid_rows,
            num_columns_to_keep=orig_num_grid_columns,
            for_high_res=False
        )
    )

    regular_grids = len(grid_latitude_matrix_deg_n.shape) == 3

    if regular_grids:
        grid_latitude_matrix_deg_n, grid_longitude_matrix_deg_e = (
            neural_net_utils.grid_coords_3d_to_4d(
                latitude_matrix_deg_n=grid_latitude_matrix_deg_n,
                longitude_matrix_deg_e=grid_longitude_matrix_deg_e
            )
        )

    grid_latitude_matrix_deg_n = data_augmentation.subset_grid_after_data_aug(
        data_matrix=grid_latitude_matrix_deg_n,
        num_rows_to_keep=orig_num_grid_rows,
        num_columns_to_keep=orig_num_grid_columns,
        for_high_res=False
    )
    grid_longitude_matrix_deg_e = data_augmentation.subset_grid_after_data_aug(
        data_matrix=grid_longitude_matrix_deg_e,
        num_rows_to_keep=orig_num_grid_rows,
        num_columns_to_keep=orig_num_grid_columns,
        for_high_res=False
    )

    if regular_grids:
        grid_latitude_matrix_deg_n = grid_latitude_matrix_deg_n[:, :, 0, :]
        grid_longitude_matrix_deg_e = grid_longitude_matrix_deg_e[:, 0, :, :]

    grid_spacings_km = numpy.repeat(
        grid_spacings_km, repeats=data_aug_num_translations
    )
    cyclone_center_latitudes_deg_n = numpy.repeat(
        cyclone_center_latitudes_deg_n, repeats=data_aug_num_translations
    )
    target_times_unix_sec = numpy.repeat(
        target_times_unix_sec, repeats=data_aug_num_translations
    )
    grid_latitude_matrix_deg_n = numpy.repeat(
        grid_latitude_matrix_deg_n, repeats=data_aug_num_translations, axis=0
    )
    grid_longitude_matrix_deg_e = numpy.repeat(
        grid_longitude_matrix_deg_e, repeats=data_aug_num_translations, axis=0
    )
    if scalar_predictor_matrix is not None:
        scalar_predictor_matrix = numpy.repeat(
            scalar_predictor_matrix, axis=0,
            repeats=data_aug_num_translations
        )

    predictor_matrices = [brightness_temp_matrix_kelvins]
    if scalar_predictor_matrix is not None:
        predictor_matrices.append(scalar_predictor_matrix)

    if semantic_segmentation_flag:
        target_matrix = neural_net_utils.make_targets_for_semantic_seg(
            row_translations_px=row_translations_low_res_px,
            column_translations_px=column_translations_low_res_px,
            grid_spacings_km=grid_spacings_km,
            cyclone_center_latitudes_deg_n=cyclone_center_latitudes_deg_n,
            gaussian_smoother_stdev_km=target_smoother_stdev_km,
            num_grid_rows=orig_num_grid_rows,
            num_grid_columns=orig_num_grid_columns
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
        LOW_RES_LATITUDES_KEY: grid_latitude_matrix_deg_n,
        LOW_RES_LONGITUDES_KEY: grid_longitude_matrix_deg_e
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

    option_dict[neural_net_utils.HIGH_RES_WAVELENGTHS_KEY] = numpy.array([])
    option_dict[neural_net_utils.LOW_RES_WAVELENGTHS_KEY] = numpy.array([11.2])
    option_dict[neural_net_utils.LAG_TIME_TOLERANCE_KEY] = 0
    option_dict[neural_net_utils.MAX_MISSING_LAG_TIMES_KEY] = 0
    option_dict[neural_net_utils.MAX_INTERP_GAP_KEY] = 0
    option_dict[neural_net_utils.SENTINEL_VALUE_KEY] = -10.
    option_dict[neural_net_utils.DATA_AUG_NUM_TRANS_KEY] = 5
    option_dict[neural_net_utils.DATA_AUG_MEAN_TRANS_KEY] = 10.
    option_dict[neural_net_utils.DATA_AUG_STDEV_TRANS_KEY] = 10.

    option_dict = neural_net_utils.check_generator_args(option_dict)

    # Do actual stuff.
    example_dir_name = option_dict[SATELLITE_DIRECTORY_KEY]
    lag_times_minutes = option_dict[LAG_TIMES_KEY]
    num_grid_rows = option_dict[NUM_GRID_ROWS_KEY]
    num_grid_columns = option_dict[NUM_GRID_COLUMNS_KEY]
    semantic_segmentation_flag = option_dict[SEMANTIC_SEG_FLAG_KEY]
    target_smoother_stdev_km = option_dict[TARGET_SMOOOTHER_STDEV_KEY]
    a_deck_file_name = option_dict[A_DECK_FILE_KEY]
    scalar_a_deck_field_names = option_dict[SCALAR_A_DECK_FIELDS_KEY]
    remove_nontropical_systems = option_dict[REMOVE_NONTROPICAL_KEY]

    orig_num_grid_rows = num_grid_rows + 0
    orig_num_grid_columns = num_grid_columns + 0

    num_extra_rowcols = 2 * max([
        numpy.max(numpy.absolute(row_translations_low_res_px)),
        numpy.max(numpy.absolute(column_translations_low_res_px))
    ])
    num_grid_rows += num_extra_rowcols
    num_grid_columns += num_extra_rowcols

    example_file_name = cira_ir_example_io.find_file(
        directory_name=example_dir_name, cyclone_id_string=cyclone_id_string,
        prefer_zipped=False, allow_other_format=False,
        raise_error_if_missing=True
    )
    unique_target_times_unix_sec = numpy.unique(target_times_unix_sec)

    data_dict = _read_satellite_data_1cyclone(
        input_file_name=example_file_name,
        lag_times_minutes=lag_times_minutes,
        num_grid_rows=num_grid_rows,
        num_grid_columns=num_grid_columns,
        return_coords=True, target_times_unix_sec=unique_target_times_unix_sec
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
        this_num_times = len(data_dict[TARGET_TIMES_KEY])

        scalar_predictor_matrix = neural_net_utils.read_scalar_data(
            a_deck_file_name=a_deck_file_name,
            field_names=scalar_a_deck_field_names,
            remove_nontropical_systems=remove_nontropical_systems,
            remove_tropical_systems=False,
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
    grid_latitude_matrix_deg_n = dd[LOW_RES_LATITUDES_KEY][idxs, ...]
    grid_longitude_matrix_deg_e = dd[LOW_RES_LONGITUDES_KEY][idxs, ...]

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
    grid_latitude_matrix_deg_n = grid_latitude_matrix_deg_n[idxs, ...]
    grid_longitude_matrix_deg_e = grid_longitude_matrix_deg_e[idxs, ...]

    brightness_temp_matrix_kelvins = (
        neural_net_utils.combine_lag_times_and_wavelengths(
            brightness_temp_matrix_kelvins
        )
    )

    _, brightness_temp_matrix_kelvins = (
        data_augmentation.augment_data_specific_trans(
            bidirectional_reflectance_matrix=None,
            brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
            row_translations_low_res_px=row_translations_low_res_px,
            column_translations_low_res_px=column_translations_low_res_px,
            sentinel_value=-10.
        )
    )

    brightness_temp_matrix_kelvins = (
        data_augmentation.subset_grid_after_data_aug(
            data_matrix=brightness_temp_matrix_kelvins,
            num_rows_to_keep=orig_num_grid_rows,
            num_columns_to_keep=orig_num_grid_columns,
            for_high_res=False
        )
    )

    regular_grids = len(grid_latitude_matrix_deg_n.shape) == 3

    if regular_grids:
        grid_latitude_matrix_deg_n, grid_longitude_matrix_deg_e = (
            neural_net_utils.grid_coords_3d_to_4d(
                latitude_matrix_deg_n=grid_latitude_matrix_deg_n,
                longitude_matrix_deg_e=grid_longitude_matrix_deg_e
            )
        )

    grid_latitude_matrix_deg_n = data_augmentation.subset_grid_after_data_aug(
        data_matrix=grid_latitude_matrix_deg_n,
        num_rows_to_keep=orig_num_grid_rows,
        num_columns_to_keep=orig_num_grid_columns,
        for_high_res=False
    )
    grid_longitude_matrix_deg_e = data_augmentation.subset_grid_after_data_aug(
        data_matrix=grid_longitude_matrix_deg_e,
        num_rows_to_keep=orig_num_grid_rows,
        num_columns_to_keep=orig_num_grid_columns,
        for_high_res=False
    )

    if regular_grids:
        grid_latitude_matrix_deg_n = grid_latitude_matrix_deg_n[:, :, 0, :]
        grid_longitude_matrix_deg_e = grid_longitude_matrix_deg_e[:, 0, :, :]

    predictor_matrices = [brightness_temp_matrix_kelvins]
    if scalar_predictor_matrix is not None:
        predictor_matrices.append(scalar_predictor_matrix)

    if semantic_segmentation_flag:
        target_matrix = neural_net_utils.make_targets_for_semantic_seg(
            row_translations_px=row_translations_low_res_px,
            column_translations_px=column_translations_low_res_px,
            grid_spacings_km=grid_spacings_km,
            cyclone_center_latitudes_deg_n=cyclone_center_latitudes_deg_n,
            gaussian_smoother_stdev_km=target_smoother_stdev_km,
            num_grid_rows=orig_num_grid_rows,
            num_grid_columns=orig_num_grid_columns
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
        LOW_RES_LATITUDES_KEY: grid_latitude_matrix_deg_n,
        LOW_RES_LONGITUDES_KEY: grid_longitude_matrix_deg_e
    }
