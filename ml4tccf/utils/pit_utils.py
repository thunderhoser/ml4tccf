"""Helper methods for computing PIT and PIT histograms.

PIT = probability integral transform
"""

import numpy
import xarray
from scipy.stats import percentileofscore
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4tccf.io import prediction_io
from ml4tccf.utils import scalar_evaluation
from ml4tccf.utils import scalar_prediction_utils as prediction_utils
from ml4tccf.outside_code import multivariate_cdf

TOLERANCE = 1e-6
KM_TO_METRES = 1000.

MAX_PIT_FOR_LOW_BINS = 0.3
MIN_PIT_FOR_HIGH_BINS = 0.7

X_OFFSET_NAME = 'x_offset_metres'
Y_OFFSET_NAME = 'y_offset_metres'
OFFSET_DIRECTION_NAME = 'offset_direction_deg'
OFFSET_DISTANCE_NAME = 'offset_distance_metres'
TARGET_FIELD_NAMES = [
    Y_OFFSET_NAME, X_OFFSET_NAME, OFFSET_DIRECTION_NAME, OFFSET_DISTANCE_NAME
]

BIN_EDGES_KEY = 'bin_edges'
BIN_COUNTS_KEY = 'bin_counts'
PITD_KEY = 'pitd_value'
PERFECT_PITD_KEY = 'perfect_pitd_value'
LOW_BIN_BIAS_KEY = 'low_bin_pit_bias'
MIDDLE_BIN_BIAS_KEY = 'middle_bin_pit_bias'
HIGH_BIN_BIAS_KEY = 'high_bin_pit_bias'

TARGET_FIELD_DIM = 'target_field'
BIN_CENTER_DIM = 'bin_center'
BIN_EDGE_DIM = 'bin_edge'

MODEL_FILE_KEY = 'model_file_name'
ISOTONIC_MODEL_FILE_KEY = 'isotonic_model_file_name'
PREDICTION_FILES_KEY = 'prediction_file_names'

PIT_DEVIATION_KEY = 'pit_deviation'
PERFECT_PIT_DEVIATION_KEY = 'perfect_pit_deviation'
BIN_COUNT_KEY = 'bin_count'
# LOW_BIN_BIAS_KEY = 'low_bin_pit_bias'
# MIDDLE_BIN_BIAS_KEY = 'middle_bin_pit_bias'
# HIGH_BIN_BIAS_KEY = 'high_bin_pit_bias'


def _get_low_mid_hi_bins(bin_edges):
    """Returns indices for low-PIT, medium-PIT, and high-PIT bins.

    B = number of bins

    :param bin_edges: length-(B + 1) numpy array of bin edges, sorted in
        ascending order.
    :return: low_bin_indices: 1-D numpy array with array indices for low-PIT
        bins.
    :return: middle_bin_indices: 1-D numpy array with array indices for
        medium-PIT bins.
    :return: high_bin_indices: 1-D numpy array with array indices for high-PIT
        bins.
    """

    num_bins = len(bin_edges) - 1

    these_diffs = bin_edges - MAX_PIT_FOR_LOW_BINS
    these_diffs[these_diffs > TOLERANCE] = numpy.inf
    max_index_for_low_bins = numpy.argmin(numpy.absolute(these_diffs)) - 1
    max_index_for_low_bins = max([max_index_for_low_bins, 0])

    low_bin_indices = numpy.linspace(
        0, max_index_for_low_bins, num=max_index_for_low_bins + 1, dtype=int
    )

    these_diffs = MIN_PIT_FOR_HIGH_BINS - bin_edges
    these_diffs[these_diffs > TOLERANCE] = numpy.inf
    min_index_for_high_bins = numpy.argmin(numpy.absolute(these_diffs))
    min_index_for_high_bins = min([min_index_for_high_bins, num_bins - 1])

    high_bin_indices = numpy.linspace(
        min_index_for_high_bins, num_bins - 1,
        num=num_bins - min_index_for_high_bins, dtype=int
    )

    middle_bin_indices = numpy.linspace(
        0, num_bins - 1, num=num_bins, dtype=int
    )
    middle_bin_indices = numpy.array(list(
        set(middle_bin_indices.tolist())
        - set(low_bin_indices.tolist())
        - set(high_bin_indices.tolist())
    ))

    return low_bin_indices, middle_bin_indices, high_bin_indices


def _pit_values_to_histogram(pit_values, num_bins):
    """Creates histogram from PIT values.

    :param pit_values: 1-D numpy array of PIT values.
    :param num_bins: Number of bins for histogram.
    :return: result_dict: See doc for `_get_histogram_one_var`.
    """

    num_examples = len(pit_values)

    bin_edges = numpy.linspace(0, 1, num=num_bins + 1, dtype=float)
    indices_example_to_bin = numpy.digitize(
        x=pit_values, bins=bin_edges, right=False
    ) - 1
    indices_example_to_bin[indices_example_to_bin < 0] = 0
    indices_example_to_bin[indices_example_to_bin >= num_bins] = num_bins - 1

    used_bin_indices, used_bin_counts = numpy.unique(
        indices_example_to_bin, return_counts=True
    )
    bin_counts = numpy.full(num_bins, 0, dtype=int)
    bin_counts[used_bin_indices] = used_bin_counts

    bin_frequencies = bin_counts.astype(float) / num_examples
    perfect_bin_frequency = 1. / num_bins

    pitd_value = numpy.sqrt(
        numpy.mean((bin_frequencies - perfect_bin_frequency) ** 2)
    )
    perfect_pitd_value = numpy.sqrt(
        (1. - perfect_bin_frequency) / (num_examples * num_bins)
    )

    low_bin_indices, middle_bin_indices, high_bin_indices = (
        _get_low_mid_hi_bins(bin_edges)
    )

    low_bin_pit_bias = numpy.mean(
        bin_frequencies[low_bin_indices] - perfect_bin_frequency
    )
    middle_bin_pit_bias = numpy.mean(
        bin_frequencies[middle_bin_indices] - perfect_bin_frequency
    )
    high_bin_pit_bias = numpy.mean(
        bin_frequencies[high_bin_indices] - perfect_bin_frequency
    )

    return {
        BIN_EDGES_KEY: bin_edges,
        BIN_COUNTS_KEY: bin_counts,
        PITD_KEY: pitd_value,
        PERFECT_PITD_KEY: perfect_pitd_value,
        LOW_BIN_BIAS_KEY: low_bin_pit_bias,
        MIDDLE_BIN_BIAS_KEY: middle_bin_pit_bias,
        HIGH_BIN_BIAS_KEY: high_bin_pit_bias
    }


def _get_histogram_euclidean(target_matrix, prediction_matrix, num_bins):
    """Computes PIT histogram for Euclidean offset.

    E = number of examples
    S = number of ensemble members

    :param target_matrix: E-by-2 numpy array of actual values, where
        target_matrix[:, 0] contains x-displacements and target_matrix[:, 1]
        contains y-displacements.
    :param prediction_matrix: E-by-2-by-S numpy array of predicted values,
        where prediction_matrix[:, 0, :] contains x-displacements and
        prediction_matrix[:, 1, :] contains y-displacements.
    :param num_bins: See doc for `_get_histogram_one_var`.
    :return: result_dict: Same.
    """

    num_examples = prediction_matrix.shape[0]
    ensemble_size = prediction_matrix.shape[2]
    pit_values = numpy.full(num_examples, numpy.nan)

    for i in range(num_examples):
        this_data_matrix = numpy.concatenate((
            numpy.transpose(prediction_matrix[i, ...]),
            target_matrix[[i], :]
        ), axis=0)

        pit_values[i] = multivariate_cdf._ecdf_mv(
            data=this_data_matrix, method='seq', use_ranks=True
        )[0][-1]

    pit_values = (pit_values - 1.) / ensemble_size
    assert numpy.all(pit_values >= 0.)
    assert numpy.all(pit_values <= 1.)

    return _pit_values_to_histogram(pit_values=pit_values, num_bins=num_bins)


def _get_histogram_one_var(target_values, prediction_matrix, num_bins,
                           is_var_direction):
    """Computes PIT histogram for one variable.

    E = number of examples
    S = number of ensemble members
    B = number of bins in histogram

    :param target_values: length-E numpy array of actual values.
    :param prediction_matrix: E-by-S numpy array of predicted values.
    :param num_bins: Number of bins in histogram.
    :param is_var_direction: Boolean flag.  If True (False), this method assumes
        that the target variable is direction (x or y).
    :return: result_dict: Dictionary with the following keys.
    result_dict["bin_edges"]: length-(B + 1) numpy array of bin edges (ranging
        from 0...1, because PIT ranges from 0...1).
    result_dict["bin_counts"]: length-B numpy array with number of examples in
        each bin.
    result_dict["pitd_value"]: Value of the calibration-deviation metric (PITD).
    result_dict["perfect_pitd_value"]: Minimum expected PITD value.
    result_dict["low_bin_pit_bias"]: PIT bias for low bins, i.e., PIT values of
        [0, 0.3).
    result_dict["middle_bin_pit_bias"]: PIT bias for middle bins, i.e., PIT
        values of [0.3, 0.7).
    result_dict["high_bin_pit_bias"]: PIT bias for high bins, i.e., PIT values
        of [0.7, 1.0].
    """

    # TODO(thunderhoser): I still need to figure this out.
    if is_var_direction:
        return None

    num_examples = len(target_values)
    pit_values = numpy.full(num_examples, numpy.nan)

    for i in range(num_examples):
        pit_values[i] = 0.01 * percentileofscore(
            a=prediction_matrix[i, :], score=target_values[i], kind='mean'
        )

    return _pit_values_to_histogram(pit_values=pit_values, num_bins=num_bins)


def get_results_all_vars(prediction_file_names, num_bins):
    """Computes PIT histogram for each target variable.

    :param prediction_file_names: 1-D list of paths to prediction files (will be
        read by `prediction_io.read_file`).
    :param num_bins: Number of bins for histogram (same for every target
        variable).
    :return: result_table_xarray: xarray table with results (variable and
        dimension names should make the table self-explanatory).
    """

    error_checking.assert_is_integer(num_bins)
    error_checking.assert_is_geq(num_bins, 10)
    error_checking.assert_is_leq(num_bins, 1000)

    # Read predictions.
    num_files = len(prediction_file_names)
    prediction_tables_xarray = [None] * num_files

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(prediction_file_names[i]))
        prediction_tables_xarray[i] = prediction_io.read_file(
            prediction_file_names[i]
        )

    prediction_table_xarray = prediction_utils.concat_over_examples(
        prediction_tables_xarray
    )
    pt = prediction_table_xarray

    grid_spacings_km = pt[prediction_utils.GRID_SPACING_KEY].values

    prediction_matrix = numpy.stack((
        pt[prediction_utils.PREDICTED_ROW_OFFSET_KEY].values,
        pt[prediction_utils.PREDICTED_COLUMN_OFFSET_KEY].values
    ), axis=1)
    prediction_matrix *= numpy.expand_dims(grid_spacings_km, axis=(1, 2))

    target_matrix = numpy.transpose(numpy.vstack((
        grid_spacings_km * pt[prediction_utils.ACTUAL_ROW_OFFSET_KEY].values,
        grid_spacings_km * pt[prediction_utils.ACTUAL_COLUMN_OFFSET_KEY].values
    )))

    prediction_matrix *= KM_TO_METRES
    target_matrix *= KM_TO_METRES

    # Add vector directions.
    prediction_matrix = numpy.concatenate((
        prediction_matrix,
        scalar_evaluation.get_offset_angles(
            x_offsets=prediction_matrix[:, [1], :],
            y_offsets=prediction_matrix[:, [0], :]
        )
    ), axis=1)

    target_matrix = numpy.hstack((
        target_matrix,
        scalar_evaluation.get_offset_angles(
            x_offsets=target_matrix[:, [1]],
            y_offsets=target_matrix[:, [0]]
        )
    ))

    # Create result table.
    num_targets = len(TARGET_FIELD_NAMES)

    main_data_dict = {
        PIT_DEVIATION_KEY: (
            (TARGET_FIELD_DIM,), numpy.full(num_targets, numpy.nan)
        ),
        PERFECT_PIT_DEVIATION_KEY: (
            (TARGET_FIELD_DIM,), numpy.full(num_targets, numpy.nan)
        ),
        BIN_COUNT_KEY: (
            (TARGET_FIELD_DIM, BIN_CENTER_DIM),
            numpy.full((num_targets, num_bins), -1, dtype=int)
        ),
        LOW_BIN_BIAS_KEY: (
            (TARGET_FIELD_DIM,), numpy.full(num_targets, numpy.nan)
        ),
        MIDDLE_BIN_BIAS_KEY: (
            (TARGET_FIELD_DIM,), numpy.full(num_targets, numpy.nan)
        ),
        HIGH_BIN_BIAS_KEY: (
            (TARGET_FIELD_DIM,), numpy.full(num_targets, numpy.nan)
        )
    }

    bin_edges = numpy.linspace(0, 1, num=num_bins + 1, dtype=float)
    bin_centers = bin_edges[:-1] + numpy.diff(bin_edges) / 2

    metadata_dict = {
        TARGET_FIELD_DIM: TARGET_FIELD_NAMES,
        BIN_CENTER_DIM: bin_centers,
        BIN_EDGE_DIM: bin_edges
    }

    result_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )
    result_table_xarray.attrs[MODEL_FILE_KEY] = (
        prediction_table_xarray.attrs[prediction_utils.MODEL_FILE_KEY]
    )
    result_table_xarray.attrs[ISOTONIC_MODEL_FILE_KEY] = (
        prediction_table_xarray.attrs[prediction_utils.ISOTONIC_MODEL_FILE_KEY]
    )
    result_table_xarray.attrs[PREDICTION_FILES_KEY] = ' '.join([
        '{0:s}'.format(f) for f in prediction_file_names
    ])

    xy_indices = numpy.array([
        TARGET_FIELD_NAMES.index(X_OFFSET_NAME),
        TARGET_FIELD_NAMES.index(Y_OFFSET_NAME)
    ], dtype=int)

    for j in xy_indices:
        print('Computing PIT histogram for "{0:s}"...'.format(
            TARGET_FIELD_NAMES[j]
        ))

        this_result_dict = _get_histogram_one_var(
            target_values=target_matrix[:, j],
            prediction_matrix=prediction_matrix[:, j, :],
            num_bins=num_bins, is_var_direction=False
        )

        result_table_xarray[PIT_DEVIATION_KEY].values[j] = (
            this_result_dict[PITD_KEY]
        )
        result_table_xarray[PERFECT_PIT_DEVIATION_KEY].values[j] = (
            this_result_dict[PERFECT_PITD_KEY]
        )
        result_table_xarray[LOW_BIN_BIAS_KEY].values[j] = (
            this_result_dict[LOW_BIN_BIAS_KEY]
        )
        result_table_xarray[MIDDLE_BIN_BIAS_KEY].values[j] = (
            this_result_dict[MIDDLE_BIN_BIAS_KEY]
        )
        result_table_xarray[HIGH_BIN_BIAS_KEY].values[j] = (
            this_result_dict[HIGH_BIN_BIAS_KEY]
        )
        result_table_xarray[BIN_COUNT_KEY].values[j, :] = (
            this_result_dict[BIN_COUNTS_KEY]
        )

    distance_indices = numpy.array(
        [TARGET_FIELD_NAMES.index(OFFSET_DISTANCE_NAME)], dtype=int
    )

    for j in distance_indices:
        print('Computing PIT histogram for "{0:s}"...'.format(
            TARGET_FIELD_NAMES[j]
        ))

        this_result_dict = _get_histogram_euclidean(
            target_matrix=target_matrix[:, xy_indices],
            prediction_matrix=prediction_matrix[:, xy_indices, :],
            num_bins=num_bins
        )

        result_table_xarray[PIT_DEVIATION_KEY].values[j] = (
            this_result_dict[PITD_KEY]
        )
        result_table_xarray[PERFECT_PIT_DEVIATION_KEY].values[j] = (
            this_result_dict[PERFECT_PITD_KEY]
        )
        result_table_xarray[LOW_BIN_BIAS_KEY].values[j] = (
            this_result_dict[LOW_BIN_BIAS_KEY]
        )
        result_table_xarray[MIDDLE_BIN_BIAS_KEY].values[j] = (
            this_result_dict[MIDDLE_BIN_BIAS_KEY]
        )
        result_table_xarray[HIGH_BIN_BIAS_KEY].values[j] = (
            this_result_dict[HIGH_BIN_BIAS_KEY]
        )
        result_table_xarray[BIN_COUNT_KEY].values[j, :] = (
            this_result_dict[BIN_COUNTS_KEY]
        )

    return result_table_xarray


def write_results(result_table_xarray, netcdf_file_name):
    """Writes PIT histogram for each target variable to NetCDF file.

    :param result_table_xarray: xarray table in format returned by
        `get_results_all_vars`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    result_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def read_results(netcdf_file_name):
    """Reads PIT histogram for each target variable from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: result_table_xarray: xarray table.  Documentation in the
        xarray table should make values self-explanatory.
    """

    result_table_xarray = xarray.open_dataset(netcdf_file_name)
    if ISOTONIC_MODEL_FILE_KEY not in result_table_xarray.attrs:
        result_table_xarray.attrs[ISOTONIC_MODEL_FILE_KEY] = None

    return result_table_xarray
