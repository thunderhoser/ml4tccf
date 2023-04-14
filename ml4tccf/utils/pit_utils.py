"""Helper methods for computing PIT and PIT histograms.

PIT = probability integral transform
"""

import numpy
from scipy.stats import percentileofscore
from ml4tccf.outside_code import multivariate_cdf

TOLERANCE = 1e-6

MAX_PIT_FOR_LOW_BINS = 0.3
MIN_PIT_FOR_HIGH_BINS = 0.7

BIN_EDGES_KEY = 'bin_edges'
BIN_COUNTS_KEY = 'bin_counts'
PITD_KEY = 'pitd_value'
PERFECT_PITD_KEY = 'perfect_pitd_value'
LOW_BIN_BIAS_KEY = 'low_bin_pit_bias'
MIDDLE_BIN_BIAS_KEY = 'middle_bin_pit_bias'
HIGH_BIN_BIAS_KEY = 'high_bin_pit_bias'


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

        # TODO(thunderhoser): Not sure if I'm doing this correctly.  Maybe I should just roll my own?  Eh, probably not.  But I need to look at the subtraction/division thing again.
        pit_values[i] = multivariate_cdf._ecdf_mv(
            data=this_data_matrix, method='seq', use_ranks=True
        )[0][-1]

    pit_values = (pit_values - 1.) / ensemble_size
    assert numpy.all(pit_values >= 0.)
    assert numpy.all(pit_values <= 1.)

    # TODO(thunderhoser): Left off here!


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