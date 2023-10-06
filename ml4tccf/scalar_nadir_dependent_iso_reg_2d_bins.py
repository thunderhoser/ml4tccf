"""Isotonic regression with scalar target variables.

This allows for nadir-dependent isotonic-regression models with 2-D bins.  In
other words, for every 2-D bin of nadir-relative coordinates, there is a
different pair of isotonic-regression models -- one to correct the x-coordinate
of the TC center and one to correct the y-coordinate.
"""

import os
import sys
import pickle
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import grids
import file_system_utils
import error_checking
import extended_best_track_io as ebtrk_io
import misc_utils
import scalar_prediction_utils as prediction_utils
import scalar_isotonic_regression as basic_iso_reg
import scalar_nadir_dependent_iso_regression as iso_reg_1d_bins

METRES_TO_KM = 0.001


def _linear_bin_index_to_2d(linear_index, nadir_relative_x_cutoffs_metres,
                            nadir_relative_y_cutoffs_metres):
    """Converts linear bin index to row and column indices.

    This is non-trivial, because I do not traverse the 2-D array in the normal
    way -- i.e., row-major, where I go across the rows and then down the
    columns.  Instead, I traverse the array along diagonals, from the
    top-left to bottom-right.

    There is a reason I traverse the array in such a weird way: sometimes I need
    to glom together adjacent bins for training, because a single bin does not
    have enough samples.  With 1-D bins, finding adjacent bins is easy.  But
    with 2-D bins, finding adjacent bins is a little more tricky.  The relevant
    bins could be in the same row, in the same column, or neither.

    :param linear_index: Linear index into 2-D bin array.
    :param nadir_relative_x_cutoffs_metres: See doc for `_find_examples_in_bin`.
    :param nadir_relative_y_cutoffs_metres: Same.
    :return: row_index: Row index into 2-D bin array.
    :return: column_index: Column index into 2-D bin array.
    """

    finite_x_cutoffs_metres = nadir_relative_x_cutoffs_metres[1:-1]
    finite_x_cutoffs_metres = numpy.concatenate((
        finite_x_cutoffs_metres[0] - numpy.diff(finite_x_cutoffs_metres[:2]),
        finite_x_cutoffs_metres,
        finite_x_cutoffs_metres[-1] + numpy.diff(finite_x_cutoffs_metres[-2:])
    ))

    finite_y_cutoffs_metres = nadir_relative_y_cutoffs_metres[1:-1]
    finite_y_cutoffs_metres = numpy.concatenate((
        finite_y_cutoffs_metres[0] - numpy.diff(finite_y_cutoffs_metres[:2]),
        finite_y_cutoffs_metres,
        finite_y_cutoffs_metres[-1] + numpy.diff(finite_y_cutoffs_metres[-2:])
    ))

    x_bin_centers_metres = (
        0.5 * (finite_x_cutoffs_metres[:-1] + finite_x_cutoffs_metres[1:])
    )
    y_bin_centers_metres = (
        0.5 * (finite_y_cutoffs_metres[:-1] + finite_y_cutoffs_metres[1:])
    )

    x_bin_center_matrix_metres, y_bin_center_matrix_metres = (
        grids.xy_vectors_to_matrices(
            x_unique_metres=x_bin_centers_metres,
            y_unique_metres=y_bin_centers_metres
        )
    )

    dist_from_origin_matrix_metres = numpy.sqrt(
        (x_bin_center_matrix_metres - x_bin_centers_metres[0]) ** 2 +
        (y_bin_center_matrix_metres - y_bin_centers_metres[0]) ** 2
    )
    sort_indices_linear = numpy.argsort(
        numpy.ravel(dist_from_origin_matrix_metres)
    )

    row_index, column_index = numpy.unravel_index(
        sort_indices_linear[linear_index],
        shape=dist_from_origin_matrix_metres.shape
    )

    return row_index, column_index


def _find_examples_in_bin(
        nadir_relative_prediction_xs_metres,
        nadir_relative_prediction_ys_metres,
        nadir_relative_x_cutoffs_metres, nadir_relative_y_cutoffs_metres,
        linear_bin_index, verbose):
    """Finds examples in one 2-D bin of nadir-relative coordinates.

    E = number of examples
    M = number of bins for nadir-relative y-coordinate
    N = number of bins for nadir-relative x-coordinate

    :param nadir_relative_prediction_xs_metres: length-E numpy array with
        nadir-relative x-coords of predictions.
    :param nadir_relative_prediction_ys_metres: length-E numpy array with
        nadir-relative y-coords of predictions.
    :param nadir_relative_x_cutoffs_metres: length-(N + 1) numpy array with bin
        cutoffs for nadir-relative x-coord.
    :param nadir_relative_y_cutoffs_metres: length-(M + 1) numpy array with bin
        cutoffs for nadir-relative y-coord.
    :param linear_bin_index: Linear index of 2-D bin.  This will be converted to
        row and column by `_linear_bin_index_to_2d`.
    :param verbose: Boolean flag.
    :return: example_indices: 1-D numpy array with indices of examples in bin.
        These are indices into the arrays `nadir_relative_prediction_xs_metres`
        and `nadir_relative_prediction_ys_metres`.
    """

    i, j = _linear_bin_index_to_2d(
        linear_index=linear_bin_index,
        nadir_relative_x_cutoffs_metres=nadir_relative_x_cutoffs_metres,
        nadir_relative_y_cutoffs_metres=nadir_relative_y_cutoffs_metres
    )

    in_row_flags = numpy.logical_and(
        nadir_relative_prediction_ys_metres >=
        nadir_relative_y_cutoffs_metres[i],
        nadir_relative_prediction_ys_metres <
        nadir_relative_y_cutoffs_metres[i + 1]
    )
    in_column_flags = numpy.logical_and(
        nadir_relative_prediction_xs_metres >=
        nadir_relative_x_cutoffs_metres[j],
        nadir_relative_prediction_xs_metres <
        nadir_relative_x_cutoffs_metres[j + 1]
    )
    example_indices = numpy.where(numpy.logical_and(
        in_row_flags, in_column_flags
    ))[0]

    if not verbose:
        return example_indices

    print((
        'Found {0:d} examples with nadir-relative x in '
        '[{1:.0f}, {2:.0f}) km and nadir-relative y in '
        '[{3:.0f}, {4:.0f}) km.'
    ).format(
        len(example_indices),
        METRES_TO_KM * nadir_relative_x_cutoffs_metres[j],
        METRES_TO_KM * nadir_relative_x_cutoffs_metres[j + 1],
        METRES_TO_KM * nadir_relative_y_cutoffs_metres[i],
        METRES_TO_KM * nadir_relative_y_cutoffs_metres[i + 1]
    ))

    return example_indices


def train_models(
        prediction_table_xarray, nadir_relative_x_cutoffs_metres,
        nadir_relative_y_cutoffs_metres, ebtrk_file_name,
        min_training_sample_size):
    """Trains one pair of IR models for every 2-D bin of nadir-relative coords.

    M = number of bins for nadir-relative y-coordinate
    N = number of bins for nadir-relative x-coordinate

    :param prediction_table_xarray: xarray table in format returned by
        `prediction_io.read_file`.
    :param nadir_relative_x_cutoffs_metres: numpy array of category cutoffs for
        nadir-relative x-coordinate.  Please leave -inf and +inf out of this
        list, as they will be added automatically.  Thus, the array should have
        length N - 1.
    :param nadir_relative_y_cutoffs_metres: numpy array of category cutoffs for
        nadir-relative y-coordinate.  Please leave -inf and +inf out of this
        list, as they will be added automatically.  Thus, the array should have
        length M - 1.
    :param ebtrk_file_name: Name of file with extended best-track data (will be
        read by `extended_best_track_io.read_file`).  This file will be used to
        find nadir-relative coordinates for every TC object.
    :param min_training_sample_size: Minimum number of training examples for one
        model.
    :return: x_coord_model_matrix: M-by-N numpy array of models (instances of
        `sklearn.isotonic.IsotonicRegression`) for bias-correcting x-coordinate
        of TC center.
    :return: y_coord_model_matrix: M-by-N numpy array of models (instances of
        `sklearn.isotonic.IsotonicRegression`) for bias-correcting y-coordinate
        of TC center.
    """

    (
        nadir_relative_x_cutoffs_metres, nadir_relative_y_cutoffs_metres
    ) = iso_reg_1d_bins.check_input_args(
        nadir_relative_x_cutoffs_metres=nadir_relative_x_cutoffs_metres,
        nadir_relative_y_cutoffs_metres=nadir_relative_y_cutoffs_metres
    )

    error_checking.assert_is_integer(min_training_sample_size)
    error_checking.assert_is_greater(min_training_sample_size, 0)

    print('Reading data from: "{0:s}"...'.format(ebtrk_file_name))
    ebtrk_table_xarray = ebtrk_io.read_file(ebtrk_file_name)
    nadir_relative_prediction_xs_metres, nadir_relative_prediction_ys_metres = (
        misc_utils.match_predictions_to_tc_centers(
            prediction_table_xarray=prediction_table_xarray,
            ebtrk_table_xarray=ebtrk_table_xarray,
            return_xy=True
        )
    )

    num_bin_rows = len(nadir_relative_y_cutoffs_metres) - 1
    num_bin_columns = len(nadir_relative_x_cutoffs_metres) - 1
    x_coord_model_matrix = numpy.full(
        (num_bin_rows, num_bin_columns), '', dtype='object'
    )
    y_coord_model_matrix = numpy.full(
        (num_bin_rows, num_bin_columns), '', dtype='object'
    )

    num_bins = num_bin_rows * num_bin_columns
    training_indices = numpy.array([], dtype=int)

    for k in range(num_bins):
        i, j = _linear_bin_index_to_2d(
            linear_index=k,
            nadir_relative_x_cutoffs_metres=nadir_relative_x_cutoffs_metres,
            nadir_relative_y_cutoffs_metres=nadir_relative_y_cutoffs_metres
        )

        new_training_indices = _find_examples_in_bin(
            nadir_relative_prediction_xs_metres=
            nadir_relative_prediction_xs_metres,
            nadir_relative_prediction_ys_metres=
            nadir_relative_prediction_ys_metres,
            nadir_relative_x_cutoffs_metres=nadir_relative_x_cutoffs_metres,
            nadir_relative_y_cutoffs_metres=nadir_relative_y_cutoffs_metres,
            linear_bin_index=k,
            verbose=True
        )

        if not isinstance(x_coord_model_matrix[i, j], str):
            continue

        training_indices = numpy.concatenate(
            (training_indices, new_training_indices), axis=0
        )
        if len(training_indices) < min_training_sample_size:
            continue

        future_training_indices = numpy.concatenate([
            _find_examples_in_bin(
                nadir_relative_prediction_xs_metres=
                nadir_relative_prediction_xs_metres,
                nadir_relative_prediction_ys_metres=
                nadir_relative_prediction_ys_metres,
                nadir_relative_x_cutoffs_metres=nadir_relative_x_cutoffs_metres,
                nadir_relative_y_cutoffs_metres=nadir_relative_y_cutoffs_metres,
                linear_bin_index=k_new,
                verbose=False
            )
            for k_new in range(k + 1, num_bins)
        ])

        if len(future_training_indices) < min_training_sample_size:
            training_with_higher_bins = True
            training_indices = numpy.concatenate((
                training_indices, future_training_indices
            ))
        else:
            training_with_higher_bins = False

        this_prediction_table_xarray = prediction_table_xarray.isel(
            {prediction_utils.EXAMPLE_DIM_KEY: training_indices}
        )
        tpt = this_prediction_table_xarray
        assert (
            len(tpt.coords[prediction_utils.EXAMPLE_DIM_KEY].values) ==
            len(training_indices)
        )

        print((
            'Training model for nadir-relative x in [{0:.0f}, {1:.0f}) km and '
            'nadir-relative y in [{2:.0f}, {3:.0f}) km with {4:d} examples...'
        ).format(
            METRES_TO_KM * nadir_relative_x_cutoffs_metres[j],
            METRES_TO_KM * nadir_relative_x_cutoffs_metres[j + 1],
            METRES_TO_KM * nadir_relative_y_cutoffs_metres[i],
            METRES_TO_KM * nadir_relative_y_cutoffs_metres[i + 1],
            len(training_indices)
        ))

        x_coord_model_matrix[i, j], y_coord_model_matrix[i, j] = (
            basic_iso_reg.train_models(
                this_prediction_table_xarray
            )
        )

        for k_prev in range(k):
            i_prev, j_prev = _linear_bin_index_to_2d(
                linear_index=k_prev,
                nadir_relative_x_cutoffs_metres=nadir_relative_x_cutoffs_metres,
                nadir_relative_y_cutoffs_metres=nadir_relative_y_cutoffs_metres
            )

            if not isinstance(x_coord_model_matrix[i_prev, j_prev], str):
                continue

            x_coord_model_matrix[i_prev, j_prev] = x_coord_model_matrix[i, j]
            y_coord_model_matrix[i_prev, j_prev] = y_coord_model_matrix[i, j]

        if training_with_higher_bins:
            for k_future in range(k + 1, num_bins):
                i_future, j_future = _linear_bin_index_to_2d(
                    linear_index=k_future,
                    nadir_relative_x_cutoffs_metres=
                    nadir_relative_x_cutoffs_metres,
                    nadir_relative_y_cutoffs_metres=
                    nadir_relative_y_cutoffs_metres
                )

                x_coord_model_matrix[i_future, j_future] = (
                    x_coord_model_matrix[i, j]
                )
                y_coord_model_matrix[i_future, j_future] = (
                    y_coord_model_matrix[i, j]
                )

        training_indices = numpy.array([], dtype=int)

    return x_coord_model_matrix, y_coord_model_matrix


def apply_models(
        prediction_table_xarray, nadir_relative_x_cutoffs_metres,
        nadir_relative_y_cutoffs_metres, ebtrk_file_name,
        x_coord_model_matrix, y_coord_model_matrix):
    """Applies nadir-dependent isotonic-regression models.

    :param prediction_table_xarray: See documentation for `train_models`.
    :param nadir_relative_x_cutoffs_metres: Same.
    :param nadir_relative_y_cutoffs_metres: Same.
    :param ebtrk_file_name: Same.
    :param x_coord_model_matrix: Same.
    :param y_coord_model_matrix: Same.
    :return: prediction_table_xarray: Same as input but with different
        predictions.
    """

    (
        nadir_relative_x_cutoffs_metres, nadir_relative_y_cutoffs_metres
    ) = iso_reg_1d_bins.check_input_args(
        nadir_relative_x_cutoffs_metres=nadir_relative_x_cutoffs_metres,
        nadir_relative_y_cutoffs_metres=nadir_relative_y_cutoffs_metres
    )

    print('Reading data from: "{0:s}"...'.format(ebtrk_file_name))
    ebtrk_table_xarray = ebtrk_io.read_file(ebtrk_file_name)
    nadir_relative_prediction_xs_metres, nadir_relative_prediction_ys_metres = (
        misc_utils.match_predictions_to_tc_centers(
            prediction_table_xarray=prediction_table_xarray,
            ebtrk_table_xarray=ebtrk_table_xarray,
            return_xy=True
        )
    )

    num_bin_rows = len(nadir_relative_y_cutoffs_metres) - 1
    num_bin_columns = len(nadir_relative_x_cutoffs_metres) - 1
    expected_dim = numpy.array([num_bin_rows, num_bin_columns], dtype=int)

    error_checking.assert_is_numpy_array(
        x_coord_model_matrix, exact_dimensions=expected_dim
    )
    error_checking.assert_is_numpy_array(
        y_coord_model_matrix, exact_dimensions=expected_dim
    )

    num_bins = num_bin_rows * num_bin_columns
    new_prediction_tables_xarray = []

    for k in range(num_bins):
        i, j = _linear_bin_index_to_2d(
            linear_index=k,
            nadir_relative_x_cutoffs_metres=nadir_relative_x_cutoffs_metres,
            nadir_relative_y_cutoffs_metres=nadir_relative_y_cutoffs_metres
        )

        example_indices = _find_examples_in_bin(
            nadir_relative_prediction_xs_metres=
            nadir_relative_prediction_xs_metres,
            nadir_relative_prediction_ys_metres=
            nadir_relative_prediction_ys_metres,
            nadir_relative_x_cutoffs_metres=nadir_relative_x_cutoffs_metres,
            nadir_relative_y_cutoffs_metres=nadir_relative_y_cutoffs_metres,
            linear_bin_index=k,
            verbose=True
        )

        if len(example_indices) == 0:
            continue

        new_prediction_table_xarray = prediction_table_xarray.isel(
            {prediction_utils.EXAMPLE_DIM_KEY: example_indices}
        )
        npt = new_prediction_table_xarray
        assert (
            len(npt.coords[prediction_utils.EXAMPLE_DIM_KEY].values) ==
            len(example_indices)
        )

        print((
            'Applying models for nadir-relative x in [{0:.0f}, {1:.0f}) km,'
            ' and nadir-relative y in [{2:.0f}, {3:.0f}) km, to {4:d} '
            'examples...'
        ).format(
            METRES_TO_KM * nadir_relative_x_cutoffs_metres[j],
            METRES_TO_KM * nadir_relative_x_cutoffs_metres[j + 1],
            METRES_TO_KM * nadir_relative_y_cutoffs_metres[i],
            METRES_TO_KM * nadir_relative_y_cutoffs_metres[i + 1],
            len(example_indices)
        ))

        new_prediction_table_xarray = basic_iso_reg.apply_models(
            prediction_table_xarray=new_prediction_table_xarray,
            x_coord_model_object=x_coord_model_matrix[i, j],
            y_coord_model_object=y_coord_model_matrix[i, j]
        )
        new_prediction_tables_xarray.append(new_prediction_table_xarray)

    new_prediction_table_xarray = prediction_utils.concat_over_examples(
        new_prediction_tables_xarray
    )
    pt = prediction_table_xarray
    npt = new_prediction_table_xarray

    assert (
        len(pt.coords[prediction_utils.EXAMPLE_DIM_KEY].values) ==
        len(npt.coords[prediction_utils.EXAMPLE_DIM_KEY].values)
    )

    return new_prediction_table_xarray


def find_file(model_dir_name, raise_error_if_missing=True):
    """Finds Dill file with set of isotonic-regression models.

    This method is a lightweight wrapper for
    `scalar_isotonic_regression.find_file`.

    :param model_dir_name: See doc for `scalar_isotonic_regression.find_file`.
    :param raise_error_if_missing: Same.
    :return: dill_file_name: Same.
    """

    return basic_iso_reg.find_file(
        model_dir_name=model_dir_name,
        raise_error_if_missing=raise_error_if_missing
    )


def write_file(
        dill_file_name, x_coord_model_matrix, y_coord_model_matrix,
        nadir_relative_x_cutoffs_metres, nadir_relative_y_cutoffs_metres):
    """Writes set of isotonic-regression models to Dill file.

    :param dill_file_name: Path to output file.
    :param x_coord_model_matrix: See doc for `train_models`.
    :param y_coord_model_matrix: Same.
    :param nadir_relative_x_cutoffs_metres: Same.
    :param nadir_relative_y_cutoffs_metres: Same.
    """

    (
        nadir_relative_x_cutoffs_metres, nadir_relative_y_cutoffs_metres
    ) = iso_reg_1d_bins.check_input_args(
        nadir_relative_x_cutoffs_metres=nadir_relative_x_cutoffs_metres,
        nadir_relative_y_cutoffs_metres=nadir_relative_y_cutoffs_metres
    )

    num_bin_rows = len(nadir_relative_y_cutoffs_metres) - 1
    num_bin_columns = len(nadir_relative_x_cutoffs_metres) - 1
    expected_dim = numpy.array([num_bin_rows, num_bin_columns], dtype=int)

    error_checking.assert_is_numpy_array(
        x_coord_model_matrix, exact_dimensions=expected_dim
    )
    error_checking.assert_is_numpy_array(
        y_coord_model_matrix, exact_dimensions=expected_dim
    )

    file_system_utils.mkdir_recursive_if_necessary(file_name=dill_file_name)

    dill_file_handle = open(dill_file_name, 'wb')
    pickle.dump(x_coord_model_matrix, dill_file_handle)
    pickle.dump(y_coord_model_matrix, dill_file_handle)
    pickle.dump(nadir_relative_x_cutoffs_metres, dill_file_handle)
    pickle.dump(nadir_relative_y_cutoffs_metres, dill_file_handle)
    dill_file_handle.close()


def read_file(dill_file_name):
    """Reads set of isotonic-regression models from Dill file.

    :param dill_file_name: Path to input file.
    :return: x_coord_model_matrix: See doc for `train_models`.
    :return: y_coord_model_matrix: Same.
    :return: nadir_relative_x_cutoffs_metres: Same.
    :return: nadir_relative_y_cutoffs_metres: Same.
    """

    error_checking.assert_file_exists(dill_file_name)

    dill_file_handle = open(dill_file_name, 'rb')
    x_coord_model_matrix = pickle.load(dill_file_handle)
    y_coord_model_matrix = pickle.load(dill_file_handle)
    nadir_relative_x_cutoffs_metres = pickle.load(dill_file_handle)
    nadir_relative_y_cutoffs_metres = pickle.load(dill_file_handle)
    dill_file_handle.close()

    return (
        x_coord_model_matrix, y_coord_model_matrix,
        nadir_relative_x_cutoffs_metres, nadir_relative_y_cutoffs_metres
    )
