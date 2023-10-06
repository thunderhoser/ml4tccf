"""Isotonic regression with scalar target variables.

This allows for "nadir-dependent" isotonic-regression models, i.e., one model
for every bin of nadir-relative coordinates.
"""

import os
import sys
import pickle
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import number_rounding
import file_system_utils
import error_checking
import extended_best_track_io as ebtrk_io
import misc_utils
import scalar_prediction_utils as prediction_utils
import scalar_isotonic_regression as basic_iso_reg

KM_TO_METRES = 1000.
METRES_TO_KM = 0.001
MAX_XY_COORD_METRES = misc_utils.MAX_XY_COORD_METRES


def check_input_args(nadir_relative_x_cutoffs_metres,
                     nadir_relative_y_cutoffs_metres):
    """Checks input arguments.

    :param nadir_relative_x_cutoffs_metres: See documentation for
        `train_models`.
    :param nadir_relative_y_cutoffs_metres: Same.
    :return: nadir_relative_x_cutoffs_metres: Same as input but with end values
        (-inf and inf).
    :return: nadir_relative_y_cutoffs_metres: Same as input but with end values
        (-inf and inf).
    """

    nadir_relative_x_cutoffs_metres = number_rounding.round_to_nearest(
        nadir_relative_x_cutoffs_metres, KM_TO_METRES
    )
    error_checking.assert_is_greater_numpy_array(
        nadir_relative_x_cutoffs_metres, -MAX_XY_COORD_METRES
    )
    error_checking.assert_is_less_than_numpy_array(
        nadir_relative_x_cutoffs_metres, MAX_XY_COORD_METRES
    )
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(nadir_relative_x_cutoffs_metres), 0.
    )
    nadir_relative_x_cutoffs_metres = numpy.concatenate((
        numpy.array([-numpy.inf]),
        nadir_relative_x_cutoffs_metres,
        numpy.array([numpy.inf])
    ))

    nadir_relative_y_cutoffs_metres = number_rounding.round_to_nearest(
        nadir_relative_y_cutoffs_metres, KM_TO_METRES
    )
    error_checking.assert_is_greater_numpy_array(
        nadir_relative_y_cutoffs_metres, -MAX_XY_COORD_METRES
    )
    error_checking.assert_is_less_than_numpy_array(
        nadir_relative_y_cutoffs_metres, MAX_XY_COORD_METRES
    )
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(nadir_relative_y_cutoffs_metres), 0.
    )
    nadir_relative_y_cutoffs_metres = numpy.concatenate((
        numpy.array([-numpy.inf]),
        nadir_relative_y_cutoffs_metres,
        numpy.array([numpy.inf])
    ))

    return nadir_relative_x_cutoffs_metres, nadir_relative_y_cutoffs_metres


def train_models(
        prediction_table_xarray, nadir_relative_x_cutoffs_metres,
        nadir_relative_y_cutoffs_metres, ebtrk_file_name,
        min_training_sample_size):
    """Trains nadir-dependent isotonic-regression models.

    M = number of bins for nadir-relative y-coordinate
    N = number of bins for nadir-relative x-coordinate

    :param prediction_table_xarray: xarray table in format returned by
        `prediction_io.read_file`.
    :param nadir_relative_x_cutoffs_metres: Category cutoffs for nadir-relative
        x-coordinate.  Please leave -inf and +inf out of this list, as they will
        be added automatically.
    :param nadir_relative_y_cutoffs_metres: Category cutoffs for nadir-relative
        y-coordinate.  Please leave -inf and +inf out of this list, as they will
        be added automatically.
    :param ebtrk_file_name: Name of file with extended best-track data (will be
        read by `extended_best_track_io.read_file`).  This file will be used to
        find nadir-relative coordinates for every TC object.
    :param min_training_sample_size: Minimum number of training examples for one
        model.
    :return: x_coord_model_objects: length-N list of models (instances of
        `sklearn.isotonic.IsotonicRegression`) for bias-correcting x-coordinate
        of TC center.
    :return: y_coord_model_objects: length-M list of models (instances of
        `sklearn.isotonic.IsotonicRegression`) for bias-correcting y-coordinate
        of TC center.
    """

    (
        nadir_relative_x_cutoffs_metres, nadir_relative_y_cutoffs_metres
    ) = check_input_args(
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

    num_x_bins = len(nadir_relative_x_cutoffs_metres) - 1
    x_coord_model_objects = [None] * num_x_bins
    training_indices = numpy.array([], dtype=int)

    for j in range(num_x_bins):
        new_training_indices = numpy.where(numpy.logical_and(
            nadir_relative_prediction_xs_metres >=
            nadir_relative_x_cutoffs_metres[j],
            nadir_relative_prediction_xs_metres <
            nadir_relative_x_cutoffs_metres[j + 1]
        ))[0]

        print((
            'Found {0:d} examples with nadir-relative x in [{1:.0f}, {2:.0f}) '
            'km.'
        ).format(
            len(new_training_indices),
            METRES_TO_KM * nadir_relative_x_cutoffs_metres[j],
            METRES_TO_KM * nadir_relative_x_cutoffs_metres[j + 1]
        ))

        if x_coord_model_objects[j] is not None:
            continue

        training_indices = numpy.concatenate(
            (training_indices, new_training_indices), axis=0
        )
        if len(training_indices) < min_training_sample_size:
            continue

        future_training_indices = numpy.where(
            nadir_relative_prediction_xs_metres >=
            nadir_relative_x_cutoffs_metres[j + 1]
        )[0]

        print('Number of future training examples = {0:d}'.format(
            len(future_training_indices)
        ))

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
            'Training model for nadir-relative x in [{0:.0f}, {1:.0f}) km with '
            '{2:d} examples...'
        ).format(
            METRES_TO_KM * nadir_relative_x_cutoffs_metres[j],
            METRES_TO_KM * nadir_relative_x_cutoffs_metres[j + 1],
            len(training_indices)
        ))

        x_coord_model_objects[j] = basic_iso_reg.train_models(
            this_prediction_table_xarray
        )[0]

        for j_prev in range(j):
            if x_coord_model_objects[j_prev] is None:
                x_coord_model_objects[j_prev] = x_coord_model_objects[j]

        if training_with_higher_bins:
            for j_future in range(num_x_bins):
                if j_future > j:
                    x_coord_model_objects[j_future] = x_coord_model_objects[j]

        training_indices = numpy.array([], dtype=int)

    num_y_bins = len(nadir_relative_y_cutoffs_metres) - 1
    y_coord_model_objects = [None] * num_y_bins
    training_indices = numpy.array([], dtype=int)

    for i in range(num_y_bins):
        new_training_indices = numpy.where(numpy.logical_and(
            nadir_relative_prediction_ys_metres >=
            nadir_relative_y_cutoffs_metres[i],
            nadir_relative_prediction_ys_metres <
            nadir_relative_y_cutoffs_metres[i + 1]
        ))[0]

        print((
            'Found {0:d} examples with nadir-relative y in [{1:.0f}, {2:.0f}) '
            'km.'
        ).format(
            len(new_training_indices),
            METRES_TO_KM * nadir_relative_y_cutoffs_metres[i],
            METRES_TO_KM * nadir_relative_y_cutoffs_metres[i + 1]
        ))

        if y_coord_model_objects[i] is not None:
            continue

        training_indices = numpy.concatenate(
            (training_indices, new_training_indices), axis=0
        )
        if len(training_indices) < min_training_sample_size:
            continue

        future_training_indices = numpy.where(
            nadir_relative_prediction_ys_metres >=
            nadir_relative_y_cutoffs_metres[i + 1]
        )[0]

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
            'Training model for nadir-relative y in [{0:.0f}, {1:.0f}) km with '
            '{2:d} examples...'
        ).format(
            METRES_TO_KM * nadir_relative_y_cutoffs_metres[i],
            METRES_TO_KM * nadir_relative_y_cutoffs_metres[i + 1],
            len(training_indices)
        ))

        y_coord_model_objects[i] = basic_iso_reg.train_models(
            this_prediction_table_xarray
        )[1]

        for i_prev in range(i):
            if y_coord_model_objects[i_prev] is None:
                y_coord_model_objects[i_prev] = y_coord_model_objects[i]

        if training_with_higher_bins:
            for i_future in range(num_y_bins):
                if i_future > i:
                    y_coord_model_objects[i_future] = y_coord_model_objects[i]

        training_indices = numpy.array([], dtype=int)

    return x_coord_model_objects, y_coord_model_objects


def apply_models(
        prediction_table_xarray, nadir_relative_x_cutoffs_metres,
        nadir_relative_y_cutoffs_metres, ebtrk_file_name,
        x_coord_model_objects, y_coord_model_objects):
    """Applies nadir-dependent isotonic-regression models.

    M = number of bins for nadir-relative y-coordinate
    N = number of bins for nadir-relative x-coordinate

    :param prediction_table_xarray: See documentation for `train_models`.
    :param nadir_relative_x_cutoffs_metres: Same.
    :param nadir_relative_y_cutoffs_metres: Same.
    :param ebtrk_file_name: Same.
    :param x_coord_model_objects: Same.
    :param y_coord_model_objects: Same.
    :return: prediction_table_xarray: Same as input but with different
        predictions.
    """

    (
        nadir_relative_x_cutoffs_metres, nadir_relative_y_cutoffs_metres
    ) = check_input_args(
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

    num_x_bins = len(nadir_relative_x_cutoffs_metres) - 1
    num_y_bins = len(nadir_relative_y_cutoffs_metres) - 1
    assert len(x_coord_model_objects) == num_x_bins
    assert len(y_coord_model_objects) == num_y_bins

    new_prediction_tables_xarray = []

    for i in range(num_y_bins):
        y_flags = numpy.logical_and(
            nadir_relative_prediction_ys_metres >=
            nadir_relative_y_cutoffs_metres[i],
            nadir_relative_prediction_ys_metres <
            nadir_relative_y_cutoffs_metres[i + 1]
        )

        for j in range(num_x_bins):
            x_flags = numpy.logical_and(
                nadir_relative_prediction_xs_metres >=
                nadir_relative_x_cutoffs_metres[j],
                nadir_relative_prediction_xs_metres <
                nadir_relative_x_cutoffs_metres[j + 1]
            )
            example_indices = numpy.where(
                numpy.logical_and(x_flags, y_flags)
            )[0]

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
                x_coord_model_object=x_coord_model_objects[j],
                y_coord_model_object=y_coord_model_objects[i]
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
        dill_file_name, x_coord_model_objects, y_coord_model_objects,
        nadir_relative_x_cutoffs_metres, nadir_relative_y_cutoffs_metres):
    """Writes set of isotonic-regression models to Dill file.

    :param dill_file_name: Path to output file.
    :param x_coord_model_objects: See doc for `train_models`.
    :param y_coord_model_objects: Same.
    :param nadir_relative_x_cutoffs_metres: Same.
    :param nadir_relative_y_cutoffs_metres: Same.
    """

    (
        nadir_relative_x_cutoffs_metres, nadir_relative_y_cutoffs_metres
    ) = check_input_args(
        nadir_relative_x_cutoffs_metres=nadir_relative_x_cutoffs_metres,
        nadir_relative_y_cutoffs_metres=nadir_relative_y_cutoffs_metres
    )

    num_x_bins = len(nadir_relative_x_cutoffs_metres) - 1
    num_y_bins = len(nadir_relative_y_cutoffs_metres) - 1
    assert len(x_coord_model_objects) == num_x_bins
    assert len(y_coord_model_objects) == num_y_bins

    file_system_utils.mkdir_recursive_if_necessary(file_name=dill_file_name)

    dill_file_handle = open(dill_file_name, 'wb')
    pickle.dump(x_coord_model_objects, dill_file_handle)
    pickle.dump(y_coord_model_objects, dill_file_handle)
    pickle.dump(nadir_relative_x_cutoffs_metres, dill_file_handle)
    pickle.dump(nadir_relative_y_cutoffs_metres, dill_file_handle)
    dill_file_handle.close()


def read_file(dill_file_name):
    """Reads set of isotonic-regression models from Dill file.

    :param dill_file_name: Path to input file.
    :return: x_coord_model_objects: See doc for `train_models`.
    :return: y_coord_model_objects: Same.
    :return: nadir_relative_x_cutoffs_metres: Same.
    :return: nadir_relative_y_cutoffs_metres: Same.
    """

    error_checking.assert_file_exists(dill_file_name)

    dill_file_handle = open(dill_file_name, 'rb')
    x_coord_model_objects = pickle.load(dill_file_handle)
    y_coord_model_objects = pickle.load(dill_file_handle)
    nadir_relative_x_cutoffs_metres = pickle.load(dill_file_handle)
    nadir_relative_y_cutoffs_metres = pickle.load(dill_file_handle)
    dill_file_handle.close()

    return (
        x_coord_model_objects, y_coord_model_objects,
        nadir_relative_x_cutoffs_metres, nadir_relative_y_cutoffs_metres
    )
