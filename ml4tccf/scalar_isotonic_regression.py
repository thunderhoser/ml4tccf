"""Isotonic regression with scalar target variables."""

import os
import sys
import pickle
import numpy
from sklearn.isotonic import IsotonicRegression

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import scalar_prediction_utils as prediction_utils

# TODO(thunderhoser): Might want to try bivariate IR, if that's even a thing.


def train_models(prediction_table_xarray):
    """Trains isotonic-regression models.

    :param prediction_table_xarray: xarray table in format returned by
        `prediction_io.read_file`.
    :return: x_coord_model_object: Model (instance of
        `sklearn.isotonic.IsotonicRegression`) for x-coordinate.
    :return: y_coord_model_object: Same but for y-coordinate.
    """

    pt = prediction_table_xarray
    grid_spacings_metres = pt[prediction_utils.GRID_SPACING_KEY].values

    predicted_x_offsets_metres = grid_spacings_metres * numpy.mean(
        pt[prediction_utils.PREDICTED_COLUMN_OFFSET_KEY].values, axis=-1
    )
    predicted_y_offsets_metres = grid_spacings_metres * numpy.mean(
        pt[prediction_utils.PREDICTED_ROW_OFFSET_KEY].values, axis=-1
    )

    actual_x_offsets_metres = (
        grid_spacings_metres *
        pt[prediction_utils.ACTUAL_COLUMN_OFFSET_KEY].values
    )
    actual_y_offsets_metres = (
        grid_spacings_metres * pt[prediction_utils.ACTUAL_ROW_OFFSET_KEY].values
    )

    x_coord_model_object = IsotonicRegression(
        increasing=True, out_of_bounds='clip'
    )
    x_coord_model_object.fit(
        X=predicted_x_offsets_metres, y=actual_x_offsets_metres
    )

    y_coord_model_object = IsotonicRegression(
        increasing=True, out_of_bounds='clip'
    )
    y_coord_model_object.fit(
        X=predicted_y_offsets_metres, y=actual_y_offsets_metres
    )

    return x_coord_model_object, y_coord_model_object


def apply_models(prediction_table_xarray, x_coord_model_object,
                 y_coord_model_object):
    """Applies isotonic-regression models.

    :param prediction_table_xarray: See doc for `train_models`.
    :param x_coord_model_object: Same.
    :param y_coord_model_object: Same.
    :return: prediction_table_xarray: Same as input but with different
        predictions.
    """

    pt = prediction_table_xarray
    grid_spacing_matrix_metres = numpy.expand_dims(
        pt[prediction_utils.GRID_SPACING_KEY].values, axis=-1
    )

    predicted_x_offset_matrix_metres = (
        grid_spacing_matrix_metres *
        pt[prediction_utils.PREDICTED_COLUMN_OFFSET_KEY].values
    )
    predicted_y_offset_matrix_metres = (
        grid_spacing_matrix_metres *
        pt[prediction_utils.PREDICTED_ROW_OFFSET_KEY].values
    )

    num_examples = predicted_x_offset_matrix_metres.shape[0]
    ensemble_size = predicted_x_offset_matrix_metres.shape[1]

    predicted_x_offset_matrix_metres = numpy.reshape(
        x_coord_model_object.predict(
            numpy.ravel(predicted_x_offset_matrix_metres)
        ),
        (num_examples, ensemble_size)
    )

    predicted_y_offset_matrix_metres = numpy.reshape(
        y_coord_model_object.predict(
            numpy.ravel(predicted_y_offset_matrix_metres)
        ),
        (num_examples, ensemble_size)
    )

    pt = pt.assign({
        prediction_utils.PREDICTED_COLUMN_OFFSET_KEY: (
            pt[prediction_utils.PREDICTED_COLUMN_OFFSET_KEY].dims,
            predicted_x_offset_matrix_metres / grid_spacing_matrix_metres
        ),
        prediction_utils.PREDICTED_ROW_OFFSET_KEY: (
            pt[prediction_utils.PREDICTED_ROW_OFFSET_KEY].dims,
            predicted_y_offset_matrix_metres / grid_spacing_matrix_metres
        )
    })

    prediction_table_xarray = pt
    return prediction_table_xarray


def find_file(model_dir_name, raise_error_if_missing=True):
    """Finds Dill file with set of isotonic-regression models.

    :param model_dir_name: Name of directory.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: dill_file_name: Path to Dill file with models.
    """

    error_checking.assert_is_string(model_dir_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    dill_file_name = '{0:s}/isotonic_regression.dill'.format(model_dir_name)

    if raise_error_if_missing and not os.path.isfile(dill_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            dill_file_name
        )
        raise ValueError(error_string)

    return dill_file_name


def write_file(dill_file_name, x_coord_model_object, y_coord_model_object):
    """Writes set of isotonic-regression models to Dill file.

    :param dill_file_name: Path to output file.
    :param x_coord_model_object: See doc for `train_models`.
    :param y_coord_model_object: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=dill_file_name)

    dill_file_handle = open(dill_file_name, 'wb')
    pickle.dump(x_coord_model_object, dill_file_handle)
    pickle.dump(y_coord_model_object, dill_file_handle)
    dill_file_handle.close()


def read_file(dill_file_name):
    """Reads set of isotonic-regression models from Dill file.

    :param dill_file_name: Path to input file.
    :return: x_coord_model_object: See doc for `train_models`.
    :return: y_coord_model_object: Same.
    """

    print(dill_file_name)
    import os
    print(os.system('ls -l {0:s}'.format(dill_file_name)))

    error_checking.assert_file_exists(dill_file_name)

    dill_file_handle = open(dill_file_name, 'rb')
    x_coord_model_object = pickle.load(dill_file_handle)
    y_coord_model_object = pickle.load(dill_file_handle)
    dill_file_handle.close()

    return x_coord_model_object, y_coord_model_object
