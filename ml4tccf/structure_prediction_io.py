"""Methods for writing TC-structure predictions."""

import os
import sys
import numpy
import netCDF4

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import prediction_io
import misc_utils
import structure_prediction_utils as prediction_utils


def find_file(directory_name, cyclone_id_string, raise_error_if_missing=True):
    """Finds file with predictions.

    This is a lightweight wrapper around `prediction_io.read_file`.

    :param directory_name: See documentation for `prediction_io.find_file`.
    :param cyclone_id_string: Same.
    :param raise_error_if_missing: Same.
    :return: prediction_file_name: Same.
    """

    return prediction_io.find_file(
        directory_name=directory_name,
        cyclone_id_string=cyclone_id_string,
        raise_error_if_missing=raise_error_if_missing
    )


def file_name_to_cyclone_id(prediction_file_name):
    """Parses cyclone ID from name of prediction file.

    This is a lightweight wrapper around
    `prediction_io.file_name_to_cyclone_id`.

    :param prediction_file_name: See documentation for
        `prediction_io.file_name_to_cyclone_id`.
    :return: cyclone_id_string: Same.
    """

    return prediction_io.file_name_to_cyclone_id(prediction_file_name)


def read_file(netcdf_file_name):
    """Reads predictions from NetCDF file.

    This is a lightweight wrapper around `prediction_io.read_file`.

    :param netcdf_file_name: See documentation for `prediction_io.read_file`.
    :return: prediction_table_xarray: Same.
    """

    return prediction_io.read_file(netcdf_file_name)


def write_file(
        netcdf_file_name, target_matrix, prediction_matrix,
        baseline_prediction_matrix, cyclone_id_string,
        target_times_unix_sec, model_file_name):
    """Writes predictions to NetCDF file.

    E = number of examples (or number of "TC samples")
    F = number of target fields
    S = ensemble size

    :param netcdf_file_name: Path to output file.
    :param target_matrix: E-by-F numpy array of correct answers.
    :param prediction_matrix: E-by-F-by-S numpy array of predictions.
    :param baseline_prediction_matrix: E-by-F numpy array of baseline
        predictions.
    :param cyclone_id_string: Cyclone ID.
    :param target_times_unix_sec: length-E numpy array of target times.
    :param model_file_name: Path to trained model (readable by
        `neural_net.read_model`).
    """

    # Check input args.
    error_checking.assert_is_numpy_array(target_matrix, num_dimensions=2)
    error_checking.assert_is_geq_numpy_array(target_matrix, 0.)

    error_checking.assert_is_numpy_array(prediction_matrix, num_dimensions=3)
    error_checking.assert_is_geq_numpy_array(prediction_matrix, 0.)

    num_examples = target_matrix.shape[0]
    num_target_fields = target_matrix.shape[1]
    ensemble_size = prediction_matrix.shape[2]
    expected_dim = numpy.array(
        [num_examples, num_target_fields, ensemble_size], dtype=int
    )
    error_checking.assert_is_numpy_array(
        prediction_matrix, exact_dimensions=expected_dim
    )

    if baseline_prediction_matrix is not None:
        error_checking.assert_is_geq_numpy_array(baseline_prediction_matrix, 0.)
        error_checking.assert_is_numpy_array(
            baseline_prediction_matrix,
            exact_dimensions=numpy.array(target_matrix.shape, dtype=int)
        )

    _ = misc_utils.parse_cyclone_id(cyclone_id_string)

    error_checking.assert_is_integer_numpy_array(target_times_unix_sec)
    error_checking.assert_is_numpy_array(
        target_times_unix_sec,
        exact_dimensions=numpy.array([num_examples], dtype=int)
    )

    error_checking.assert_is_string(model_file_name)

    # Do actual stuff.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(netcdf_file_name, 'w', format='NETCDF4')

    num_cyclone_id_chars = len(cyclone_id_string)

    dataset_object.setncattr(prediction_utils.MODEL_FILE_KEY, model_file_name)
    dataset_object.createDimension(
        prediction_utils.EXAMPLE_DIM, num_examples
    )
    dataset_object.createDimension(
        prediction_utils.TARGET_FIELD_DIM, num_target_fields
    )
    dataset_object.createDimension(
        prediction_utils.ENSEMBLE_MEMBER_DIM, ensemble_size
    )
    dataset_object.createDimension(
        prediction_utils.CYCLONE_ID_CHAR_DIM, num_cyclone_id_chars
    )

    these_dim = (
        prediction_utils.EXAMPLE_DIM, prediction_utils.TARGET_FIELD_DIM,
        prediction_utils.ENSEMBLE_MEMBER_DIM
    )
    dataset_object.createVariable(
        prediction_utils.PREDICTION_KEY, datatype=numpy.float32,
        dimensions=these_dim
    )
    dataset_object.variables[prediction_utils.PREDICTION_KEY][:] = (
        prediction_matrix
    )

    these_dim = (
        prediction_utils.EXAMPLE_DIM, prediction_utils.TARGET_FIELD_DIM
    )
    dataset_object.createVariable(
        prediction_utils.TARGET_KEY, datatype=numpy.float32,
        dimensions=these_dim
    )
    dataset_object.variables[prediction_utils.TARGET_KEY][:] = target_matrix

    if baseline_prediction_matrix is not None:
        dataset_object.createVariable(
            prediction_utils.BASELINE_PREDICTION_KEY, datatype=numpy.float32,
            dimensions=these_dim
        )
        dataset_object.variables[prediction_utils.BASELINE_PREDICTION_KEY][:] = (
            baseline_prediction_matrix
        )

    dataset_object.createVariable(
        prediction_utils.TARGET_TIME_KEY, datatype=numpy.int32,
        dimensions=prediction_utils.EXAMPLE_DIM
    )
    dataset_object.variables[prediction_utils.TARGET_TIME_KEY][:] = (
        target_times_unix_sec
    )

    this_string_format = 'S{0:d}'.format(num_cyclone_id_chars)
    cyclone_ids_char_array = netCDF4.stringtochar(numpy.array(
        [cyclone_id_string] * num_examples, dtype=this_string_format
    ))

    dataset_object.createVariable(
        prediction_utils.CYCLONE_ID_KEY, datatype='S1',
        dimensions=(
            prediction_utils.EXAMPLE_DIM,
            prediction_utils.CYCLONE_ID_CHAR_DIM
        )
    )
    dataset_object.variables[prediction_utils.CYCLONE_ID_KEY][:] = numpy.array(
        cyclone_ids_char_array
    )

    dataset_object.close()
