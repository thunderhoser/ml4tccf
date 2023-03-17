"""Methods for writing gridded predictions (probabilities)."""

import numpy
import netCDF4
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4tccf.utils import misc_utils
from ml4tccf.utils import gridded_prediction_utils as prediction_utils

TOLERANCE = 1e-6


def write_file(
        netcdf_file_name, target_matrix, prediction_matrix, grid_spacings_km,
        cyclone_center_latitudes_deg_n, cyclone_id_string,
        target_times_unix_sec, model_file_name):
    """Writes predictions to NetCDF file.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    S = ensemble size

    :param netcdf_file_name: Path to output file.
    :param target_matrix: E-by-M-by-N numpy array of "true probabilities,"
        corresponding to best-track locations of TC centers.
    :param prediction_matrix: E-by-M-by-N-by-S numpy array of predicted
        probabilities.
    :param grid_spacings_km: length-E numpy array of grid spacings.
    :param cyclone_center_latitudes_deg_n: length-E numpy array of latitudes at
        best-track TC centers (deg north).
    :param cyclone_id_string: Cyclone ID.
    :param target_times_unix_sec: length-E numpy array of target times.
    :param model_file_name: Path to trained model (readable by
        `neural_net.read_model`).
    """

    # Check input args.
    error_checking.assert_is_numpy_array(target_matrix, num_dimensions=3)
    error_checking.assert_is_geq_numpy_array(target_matrix, 0.)
    error_checking.assert_is_leq_numpy_array(target_matrix, 1.)

    # assert numpy.allclose(
    #     numpy.sum(target_matrix, axis=(1, 2)), 1., atol=TOLERANCE
    # )

    error_checking.assert_is_numpy_array(prediction_matrix, num_dimensions=4)
    error_checking.assert_is_numpy_array_without_nan(prediction_matrix)

    num_examples = target_matrix.shape[0]
    num_grid_rows = target_matrix.shape[1]
    num_grid_columns = target_matrix.shape[2]
    ensemble_size = prediction_matrix.shape[3]
    expected_dim = numpy.array(
        [num_examples, num_grid_rows, num_grid_columns, ensemble_size],
        dtype=int
    )
    error_checking.assert_is_numpy_array(
        prediction_matrix, exact_dimensions=expected_dim
    )

    # TODO(thunderhoser): Maybe I should enforce this constraint somewhere other
    # than the architecture (leads to very small values in loss function,
    # since values are often squared).
    # assert numpy.allclose(
    #     numpy.sum(prediction_matrix, axis=(1, 2)), 1., atol=TOLERANCE
    # )

    _ = misc_utils.parse_cyclone_id(cyclone_id_string)

    error_checking.assert_is_greater_numpy_array(grid_spacings_km, 0.)
    error_checking.assert_is_numpy_array(
        target_times_unix_sec,
        exact_dimensions=numpy.array([num_examples], dtype=int)
    )

    error_checking.assert_is_valid_lat_numpy_array(
        cyclone_center_latitudes_deg_n
    )
    error_checking.assert_is_numpy_array(
        cyclone_center_latitudes_deg_n,
        exact_dimensions=numpy.array([num_examples], dtype=int)
    )

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

    dataset_object.setncattr(prediction_utils.MODEL_FILE_KEY, model_file_name)
    dataset_object.createDimension(prediction_utils.EXAMPLE_DIM, num_examples)
    dataset_object.createDimension(prediction_utils.ROW_DIM, num_grid_rows)
    dataset_object.createDimension(
        prediction_utils.COLUMN_DIM, num_grid_columns
    )
    dataset_object.createDimension(
        prediction_utils.ENSEMBLE_MEMBER_DIM, ensemble_size
    )
    dataset_object.createDimension(
        prediction_utils.CYCLONE_ID_CHAR_DIM, num_cyclone_id_chars
    )

    these_dim = (
        prediction_utils.EXAMPLE_DIM, prediction_utils.ROW_DIM,
        prediction_utils.COLUMN_DIM
    )
    dataset_object.createVariable(
        prediction_utils.TARGET_MATRIX_KEY, datatype=numpy.float32,
        dimensions=these_dim
    )
    dataset_object.variables[prediction_utils.TARGET_MATRIX_KEY][:] = (
        target_matrix
    )

    these_dim = (
        prediction_utils.EXAMPLE_DIM, prediction_utils.ROW_DIM,
        prediction_utils.COLUMN_DIM, prediction_utils.ENSEMBLE_MEMBER_DIM
    )
    dataset_object.createVariable(
        prediction_utils.PREDICTION_MATRIX_KEY, datatype=numpy.float32,
        dimensions=these_dim
    )
    dataset_object.variables[prediction_utils.PREDICTION_MATRIX_KEY][:] = (
        prediction_matrix
    )

    dataset_object.createVariable(
        prediction_utils.GRID_SPACING_KEY, datatype=numpy.float32,
        dimensions=prediction_utils.EXAMPLE_DIM
    )
    dataset_object.variables[prediction_utils.GRID_SPACING_KEY][:] = (
        grid_spacings_km
    )

    dataset_object.createVariable(
        prediction_utils.ACTUAL_CENTER_LATITUDE_KEY, datatype=numpy.float32,
        dimensions=prediction_utils.EXAMPLE_DIM
    )
    dataset_object.variables[prediction_utils.ACTUAL_CENTER_LATITUDE_KEY][:] = (
        cyclone_center_latitudes_deg_n
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
