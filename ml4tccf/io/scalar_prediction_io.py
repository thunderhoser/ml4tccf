"""Methods for writing scalar predictions (x- and y-coords)."""

import numpy
import netCDF4
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4tccf.utils import misc_utils
from ml4tccf.utils import scalar_prediction_utils as prediction_utils


def write_file(
        netcdf_file_name, target_matrix, prediction_matrix, cyclone_id_string,
        target_times_unix_sec, model_file_name, isotonic_model_file_name,
        uncertainty_calib_model_file_name):
    """Writes predictions to NetCDF file.

    E = number of examples
    S = ensemble size

    :param netcdf_file_name: Path to output file.
    :param target_matrix: E-by-4 numpy array.  target_matrix[:, 0] contains
        row positions of TC centers; target_matrix[:, 1] contains column
        positions of TC centers; target_matrix[:, 2] contains grid spacing in
        km, used to convert row-column distances to actual distances; and
        target_matrix[:, 3] contains true TC latitudes in deg north.
    :param prediction_matrix: E-by-2-by-S numpy array.
        prediction_matrix[:, 0, :] contains predicted row positions of TC
        centers, and prediction_matrix[:, 1, :] contains predicted column
        positions of TC centers.
    :param cyclone_id_string: Cyclone ID.
    :param target_times_unix_sec: length-E numpy array of target times.
    :param model_file_name: Path to trained model (readable by
        `neural_net.read_model`).
    :param isotonic_model_file_name: Path to isotonic-regression model used for
        bias correction (readable by `isotonic_regression.read_file`).  If bias
        correction was not used, make this None.
    :param uncertainty_calib_model_file_name: Path to uncertainty-calibration
        model used to bias-correct spread (readable by
        `uncertainty_calibration.read_file`).  If uncertainty calibration was
        not used, make this None.
    """

    # Check input args.
    error_checking.assert_is_numpy_array(target_matrix, num_dimensions=2)
    error_checking.assert_is_numpy_array_without_nan(target_matrix)
    error_checking.assert_equals(target_matrix.shape[1], 4)
    error_checking.assert_is_greater_numpy_array(target_matrix[:, 2], 0)
    error_checking.assert_is_valid_lat_numpy_array(target_matrix[:, 3])

    error_checking.assert_is_numpy_array(prediction_matrix, num_dimensions=3)
    error_checking.assert_is_numpy_array_without_nan(prediction_matrix)

    num_examples = target_matrix.shape[0]
    ensemble_size = prediction_matrix.shape[2]
    expected_dim = numpy.array([num_examples, 2, ensemble_size], dtype=int)
    error_checking.assert_is_numpy_array(
        prediction_matrix, exact_dimensions=expected_dim
    )

    _ = misc_utils.parse_cyclone_id(cyclone_id_string)

    error_checking.assert_is_integer_numpy_array(target_times_unix_sec)
    error_checking.assert_is_numpy_array(
        target_times_unix_sec,
        exact_dimensions=numpy.array([num_examples], dtype=int)
    )

    error_checking.assert_is_string(model_file_name)
    if isotonic_model_file_name is None:
        isotonic_model_file_name = ''
    if uncertainty_calib_model_file_name is None:
        uncertainty_calib_model_file_name = ''

    error_checking.assert_is_string(isotonic_model_file_name)
    error_checking.assert_is_string(uncertainty_calib_model_file_name)

    # Do actual stuff.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    num_cyclone_id_chars = len(cyclone_id_string)

    dataset_object.setncattr(prediction_utils.MODEL_FILE_KEY, model_file_name)
    dataset_object.setncattr(
        prediction_utils.ISOTONIC_MODEL_FILE_KEY, isotonic_model_file_name
    )
    dataset_object.setncattr(
        prediction_utils.UNCERTAINTY_CALIB_MODEL_FILE_KEY,
        uncertainty_calib_model_file_name
    )
    dataset_object.createDimension(
        prediction_utils.EXAMPLE_DIM_KEY, num_examples
    )
    dataset_object.createDimension(
        prediction_utils.ENSEMBLE_MEMBER_DIM_KEY, ensemble_size
    )
    dataset_object.createDimension(
        prediction_utils.CYCLONE_ID_CHAR_DIM_KEY, num_cyclone_id_chars
    )

    dataset_object.createVariable(
        prediction_utils.PREDICTED_ROW_OFFSET_KEY, datatype=numpy.float32,
        dimensions=(
            prediction_utils.EXAMPLE_DIM_KEY,
            prediction_utils.ENSEMBLE_MEMBER_DIM_KEY
        )
    )
    dataset_object.variables[prediction_utils.PREDICTED_ROW_OFFSET_KEY][:] = (
        prediction_matrix[:, 0, :]
    )

    dataset_object.createVariable(
        prediction_utils.PREDICTED_COLUMN_OFFSET_KEY, datatype=numpy.float32,
        dimensions=(
            prediction_utils.EXAMPLE_DIM_KEY,
            prediction_utils.ENSEMBLE_MEMBER_DIM_KEY
        )
    )
    dataset_object.variables[
        prediction_utils.PREDICTED_COLUMN_OFFSET_KEY
    ][:] = prediction_matrix[:, 1, :]

    dataset_object.createVariable(
        prediction_utils.ACTUAL_ROW_OFFSET_KEY, datatype=numpy.float32,
        dimensions=prediction_utils.EXAMPLE_DIM_KEY
    )
    dataset_object.variables[prediction_utils.ACTUAL_ROW_OFFSET_KEY][:] = (
        target_matrix[:, 0]
    )

    dataset_object.createVariable(
        prediction_utils.ACTUAL_COLUMN_OFFSET_KEY, datatype=numpy.float32,
        dimensions=prediction_utils.EXAMPLE_DIM_KEY
    )
    dataset_object.variables[prediction_utils.ACTUAL_COLUMN_OFFSET_KEY][:] = (
        target_matrix[:, 1]
    )

    dataset_object.createVariable(
        prediction_utils.GRID_SPACING_KEY, datatype=numpy.float32,
        dimensions=prediction_utils.EXAMPLE_DIM_KEY
    )
    dataset_object.variables[prediction_utils.GRID_SPACING_KEY][:] = (
        target_matrix[:, 2]
    )

    dataset_object.createVariable(
        prediction_utils.ACTUAL_CENTER_LATITUDE_KEY, datatype=numpy.float32,
        dimensions=prediction_utils.EXAMPLE_DIM_KEY
    )
    dataset_object.variables[prediction_utils.ACTUAL_CENTER_LATITUDE_KEY][:] = (
        target_matrix[:, 3]
    )

    dataset_object.createVariable(
        prediction_utils.TARGET_TIME_KEY, datatype=numpy.int32,
        dimensions=prediction_utils.EXAMPLE_DIM_KEY
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
            prediction_utils.EXAMPLE_DIM_KEY,
            prediction_utils.CYCLONE_ID_CHAR_DIM_KEY
        )
    )
    dataset_object.variables[prediction_utils.CYCLONE_ID_KEY][:] = numpy.array(
        cyclone_ids_char_array
    )

    dataset_object.close()
