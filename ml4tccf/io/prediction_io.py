"""I/O methods for predictions in any format (scalar or gridded)."""

import os
import numpy
import xarray
from gewittergefahr.gg_utils import error_checking
from ml4tccf.utils import misc_utils
from ml4tccf.utils import gridded_prediction_utils


def find_file(directory_name, cyclone_id_string, raise_error_if_missing=True):
    """Finds file with predictions.

    :param directory_name: Name of directory.
    :param cyclone_id_string: Cyclone ID.
    :param raise_error_if_missing: Boolean flag.  If file is not found and
        `raise_error_if_missing == True`, this method will raise an error.  If
        file is not found and `raise_error_if_missing == False`, this method
        will return the expected file path.
    :return: prediction_file_name: Actual or expected file path.
    :raises: ValueError: if file is not found and
        `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)
    _ = misc_utils.parse_cyclone_id(cyclone_id_string)

    prediction_file_name = '{0:s}/predictions_{1:s}.nc'.format(
        directory_name, cyclone_id_string
    )

    if os.path.isfile(prediction_file_name):
        return prediction_file_name

    if not raise_error_if_missing:
        return prediction_file_name

    error_string = 'Cannot find prediction file.  Expected at: "{0:s}"'.format(
        prediction_file_name
    )
    raise ValueError(error_string)


def file_name_to_cyclone_id(prediction_file_name):
    """Parses cyclone ID from name of prediction file.

    :param prediction_file_name: Path to prediction file.
    :return: cyclone_id_string: Cyclone ID.
    """

    error_checking.assert_is_string(prediction_file_name)

    pathless_file_name = os.path.split(prediction_file_name)[1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]
    cyclone_id_string = extensionless_file_name.split('_')[-1]

    misc_utils.parse_cyclone_id(cyclone_id_string)
    return cyclone_id_string


def read_file(netcdf_file_name):
    """Reads predictions from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: prediction_table_xarray: xarray table.  Variable names and metadata
        in the table should make it self-explanatory.
    """

    prediction_table_xarray = xarray.open_dataset(netcdf_file_name)

    pt = prediction_table_xarray
    pt[gridded_prediction_utils.PREDICTION_MATRIX_KEY] = numpy.minimum(
        pt[gridded_prediction_utils.PREDICTION_MATRIX_KEY], 1.
    )
    prediction_table_xarray = pt

    return prediction_table_xarray
