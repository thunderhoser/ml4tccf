"""I/O methods for predictions in any format (scalar or gridded)."""

import os
import sys
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import misc_utils
import scalar_prediction_utils
import gridded_prediction_utils


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
    ptx = prediction_table_xarray

    cyclone_id_strings = [
        c.decode('utf-8')
        for c in ptx[scalar_prediction_utils.CYCLONE_ID_KEY].values
    ]

    ptx = ptx.assign({
        scalar_prediction_utils.CYCLONE_ID_KEY: (
            ptx[scalar_prediction_utils.CYCLONE_ID_KEY].dims,
            cyclone_id_strings
        )
    })

    if gridded_prediction_utils.PREDICTION_MATRIX_KEY in ptx:
        ptx[gridded_prediction_utils.PREDICTION_MATRIX_KEY] = numpy.minimum(
            ptx[gridded_prediction_utils.PREDICTION_MATRIX_KEY], 1.
        )
    else:
        if scalar_prediction_utils.ISOTONIC_MODEL_FILE_KEY not in ptx.attrs:
            ptx.attrs[scalar_prediction_utils.ISOTONIC_MODEL_FILE_KEY] = None
        elif ptx.attrs[scalar_prediction_utils.ISOTONIC_MODEL_FILE_KEY] == '':
            ptx.attrs[scalar_prediction_utils.ISOTONIC_MODEL_FILE_KEY] = None

    prediction_table_xarray = ptx
    return prediction_table_xarray
