"""Methods for reading and writing satellite data."""

import os
import sys
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import file_system_utils
import error_checking
import misc_utils

DATE_FORMAT = '%Y-%m-%d'


def find_file(directory_name, cyclone_id_string, valid_date_string,
              raise_error_if_missing=True):
    """Finds file with satellite data.

    :param directory_name: Name of directory.
    :param cyclone_id_string: Cyclone ID.
    :param valid_date_string: Valid date (format "yyyy-mm-dd").
    :param raise_error_if_missing: Boolean flag.  If file is not found and
        `raise_error_if_missing == True`, this method will raise an error.  If
        file is not found and `raise_error_if_missing == False`, this method
        will return the expected file path.
    :return: satellite_file_name: Actual or expected file path.
    :raises: ValueError: if file is not found and
        `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)
    _ = misc_utils.parse_cyclone_id(cyclone_id_string)
    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)

    satellite_file_name = '{0:s}/satellite_{1:s}_{2:s}.nc'.format(
        directory_name, cyclone_id_string, valid_date_string
    )

    if os.path.isfile(satellite_file_name):
        return satellite_file_name

    if not raise_error_if_missing:
        return satellite_file_name

    error_string = 'Cannot find satellite file.  Expected at: "{0:s}"'.format(
        satellite_file_name
    )
    raise ValueError(error_string)


def read_file(netcdf_file_name):
    """Reads satellite data from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: satellite_table_xarray: xarray table.  Documentation in the xarray
        table should make values self-explanatory.
    """

    return xarray.open_dataset(netcdf_file_name)


def write_file(satellite_table_xarray, netcdf_file_name):
    """Writes satellite data to NetCDF file.

    :param satellite_table_xarray: xarray table in format returned by
        `read_file`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    satellite_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )
