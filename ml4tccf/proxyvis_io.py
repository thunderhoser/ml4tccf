"""Methods for reading and writing proxy-vis data."""

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

TIME_FORMAT_IN_FILE_NAME = '%Y-%m-%d-%H%M%S'


def find_file(directory_name, cyclone_id_string, valid_time_unix_sec,
              raise_error_if_missing=True):
    """Finds file with proxy-vis data.

    :param directory_name: Name of directory.
    :param cyclone_id_string: Cyclone ID.
    :param valid_time_unix_sec: Valid time.
    :param raise_error_if_missing: Boolean flag.  If file is not found and
        `raise_error_if_missing == True`, this method will raise an error.  If
        file is not found and `raise_error_if_missing == False`, this method
        will return the expected file path.
    :return: proxyvis_file_name: Actual or expected file path.
    :raises: ValueError: if file is not found and
        `raise_error_if_missing == True`.
    """

    # TODO(thunderhoser): ml4tccf still needs code that handles cyclone IDs.

    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_string(cyclone_id_string)
    error_checking.assert_is_integer(valid_time_unix_sec)
    error_checking.assert_is_boolean(raise_error_if_missing)

    proxyvis_file_name = '{0:s}/proxyvis_{1:s}_{2:s}.nc'.format(
        directory_name, cyclone_id_string,
        time_conversion.unix_sec_to_string(
            valid_time_unix_sec, TIME_FORMAT_IN_FILE_NAME
        )
    )

    if os.path.isfile(proxyvis_file_name):
        return proxyvis_file_name

    if not raise_error_if_missing:
        return proxyvis_file_name

    error_string = 'Cannot find proxy-vis file.  Expected at: "{0:s}"'.format(
        proxyvis_file_name
    )
    raise ValueError(error_string)


def read_file(netcdf_file_name):
    """Reads proxy-vis data from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: proxyvis_table_xarray: xarray table.  Documentation in the xarray
        table should make values self-explanatory.
    """

    return xarray.open_dataset(netcdf_file_name)


def write_file(proxyvis_table_xarray, netcdf_file_name):
    """Writes proxy-vis data to NetCDF file.

    :param proxyvis_table_xarray: xarray table in format returned by
        `read_file`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    proxyvis_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )
