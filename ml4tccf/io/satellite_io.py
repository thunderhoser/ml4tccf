"""Methods for reading and writing satellite data."""

import os
import glob
import xarray
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4tccf.utils import misc_utils

DATE_FORMAT = '%Y-%m-%d'
CYCLONE_ID_REGEX = '[0-9][0-9][0-9][0-9][A-Z][A-Z][0-9][0-9]'
DATE_REGEX = '[0-9][0-9][0-9][0-9]-[0-1][0-9]-[0-3][0-9]'


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


def find_files_one_cyclone(
        directory_name, cyclone_id_string, raise_error_if_all_missing=True):
    """Finds all satellite files for one tropical cyclone.

    :param directory_name: Name of directory.
    :param cyclone_id_string: Cyclone ID.
    :param raise_error_if_all_missing: Boolean flag.  If no files are found and
        `raise_error_if_all_missing == True`, this method will raise an error.
        If no files are found and `raise_error_if_all_missing == False`, this
        method will return an empty list.
    :return: satellite_file_names: 1-D list of paths to existing files.
    :raises: ValueError: if no files are found and
        `raise_error_if_all_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    _ = misc_utils.parse_cyclone_id(cyclone_id_string)
    error_checking.assert_is_boolean(raise_error_if_all_missing)

    file_pattern = '{0:s}/satellite_{1:s}_{2:s}.nc'.format(
        directory_name, cyclone_id_string, DATE_REGEX
    )
    satellite_file_names = glob.glob(file_pattern)
    satellite_file_names.sort()

    if len(satellite_file_names) > 0 or not raise_error_if_all_missing:
        return satellite_file_names

    error_string = (
        'No files were found for cyclone {0:s} in directory: "{1:s}"'
    ).format(cyclone_id_string, directory_name)

    raise ValueError(error_string)


def file_name_to_cyclone_id(satellite_file_name):
    """Parses cyclone ID from name of satellite file.

    :param satellite_file_name: File path.
    :return: cyclone_id_string: Cyclone ID.
    """

    error_checking.assert_is_string(satellite_file_name)
    pathless_file_name = os.path.split(satellite_file_name)[1]

    cyclone_id_string = pathless_file_name.split('.')[0].split('_')[-2]
    _ = misc_utils.parse_cyclone_id(cyclone_id_string)

    return cyclone_id_string


def file_name_to_date(satellite_file_name):
    """Parses date from name of satellite file.

    :param satellite_file_name: File path.
    :return: valid_date_string: Valid date.
    """

    error_checking.assert_is_string(satellite_file_name)
    pathless_file_name = os.path.split(satellite_file_name)[1]

    valid_date_string = pathless_file_name.split('.')[0].split('_')[-1]
    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)

    return valid_date_string


def find_cyclones(directory_name, raise_error_if_all_missing=True):
    """Finds all cyclones.

    :param directory_name: Name of directory with satellite files.
    :param raise_error_if_all_missing: Boolean flag.  If no cyclones are found
        and `raise_error_if_all_missing == True`, will throw error.  If no
        cyclones are found and `raise_error_if_all_missing == False`, will
        return empty list.
    :return: cyclone_id_strings: List of cyclone IDs.
    :raises: ValueError: if no files are found and
        `raise_error_if_all_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_boolean(raise_error_if_all_missing)

    file_pattern = '{0:s}/satellite_{1:s}_{2:s}.nc'.format(
        directory_name, CYCLONE_ID_REGEX, DATE_REGEX
    )
    satellite_file_names = glob.glob(file_pattern)

    # file_pattern = '{0:s}{1:s}'.format(file_pattern, GZIP_FILE_EXTENSION)
    # satellite_file_names += glob.glob(file_pattern)

    cyclone_id_strings = []

    for this_file_name in satellite_file_names:
        try:
            cyclone_id_strings.append(
                file_name_to_cyclone_id(this_file_name)
            )
        except:
            pass

    cyclone_id_strings = list(set(cyclone_id_strings))
    cyclone_id_strings.sort()

    if raise_error_if_all_missing and len(cyclone_id_strings) == 0:
        error_string = (
            'Could not find any cyclone IDs from files with pattern: "{0:s}"'
        ).format(file_pattern)

        raise ValueError(error_string)

    return cyclone_id_strings


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
