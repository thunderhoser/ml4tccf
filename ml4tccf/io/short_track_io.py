"""Input/output methods for pre-processed short-track data."""

import os
import numpy
import xarray
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4tccf.utils import misc_utils

CYCLONE_ID_KEY = 'cyclone_id_string'

INIT_TIME_DIM = 'init_time_unix_sec'
LEAD_TIME_DIM = 'lead_time_seconds'

LATITUDE_KEY = 'latitude_deg_n'
LONGITUDE_KEY = 'longitude_deg_e'


def find_file(directory_name, cyclone_id_string, raise_error_if_missing=True):
    """Finds file with short-track data.

    :param directory_name: Name of directory.
    :param cyclone_id_string: Cyclone ID.
    :param raise_error_if_missing: Boolean flag.  If file is not found and
        `raise_error_if_missing == True`, this method will raise an error.  If
        file is not found and `raise_error_if_missing == False`, this method
        will return the expected file path.
    :return: short_track_file_name: Actual or expected file path.
    :raises: ValueError: if file is not found and
        `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    _ = misc_utils.parse_cyclone_id(cyclone_id_string)
    short_track_file_name = '{0:s}/short_track_{1:d}.nc'.format(
        directory_name, cyclone_id_string
    )

    if os.path.isfile(short_track_file_name) or not raise_error_if_missing:
        return short_track_file_name

    error_string = 'Cannot find short-track file.  Expected at: "{0:s}"'.format(
        short_track_file_name
    )
    raise ValueError(error_string)


def read_file(netcdf_file_name):
    """Reads short-track data from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: short_track_table_xarray: xarray table.  Documentation in the
        xarray table should make values self-explanatory.
    """

    return xarray.open_dataset(netcdf_file_name)


def write_file(short_track_table_xarray, netcdf_file_name):
    """Writes short-track data to NetCDF file.

    :param short_track_table_xarray: xarray table in format returned by
        `read_file`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    short_track_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def concat_over_init_time(short_track_tables_xarray):
    """Concatenates many short-track tables, each with different init time.

    :param short_track_tables_xarray: 1-D list of tables.
    :return: short_track_table_xarray: Single table.
    :raises: ValueError: if tables do not all contain data for the same tropical
        cyclone.
    """

    cyclone_id_strings = [
        t.attrs[CYCLONE_ID_KEY] for t in short_track_tables_xarray
    ]
    unique_cyclone_id_strings = list(set(cyclone_id_strings))

    if len(unique_cyclone_id_strings) > 1:
        error_string = (
            'Cannot concatenate data from different cyclones.  In this '
            'case, data come from the following unique cyclones:\n{0:s}'
        ).format(str(unique_cyclone_id_strings))

        raise ValueError(error_string)

    init_times_unix_sec = numpy.concatenate(
        [t.coords[INIT_TIME_DIM].values for t in short_track_tables_xarray]
    )
    unique_init_times_unix_sec, unique_counts = numpy.unique(
        init_times_unix_sec, return_counts=True
    )

    if numpy.any(unique_counts > 1):
        bad_init_time_strings = [
            time_conversion.unix_sec_to_string(t, '%Y-%m-%d-%H%M%S')
            for t in unique_init_times_unix_sec[unique_counts > 1]
        ]
        error_string = (
            'Init times in list of tables are not unique.  The following init '
            'times appear more than once:\n{0:s}'
        ).format(str(bad_init_time_strings))

        raise ValueError(error_string)

    return xarray.concat(
        short_track_tables_xarray, dim=INIT_TIME_DIM, data_vars='all',
        coords='minimal', compat='identical', join='exact'
    )
