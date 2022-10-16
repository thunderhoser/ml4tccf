"""Methods for reading and converting raw proxy-vis data."""

import os
import numpy
import xarray
import pyproj
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from ml4tccf.utils import proxyvis_utils

TOLERANCE_DEG = 1e-4
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
TIME_FORMAT_IN_FILE_NAME = '%Y-%m-%dT%H_%M_%S'

LATITUDE_KEY = 'latitude'
LONGITUDE_KEY = 'longitude'
X_COORD_KEY = 'x'
Y_COORD_KEY = 'y'
BIDIRECTIONAL_REFLECTANCE_KEY = 'proxy_vis'
START_TIME_KEY = 'start_time'
END_TIME_KEY = 'end_time'

PYPROJ_KEYWORD = 'PROJ CRS string: '


def file_name_to_time(proxyvis_file_name):
    """Parses valid time from file name.

    :param proxyvis_file_name: Path to file with raw proxy-vis data.
    :return: valid_time_unix_sec: Valid time.
    """

    pathless_file_name = os.path.split(proxyvis_file_name)[1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]
    valid_time_string = '_'.join(
        extensionless_file_name.split('_')[-3:]
    )

    return time_conversion.string_to_unix_sec(
        valid_time_string, TIME_FORMAT_IN_FILE_NAME
    )


def read_file(netcdf_file_name):
    """Reads proxy-vis data from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: proxyvis_table_xarray: xarray table with proxy-vis data.
    :raises: TypeError: if projection metadata are improperly formatted.
    :raises: ValueError: if projection metadata do not match coordinates in
        file.
    """

    proxyvis_table_xarray = xarray.open_dataset(netcdf_file_name)

    full_projection_string = proxyvis_table_xarray['cc'].attrs['crs_wkt']
    num_pyproj_strings = full_projection_string.count(PYPROJ_KEYWORD)

    if num_pyproj_strings != 1:
        error_string = (
            'Found PyProj keyword ("{0:s}") {1:d} times (expected once) in the '
            'following string:\n{2:s}'
        ).format(PYPROJ_KEYWORD, num_pyproj_strings, full_projection_string)

        raise TypeError(error_string)

    pyproj_string = full_projection_string.split(PYPROJ_KEYWORD)[-1]
    pyproj_string = pyproj_string.split('"')[0]
    pyproj_object = pyproj.Proj(pyproj_string)

    x_matrix_metres, y_matrix_metres = numpy.meshgrid(
        proxyvis_table_xarray[X_COORD_KEY].values,
        proxyvis_table_xarray[Y_COORD_KEY].values
    )
    longitude_matrix_deg_e, latitude_matrix_deg_n = pyproj_object(
        x_matrix_metres, y_matrix_metres, inverse=True
    )

    longitude_matrix_deg_e = lng_conversion.convert_lng_positive_in_west(
        longitude_matrix_deg_e
    )
    orig_longitude_matrix_deg_e = lng_conversion.convert_lng_positive_in_west(
        proxyvis_table_xarray[LONGITUDE_KEY].values
    )

    if not numpy.allclose(
            latitude_matrix_deg_n, proxyvis_table_xarray[LATITUDE_KEY].values,
            atol=TOLERANCE_DEG
    ):
        abs_diff_matrix_deg = numpy.absolute(
            latitude_matrix_deg_n - proxyvis_table_xarray[LATITUDE_KEY].values
        )

        error_string = (
            'Projected latitudes do not match those in file.  (Min, mean, max) '
            'absolute diffs = ({0:.4f}, {1:.4f}, {2:.4f}) deg.'
        ).format(
            numpy.min(abs_diff_matrix_deg), numpy.mean(abs_diff_matrix_deg),
            numpy.max(abs_diff_matrix_deg)
        )

        raise ValueError(error_string)

    # TODO(thunderhoser): This code does not properly handle wrap-around issues
    # for longitude.
    if not numpy.allclose(
            longitude_matrix_deg_e, orig_longitude_matrix_deg_e,
            atol=TOLERANCE_DEG
    ):
        abs_diff_matrix_deg = numpy.absolute(
            longitude_matrix_deg_e - orig_longitude_matrix_deg_e
        )

        error_string = (
            'Projected longitudes do not match those in file.  (Min, mean, max)'
            ' absolute diffs = ({0:.4f}, {1:.4f}, {2:.4f}) deg.'
        ).format(
            numpy.min(abs_diff_matrix_deg), numpy.mean(abs_diff_matrix_deg),
            numpy.max(abs_diff_matrix_deg)
        )

        raise ValueError(error_string)

    main_data_dict = {
        proxyvis_utils.X_COORD_KEY: (
            (proxyvis_utils.GRID_COLUMN_DIM,),
            proxyvis_table_xarray[X_COORD_KEY].values
        ),
        proxyvis_utils.Y_COORD_KEY: (
            (proxyvis_utils.GRID_ROW_DIM,),
            proxyvis_table_xarray[Y_COORD_KEY].values
        ),
        proxyvis_utils.BIDIRECTIONAL_REFLECTANCE_KEY: (
            (proxyvis_utils.GRID_ROW_DIM, proxyvis_utils.GRID_COLUMN_DIM),
            proxyvis_table_xarray[BIDIRECTIONAL_REFLECTANCE_KEY].values
        )
    }

    num_grid_rows = len(proxyvis_table_xarray[Y_COORD_KEY].values)
    num_grid_columns = len(proxyvis_table_xarray[X_COORD_KEY].values)

    metadata_dict = {
        proxyvis_utils.GRID_ROW_DIM: numpy.linspace(
            0, num_grid_rows - 1, num=num_grid_rows, dtype=int
        ),
        proxyvis_utils.GRID_COLUMN_DIM: numpy.linspace(
            0, num_grid_columns - 1, num=num_grid_columns, dtype=int
        )
    }

    pt = proxyvis_table_xarray

    start_time_string = (
        pt[BIDIRECTIONAL_REFLECTANCE_KEY].attrs[START_TIME_KEY]
    )
    start_time_unix_sec = time_conversion.string_to_unix_sec(
        start_time_string.split('.')[0], TIME_FORMAT
    )
    this_digit = int(start_time_string.split('.')[-1][0])
    if this_digit >= 5:
        start_time_unix_sec += 1

    end_time_string = (
        pt[BIDIRECTIONAL_REFLECTANCE_KEY].attrs[END_TIME_KEY]
    )
    end_time_unix_sec = time_conversion.string_to_unix_sec(
        end_time_string.split('.')[0], TIME_FORMAT
    )
    this_digit = int(end_time_string.split('.')[-1][0])
    if this_digit >= 5:
        end_time_unix_sec += 1

    attributes_dict = {
        proxyvis_utils.PYPROJ_STRING_KEY: pyproj_string,
        proxyvis_utils.START_TIME_KEY: start_time_unix_sec,
        proxyvis_utils.END_TIME_KEY: end_time_unix_sec
    }

    return xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict, attrs=attributes_dict
    )
