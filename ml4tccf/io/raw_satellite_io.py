"""Methods for reading and converting raw satellite data."""

import os
import glob
import numpy
import xarray
import pyproj
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking
from ml4tccf.utils import misc_utils
from ml4tccf.utils import satellite_utils

MICRONS_TO_METRES = 1e-6
PYPROJ_KEYWORD = 'PROJ CRS string: '

TIME_FORMAT = '%Y-%m-%dT%H-%M-%S'
DATE_REGEX = '[0-9][0-9][0-9][0-9]-[0-1][0-9]-[0-3][0-9]'
TIME_REGEX = (
    '[0-9][0-9][0-9][0-9]-[0-1][0-9]-[0-3][0-9]'
    'T[0-2][0-9]-[0-5][0-9]-[0-5][0-9]'
)

LOW_RES_CHANNEL_KEYS = [
    'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16'
]
HIGH_RES_CHANNEL_KEYS = ['C02']

X_COORD_KEY = 'x'
Y_COORD_KEY = 'y'
LATITUDE_KEY = 'latitude'
LONGITUDE_KEY = 'longitude'
CYCLONE_ID_KEY = 'atcf_id'


def _cyclone_id_orig_to_new(orig_cyclone_id_string):
    """Converts cyclone ID from original format to new format.

    :param orig_cyclone_id_string: Original ID (format bbYYYYNN), where bb is
        the basin; YYYY is the year; and NN is the ordinal number.
    :return: cyclone_id_string: Proper ID (format YYYYBBNN).
    """

    cyclone_id_string = '{0:s}{1:s}'.format(
        orig_cyclone_id_string[-4:], orig_cyclone_id_string[:4].upper()
    )
    _ = misc_utils.parse_cyclone_id(cyclone_id_string)

    return cyclone_id_string


def _cyclone_id_new_to_orig(cyclone_id_string):
    """Converts cyclone ID from new format to original format.

    :param cyclone_id_string: See doc for `_cyclone_id_orig_to_new`.
    :return: orig_cyclone_id_string: Same.
    """

    _ = misc_utils.parse_cyclone_id(cyclone_id_string)
    return '{0:s}{1:s}'.format(
        cyclone_id_string[-4:].lower(), cyclone_id_string[:4]
    )


def find_files_one_tc(
        directory_name, cyclone_id_string, look_for_high_res,
        raise_error_if_all_missing=True):
    """Finds all files with satellite data for one tropical cyclone.

    :param directory_name: Name of directory.
    :param cyclone_id_string: Cyclone ID.
    :param look_for_high_res: Boolean flag.  If True (False), will look for
        high- (low-)resolution data.
    :param raise_error_if_all_missing: Boolean flag.  If all files are missing
        and `raise_error_if_all_missing == True`, will throw error.
    :return: satellite_file_names: 1-D list of paths to satellite files.  This
        list does *not* contain expected paths to non-existent files.
    :raises: ValueError: if all files are missing.
    """

    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_boolean(look_for_high_res)
    error_checking.assert_is_boolean(raise_error_if_all_missing)
    orig_cyclone_id_string = _cyclone_id_new_to_orig(cyclone_id_string)

    satellite_file_pattern = (
        '{0:s}/{1:s}/{2:s}/{3:s}/{4:s}_{3:s}_{1:s}.nc'
    ).format(
        directory_name,
        orig_cyclone_id_string,
        DATE_REGEX,
        TIME_REGEX,
        '0500m' if look_for_high_res else '2000m'
    )

    satellite_file_names = glob.glob(satellite_file_pattern)

    if len(satellite_file_names) > 0:
        satellite_file_names.sort()
        return satellite_file_names

    if not raise_error_if_all_missing:
        return satellite_file_names

    error_string = (
        'Cannot find any file in directory "{0:s}" for cyclone ID "{1:s}".'
    ).format(
        directory_name, cyclone_id_string
    )
    raise ValueError(error_string)


def find_file(directory_name, cyclone_id_string, valid_time_unix_sec,
              look_for_high_res, raise_error_if_missing=True):
    """Finds file with satellite data.

    :param directory_name: Name of directory.
    :param cyclone_id_string: Cyclone ID.
    :param valid_time_unix_sec: Valid time.
    :param look_for_high_res: Boolean flag.  If True (False), will look for
        high- (low-)resolution data.
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
    error_checking.assert_is_boolean(look_for_high_res)
    orig_cyclone_id_string = _cyclone_id_new_to_orig(cyclone_id_string)

    valid_time_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, TIME_FORMAT
    )
    valid_date_string = valid_time_string[:10]

    satellite_file_name = '{0:s}/{1:s}/{2:s}/{3:s}/{4:s}_{3:s}_{1:s}.nc'.format(
        directory_name,
        orig_cyclone_id_string,
        valid_date_string,
        valid_time_string,
        '0500m' if look_for_high_res else '2000m'
    )

    if os.path.isfile(satellite_file_name):
        return satellite_file_name

    if not raise_error_if_missing:
        return satellite_file_name

    error_string = 'Cannot find satellite file.  Expected at: "{0:s}"'.format(
        satellite_file_name
    )
    raise ValueError(error_string)


def file_name_to_time(satellite_file_name):
    """Parses valid time from file name.

    :param satellite_file_name: Path to file with raw satellite data.
    :return: valid_time_unix_sec: Valid time.
    """

    pathless_file_name = os.path.split(satellite_file_name)[1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]
    valid_time_string = extensionless_file_name.split('_')[1]

    return time_conversion.string_to_unix_sec(valid_time_string, TIME_FORMAT)


def read_file(satellite_file_name, is_high_res):
    """Reads satellite data from raw file.

    :param satellite_file_name: Path to input file (NetCDF format).
    :param is_high_res: Boolean flag.  If True (False), will expect high-
        (low-)resolution data.
    :return: satellite_table_xarray: xarray table.  Variable names and metadata
        should make this table self-explanatory.
    """

    # Error-checking.
    error_checking.assert_is_boolean(is_high_res)

    orig_satellite_table_xarray = xarray.open_dataset(satellite_file_name)
    cyclone_id_string = _cyclone_id_orig_to_new(
        orig_satellite_table_xarray.attrs[CYCLONE_ID_KEY]
    )
    valid_time_unix_sec = file_name_to_time(satellite_file_name)

    full_projection_string = orig_satellite_table_xarray['eqc'].attrs['crs_wkt']
    num_pyproj_strings = full_projection_string.count(PYPROJ_KEYWORD)

    if num_pyproj_strings != 1:
        error_string = (
            'Found PyProj keyword ("{0:s}") {1:d} times (expected once) in the '
            'following string:\n{2:s}'
        ).format(PYPROJ_KEYWORD, num_pyproj_strings, full_projection_string)

        raise TypeError(error_string)

    pyproj_string = full_projection_string.split(PYPROJ_KEYWORD)[-1]
    pyproj_string = pyproj_string.split('"')[0]
    pyproj_string = pyproj_string.ljust(100, ' ')
    _ = pyproj.Proj(pyproj_string)

    x_coords_metres = orig_satellite_table_xarray.coords[X_COORD_KEY].values
    assert (
        numpy.all(numpy.diff(x_coords_metres) > 0)
        or numpy.all(numpy.diff(x_coords_metres) < 0)
    )

    y_coords_metres = orig_satellite_table_xarray.coords[Y_COORD_KEY].values
    assert (
        numpy.all(numpy.diff(y_coords_metres) > 0)
        or numpy.all(numpy.diff(y_coords_metres) < 0)
    )

    latitudes_deg_n = orig_satellite_table_xarray[LATITUDE_KEY].values
    error_checking.assert_is_valid_lat_numpy_array(latitudes_deg_n)

    were_latitudes_upside_down = numpy.all(numpy.diff(latitudes_deg_n) < 0)
    if were_latitudes_upside_down:
        latitudes_deg_n = latitudes_deg_n[::-1]
        y_coords_metres = y_coords_metres[::-1]

    assert numpy.all(numpy.diff(latitudes_deg_n) > 0)

    longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        longitudes_deg=orig_satellite_table_xarray[LONGITUDE_KEY].values,
        allow_nan=False
    )

    # TODO(thunderhoser): I need to think about this choice some more.
    if not numpy.all(numpy.diff(longitudes_deg_e) > 0):
        longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
            longitudes_deg=longitudes_deg_e, allow_nan=False
        )

    assert numpy.all(numpy.diff(longitudes_deg_e) > 0)

    # Do actual stuff.
    if is_high_res:
        channel_keys = HIGH_RES_CHANNEL_KEYS
    else:
        channel_keys = LOW_RES_CHANNEL_KEYS

    num_rows = len(latitudes_deg_n)
    num_columns = len(longitudes_deg_e)
    num_channels = len(channel_keys)
    data_matrix = numpy.full(
        (1, num_rows, num_columns, num_channels), numpy.nan
    )
    wavelengths_metres = numpy.full(num_channels, numpy.nan)

    for k in range(num_channels):
        if were_latitudes_upside_down:
            data_matrix[0, ..., k] = numpy.flip(
                orig_satellite_table_xarray[channel_keys[k]].values, axis=0
            )
        else:
            data_matrix[0, ..., k] = (
                orig_satellite_table_xarray[channel_keys[k]].values
            )

        wavelengths_metres[k] = MICRONS_TO_METRES * float(
            orig_satellite_table_xarray[
                channel_keys[k]
            ].attrs['wavelength'].split()[0]
        )

    if is_high_res:
        data_matrix *= 0.01

        x_coord_key = satellite_utils.X_COORD_HIGH_RES_KEY
        y_coord_key = satellite_utils.Y_COORD_HIGH_RES_KEY
        latitude_key = satellite_utils.LATITUDE_HIGH_RES_KEY
        longitude_key = satellite_utils.LONGITUDE_HIGH_RES_KEY
        main_data_key = satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY

        row_dim = satellite_utils.HIGH_RES_ROW_DIM
        column_dim = satellite_utils.HIGH_RES_COLUMN_DIM
        wavelength_dim = satellite_utils.HIGH_RES_WAVELENGTH_DIM
    else:
        x_coord_key = satellite_utils.X_COORD_LOW_RES_KEY
        y_coord_key = satellite_utils.Y_COORD_LOW_RES_KEY
        latitude_key = satellite_utils.LATITUDE_LOW_RES_KEY
        longitude_key = satellite_utils.LONGITUDE_LOW_RES_KEY
        main_data_key = satellite_utils.BRIGHTNESS_TEMPERATURE_KEY

        row_dim = satellite_utils.LOW_RES_ROW_DIM
        column_dim = satellite_utils.LOW_RES_COLUMN_DIM
        wavelength_dim = satellite_utils.LOW_RES_WAVELENGTH_DIM

    main_data_dict = {
        x_coord_key: (
            (satellite_utils.TIME_DIM, column_dim),
            numpy.expand_dims(x_coords_metres, axis=0).astype('float32')
        ),
        y_coord_key: (
            (satellite_utils.TIME_DIM, row_dim),
            numpy.expand_dims(y_coords_metres, axis=0).astype('float32')
        ),
        latitude_key: (
            (satellite_utils.TIME_DIM, row_dim),
            numpy.expand_dims(latitudes_deg_n, axis=0).astype('float32')
        ),
        longitude_key: (
            (satellite_utils.TIME_DIM, column_dim),
            numpy.expand_dims(longitudes_deg_e, axis=0).astype('float32')
        ),
        main_data_key: (
            (satellite_utils.TIME_DIM, row_dim, column_dim, wavelength_dim),
            data_matrix.astype('float32')
        ),
        satellite_utils.CYCLONE_ID_KEY: (
            (satellite_utils.TIME_DIM,), numpy.array([cyclone_id_string])
        ),
        satellite_utils.PYPROJ_STRING_KEY: (
            (satellite_utils.TIME_DIM,), numpy.array([pyproj_string])
        )
    }

    num_grid_rows = len(y_coords_metres)
    num_grid_columns = len(x_coords_metres)

    metadata_dict = {
        row_dim: numpy.linspace(
            0, num_grid_rows - 1, num=num_grid_rows, dtype=int
        ),
        column_dim: numpy.linspace(
            0, num_grid_columns - 1, num=num_grid_columns, dtype=int
        ),
        wavelength_dim: wavelengths_metres,
        satellite_utils.TIME_DIM: numpy.array([valid_time_unix_sec], dtype=int)
    }

    return xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )


def merge_low_and_high_res(low_res_satellite_table_xarray,
                           high_res_satellite_table_xarray):
    """Merges low- and high-res satellite data.

    :param low_res_satellite_table_xarray: xarray table in format created by
        `read_file`.
    :param high_res_satellite_table_xarray: xarray table in format created by
        `read_file`.
    :return: satellite_table_xarray: xarray table with both high- and low-res
        data.
    """

    return xarray.merge(
        [low_res_satellite_table_xarray, high_res_satellite_table_xarray],
        compat='identical', join='exact'
    )
