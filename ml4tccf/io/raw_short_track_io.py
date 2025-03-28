"""Methods for reading and converting raw short-track data."""

import os
import glob
import pickle
import warnings
import numpy
import xarray
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking
from ml4tccf.io import short_track_io
from ml4tccf.utils import misc_utils

MINUTES_TO_SECONDS = 60
TEN_MINUTES_TO_SECONDS = 600

TIME_FORMAT_IN_FILES = '%Y-%m-%d-%H%M%S'
TIME_FORMAT_IN_FILE_NAMES = '%Y-%m-%d-%H-%M-%S'

VALID_TIME_KEY = 'st_one_sec_time'
LATITUDE_KEY = 'st_one_sec_lats'
LONGITUDE_KEY = 'st_one_sec_lond'

TIME_REGEX = '[0-9][0-9][0-9][0-9]-[0-1][0-9]-[0-3][0-9]-[0-2][0-9]-[0-5][0-9]-[0-5][0-9]'


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


def find_file(directory_name, cyclone_id_string, init_time_unix_sec,
              raise_error_if_missing=True):
    """Finds file with short-track data.

    :param directory_name: Name of directory.
    :param cyclone_id_string: Cyclone ID.
    :param init_time_unix_sec: Forecast-initialization time.
    :param raise_error_if_missing: Boolean flag.  If file is not found and
        `raise_error_if_missing == True`, this method will raise an error.  If
        file is not found and `raise_error_if_missing == False`, this method
        will return the expected file path.
    :return: short_track_file_name: Actual or expected file path.
    :raises: ValueError: if file is not found and
        `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_integer(init_time_unix_sec)
    error_checking.assert_equals(
        numpy.mod(init_time_unix_sec, TEN_MINUTES_TO_SECONDS),
        0
    )
    error_checking.assert_is_boolean(raise_error_if_missing)

    cyclone_year = misc_utils.parse_cyclone_id(cyclone_id_string)[0]
    short_track_file_name = (
        '{0:s}/a{1:s}/storm_track_interp_a{1:s}_{2:s}.pkl'
    ).format(
        directory_name,
        cyclone_year,
        _cyclone_id_new_to_orig(cyclone_id_string),
        time_conversion.unix_sec_to_string(
            init_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES
        )
    )

    if os.path.isfile(short_track_file_name) or not raise_error_if_missing:
        return short_track_file_name

    error_string = 'Cannot find short-track file.  Expected at: "{0:s}"'.format(
        short_track_file_name
    )
    raise ValueError(error_string)


def find_files_one_cyclone(
        directory_name, cyclone_id_string, raise_error_if_all_missing=True):
    """Finds all short-track files for one tropical cyclone.

    :param directory_name: Name of directory.
    :param cyclone_id_string: Cyclone ID.
    :param raise_error_if_all_missing: Boolean flag.  If no files are found and
        `raise_error_if_all_missing == True`, this method will raise an error.
        If no files are found and `raise_error_if_all_missing == False`, this
        method will return an empty list.
    :return: short_track_file_names: 1-D list of paths to existing files.
    :raises: ValueError: if no files are found and
        `raise_error_if_all_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_boolean(raise_error_if_all_missing)

    cyclone_year = misc_utils.parse_cyclone_id(cyclone_id_string)[0]
    file_pattern = (
        '{0:s}/a{1:s}/storm_track_interp_a{1:s}_{2:s}.pkl'
    ).format(
        directory_name,
        cyclone_year,
        _cyclone_id_new_to_orig(cyclone_id_string),
        TIME_REGEX
    )

    short_track_file_names = glob.glob(file_pattern)
    short_track_file_names.sort()

    if len(short_track_file_names) > 0 or not raise_error_if_all_missing:
        return short_track_file_names

    error_string = (
        'No files were found for cyclone {0:s} in directory: "{1:s}"'
    ).format(cyclone_id_string, directory_name)

    raise ValueError(error_string)


def file_name_to_init_time(short_track_file_name):
    """Parses init time from file name.

    :param short_track_file_name: Path to file with raw short-track data.
    :return: init_time_unix_sec: Forecast-init time.
    """

    pathless_file_name = os.path.split(short_track_file_name)[1]
    extensionless_file_name = pathless_file_name.split('.')[0]
    init_time_string = extensionless_file_name.split('_')[-1]

    return time_conversion.string_to_unix_sec(
        init_time_string, TIME_FORMAT_IN_FILE_NAMES
    )


def file_name_to_cyclone_id(short_track_file_name):
    """Parses cyclone ID from file name.

    :param short_track_file_name: Path to file with raw short-track data.
    :return: cyclone_id_string: Cyclone ID.
    """

    pathless_file_name = os.path.split(short_track_file_name)[1]
    extensionless_file_name = pathless_file_name.split('.')[0]

    return _cyclone_id_orig_to_new(
        extensionless_file_name.split('_')[-2][1:]
    )


def read_file(pickle_file_name, num_minutes_back, num_minutes_ahead):
    """Reads estimated TC-center locations from short-track file.

    :param pickle_file_name: Path to input file.
    :param num_minutes_back: Number of minutes back.  Letting
        M = `num_minutes_back`, this method will read estimated TC-center
        locations valid over the time period [init_time - M minutes, init_time].
    :param num_minutes_ahead: Number of minutes ahead.  Letting
        M = `num_minutes_ahead`, this method will read estimated TC-center
        locations valid over the time period [init_time, init_time + M minutes].
    :return: short_track_table_xarray: xarray table with estimated TC-center
        locations.
    """

    error_checking.assert_is_integer(num_minutes_back)
    error_checking.assert_is_greater(num_minutes_back, 0)
    error_checking.assert_equals(
        numpy.mod(num_minutes_back, 10),
        0
    )

    error_checking.assert_is_integer(num_minutes_ahead)
    error_checking.assert_is_greater(num_minutes_ahead, 0)
    error_checking.assert_equals(
        numpy.mod(num_minutes_ahead, 10),
        0
    )

    init_time_unix_sec = file_name_to_init_time(pickle_file_name)
    last_valid_time_unix_sec = (
        init_time_unix_sec + num_minutes_ahead * MINUTES_TO_SECONDS
    )
    first_valid_time_unix_sec = (
        init_time_unix_sec - num_minutes_back * MINUTES_TO_SECONDS
    )
    desired_valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_valid_time_unix_sec,
        end_time_unix_sec=last_valid_time_unix_sec,
        time_interval_sec=TEN_MINUTES_TO_SECONDS,
        include_endpoint=True
    )

    pickle_file_handle = open(pickle_file_name, 'rb')
    short_track_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    valid_time_strings = [
        t.strftime(TIME_FORMAT_IN_FILES)
        for t in short_track_dict[VALID_TIME_KEY]
    ]
    valid_times_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT_IN_FILES)
        for t in valid_time_strings
    ], dtype=int)

    good_indices = []
    for this_time in desired_valid_times_unix_sec:
        these_indices = numpy.where(valid_times_unix_sec == this_time)[0]

        assert len(these_indices) < 2
        if this_time <= init_time_unix_sec:
            assert len(these_indices) == 1

        if len(these_indices) > 0:
            good_indices.append(these_indices[0])
            continue

        warning_string = (
            'POTENTIAL ERROR: Cannot find short-track forecast init {0:s} and '
            'valid {1:s}.'
        ).format(
            time_conversion.unix_sec_to_string(
                init_time_unix_sec, '%Y-%m-%d-%H%M'
            ),
            time_conversion.unix_sec_to_string(this_time, '%Y-%m-%d-%H%M')
        )

        warnings.warn(warning_string)
        good_indices.append(-1)

    good_indices = numpy.array(good_indices, dtype=int)

    desired_latitudes_deg_n = short_track_dict[LATITUDE_KEY][good_indices]
    error_checking.assert_is_valid_lat_numpy_array(
        desired_latitudes_deg_n, allow_nan=False
    )
    desired_latitudes_deg_n[good_indices == -1] = numpy.nan

    desired_longitudes_deg_e = short_track_dict[LONGITUDE_KEY][good_indices]
    desired_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        desired_longitudes_deg_e, allow_nan=False
    )
    desired_longitudes_deg_e[good_indices == -1] = numpy.nan

    coord_dict = {
        short_track_io.INIT_TIME_DIM: numpy.array(
            [init_time_unix_sec], dtype=int
        ),
        short_track_io.LEAD_TIME_DIM: (
            desired_valid_times_unix_sec - init_time_unix_sec
        )
    }

    attribute_dict = {
        short_track_io.CYCLONE_ID_KEY: file_name_to_cyclone_id(pickle_file_name)
    }

    these_dims = (short_track_io.INIT_TIME_DIM, short_track_io.LEAD_TIME_DIM)
    main_data_dict = {
        short_track_io.LATITUDE_KEY: (
            these_dims, numpy.expand_dims(desired_latitudes_deg_n, axis=0)
        ),
        short_track_io.LONGITUDE_KEY: (
            these_dims, numpy.expand_dims(desired_longitudes_deg_e, axis=0)
        )
    }

    return xarray.Dataset(
        coords=coord_dict, data_vars=main_data_dict, attrs=attribute_dict
    )
