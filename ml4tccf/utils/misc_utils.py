"""Miscellaneous helper methods."""

import os
import tempfile
import numpy
from scipy.ndimage import distance_transform_edt
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking

TIME_FORMAT = '%Y %m %d %H %M %S'
ERROR_STRING = (
    '\nUnix command failed (log messages shown above should explain why).'
)

BATCH_SIZE_FOR_ALTITUDE_ANGLE = 100
DEFAULT_EXE_NAME_FOR_ALTITUDE_ANGLE = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/solarpos/solarpos'
)

NORTH_ATLANTIC_ID_STRING = 'AL'
SOUTH_ATLANTIC_ID_STRING = 'SL'
NORTHEAST_PACIFIC_ID_STRING = 'EP'
NORTH_CENTRAL_PACIFIC_ID_STRING = 'CP'
NORTHWEST_PACIFIC_ID_STRING = 'WP'
NORTH_INDIAN_ID_STRING = 'IO'
SOUTHERN_HEMISPHERE_ID_STRING = 'SH'

VALID_BASIN_ID_STRINGS = [
    NORTH_ATLANTIC_ID_STRING, SOUTH_ATLANTIC_ID_STRING,
    NORTHEAST_PACIFIC_ID_STRING, NORTH_CENTRAL_PACIFIC_ID_STRING,
    NORTHWEST_PACIFIC_ID_STRING, NORTH_INDIAN_ID_STRING,
    SOUTHERN_HEMISPHERE_ID_STRING
]


def check_basin_id(basin_id_string):
    """Ensures that basin ID is valid.

    :param basin_id_string: Basin ID.
    :raises: ValueError: if `basin_id_strings not in VALID_BASIN_ID_STRINGS`.
    """

    error_checking.assert_is_string(basin_id_string)
    if basin_id_string in VALID_BASIN_ID_STRINGS:
        return

    error_string = (
        'Basin ID ("{0:s}") must be in the following list:\n{1:s}'
    ).format(basin_id_string, str(VALID_BASIN_ID_STRINGS))

    raise ValueError(error_string)


def get_cyclone_id(year, basin_id_string, cyclone_number):
    """Creates cyclone ID from metadata.

    :param year: Year (integer).
    :param basin_id_string: Basin ID (must be accepted by `check_basin_id`).
    :param cyclone_number: Cyclone number (integer).
    :return: cyclone_id_string: Cyclone ID.
    """

    error_checking.assert_is_integer(year)
    error_checking.assert_is_geq(year, 0)
    check_basin_id(basin_id_string)
    error_checking.assert_is_integer(cyclone_number)
    error_checking.assert_is_greater(cyclone_number, 0)

    return '{0:04d}{1:s}{2:02d}'.format(year, basin_id_string, cyclone_number)


def parse_cyclone_id(cyclone_id_string):
    """Parses metadata from cyclone ID.

    :param cyclone_id_string: Cyclone ID, formatted like "yyyybbcc", where yyyy
        is the year; bb is the basin ID; and cc is the cyclone number ([cc]th
        cyclone of the season in the given basin).
    :return: year: Year (integer).
    :return: basin_id_string: Basin ID.
    :return: cyclone_number: Cyclone number (integer).
    """

    error_checking.assert_is_string(cyclone_id_string)
    assert len(cyclone_id_string) == 8

    year = int(cyclone_id_string[:4])
    error_checking.assert_is_geq(year, 0)

    basin_id_string = cyclone_id_string[4:6]
    check_basin_id(basin_id_string)

    cyclone_number = int(cyclone_id_string[6:])
    error_checking.assert_is_greater(cyclone_number, 0)

    return year, basin_id_string, cyclone_number


def fill_nans(data_matrix):
    """Fills NaN's with nearest neighbours.

    This method is adapted from the method `fill`, which you can find here:
    https://stackoverflow.com/posts/9262129/revisions

    :param data_matrix: numpy array of real-valued data.
    :return: data_matrix: Same but without NaN's.
    """

    error_checking.assert_is_real_numpy_array(data_matrix)

    indices = distance_transform_edt(
        numpy.isnan(data_matrix), return_distances=False, return_indices=True
    )
    return data_matrix[tuple(indices)]


def get_solar_altitude_angles(
        valid_time_unix_sec, latitudes_deg_n, longitudes_deg_e,
        temporary_dir_name,
        fortran_exe_name=DEFAULT_EXE_NAME_FOR_ALTITUDE_ANGLE):
    """Computes solar altitude angle at every point on Earth's surface.

    P = number of points

    :param valid_time_unix_sec: Valid time.
    :param latitudes_deg_n: length-P numpy array of latitudes (deg north).
    :param longitudes_deg_e: length-P numpy array of longitudes (deg south).
    :param temporary_dir_name: Name of directory for temporary text file.
    :param fortran_exe_name: Path to Fortran executable (pathless file name
        should probably be "solarpos").
    :return: altitude_angles_deg: length-P numpy array of solar altitude angles.
    """

    # Check input args.
    valid_time_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, TIME_FORMAT
    )
    error_checking.assert_is_string(temporary_dir_name)
    error_checking.assert_file_exists(fortran_exe_name)

    error_checking.assert_is_numpy_array(latitudes_deg_n, num_dimensions=1)
    num_points = len(latitudes_deg_n)
    error_checking.assert_is_numpy_array(
        longitudes_deg_e, exact_dimensions=numpy.array([num_points], dtype=int)
    )

    longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
        longitudes_deg_e, allow_nan=False
    )
    error_checking.assert_is_valid_lat_numpy_array(
        latitudes_deg_n, allow_nan=False
    )

    # Do actual stuff.
    temporary_file_name = tempfile.NamedTemporaryFile(
        dir=temporary_dir_name, delete=True
    ).name

    for i in range(0, num_points, BATCH_SIZE_FOR_ALTITUDE_ANGLE):
        if numpy.mod(i, 10 * BATCH_SIZE_FOR_ALTITUDE_ANGLE) == 0:
            print((
                'Have computed solar altitude angle for {0:d} of {1:d} '
                'points...'
            ).format(
                i, num_points
            ))

        first_index = i
        last_index = min([
            i + BATCH_SIZE_FOR_ALTITUDE_ANGLE, num_points
        ])

        command_string = '; '.join([
            '"{0:s}" {1:s} {2:.10f} {3:.10f} >> {4:s}'.format(
                fortran_exe_name, valid_time_string, y, x, temporary_file_name
            )
            for y, x in zip(
                latitudes_deg_n[first_index:last_index],
                longitudes_deg_e[first_index:last_index]
            )
        ])

        exit_code = os.system(command_string)
        if exit_code == 0:
            continue

        raise ValueError(ERROR_STRING)

    print('Have computed solar altitude angle for all {0:d} points!'.format(
        num_points
    ))

    found_header = False
    current_index = 0
    altitude_angles_deg = numpy.full(num_points, numpy.nan)

    for this_line in open(temporary_file_name, 'r').readlines():
        if not found_header:
            found_header = this_line.split()[0] == 'Time'
            continue

        try:
            altitude_angles_deg[current_index] = float(this_line.split()[3])
            found_header = False
            current_index += 1
        except ValueError:
            continue

    if os.path.isfile(temporary_file_name):
        os.remove(temporary_file_name)

    assert current_index == num_points
    return altitude_angles_deg
