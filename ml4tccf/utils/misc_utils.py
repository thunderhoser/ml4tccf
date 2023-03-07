"""Miscellaneous helper methods."""

import os
import gzip
import shutil
import warnings
import tempfile
import numpy
from scipy.ndimage import center_of_mass, distance_transform_edt
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6

GZIP_FILE_EXTENSION = '.gz'
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


def is_regular_grid_valid(latitudes_deg_n, longitudes_deg_e):
    """Determines validity of coordinates in regular grid.

    :param latitudes_deg_n: 1-D numpy array of latitudes (deg north).
    :param longitudes_deg_e: 1-D numpy array of longitudes (deg east).
    :return: is_grid_valid: Boolean flag.
    :return: increasing_latitudes_deg_n: Same as input, except that array is
        monotonically increasing.
    :return: increasing_longitudes_deg_e: Same as input, except that array is
        monotonically increasing.
    """

    try:
        error_checking.assert_is_numpy_array(latitudes_deg_n, num_dimensions=1)
        error_checking.assert_is_numpy_array(longitudes_deg_e, num_dimensions=1)

        error_checking.assert_is_valid_lat_numpy_array(
            latitudes_deg_n, allow_nan=False
        )
        error_checking.assert_is_valid_lng_numpy_array(
            longitudes_deg_e, allow_nan=False
        )
    except:
        return False, None, None

    forward_latitudes_deg_n = latitudes_deg_n + 0.
    backwards_latitudes_deg_n = latitudes_deg_n[::-1]
    forward_latitude_diffs_deg = numpy.diff(forward_latitudes_deg_n)
    backwards_latitude_diffs_deg = numpy.diff(backwards_latitudes_deg_n)

    if not (
            numpy.all(forward_latitude_diffs_deg > 0) or
            numpy.all(backwards_latitude_diffs_deg > 0)
    ):
        these_counts = numpy.array([
            numpy.sum(forward_latitude_diffs_deg > 0),
            numpy.sum(backwards_latitude_diffs_deg > 0)
        ], dtype=int)

        if numpy.argmax(these_counts) == 0:
            these_latitudes_deg = forward_latitudes_deg_n
            these_latitude_diffs_deg = forward_latitude_diffs_deg
        else:
            these_latitudes_deg = backwards_latitudes_deg_n
            these_latitude_diffs_deg = backwards_latitude_diffs_deg

        bad_indices = numpy.where(
            numpy.invert(these_latitude_diffs_deg > 0)
        )[0]

        if len(bad_indices) > 0:
            warning_string = (
                'Latitudes are not monotonically increasing.  Non-positive '
                'differences are as follows:'
            )
            for i in bad_indices:
                warning_string += '\n{0:.4f} to {1:.4f} deg N'.format(
                    these_latitudes_deg[i], these_latitudes_deg[i + 1]
                )

            warnings.warn(warning_string)

        return False, None, None

    positive_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        longitudes_deg_e + 0.
    )
    negative_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
        longitudes_deg_e + 0.
    )

    positive_longitude_diffs_deg = numpy.diff(positive_longitudes_deg_e)
    negative_longitude_diffs_deg = numpy.diff(negative_longitudes_deg_e)

    if not (
            numpy.all(positive_longitude_diffs_deg > 0) or
            numpy.all(negative_longitude_diffs_deg > 0)
    ):
        these_counts = numpy.array([
            numpy.sum(positive_longitude_diffs_deg > 0),
            numpy.sum(negative_longitude_diffs_deg > 0)
        ], dtype=int)

        if numpy.argmax(these_counts) == 0:
            these_longitudes_deg_e = positive_longitudes_deg_e
            these_longitude_diffs_deg = positive_longitude_diffs_deg
        else:
            these_longitudes_deg_e = negative_longitudes_deg_e
            these_longitude_diffs_deg = negative_longitude_diffs_deg

        bad_indices = numpy.where(
            numpy.invert(these_longitude_diffs_deg > 0)
        )[0]

        if len(bad_indices) > 0:
            warning_string = (
                'Longitudes are not monotonically increasing.  Non-positive '
                'differences are as follows:'
            )
            for i in bad_indices:
                warning_string += '\n{0:.4f} to {1:.4f} deg E'.format(
                    these_longitudes_deg_e[i], these_longitudes_deg_e[i + 1]
                )

            warnings.warn(warning_string)

        return False, None, None

    if numpy.all(forward_latitude_diffs_deg > 0):
        increasing_latitudes_deg_n = forward_latitudes_deg_n
    else:
        increasing_latitudes_deg_n = backwards_latitudes_deg_n

    if numpy.all(numpy.diff(longitudes_deg_e) > 0):
        increasing_longitudes_deg_e = longitudes_deg_e
    elif numpy.all(positive_longitude_diffs_deg > 0):
        increasing_longitudes_deg_e = positive_longitudes_deg_e
    else:
        increasing_longitudes_deg_e = negative_longitudes_deg_e

    return True, increasing_latitudes_deg_n, increasing_longitudes_deg_e


def gzip_file(input_file_name):
    """Compresses file via gzip.

    :param input_file_name: Path to input file.
    :raises: ValueError: if file is already gzipped.
    """

    error_checking.assert_file_exists(input_file_name)
    if input_file_name.endswith(GZIP_FILE_EXTENSION):
        raise ValueError(
            'File is already gzipped: "{0:s}"'.format(input_file_name)
        )

    gzipped_file_name = '{0:s}{1:s}'.format(input_file_name, GZIP_FILE_EXTENSION)

    with open(input_file_name, 'rb') as netcdf_handle:
        with gzip.open(gzipped_file_name, 'wb') as gzip_handle:
            shutil.copyfileobj(netcdf_handle, gzip_handle)


def gunzip_file(gzipped_file_name):
    """Deompresses file via gunzip.

    :param gzipped_file_name: Path to gzipped file.
    :raises: ValueError: if file is not gzipped.
    """

    error_checking.assert_is_string(gzipped_file_name)
    if not gzipped_file_name.endswith(GZIP_FILE_EXTENSION):
        raise ValueError(
            'File is not gzipped: "{0:s}"'.format(gzipped_file_name)
        )

    unzipped_file_name = gzipped_file_name[:-len(GZIP_FILE_EXTENSION)]

    with gzip.open(gzipped_file_name, 'rb') as gzip_handle:
        with open(unzipped_file_name, 'wb') as unzipped_handle:
            shutil.copyfileobj(gzip_handle, unzipped_handle)


def confidence_interval_to_polygon(
        x_value_matrix, y_value_matrix, confidence_level, same_order):
    """Turns confidence interval into polygon.

    P = number of points
    B = number of bootstrap replicates
    V = number of vertices in resulting polygon = 2 * P + 1

    :param x_value_matrix: P-by-B numpy array of x-values.
    :param y_value_matrix: P-by-B numpy array of y-values.
    :param confidence_level: Confidence level (in range 0...1).
    :param same_order: Boolean flag.  If True (False), minimum x-values will be
        matched with minimum (maximum) y-values.
    :return: polygon_coord_matrix: V-by-2 numpy array of coordinates
        (x-coordinates in first column, y-coords in second).
    """

    error_checking.assert_is_numpy_array(x_value_matrix, num_dimensions=2)
    error_checking.assert_is_numpy_array(y_value_matrix, num_dimensions=2)

    expected_dim = numpy.array([
        x_value_matrix.shape[0], y_value_matrix.shape[1]
    ], dtype=int)

    error_checking.assert_is_numpy_array(
        y_value_matrix, exact_dimensions=expected_dim
    )

    error_checking.assert_is_geq(confidence_level, 0.9)
    error_checking.assert_is_leq(confidence_level, 1.)
    error_checking.assert_is_boolean(same_order)

    min_percentile = 50 * (1. - confidence_level)
    max_percentile = 50 * (1. + confidence_level)

    x_values_bottom = numpy.nanpercentile(
        x_value_matrix, min_percentile, axis=1, interpolation='linear'
    )
    x_values_top = numpy.nanpercentile(
        x_value_matrix, max_percentile, axis=1, interpolation='linear'
    )
    y_values_bottom = numpy.nanpercentile(
        y_value_matrix, min_percentile, axis=1, interpolation='linear'
    )
    y_values_top = numpy.nanpercentile(
        y_value_matrix, max_percentile, axis=1, interpolation='linear'
    )

    real_indices = numpy.where(numpy.invert(numpy.logical_or(
        numpy.isnan(x_values_bottom), numpy.isnan(y_values_bottom)
    )))[0]

    if len(real_indices) == 0:
        return None

    x_values_bottom = x_values_bottom[real_indices]
    x_values_top = x_values_top[real_indices]
    y_values_bottom = y_values_bottom[real_indices]
    y_values_top = y_values_top[real_indices]

    x_vertices = numpy.concatenate((
        x_values_top, x_values_bottom[::-1], x_values_top[[0]]
    ))

    if same_order:
        y_vertices = numpy.concatenate((
            y_values_top, y_values_bottom[::-1], y_values_top[[0]]
        ))
    else:
        y_vertices = numpy.concatenate((
            y_values_bottom, y_values_top[::-1], y_values_bottom[[0]]
        ))

    return numpy.transpose(numpy.vstack((
        x_vertices, y_vertices
    )))


def target_matrix_to_centroid(target_matrix, test_mode=False):
    """Converts target matrix to centroid (x- and y-coord).

    M = number of rows in grid
    N = number of columns in grid

    :param target_matrix: M-by-N numpy array of "true probabilities" for
        TC-center location.
    :param test_mode: Leave this alone.
    :return: row_offset_px: Row offset (pixels north of grid center).
    :return: column_offset_px: Column offset (pixels east of grid center).
    """

    error_checking.assert_is_numpy_array(target_matrix, num_dimensions=2)
    error_checking.assert_is_geq_numpy_array(target_matrix, 0.)
    error_checking.assert_is_leq_numpy_array(target_matrix, 1.)
    assert numpy.isclose(numpy.sum(target_matrix), 1.)

    error_checking.assert_is_boolean(test_mode)

    num_grid_rows = target_matrix.shape[0]
    num_grid_columns = target_matrix.shape[1]
    error_checking.assert_equals(numpy.mod(num_grid_rows, 2), 0)
    error_checking.assert_equals(numpy.mod(num_grid_columns, 2), 0)

    sorted_target_values = numpy.sort(numpy.ravel(target_matrix))[::-1]
    top_four_range = (
        numpy.max(sorted_target_values[:4]) -
        numpy.min(sorted_target_values[:4])
    )
    assert top_four_range <= TOLERANCE

    if test_mode:
        top_five_range = (
            numpy.max(sorted_target_values[:5]) -
            numpy.min(sorted_target_values[:5])
        )
        assert top_five_range > TOLERANCE

        centroid_indices_linear = numpy.argsort(
            -1 * numpy.ravel(target_matrix)
        )[:4]
    else:
        centroid_indices_linear = numpy.where(
            numpy.max(target_matrix) - numpy.ravel(target_matrix) < TOLERANCE
        )[0]
        print(len(centroid_indices_linear))

    centroid_row_indices, centroid_column_indices = numpy.unravel_index(
        centroid_indices_linear, target_matrix.shape
    )
    centroid_row_index = numpy.mean(centroid_row_indices.astype(float))
    centroid_column_index = numpy.mean(centroid_column_indices.astype(float))

    image_center_row_index = 0.5 * num_grid_rows - 0.5
    image_center_column_index = 0.5 * num_grid_columns - 0.5

    return (
        int(numpy.round(centroid_row_index - image_center_row_index)),
        int(numpy.round(centroid_column_index - image_center_column_index))
    )


def prediction_matrix_to_centroid(prediction_matrix):
    """Converts prediction matrix to centroid (x- and y-coord).

    M = number of rows in grid
    N = number of columns in grid

    :param prediction_matrix: M-by-N numpy array of probabilities for TC-center
        location.
    :return: row_offset_px: Row offset (pixels north of grid center).
    :return: column_offset_px: Column offset (pixels east of grid center).
    """

    error_checking.assert_is_numpy_array(prediction_matrix, num_dimensions=2)
    error_checking.assert_is_geq_numpy_array(prediction_matrix, 0.)
    error_checking.assert_is_leq_numpy_array(prediction_matrix, 1.)
    assert numpy.isclose(numpy.sum(prediction_matrix), 1.)

    num_grid_rows = prediction_matrix.shape[0]
    num_grid_columns = prediction_matrix.shape[1]
    error_checking.assert_equals(numpy.mod(num_grid_rows, 2), 0)
    error_checking.assert_equals(numpy.mod(num_grid_columns, 2), 0)

    centroid_row_index, centroid_column_index = center_of_mass(
        prediction_matrix
    )

    image_center_row_index = 0.5 * num_grid_rows - 0.5
    image_center_column_index = 0.5 * num_grid_columns - 0.5

    return (
        centroid_row_index - image_center_row_index,
        centroid_column_index - image_center_column_index
    )
