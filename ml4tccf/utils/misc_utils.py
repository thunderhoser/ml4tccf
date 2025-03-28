"""Miscellaneous helper methods."""

import os
import gzip
import shutil
import warnings
import tempfile
import numpy
from pyproj import Proj
from scipy.interpolate import interp1d
from scipy.ndimage import center_of_mass, distance_transform_edt
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from ml4tccf.utils import extended_best_track_utils as ebtrk_utils
from ml4tccf.utils import scalar_prediction_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H%M'

TOLERANCE = 1e-6
SYNOPTIC_TIME_TOLERANCE_SEC = 900

HOURS_TO_SECONDS = 3600
DEGREES_LAT_TO_METRES = 60 * 1852.
MAX_XY_COORD_METRES = 60 * DEGREES_LAT_TO_METRES

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

# The following values come from Anderson et al. (1986).
TROPICAL_STD_ATMO_TEMPS_KELVINS = numpy.array([
    299.7, 293.7, 287.7, 283.7, 277.0, 270.3, 263.6, 257.0, 250.3, 243.6, 237.0,
    230.1, 223.6, 217.0, 210.3, 203.7, 197.0, 194.8
])
TROPICAL_STD_ATMO_HEIGHTS_M_ASL = numpy.linspace(0, 17000, num=18, dtype=float)


def cyclone_id_to_satellite_metadata(cyclone_id_string):
    """Returns satellite metadata for the given cyclone.

    :param cyclone_id_string: Cyclone ID.
    :return: satellite_longitude_deg_e: Longitude (deg east) of satellite
        subpoint.
    :return: satellite_altitude_m_agl: Satellite altitude (metres above ground
        level).
    """

    basin_id_string = parse_cyclone_id(cyclone_id_string)[1]

    if basin_id_string == NORTH_ATLANTIC_ID_STRING:
        return -75.2, 35786650.

    if basin_id_string == NORTHWEST_PACIFIC_ID_STRING:
        return 140.7, 35793000.

    if basin_id_string == NORTHEAST_PACIFIC_ID_STRING:
        return -137.2, 35794900.


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
    # assert numpy.isclose(numpy.sum(target_matrix), 1.)

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
    # assert numpy.isclose(numpy.sum(prediction_matrix), 1.)

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


def latlng_points_to_probability_grid(
        point_latitudes_deg_n, point_longitudes_deg_e,
        grid_latitude_array_deg_n, grid_longitude_array_deg_e):
    """Converts predicted point locations to grid of predicted probabilities.

    P = number of points
    M = number of rows in grid
    N = number of columns in grid

    :param point_latitudes_deg_n: length-P numpy array of latitudes (deg north).
    :param point_longitudes_deg_e: length-P numpy array of longitudes (deg
        east).
    :param grid_latitude_array_deg_n: numpy array of grid-point latitudes (deg
        north).  If regular grid, dimensions should be length-M.  If irregular
        grid, dimensions should be M x N.
    :param grid_longitude_array_deg_e: numpy array of grid-point longitudes (deg
        east).  If regular grid, dimensions should be length-N.  If irregular
        grid, dimensions should be M x N.
    :return: probability_matrix: M-by-N numpy array of probabilities.
    """

    # Check input args.
    regular_grid = len(grid_latitude_array_deg_n.shape) == 1

    if regular_grid:
        this_flag = is_regular_grid_valid(
            latitudes_deg_n=grid_latitude_array_deg_n,
            longitudes_deg_e=grid_longitude_array_deg_e
        )[0]

        assert this_flag

        grid_latitude_matrix_deg_n, grid_longitude_matrix_deg_e = (
            grids.latlng_vectors_to_matrices(
                unique_latitudes_deg=grid_latitude_array_deg_n,
                unique_longitudes_deg=grid_longitude_array_deg_e
            )
        )
    else:
        error_checking.assert_is_numpy_array(
            grid_latitude_array_deg_n, num_dimensions=2
        )
        error_checking.assert_is_numpy_array(
            grid_longitude_array_deg_e,
            exact_dimensions=numpy.array(
                grid_latitude_array_deg_n.shape, dtype=int
            )
        )

        grid_latitude_matrix_deg_n = grid_latitude_array_deg_n + 0.
        grid_longitude_matrix_deg_e = grid_longitude_array_deg_e + 0.

    num_rows = grid_latitude_matrix_deg_n.shape[0]
    half_num_rows_float = 0.5 * num_rows
    half_num_rows = int(numpy.round(half_num_rows_float))
    assert numpy.isclose(half_num_rows, half_num_rows_float, atol=TOLERANCE)

    num_columns = grid_latitude_matrix_deg_n.shape[1]
    half_num_columns_float = 0.5 * num_columns
    half_num_columns = int(numpy.round(half_num_columns_float))
    assert numpy.isclose(
        half_num_columns, half_num_columns_float, atol=TOLERANCE
    )

    error_checking.assert_is_numpy_array(
        point_latitudes_deg_n, num_dimensions=1
    )
    error_checking.assert_is_numpy_array(
        point_longitudes_deg_e,
        exact_dimensions=numpy.array(point_latitudes_deg_n.shape, dtype=int)
    )

    error_checking.assert_is_valid_lat_numpy_array(
        point_latitudes_deg_n, allow_nan=False
    )
    error_checking.assert_is_valid_lng_numpy_array(
        point_longitudes_deg_e, allow_nan=False
    )

    # Do actual stuff.
    i_start = half_num_rows - 1
    i_end = half_num_rows + 1
    j_start = half_num_columns - 1
    j_end = half_num_columns + 1
    grid_center_latitude_deg_n = numpy.mean(
        grid_latitude_matrix_deg_n[i_start:i_end, j_start:j_end]
    )

    projection_string = (
        '+proj=eqc +no_defs +R=6371228 +k=1 +units=m +lat_ts={0:10f}'
    ).format(grid_center_latitude_deg_n)

    projection_object = Proj(projection_string)

    grid_x_matrix_metres, grid_y_matrix_metres = projection_object(
        grid_longitude_matrix_deg_e, grid_latitude_matrix_deg_n
    )
    point_x_coords_metres, point_y_coords_metres = projection_object(
        point_longitudes_deg_e, point_latitudes_deg_n
    )

    max_x_spacing_metres = numpy.maximum(
        numpy.max(numpy.diff(grid_x_matrix_metres, axis=0)),
        numpy.max(numpy.diff(grid_x_matrix_metres, axis=1))
    )
    assert numpy.all(
        point_x_coords_metres >=
        numpy.min(grid_x_matrix_metres) - max_x_spacing_metres / 2
    )
    assert numpy.all(
        point_x_coords_metres <=
        numpy.max(grid_x_matrix_metres) + max_x_spacing_metres / 2
    )

    max_y_spacing_metres = numpy.maximum(
        numpy.max(numpy.diff(grid_y_matrix_metres, axis=0)),
        numpy.max(numpy.diff(grid_y_matrix_metres, axis=1))
    )
    assert numpy.all(
        point_y_coords_metres >=
        numpy.min(grid_y_matrix_metres) - max_y_spacing_metres / 2
    )
    assert numpy.all(
        point_y_coords_metres <=
        numpy.max(grid_y_matrix_metres) + max_y_spacing_metres / 2
    )

    # TODO(thunderhoser): I could probably make the assignment code faster if
    # need be.
    num_points = len(point_x_coords_metres)
    probability_matrix = numpy.full(grid_x_matrix_metres.shape, 0.)

    for i in range(num_points):
        if numpy.mod(i, 10) == 0:
            print('Have assigned {0:d} of {1:d} points to grid cell...'.format(
                i, num_points
            ))

        linear_index = numpy.argmin(numpy.ravel(
            (point_x_coords_metres[i] - grid_x_matrix_metres) ** 2 +
            (point_y_coords_metres[i] - grid_y_matrix_metres) ** 2
        ))
        row_index, column_index = numpy.unravel_index(
            linear_index, grid_x_matrix_metres.shape
        )
        probability_matrix[row_index, column_index] += 1.

    print('Have assigned all {0:d} points to grid cell!'.format(num_points))

    probability_matrix = probability_matrix / num_points
    return probability_matrix


def standard_to_geodetic_angles(standard_angles_deg):
    """Converts angles from standard to geodetic format.

    "Standard format" = measured counterclockwise from due east
    "Geodetic format" = measured clockwise from due north

    :param standard_angles_deg: numpy array of standard angles (degrees).
    :return: geodetic_angles_deg: equivalent-size numpy array of geodetic
        angles.
    """

    error_checking.assert_is_numpy_array_without_nan(standard_angles_deg)
    return numpy.mod((450. - standard_angles_deg), 360.)


def geodetic_to_standard_angles(geodetic_angles_deg):
    """Converts angles from geodetic to standard format.

    For the definitions of "geodetic format" and "standard format," see doc for
    `standard_to_geodetic_angles`.

    :param geodetic_angles_deg: numpy array of geodetic angles (degrees).
    :return: standard_angles_deg: equivalent-size numpy array of standard
        angles.
    """

    error_checking.assert_is_numpy_array_without_nan(geodetic_angles_deg)
    return numpy.mod((450. - geodetic_angles_deg), 360.)


def brightness_temp_to_cloud_top_height(brightness_temps_kelvins):
    """Converts brightness temperature to cloud-top height.

    WARNING: This method assumes a tropical standard atmosphere and surface
    elevation of 0 metres above sea level.

    :param brightness_temps_kelvins: numpy array of brightness temperatures.
    :return: cloud_top_heights_m_agl: numpy array of cloud-top heights (metres
        above ground level), with same shape as input array.
    """

    error_checking.assert_is_numpy_array_without_nan(brightness_temps_kelvins)

    interp_object = interp1d(
        x=TROPICAL_STD_ATMO_TEMPS_KELVINS[::-1],
        y=TROPICAL_STD_ATMO_HEIGHTS_M_ASL[::-1],
        kind='linear', assume_sorted=True,
        bounds_error=False, fill_value='extrapolate'
    )

    cloud_top_heights_m_agl = interp_object(brightness_temps_kelvins)
    cloud_top_heights_m_agl = numpy.maximum(cloud_top_heights_m_agl, 0.)
    return cloud_top_heights_m_agl


def get_xy_grid_one_tc_object(
        cyclone_id_string, grid_latitudes_deg_n, grid_longitudes_deg_e,
        normalize_to_minmax, test_mode=False):
    """Creates grid of x-coords, and grid of y-coords, for one TC object.

    This method exactly replicates xy-coords in the original data files from
    Robert and Galina.

    x = zonal distance from nadir
    y = meridional distance from nadir

    M = number of rows in grid
    N = number of columns in grid

    :param cyclone_id_string: Cyclone ID.
    :param grid_latitudes_deg_n: length-M numpy array of latitudes (deg north).
    :param grid_longitudes_deg_e: length-N numpy array of longitudes (deg east).
    :param normalize_to_minmax: Boolean flag.  If True, will normalize each
        coordinates (x and y) to range from 0...1.
    :param test_mode: Leave this alone.
    :return: grid_x_coords_metres: length-M numpy array of x-coordinates (zonal
        distances from nadir).
    :return: grid_y_coords_metres: length-N numpy array of y-coordinates
        (meridional distances from nadir).
    """

    # Check input args.
    error_checking.assert_is_numpy_array(grid_latitudes_deg_n, num_dimensions=1)
    error_checking.assert_is_valid_lat_numpy_array(
        grid_latitudes_deg_n, allow_nan=False
    )
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(grid_latitudes_deg_n), 0.
    )

    num_rows = len(grid_latitudes_deg_n)
    half_num_rows_float = 0.5 * num_rows
    half_num_rows = int(numpy.round(half_num_rows_float))
    assert numpy.isclose(half_num_rows, half_num_rows_float, atol=TOLERANCE)

    error_checking.assert_is_numpy_array(
        grid_longitudes_deg_e, num_dimensions=1
    )
    error_checking.assert_is_valid_lng_numpy_array(
        grid_longitudes_deg_e, positive_in_west_flag=False,
        negative_in_west_flag=False, allow_nan=False
    )

    error_checking.assert_is_boolean(normalize_to_minmax)
    error_checking.assert_is_boolean(test_mode)

    # Do actual stuff.
    i_start = half_num_rows - 1
    i_end = half_num_rows + 1
    cyclone_center_latitude_deg_n = numpy.mean(
        grid_latitudes_deg_n[i_start:i_end]
    )

    if test_mode:
        satellite_subpoint_longitude_deg_e = 0.
    else:
        satellite_subpoint_longitude_deg_e = cyclone_id_to_satellite_metadata(
            cyclone_id_string
        )[0]

    projection_string = (
        '+proj=eqc +no_defs +R=6371228 +k=1 +units=m '
        '+lat_ts={0:.4f} +lat_0=0.0 +lon_0={1:.4f}'
    ).format(
        cyclone_center_latitude_deg_n, satellite_subpoint_longitude_deg_e
    )

    projection_object = Proj(projection_string)

    fake_grid_latitudes_deg_n = numpy.full(
        len(grid_longitudes_deg_e), cyclone_center_latitude_deg_n
    )
    grid_x_coords_metres = projection_object(
        grid_longitudes_deg_e, fake_grid_latitudes_deg_n, inverse=False
    )[0]

    fake_grid_longitudes_deg_e = numpy.full(
        len(grid_latitudes_deg_n), satellite_subpoint_longitude_deg_e
    )
    grid_y_coords_metres = projection_object(
        fake_grid_longitudes_deg_e, grid_latitudes_deg_n, inverse=False
    )[1]

    if not normalize_to_minmax:
        return grid_x_coords_metres, grid_y_coords_metres

    return (
        grid_x_coords_metres / MAX_XY_COORD_METRES,
        grid_y_coords_metres / MAX_XY_COORD_METRES
    )


def create_latlng_grid(
        min_latitude_deg_n, max_latitude_deg_n, latitude_spacing_deg,
        min_longitude_deg_e, max_longitude_deg_e, longitude_spacing_deg):
    """Creates lat-long grid.

    M = number of rows in grid
    N = number of columns in grid

    :param min_latitude_deg_n: Minimum latitude (deg N) in grid.
    :param max_latitude_deg_n: Max latitude (deg N) in grid.
    :param latitude_spacing_deg: Spacing (deg N) between grid points in adjacent
        rows.
    :param min_longitude_deg_e: Minimum longitude (deg E) in grid.
    :param max_longitude_deg_e: Max longitude (deg E) in grid.
    :param longitude_spacing_deg: Spacing (deg E) between grid points in
        adjacent columns.
    :return: grid_point_latitudes_deg: length-M numpy array with latitudes
        (deg N) of grid points.
    :return: grid_point_longitudes_deg: length-N numpy array with longitudes
        (deg E) of grid points.
    """

    # TODO(thunderhoser): Make this handle wrap-around issues.

    min_longitude_deg_e = lng_conversion.convert_lng_positive_in_west(
        min_longitude_deg_e
    )
    max_longitude_deg_e = lng_conversion.convert_lng_positive_in_west(
        max_longitude_deg_e
    )

    min_latitude_deg_n = number_rounding.floor_to_nearest(
        min_latitude_deg_n, latitude_spacing_deg
    )
    max_latitude_deg_n = number_rounding.ceiling_to_nearest(
        max_latitude_deg_n, latitude_spacing_deg
    )
    min_longitude_deg_e = number_rounding.floor_to_nearest(
        min_longitude_deg_e, longitude_spacing_deg
    )
    max_longitude_deg_e = number_rounding.ceiling_to_nearest(
        max_longitude_deg_e, longitude_spacing_deg
    )

    num_grid_rows = 1 + int(numpy.round(
        (max_latitude_deg_n - min_latitude_deg_n) / latitude_spacing_deg
    ))
    num_grid_columns = 1 + int(numpy.round(
        (max_longitude_deg_e - min_longitude_deg_e) / longitude_spacing_deg
    ))

    return grids.get_latlng_grid_points(
        min_latitude_deg=min_latitude_deg_n,
        min_longitude_deg=min_longitude_deg_e,
        lat_spacing_deg=latitude_spacing_deg,
        lng_spacing_deg=longitude_spacing_deg,
        num_rows=num_grid_rows, num_columns=num_grid_columns
    )


def match_predictions_to_tc_centers(
        prediction_table_xarray, ebtrk_table_xarray, return_xy):
    """Finds TC center for every prediction.

    E = number of TC objects

    :param prediction_table_xarray: xarray table with predictions, in format
        returned by `prediction_io.read_file`.
    :param ebtrk_table_xarray: xarray table with extended best-track data, in
        format returned by `extended_best_track_io.read_file`.
    :param return_xy: Boolean flag.  If True (False), will return nadir-relative
        x-y coordinates (lat-long coordinates).
    :return: zonal_center_coords: length-E numpy array with zonal coordinates of
        TC centers.  These will be either nadir-relative x-coordinates or
        longitudes.
    :return: meridional_center_coords: length-E numpy array with meridional
        coordinates of TC centers.  These will be either nadir-relative
        y-coordinates or latitudes.
    """

    error_checking.assert_is_boolean(return_xy)

    ebtrk_times_unix_sec = (
        HOURS_TO_SECONDS * ebtrk_table_xarray[ebtrk_utils.VALID_TIME_KEY]
    )
    ebtrk_cyclone_id_strings = numpy.array([
        string_to_utf8(s)
        for s in ebtrk_table_xarray[ebtrk_utils.STORM_ID_KEY].values
    ])

    pt = prediction_table_xarray
    num_examples = len(pt[scalar_prediction_utils.TARGET_TIME_KEY].values)
    zonal_center_coords = numpy.full(num_examples, numpy.nan)
    meridional_center_coords = numpy.full(num_examples, numpy.nan)

    print(SEPARATOR_STRING)

    for i in range(num_examples):
        if numpy.mod(i, 100) == 0:
            print('Have found TC center for {0:d} of {1:d} examples...'.format(
                i, num_examples
            ))

        this_cyclone_id_string = string_to_utf8(
            pt[scalar_prediction_utils.CYCLONE_ID_KEY].values[i]
        )
        this_time_unix_sec = (
            pt[scalar_prediction_utils.TARGET_TIME_KEY].values[i]
        )

        these_indices = numpy.where(numpy.logical_and(
            ebtrk_cyclone_id_strings == this_cyclone_id_string,
            numpy.absolute(ebtrk_times_unix_sec - this_time_unix_sec) <=
            SYNOPTIC_TIME_TOLERANCE_SEC
        ))[0]

        if len(these_indices) == 0:
            warning_string = (
                'POTENTIAL ERROR: Cannot find cyclone {0:s} within {1:d} '
                'seconds of {2:s} in extended best-track data.'
            ).format(
                this_cyclone_id_string,
                SYNOPTIC_TIME_TOLERANCE_SEC,
                time_conversion.unix_sec_to_string(
                    this_time_unix_sec, TIME_FORMAT_FOR_LOG_MESSAGES
                )
            )

            warnings.warn(warning_string)
            continue

        if len(these_indices) > 1:
            warning_string = (
                'POTENTIAL ERROR: Found {0:d} instances of cyclone {1:s} '
                'within {2:d} seconds of {3:s} in extended best-track data:'
                '\n{4:s}'
            ).format(
                len(these_indices),
                this_cyclone_id_string,
                SYNOPTIC_TIME_TOLERANCE_SEC,
                time_conversion.unix_sec_to_string(
                    this_time_unix_sec, TIME_FORMAT_FOR_LOG_MESSAGES
                ),
                str(ebtrk_table_xarray.isel(
                    indexers={ebtrk_utils.STORM_OBJECT_DIM: these_indices}
                ))
            )

            warnings.warn(warning_string)

        this_latitude_deg_n = ebtrk_table_xarray[
            ebtrk_utils.CENTER_LATITUDE_KEY
        ].values[these_indices[0]]

        this_longitude_deg_e = ebtrk_table_xarray[
            ebtrk_utils.CENTER_LONGITUDE_KEY
        ].values[these_indices[0]]

        if not return_xy:
            zonal_center_coords[i] = this_longitude_deg_e + 0.
            meridional_center_coords[i] = this_latitude_deg_n + 0.
            continue

        # TODO(thunderhoser): The longitude command below is not foolproof.
        fake_grid_latitudes_deg_n = numpy.array([
            this_latitude_deg_n - 0.1, this_latitude_deg_n + 0.1
        ])
        fake_grid_longitudes_deg_e = numpy.array([
            this_longitude_deg_e - 0.1, this_longitude_deg_e + 0.1
        ])

        these_x_metres, these_y_metres = get_xy_grid_one_tc_object(
            cyclone_id_string=this_cyclone_id_string,
            grid_latitudes_deg_n=fake_grid_latitudes_deg_n,
            grid_longitudes_deg_e=fake_grid_longitudes_deg_e,
            normalize_to_minmax=False
        )
        zonal_center_coords[i] = numpy.mean(these_x_metres)
        meridional_center_coords[i] = numpy.mean(these_y_metres)

    print('Have found TC center for all {0:d} examples!'.format(num_examples))
    print(SEPARATOR_STRING)

    return zonal_center_coords, meridional_center_coords


def string_to_utf8(input_string):
    """Decodes string to UTF-8.

    :param input_string: String.
    :return: output_string: String in UTF-8 format.
    """

    try:
        return input_string.decode('utf-8')
    except AttributeError:
        return input_string
