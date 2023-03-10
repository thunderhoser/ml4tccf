"""Image-filtering."""

import numpy
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_dilation
from gewittergefahr.gg_utils import error_checking

DEG_LATITUDE_TO_KM = 60 * 1.852
DEGREES_TO_RADIANS = numpy.pi / 180
TARGET_DISCRETIZATION_DEG = 0.1


def _get_structure_matrix_for_target_discretizn(
        grid_spacing_km, cyclone_center_latitude_deg_n):
    """Returns structure matrix used to deal with discretization of targets.

    Specifically, target values (best-track TC centers) are rounded to the
    nearest 0.1 deg latitude and longitude.

    :param grid_spacing_km: Grid spacing.
    :param cyclone_center_latitude_deg_n: Latitude of true cyclone center (deg
        north).
    :return: structure_matrix: 2-D numpy array of Boolean flags.
    """

    row_uncertainty_px = (
        0.5 * TARGET_DISCRETIZATION_DEG * DEG_LATITUDE_TO_KM / grid_spacing_km
    )
    column_uncertainty_px = (
        numpy.cos(cyclone_center_latitude_deg_n * DEGREES_TO_RADIANS) *
        row_uncertainty_px
    )

    # To deal with issue of grid point vs. grid corner (4 grid points).
    row_uncertainty_px -= 0.5
    column_uncertainty_px -= 0.5
    column_uncertainty_px = max([column_uncertainty_px, 0.])

    half_grid_size_px = int(numpy.ceil(
        max([row_uncertainty_px, column_uncertainty_px])
    ))
    pixel_offsets = numpy.linspace(
        -half_grid_size_px, half_grid_size_px, num=2 * half_grid_size_px + 1,
        dtype=float
    )

    column_offset_matrix_px, row_offset_matrix_px = numpy.meshgrid(
        pixel_offsets, pixel_offsets
    )

    return numpy.logical_and(
        numpy.absolute(row_offset_matrix_px) <= row_uncertainty_px,
        numpy.absolute(column_offset_matrix_px) <= column_uncertainty_px
    )


def undo_target_discretization(integer_target_matrix, grid_spacing_km,
                               cyclone_center_latitude_deg_n):
    """Undoes discretization of targets.

    Specifically, target values (best-track TC centers) are rounded to the
    nearest 0.1 deg latitude and longitude.

    M = number of rows in grid
    N = number of columns in grid

    :param integer_target_matrix: M-by-N numpy array of target values (integers
        in 0...1).
    :param grid_spacing_km: Grid spacing.
    :param cyclone_center_latitude_deg_n: Latitude of true cyclone center (deg
        north).
    :return: target_matrix: M-by-N numpy array of new target values (floats in
        0...1).
    """

    error_checking.assert_is_integer_numpy_array(integer_target_matrix)
    error_checking.assert_is_numpy_array(
        integer_target_matrix, num_dimensions=2
    )
    error_checking.assert_is_geq_numpy_array(integer_target_matrix, 0)
    error_checking.assert_is_leq_numpy_array(integer_target_matrix, 1)
    error_checking.assert_equals(numpy.sum(integer_target_matrix), 4)

    error_checking.assert_is_greater(grid_spacing_km, 0.)
    error_checking.assert_is_valid_latitude(cyclone_center_latitude_deg_n)

    structure_matrix = _get_structure_matrix_for_target_discretizn(
        grid_spacing_km=grid_spacing_km,
        cyclone_center_latitude_deg_n=cyclone_center_latitude_deg_n
    )

    target_matrix = binary_dilation(
        integer_target_matrix, structure=structure_matrix, iterations=1,
        border_value=0
    ).astype(float)

    # return target_matrix / numpy.sum(target_matrix)
    return target_matrix


def smooth_targets_with_gaussian(
        target_matrix, grid_spacing_km, stdev_distance_km):
    """Smooths target field with Gaussian filter.

    M = number of rows in grid
    N = number of columns in grid

    :param target_matrix: M-by-N numpy array of target values (floats in 0...1).
    :param grid_spacing_km: Grid spacing.
    :param stdev_distance_km: Standard-deviation distance for Gaussian filter.
    :return: smoothed_target_matrix: Smoothed version of input, still M-by-N.
    """

    error_checking.assert_is_numpy_array(target_matrix, num_dimensions=2)
    error_checking.assert_is_geq_numpy_array(target_matrix, 0.)
    error_checking.assert_is_leq_numpy_array(target_matrix, 1.)

    error_checking.assert_is_greater(grid_spacing_km, 0.)
    error_checking.assert_is_greater(stdev_distance_km, 0.)

    smoothed_target_matrix = gaussian_filter(
        input=target_matrix, sigma=stdev_distance_km / grid_spacing_km,
        order=0, mode='constant', cval=0., truncate=4.
    )

    # return smoothed_target_matrix / numpy.sum(smoothed_target_matrix)
    return smoothed_target_matrix


def create_mean_conv_filter(half_num_rows, half_num_columns, num_channels):
    """Creates convolutional filter that computes mean.

    M = number of rows in filter
    N = number of columns in filter
    C = number of channels

    :param half_num_rows: Number of rows on either side of center.  This is
        (M - 1) / 2.
    :param half_num_columns: Number of columns on either side of center.  This
        is (N - 1) / 2.
    :param num_channels: Number of channels.
    :return: weight_matrix: M-by-N-by-C-by-C numpy array of filter weights.
    """

    error_checking.assert_is_integer(half_num_rows)
    error_checking.assert_is_geq(half_num_rows, 0)
    error_checking.assert_is_integer(half_num_columns)
    error_checking.assert_is_geq(half_num_columns, 0)
    error_checking.assert_is_integer(num_channels)
    error_checking.assert_is_greater(num_channels, 0)

    num_rows = 2 * half_num_rows + 1
    num_columns = 2 * half_num_columns + 1
    weight = 1. / (num_rows * num_columns)

    return numpy.full(
        (num_rows, num_columns, num_channels, num_channels), weight
    )
