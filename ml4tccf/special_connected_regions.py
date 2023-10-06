"""Finds special connected regions in 2-D image.

A 'special connected region' is a region in which the summed data values over
all grid cells meets or exceeds a minimum threshold.  Also, the special
connected region contains the fewest grid cells necessary to satisfy this
criterion.
"""

import os
import sys
import numpy
from scipy.ndimage.measurements import label

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking


def _find_one_connected_region(
        data_matrix, seed_row, seed_column, minimum_sum, region_id_matrix,
        grid_x_coords, grid_y_coords):
    """Finds one special connected region.

    M = number of rows in grid
    N = number of columns in grid

    :param data_matrix: M-by-N numpy array of data values.
    :param seed_row: Row index of seed for connected region.  This grid cell
        must be in the region.
    :param seed_column: Column index of seed for connected region.  This grid
        cell must be in the region.
    :param minimum_sum: Minimum sum over region
    :param region_id_matrix: M-by-N numpy array of region IDs (positive
        integers).  -1 means that the grid cell is still unassigned.
    :param grid_x_coords: length-N numpy array with x-coordinates of grid
        points.
    :param grid_y_coords: length-M numpy array with y-coordinates of grid
        points.
    :return: region_id_matrix: Same as input but maybe updated.
    """

    x_coord_matrix, y_coord_matrix = numpy.meshgrid(
        grid_x_coords, grid_y_coords
    )
    dist_from_seed_matrix = numpy.sqrt(
        (x_coord_matrix - grid_x_coords[seed_column]) ** 2 +
        (y_coord_matrix - grid_y_coords[seed_row]) ** 2
    )

    sort_indices_linear = numpy.argsort(numpy.ravel(dist_from_seed_matrix))
    sort_indices_linear = [
        k for k in sort_indices_linear
        if numpy.ravel(region_id_matrix)[k] == -1
    ]

    for i in range(len(sort_indices_linear)):
        temp_data_array = numpy.full(data_matrix.size, 0, dtype=int)
        temp_data_array[sort_indices_linear[:(i + 1)]] = 1
        temp_data_matrix = numpy.reshape(temp_data_array, data_matrix.shape)

        label_matrix = label(
            temp_data_matrix,
            structure=numpy.full((3, 3), 1, dtype=int)
        )[0]

        these_rows, these_columns = numpy.where(label_matrix == 1)
        this_sum = numpy.sum(data_matrix[these_rows, these_columns])
        if this_sum < minimum_sum:
            continue

        current_region_id = numpy.max(region_id_matrix) + 1
        current_region_id = max([current_region_id, 1])
        region_id_matrix[these_rows, these_columns] = current_region_id
        return region_id_matrix

    return region_id_matrix


def _assign_leftover_grid_cells(
        region_id_matrix, grid_x_coords, grid_y_coords):
    """Assigns each left-over grid cell to the nearest region.

    M = number of rows in grid
    N = number of columns in grid

    :param region_id_matrix: M-by-N numpy array of region IDs (positive
        integers).  -1 means that the grid cell is still unassigned.
    :param grid_x_coords: length-N numpy array with x-coordinates of grid
        points.
    :param grid_y_coords: length-M numpy array with y-coordinates of grid
        points.
    :return: region_id_matrix: Same as input but maybe updated.
    """

    if numpy.all(region_id_matrix > -1):
        return region_id_matrix

    if numpy.all(region_id_matrix == -1):
        region_id_matrix[:] = 1
        return region_id_matrix

    x_coord_matrix, y_coord_matrix = numpy.meshgrid(
        grid_x_coords, grid_y_coords
    )
    leftover_row_indices, leftover_column_indices = numpy.where(
        region_id_matrix == -1
    )

    for i, j in zip(leftover_row_indices, leftover_column_indices):
        dist_from_leftover_matrix = numpy.sqrt(
            (x_coord_matrix - x_coord_matrix[i, j]) ** 2 +
            (y_coord_matrix - y_coord_matrix[i, j]) ** 2
        )
        dist_from_leftover_matrix[region_id_matrix == -1] = numpy.inf

        nearest_index_linear = numpy.argmin(
            numpy.ravel(dist_from_leftover_matrix)
        )
        nearest_region_id = numpy.ravel(region_id_matrix)[nearest_index_linear]

        region_id_matrix[i, j] = nearest_region_id

    return region_id_matrix


def find_connected_regions(
        data_matrix, minimum_sum, grid_x_coords, grid_y_coords):
    """Finds all special connected regions in one image.

    M = number of rows in grid
    N = number of columns in grid

    :param data_matrix: M-by-N numpy array of data values.
    :param minimum_sum: Minimum sum over region
    :param grid_x_coords: length-N numpy array with x-coordinates of grid
        points.
    :param grid_y_coords: length-M numpy array with y-coordinates of grid
        points.
    :return: region_id_matrix: Same as input but maybe updated.
    """

    # Check input args.
    error_checking.assert_is_numpy_array(grid_x_coords, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(numpy.diff(grid_x_coords), 0.)
    error_checking.assert_is_numpy_array(grid_y_coords, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(numpy.diff(grid_y_coords), 0.)

    num_grid_rows = len(grid_y_coords)
    num_grid_columns = len(grid_x_coords)
    expected_dim = numpy.array([num_grid_rows, num_grid_columns], dtype=int)

    error_checking.assert_is_geq_numpy_array(data_matrix, 0.)
    error_checking.assert_is_numpy_array(
        data_matrix, exact_dimensions=expected_dim
    )

    error_checking.assert_is_greater(minimum_sum, 0.)

    # Do actual stuff.
    region_id_matrix = numpy.full(data_matrix.shape, -1, dtype=int)

    while not numpy.all(region_id_matrix > -1):
        prev_region_id_matrix = region_id_matrix + 0

        unassigned_indices_linear = numpy.where(
            numpy.ravel(region_id_matrix) == -1
        )[0]
        seed_index_linear = numpy.random.choice(
            unassigned_indices_linear, size=1, replace=False
        )[0]
        seed_row, seed_column = numpy.unravel_index(
            seed_index_linear, (num_grid_rows, num_grid_columns)
        )

        region_id_matrix = _find_one_connected_region(
            data_matrix=data_matrix,
            seed_row=seed_row, seed_column=seed_column,
            minimum_sum=minimum_sum,
            region_id_matrix=region_id_matrix,
            grid_x_coords=grid_x_coords,
            grid_y_coords=grid_y_coords
        )

        if numpy.all(prev_region_id_matrix == region_id_matrix):
            break

    return _assign_leftover_grid_cells(
        region_id_matrix=region_id_matrix,
        grid_x_coords=grid_x_coords,
        grid_y_coords=grid_y_coords
    )
