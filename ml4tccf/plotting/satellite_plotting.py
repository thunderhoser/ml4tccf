"""Plotting methods for satellite data."""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot
from scipy.interpolate import RegularGridInterpolator
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from ml4tccf.utils import misc_utils

TOLERANCE = 1e-6

DEFAULT_MIN_TEMP_KELVINS = 190.
DEFAULT_MAX_TEMP_KELVINS = 310.
DEFAULT_CUTOFF_TEMP_KELVINS = 240.

DEFAULT_FONT_SIZE = 30


def _grid_points_to_edges_1d(grid_point_coords):
    """Converts grid points (i.e., cell centers) to cell edges for 1-D grid.

    P = number of grid points

    :param grid_point_coords: length-P numpy array of grid-point coordinates, in
        increasing order.
    :return: grid_cell_edge_coords: length-(P + 1) numpy array of grid-cell-edge
        coordinates, also in increasing order.
    """

    grid_cell_edge_coords = (grid_point_coords[:-1] + grid_point_coords[1:]) / 2
    first_edge_coords = (
        grid_point_coords[0] - numpy.diff(grid_point_coords[:2]) / 2
    )
    last_edge_coords = (
        grid_point_coords[-1] + numpy.diff(grid_point_coords[-2:]) / 2
    )

    return numpy.concatenate((
        first_edge_coords, grid_cell_edge_coords, last_edge_coords
    ))


# def _grid_points_to_edges_2d_old(grid_point_coord_matrix):
#     """Converts grid points (i.e., cell centers) to cell edges for 2-D grid.
#
#     M = number of rows
#     N = number of columns
#
#     :param grid_point_coord_matrix: M-by-N numpy array of grid-point
#         coordinates.
#     :return: grid_cell_edge_coord_matrix: (M + 1)-by-(N + 1) numpy array of
#         grid-cell-edge coordinates.
#     """
#
#     # TODO(thunderhoser): Problem: RectBivariateSpline cannot extrapolate.
#
#     num_rows = grid_point_coord_matrix.shape[0]
#     num_columns = grid_point_coord_matrix.shape[1]
#
#     row_indices_orig = numpy.linspace(
#         0, num_rows - 1, num=num_rows, dtype=float
#     )
#     column_indices_orig = numpy.linspace(
#         0, num_columns - 1, num=num_columns, dtype=float
#     )
#
#     row_indices_new = _grid_points_to_edges_1d(row_indices_orig)
#     column_indices_new = _grid_points_to_edges_1d(column_indices_orig)
#
#     interp_object = RectBivariateSpline(
#         x=row_indices_orig, y=column_indices_orig,
#         z=grid_point_coord_matrix, kx=1, ky=1, s=0
#     )
#
#     return interp_object(x=row_indices_new, y=column_indices_new, grid=True)


def _grid_points_to_edges_2d(grid_point_coord_matrix):
    """Converts grid points (i.e., cell centers) to cell edges for 2-D grid.

    M = number of rows
    N = number of columns

    :param grid_point_coord_matrix: M-by-N numpy array of grid-point
        coordinates.
    :return: grid_cell_edge_coord_matrix: (M + 1)-by-(N + 1) numpy array of
        grid-cell-edge coordinates.
    """

    num_rows = grid_point_coord_matrix.shape[0]
    num_columns = grid_point_coord_matrix.shape[1]

    row_indices_orig = numpy.linspace(
        0, num_rows - 1, num=num_rows, dtype=float
    )
    column_indices_orig = numpy.linspace(
        0, num_columns - 1, num=num_columns, dtype=float
    )

    row_indices_new = _grid_points_to_edges_1d(row_indices_orig)
    column_indices_new = _grid_points_to_edges_1d(column_indices_orig)

    interp_object = RegularGridInterpolator(
        points=(row_indices_orig, column_indices_orig),
        values=grid_point_coord_matrix,
        method='linear', bounds_error=False, fill_value=None
    )

    row_index_matrix_new, column_index_matrix_new = numpy.meshgrid(
        row_indices_new, column_indices_new
    )
    rowcol_index_matrix_nw = numpy.transpose(numpy.vstack((
        numpy.ravel(row_index_matrix_new),
        numpy.ravel(column_index_matrix_new)
    )))

    grid_cell_edge_coords = interp_object(rowcol_index_matrix_nw)

    return numpy.reshape(
        grid_cell_edge_coords,
        (len(row_indices_new), len(column_indices_new))
    )


def get_colour_scheme_for_brightness_temp(
        min_temp_kelvins=DEFAULT_MIN_TEMP_KELVINS,
        max_temp_kelvins=DEFAULT_MAX_TEMP_KELVINS,
        cutoff_temp_kelvins=DEFAULT_CUTOFF_TEMP_KELVINS):
    """Returns colour scheme for brightness temperature.

    :param min_temp_kelvins: Minimum temperature in colour scheme.
    :param max_temp_kelvins: Max temperature in colour scheme.
    :param cutoff_temp_kelvins: Cutoff between grey and non-grey colours.
    :return: colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :return: colour_norm_object: Colour-normalizer (maps from data space to
        colour-bar space, which goes from 0...1).  This is an instance of
        `matplotlib.colors.Normalize`.
    """

    error_checking.assert_is_greater(max_temp_kelvins, cutoff_temp_kelvins)
    error_checking.assert_is_greater(cutoff_temp_kelvins, min_temp_kelvins)

    normalized_values = numpy.linspace(0, 1, num=1001, dtype=float)

    grey_colour_map_object = pyplot.get_cmap('Greys')
    grey_temps_kelvins = numpy.linspace(
        cutoff_temp_kelvins, max_temp_kelvins, num=1001, dtype=float
    )
    grey_rgb_matrix = grey_colour_map_object(normalized_values)[:, :-1]

    plasma_colour_map_object = pyplot.get_cmap('plasma')
    plasma_temps_kelvins = numpy.linspace(
        min_temp_kelvins, cutoff_temp_kelvins, num=1001, dtype=float
    )
    plasma_rgb_matrix = plasma_colour_map_object(normalized_values)[:, :-1]

    boundary_temps_kelvins = numpy.concatenate(
        (plasma_temps_kelvins, grey_temps_kelvins), axis=0
    )
    rgb_matrix = numpy.concatenate(
        (plasma_rgb_matrix, grey_rgb_matrix), axis=0
    )

    colour_map_object = matplotlib.colors.ListedColormap(rgb_matrix)
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        boundary_temps_kelvins, colour_map_object.N
    )

    return colour_map_object, colour_norm_object


def get_colour_scheme_for_bdrf():
    """Returns colour scheme for bidirectional reflectance.

    :return: colour_map_object: See doc for
        `get_colour_scheme_for_brightness_temp`.
    :return: colour_norm_object: Same.
    """

    colour_map_object = pyplot.get_cmap('gist_gray', lut=1001)
    colour_norm_object = matplotlib.colors.Normalize(vmin=0., vmax=1.)

    return colour_map_object, colour_norm_object


def add_colour_bar(
        data_matrix, axes_object, colour_map_object, colour_norm_object,
        orientation_string, font_size, for_brightness_temp):
    """Adds colour bar to plot.

    :param data_matrix: See doc for `plot_2d_grid_regular`.
    :param axes_object: Same.
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :param colour_norm_object: Colour-normalizer (maps from data space to
        colour-bar space, which goes from 0...1).  This should be an instance of
        `matplotlib.colors.Normalize`.
    :param orientation_string: Orientation ("vertical" or "horizontal").
    :param font_size: Font size for labels on colour bar.
    :param for_brightness_temp: Boolean flag.  If True (False), will assume that
        colour bar is for brightness temperature (bidirectional reflectance).
    :return: colour_bar_object: See doc for `plot_2d_grid_regular`.
    """

    error_checking.assert_is_numpy_array(data_matrix, num_dimensions=2)
    error_checking.assert_is_string(orientation_string)
    error_checking.assert_is_greater(font_size, 0)
    error_checking.assert_is_boolean(for_brightness_temp)

    if orientation_string == 'horizontal' and font_size > DEFAULT_FONT_SIZE:
        padding = 0.15
    else:
        padding = None

    colour_bar_object = gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=data_matrix,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string=orientation_string,
        extend_min=True, extend_max=True, font_size=font_size, padding=padding
    )

    if hasattr(colour_norm_object, 'boundaries'):
        min_colour_value = colour_norm_object.boundaries[0]
        max_colour_value = colour_norm_object.boundaries[-1]
    else:
        min_colour_value = colour_norm_object.vmin
        max_colour_value = colour_norm_object.vmax

    if for_brightness_temp:
        num_tick_values = 1 + int(numpy.round(
            (max_colour_value - min_colour_value) / 10
        ))
    else:
        num_tick_values = 1 + int(numpy.round(
            (max_colour_value - min_colour_value) / 0.1
        ))

    tick_values = numpy.linspace(
        min_colour_value, max_colour_value, num=num_tick_values, dtype=float
    )

    if for_brightness_temp:
        tick_values = numpy.round(tick_values).astype(int)
        tick_strings = ['{0:d}'.format(v) for v in tick_values]
    else:
        tick_strings = ['{0:.2f}'.format(v) for v in tick_values]

    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    return colour_bar_object


def plot_2d_grid_latlng(
        data_matrix, axes_object, latitude_array_deg_n, longitude_array_deg_e,
        plotting_brightness_temp, cbar_orientation_string='vertical',
        font_size=DEFAULT_FONT_SIZE, colour_map_object=None,
        colour_norm_object=None):
    """Plots data (brightness temperature or BDRF) on lat-long grid.

    M = number of rows in grid
    N = number of columns in grid

    :param data_matrix: M-by-N numpy array of data values.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param latitude_array_deg_n: If regular lat-long grid, this should be a
        length-M numpy array of latitudes (deg north).  If irregular grid, this
        should be an M-by-N array of latitudes.
    :param longitude_array_deg_e: If regular lat-long grid, this should be a
        length-N numpy array of longitudes (deg east).  If irregular grid, this
        should be an M-by-N array of longitudes.
    :param plotting_brightness_temp: Boolean flag.  If True (False), will assume
        that data in `data_matrix` are brightness temperatures (bidirectional
        reflectances).
    :param cbar_orientation_string: Colour-bar orientation.  May be
        "horizontal", "vertical", or None.
    :param font_size: Font size.
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).  If None, default will be chosen based on
        `plotting_brightness_temp`.
    :param colour_norm_object: Colour-normalizer (maps from data space to
        colour-bar space, which goes from 0...1).  This should be an instance of
        `matplotlib.colors.Normalize`.  If None, default will be chosen based on
        `plotting_brightness_temp`.
    :return: colour_bar_object: Colour-bar handle (instance of
        `matplotlib.pyplot.colorbar`).  If `cbar_orientation_string is None`,
        this will also be None.
    """

    # TODO(thunderhoser): Worry about longitudes being pos or neg in WH.

    # Check input args.
    error_checking.assert_is_valid_lat_numpy_array(latitude_array_deg_n)
    longitude_array_deg_e = lng_conversion.convert_lng_negative_in_west(
        longitude_array_deg_e
    )

    regular_grid = len(latitude_array_deg_n.shape) == 1

    if regular_grid:
        this_flag, latitudes_to_plot_deg_n, longitudes_to_plot_deg_e = (
            misc_utils.is_regular_grid_valid(
                latitudes_deg_n=latitude_array_deg_n,
                longitudes_deg_e=longitude_array_deg_e
            )
        )

        assert this_flag
        num_rows = len(latitudes_to_plot_deg_n)
        num_columns = len(longitudes_to_plot_deg_e)
    else:
        error_checking.assert_is_numpy_array(
            latitude_array_deg_n, num_dimensions=2
        )
        error_checking.assert_is_numpy_array(
            longitude_array_deg_e,
            exact_dimensions=numpy.array(latitude_array_deg_n.shape, dtype=int)
        )

        latitudes_to_plot_deg_n = latitude_array_deg_n + 0.
        longitudes_to_plot_deg_e = longitude_array_deg_e + 0.

        num_rows = latitude_array_deg_n.shape[0]
        num_columns = latitude_array_deg_n.shape[1]

    expected_dim = numpy.array([num_rows, num_columns], dtype=int)
    error_checking.assert_is_numpy_array(
        data_matrix, exact_dimensions=expected_dim
    )

    error_checking.assert_is_boolean(plotting_brightness_temp)
    if cbar_orientation_string is not None:
        error_checking.assert_is_string(cbar_orientation_string)

    # Set up grid coordinates.
    if regular_grid:
        latitudes_to_plot_deg_n = _grid_points_to_edges_1d(
            latitudes_to_plot_deg_n
        )
        longitudes_to_plot_deg_e = _grid_points_to_edges_1d(
            longitudes_to_plot_deg_e
        )
    else:

        # TODO(thunderhoser): This does not fully handle crossover issue.
        longitude_range_deg = (
            numpy.max(longitudes_to_plot_deg_e) -
            numpy.min(longitudes_to_plot_deg_e)
        )
        if longitude_range_deg > 100:
            longitudes_to_plot_deg_e = (
                lng_conversion.convert_lng_positive_in_west(
                    longitudes_to_plot_deg_e
                )
            )

        latitudes_to_plot_deg_n = _grid_points_to_edges_2d(
            latitudes_to_plot_deg_n
        )
        longitudes_to_plot_deg_e = _grid_points_to_edges_2d(
            longitudes_to_plot_deg_e
        )

    data_matrix_to_plot = grids.latlng_field_grid_points_to_edges(
        field_matrix=data_matrix,
        min_latitude_deg=1., min_longitude_deg=1.,
        lat_spacing_deg=1e-6, lng_spacing_deg=1e-6
    )[0]

    # Do actual plotting.
    data_matrix_to_plot = numpy.ma.masked_where(
        numpy.isnan(data_matrix_to_plot), data_matrix_to_plot
    )

    if colour_map_object is None or colour_norm_object is None:
        if plotting_brightness_temp:
            colour_map_object, colour_norm_object = (
                get_colour_scheme_for_brightness_temp()
            )
        else:
            colour_map_object, colour_norm_object = (
                get_colour_scheme_for_bdrf()
            )

    if hasattr(colour_norm_object, 'boundaries'):
        min_colour_value = colour_norm_object.boundaries[0]
        max_colour_value = colour_norm_object.boundaries[-1]
    else:
        min_colour_value = colour_norm_object.vmin
        max_colour_value = colour_norm_object.vmax

    if regular_grid:
        axes_object.pcolormesh(
            longitudes_to_plot_deg_e, latitudes_to_plot_deg_n,
            data_matrix_to_plot,
            cmap=colour_map_object, norm=colour_norm_object,
            vmin=min_colour_value, vmax=max_colour_value, shading='flat',
            edgecolors='None', zorder=-1e11
        )
    else:
        axes_object.pcolor(
            longitudes_to_plot_deg_e, latitudes_to_plot_deg_n,
            data_matrix_to_plot,
            cmap=colour_map_object, norm=colour_norm_object,
            vmin=min_colour_value, vmax=max_colour_value,
            edgecolors='None', zorder=-1e11
        )

    if cbar_orientation_string is None:
        return None

    return add_colour_bar(
        data_matrix=data_matrix, axes_object=axes_object,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string=cbar_orientation_string, font_size=font_size,
        for_brightness_temp=plotting_brightness_temp
    )


def plot_2d_grid_no_coords(
        data_matrix, axes_object, plotting_brightness_temp,
        cbar_orientation_string='vertical', font_size=DEFAULT_FONT_SIZE,
        colour_map_object=None, colour_norm_object=None):
    """Plots data (brightness temperature or BDRF) on grid without coords.

    :param data_matrix: See doc for `plot_2d_grid_no_coords`.
    :param axes_object: Same.
    :param plotting_brightness_temp: Same.
    :param cbar_orientation_string: Same.
    :param font_size: Same.
    :param colour_map_object: Same.
    :param colour_norm_object: Same.
    :return: colour_bar_object: Same.
    """

    error_checking.assert_is_numpy_array(data_matrix, num_dimensions=2)
    error_checking.assert_is_boolean(plotting_brightness_temp)
    if cbar_orientation_string is not None:
        error_checking.assert_is_string(cbar_orientation_string)

    # Do actual plotting.
    data_matrix_to_plot = numpy.ma.masked_where(
        numpy.isnan(data_matrix), data_matrix
    )

    if colour_map_object is None or colour_norm_object is None:
        if plotting_brightness_temp:
            colour_map_object, colour_norm_object = (
                get_colour_scheme_for_brightness_temp()
            )
        else:
            colour_map_object, colour_norm_object = (
                get_colour_scheme_for_bdrf()
            )

    if hasattr(colour_norm_object, 'boundaries'):
        min_colour_value = colour_norm_object.boundaries[0]
        max_colour_value = colour_norm_object.boundaries[-1]
    else:
        min_colour_value = colour_norm_object.vmin
        max_colour_value = colour_norm_object.vmax

    axes_object.imshow(
        data_matrix_to_plot, origin='lower',
        cmap=colour_map_object, norm=colour_norm_object,
        vmin=min_colour_value, vmax=max_colour_value,
        zorder=-1e11
    )

    axes_object.set_xticks([], [])
    axes_object.set_yticks([], [])

    if cbar_orientation_string is None:
        return None

    return add_colour_bar(
        data_matrix=data_matrix, axes_object=axes_object,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string=cbar_orientation_string, font_size=font_size,
        for_brightness_temp=plotting_brightness_temp
    )
