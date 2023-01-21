"""Helper methods for satellite data."""

import warnings
import numpy
import xarray
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from ml4tccf.utils import misc_utils

METRES_TO_MICRONS = 1e6
TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H%M'

MIN_BRIGHTNESS_TEMP_KELVINS = 170.
MAX_BRIGHTNESS_TEMP_KELVINS = 330.

DEFAULT_MAX_BAD_PIXELS_LOW_RES = 6250
DEFAULT_MAX_BAD_PIXELS_HIGH_RES = 25000
DEFAULT_MIN_VISIBLE_PIXEL_FRACTION = 0.9

TIME_DIM = 'valid_time_unix_sec'
HIGH_RES_ROW_DIM = 'high_res_row'
HIGH_RES_COLUMN_DIM = 'high_res_column'
HIGH_RES_WAVELENGTH_DIM = 'high_res_wavelength_metres'

LOW_RES_ROW_DIM = 'low_res_row'
LOW_RES_COLUMN_DIM = 'low_res_column'
LOW_RES_WAVELENGTH_DIM = 'low_res_wavelength_metres'

CYCLONE_ID_KEY = 'cyclone_id_string'
PYPROJ_STRING_KEY = 'projection_string'

X_COORD_HIGH_RES_KEY = 'x_coord_high_res_metres'
Y_COORD_HIGH_RES_KEY = 'y_coord_high_res_metres'
LATITUDE_HIGH_RES_KEY = 'latitude_high_res_deg_n'
LONGITUDE_HIGH_RES_KEY = 'longitude_high_res_deg_e'
BRIGHTNESS_TEMPERATURE_KEY = 'brightness_temp_kelvins'

X_COORD_LOW_RES_KEY = 'x_coord_low_res_metres'
Y_COORD_LOW_RES_KEY = 'y_coord_low_res_metres'
LATITUDE_LOW_RES_KEY = 'latitude_low_res_deg_n'
LONGITUDE_LOW_RES_KEY = 'longitude_low_res_deg_e'
BIDIRECTIONAL_REFLECTANCE_KEY = 'bidirectional_reflectance'


def concat_over_time(satellite_tables_xarray):
    """Concatenates satellite data over many time steps.

    All time steps must contain data for the same tropical cyclone.

    :param satellite_tables_xarray: 1-D list of input tables, in format returned
        by `satellite_io.read_file`.
    :return: satellite_table_xarray: xarray table with all time steps.
    :raises: ValueError: if cyclone IDs are not identical.
    """

    satellite_table_xarray = xarray.concat(
        satellite_tables_xarray, dim=TIME_DIM, data_vars='all',
        coords='minimal', compat='identical', join='exact'
    )

    cyclone_id_strings = (
        satellite_table_xarray[CYCLONE_ID_KEY].values
    )
    unique_cyclone_id_strings = numpy.unique(numpy.array(cyclone_id_strings))
    if len(unique_cyclone_id_strings) == 1:
        return satellite_table_xarray

    error_string = (
        'Tables should all contain the same cyclone ID.  Instead, found {0:d} '
        'unique cyclone IDs:\n{1:s}'
    ).format(len(unique_cyclone_id_strings), str(unique_cyclone_id_strings))

    raise ValueError(error_string)


def quality_control_low_res(
        satellite_table_xarray,
        max_bad_pixels_per_time_channel=DEFAULT_MAX_BAD_PIXELS_LOW_RES):
    """Quality-controls low-resolution satellite data.

    :param satellite_table_xarray: xarray table in format returned by
        `satellite_io.read_file`.
    :param max_bad_pixels_per_time_channel: Max number of bad pixels in one
        time/channel pair.  For any time/channel pair with more bad pixels,
        the whole time/channel will be replaced with NaN.  For any time/channel
        pair with fewer or equal bad pixels, bad pixels will be inpainted.
    :return: satellite_table_xarray: Same but quality-controlled.
    """

    error_checking.assert_is_integer(max_bad_pixels_per_time_channel)
    error_checking.assert_is_geq(max_bad_pixels_per_time_channel, 0)

    t = satellite_table_xarray

    t[BRIGHTNESS_TEMPERATURE_KEY].values[
        t[BRIGHTNESS_TEMPERATURE_KEY].values < MIN_BRIGHTNESS_TEMP_KELVINS
    ] = numpy.nan

    t[BRIGHTNESS_TEMPERATURE_KEY].values[
        t[BRIGHTNESS_TEMPERATURE_KEY].values > MAX_BRIGHTNESS_TEMP_KELVINS
    ] = numpy.nan

    these_indices = numpy.where(
        numpy.isnan(t[BRIGHTNESS_TEMPERATURE_KEY].values)
    )
    bad_time_indices = these_indices[0]
    bad_channel_indices = these_indices[-1]

    if len(bad_time_indices) == 0:
        return satellite_table_xarray

    bad_index_matrix = numpy.transpose(
        numpy.vstack((bad_time_indices, bad_channel_indices))
    )
    bad_index_matrix = numpy.unique(bad_index_matrix, axis=0)
    bad_time_indices = bad_index_matrix[:, 0]
    bad_channel_indices = bad_index_matrix[:, 1]

    for i, k in zip(bad_time_indices, bad_channel_indices):
        num_bad_pixels = numpy.sum(
            numpy.isnan(t[BRIGHTNESS_TEMPERATURE_KEY].values[i, ..., k])
        )

        if num_bad_pixels > max_bad_pixels_per_time_channel:
            warning_string = (
                'POTENTIAL ERROR: Found {0:d} bad pixels for {1:.2f}-micron '
                'channel at {2:s}.  Replacing all pixels with NaN!'
            ).format(
                num_bad_pixels,
                METRES_TO_MICRONS * t.coords[LOW_RES_WAVELENGTH_DIM].values[k],
                time_conversion.unix_sec_to_string(
                    t.coords[TIME_DIM].values[i], TIME_FORMAT_FOR_LOG_MESSAGES
                )
            )

            warnings.warn(warning_string)
            t[BRIGHTNESS_TEMPERATURE_KEY].values[i, ..., k] = numpy.nan
            continue

        log_string = (
            'Found {0:d} bad pixels for {1:.2f}-micron channel at {2:s}.  '
            'Inpainting to replace bad pixels!'
        ).format(
            num_bad_pixels,
            METRES_TO_MICRONS * t.coords[LOW_RES_WAVELENGTH_DIM].values[k],
            time_conversion.unix_sec_to_string(
                t.coords[TIME_DIM].values[i], TIME_FORMAT_FOR_LOG_MESSAGES
            )
        )

        print(log_string)
        t[BRIGHTNESS_TEMPERATURE_KEY].values[i, ..., k] = misc_utils.fill_nans(
            t[BRIGHTNESS_TEMPERATURE_KEY].values[i, ..., k]
        )

    satellite_table_xarray = t
    return satellite_table_xarray


def quality_control_high_res(
        satellite_table_xarray,
        max_bad_pixels_per_time_channel=DEFAULT_MAX_BAD_PIXELS_HIGH_RES):
    """Quality-controls high-resolution satellite data.

    :param satellite_table_xarray: xarray table in format returned by
        `satellite_io.read_file`.
    :param max_bad_pixels_per_time_channel: Max number of bad pixels in one
        time/channel pair.  For any time/channel pair with more bad pixels,
        the whole time/channel will be replaced with NaN.  For any time/channel
        pair with fewer or equal bad pixels, bad pixels will be inpainted.
    :return: satellite_table_xarray: Same but quality-controlled.
    """

    error_checking.assert_is_integer(max_bad_pixels_per_time_channel)
    error_checking.assert_is_geq(max_bad_pixels_per_time_channel, 0)

    t = satellite_table_xarray

    t[BIDIRECTIONAL_REFLECTANCE_KEY].values[
        t[BIDIRECTIONAL_REFLECTANCE_KEY].values < 0
    ] = 0.

    these_indices = numpy.where(
        numpy.isnan(t[BIDIRECTIONAL_REFLECTANCE_KEY].values)
    )
    bad_time_indices = these_indices[0]
    bad_channel_indices = these_indices[-1]

    if len(bad_time_indices) == 0:
        return satellite_table_xarray

    bad_index_matrix = numpy.transpose(
        numpy.vstack((bad_time_indices, bad_channel_indices))
    )
    bad_index_matrix = numpy.unique(bad_index_matrix, axis=0)
    bad_time_indices = bad_index_matrix[:, 0]
    bad_channel_indices = bad_index_matrix[:, 1]

    for i, k in zip(bad_time_indices, bad_channel_indices):
        num_bad_pixels = numpy.sum(
            numpy.isnan(t[BIDIRECTIONAL_REFLECTANCE_KEY].values[i, ..., k])
        )

        if num_bad_pixels > max_bad_pixels_per_time_channel:
            warning_string = (
                'POTENTIAL ERROR: Found {0:d} bad pixels for {1:.2f}-micron '
                'channel at {2:s}.  Replacing all pixels with NaN!'
            ).format(
                num_bad_pixels,
                METRES_TO_MICRONS * t.coords[LOW_RES_WAVELENGTH_DIM].values[k],
                time_conversion.unix_sec_to_string(
                    t.coords[TIME_DIM].values[i], TIME_FORMAT_FOR_LOG_MESSAGES
                )
            )

            warnings.warn(warning_string)
            t[BIDIRECTIONAL_REFLECTANCE_KEY].values[i, ..., k] = numpy.nan
            continue

        log_string = (
            'Found {0:d} bad pixels for {1:.2f}-micron channel at {2:s}.  '
            'Inpainting to replace bad pixels!'
        ).format(
            num_bad_pixels,
            METRES_TO_MICRONS * t.coords[LOW_RES_WAVELENGTH_DIM].values[k],
            time_conversion.unix_sec_to_string(
                t.coords[TIME_DIM].values[i], TIME_FORMAT_FOR_LOG_MESSAGES
            )
        )

        print(log_string)
        t[BIDIRECTIONAL_REFLECTANCE_KEY].values[i, ..., k] = (
            misc_utils.fill_nans(
                t[BIDIRECTIONAL_REFLECTANCE_KEY].values[i, ..., k]
            )
        )

    satellite_table_xarray = t
    return satellite_table_xarray


def mask_visible_data_at_night(
        satellite_table_xarray, temporary_dir_name,
        min_visible_pixel_fraction=DEFAULT_MIN_VISIBLE_PIXEL_FRACTION,
        altitude_angle_exe_name=misc_utils.DEFAULT_EXE_NAME_FOR_ALTITUDE_ANGLE):
    """Masks visible data at night.

    :param satellite_table_xarray: xarray table in format returned by
        `satellite_io.read_file`.
    :param temporary_dir_name: Name of directory for temporary text file with
        solar altitude angles.
    :param min_visible_pixel_fraction: Minimum fraction of visible pixels at a
        given time.  For any time with fewer visible pixels, all visible data
        (bidirectional-reflectance values) will be replaced with NaN.
    :param altitude_angle_exe_name: Path to Fortran executable (pathless file
        name should probably be "solarpos") that computes solar altitude angles.
    :return: satellite_table_xarray: Same as input but with visible data masked.
    """

    error_checking.assert_is_geq(min_visible_pixel_fraction, 0.5)
    error_checking.assert_is_leq(min_visible_pixel_fraction, 1.)
    error_checking.assert_is_leq(min_visible_pixel_fraction, 1.)

    t = satellite_table_xarray
    valid_times_unix_sec = t.coords[TIME_DIM].values

    for i in range(len(valid_times_unix_sec)):
        these_latitudes_deg_n = t[LATITUDE_HIGH_RES_KEY].values[i, :][::30]
        these_longitudes_deg_e = t[LONGITUDE_HIGH_RES_KEY].values[i, :][::30]
        this_longitude_matrix_deg_e, this_latitude_matrix_deg_n = (
            numpy.meshgrid(these_longitudes_deg_e, these_latitudes_deg_n)
        )

        these_altitude_angles_deg = misc_utils.get_solar_altitude_angles(
            valid_time_unix_sec=valid_times_unix_sec[i],
            latitudes_deg_n=numpy.ravel(this_latitude_matrix_deg_n),
            longitudes_deg_e=numpy.ravel(this_longitude_matrix_deg_e),
            temporary_dir_name=temporary_dir_name,
            fortran_exe_name=altitude_angle_exe_name
        )

        this_visible_fraction = numpy.mean(these_altitude_angles_deg > 0)
        if this_visible_fraction >= min_visible_pixel_fraction:
            continue

        log_string = (
            'Only {0:.2f}% of pixels are visible at {1:s} in grid centered on '
            '({2:.1f} deg N, {3:.1f} deg E).  Replacing all visible '
            'reflectances with NaN!'
        ).format(
            100 * this_visible_fraction,
            time_conversion.unix_sec_to_string(
                valid_times_unix_sec[i], TIME_FORMAT_FOR_LOG_MESSAGES
            ),
            numpy.mean(t[LATITUDE_HIGH_RES_KEY].values[i, :]),
            numpy.mean(t[LONGITUDE_HIGH_RES_KEY].values[i, :])
        )

        print(log_string)

        t[BIDIRECTIONAL_REFLECTANCE_KEY].values[i, ...] = numpy.nan

    satellite_table_xarray = t
    return satellite_table_xarray
