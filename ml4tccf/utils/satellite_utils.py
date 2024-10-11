"""Helper methods for satellite data."""

import copy
import warnings
import numpy
import xarray
from scipy.interpolate import interp1d
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from ml4tccf.utils import misc_utils

TOLERANCE = 1e-6
DUMMY_TIME_UNIX_SEC = int(-1e10)

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


def _compute_interpolation_gap(source_times_unix_sec, target_time_unix_sec):
    """Computes interpolation gap.

    This is the gap between the target time and the two nearest source times.

    :param source_times_unix_sec: 1-D numpy array of source times.
    :param target_time_unix_sec: Target time (trying to interpolate to this
        time).
    :return: interp_gap_sec: Interpolation gap (seconds).
    """

    time_diff_before_sec = None
    time_diff_after_sec = None

    if numpy.any(source_times_unix_sec <= target_time_unix_sec):
        nearest_orig_time_before_unix_sec = source_times_unix_sec[
            source_times_unix_sec <= target_time_unix_sec
        ].max()

        time_diff_before_sec = (
            target_time_unix_sec - nearest_orig_time_before_unix_sec
        )

    if numpy.any(source_times_unix_sec >= target_time_unix_sec):
        nearest_orig_time_after_unix_sec = source_times_unix_sec[
            source_times_unix_sec >= target_time_unix_sec
        ].min()

        time_diff_after_sec = (
            nearest_orig_time_after_unix_sec - target_time_unix_sec
        )

    if time_diff_before_sec is None:
        interp_gap_sec = 2 * time_diff_after_sec
    elif time_diff_after_sec is None:
        interp_gap_sec = 2 * time_diff_before_sec
    else:
        interp_gap_sec = time_diff_before_sec + time_diff_after_sec

    return interp_gap_sec


def concat_over_time(satellite_tables_xarray, allow_different_cyclones=False):
    """Concatenates satellite data over many time steps.

    All time steps must contain data for the same tropical cyclone.

    :param satellite_tables_xarray: 1-D list of input tables, in format returned
        by `satellite_io.read_file`.
    :param allow_different_cyclones: Boolean flag.  If True, will allow data for
        different cyclones to be concatenated together.
    :return: satellite_table_xarray: xarray table with all time steps.
    :raises: ValueError: if cyclone IDs are not identical.
    """

    error_checking.assert_is_boolean(allow_different_cyclones)

    # TODO(thunderhoser): Might also need to have some condition for BDRF here.
    satellite_tables_xarray = [
        t for t in satellite_tables_xarray
        if t[BRIGHTNESS_TEMPERATURE_KEY].values.size > 0
    ]

    try:
        satellite_table_xarray = xarray.concat(
            satellite_tables_xarray, dim=TIME_DIM, data_vars='all',
            coords='minimal', compat='identical', join='exact'
        )
    except:
        satellite_table_xarray = xarray.concat(
            satellite_tables_xarray, dim=TIME_DIM, data_vars='all',
            coords='minimal', compat='identical'
        )

    if allow_different_cyclones:
        return satellite_table_xarray

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
    brightness_temp_matrix_kelvins = t[BRIGHTNESS_TEMPERATURE_KEY].values

    brightness_temp_matrix_kelvins[
        brightness_temp_matrix_kelvins < MIN_BRIGHTNESS_TEMP_KELVINS
    ] = numpy.nan
    brightness_temp_matrix_kelvins[
        brightness_temp_matrix_kelvins > MAX_BRIGHTNESS_TEMP_KELVINS
    ] = numpy.nan

    these_indices = numpy.where(numpy.isnan(brightness_temp_matrix_kelvins))
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
            numpy.isnan(brightness_temp_matrix_kelvins[i, ..., k])
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
            brightness_temp_matrix_kelvins[i, ..., k] = numpy.nan
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
        brightness_temp_matrix_kelvins[i, ..., k] = misc_utils.fill_nans(
            brightness_temp_matrix_kelvins[i, ..., k]
        )

    satellite_table_xarray = t
    satellite_table_xarray = satellite_table_xarray.assign({
        BRIGHTNESS_TEMPERATURE_KEY: (
            satellite_table_xarray[BRIGHTNESS_TEMPERATURE_KEY].dims,
            brightness_temp_matrix_kelvins
        )
    })

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
    bidirectional_reflectance_matrix = t[BIDIRECTIONAL_REFLECTANCE_KEY].values
    bidirectional_reflectance_matrix[bidirectional_reflectance_matrix < 0] = 0.

    these_indices = numpy.where(numpy.isnan(bidirectional_reflectance_matrix))
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
            numpy.isnan(bidirectional_reflectance_matrix[i, ..., k])
        )

        if num_bad_pixels > max_bad_pixels_per_time_channel:
            warning_string = (
                'POTENTIAL ERROR: Found {0:d} bad pixels for {1:.2f}-micron '
                'channel at {2:s}.  Replacing all pixels with NaN!'
            ).format(
                num_bad_pixels,
                METRES_TO_MICRONS * t.coords[HIGH_RES_WAVELENGTH_DIM].values[k],
                time_conversion.unix_sec_to_string(
                    t.coords[TIME_DIM].values[i], TIME_FORMAT_FOR_LOG_MESSAGES
                )
            )

            warnings.warn(warning_string)
            bidirectional_reflectance_matrix[i, ..., k] = numpy.nan
            continue

        log_string = (
            'Found {0:d} bad pixels for {1:.2f}-micron channel at {2:s}.  '
            'Inpainting to replace bad pixels!'
        ).format(
            num_bad_pixels,
            METRES_TO_MICRONS * t.coords[HIGH_RES_WAVELENGTH_DIM].values[k],
            time_conversion.unix_sec_to_string(
                t.coords[TIME_DIM].values[i], TIME_FORMAT_FOR_LOG_MESSAGES
            )
        )

        print(log_string)
        bidirectional_reflectance_matrix[i, ..., k] = misc_utils.fill_nans(
            bidirectional_reflectance_matrix[i, ..., k]
        )

    satellite_table_xarray = t
    satellite_table_xarray = satellite_table_xarray.assign({
        BIDIRECTIONAL_REFLECTANCE_KEY: (
            satellite_table_xarray[BIDIRECTIONAL_REFLECTANCE_KEY].dims,
            bidirectional_reflectance_matrix
        )
    })

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
    bidirectional_reflectance_matrix = t[BIDIRECTIONAL_REFLECTANCE_KEY].values
    valid_times_unix_sec = t.coords[TIME_DIM].values

    for i in range(len(valid_times_unix_sec)):
        these_latitudes_deg_n = t[LATITUDE_HIGH_RES_KEY].values[i, :][::120]
        these_longitudes_deg_e = t[LONGITUDE_HIGH_RES_KEY].values[i, :][::120]
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

        bidirectional_reflectance_matrix[i, ...] = numpy.nan

    satellite_table_xarray = t
    satellite_table_xarray = satellite_table_xarray.assign({
        BIDIRECTIONAL_REFLECTANCE_KEY: (
            satellite_table_xarray[BIDIRECTIONAL_REFLECTANCE_KEY].dims,
            bidirectional_reflectance_matrix
        )
    })

    return satellite_table_xarray


def subset_grid(satellite_table_xarray, num_rows_to_keep, num_columns_to_keep,
                for_high_res):
    """Subsets grid by center-cropping.

    :param satellite_table_xarray: xarray table in format returned by
        `read_file`.
    :param num_rows_to_keep: Number of grid rows to keep.
    :param num_columns_to_keep: Number of grid columns to keep.
    :param for_high_res: Boolean flag.  If True (False), will subset grid for
        high- (low-)resolution data.
    :return: new_table_xarray: Same as input but with smaller grid.
    """

    # Error-checking.
    error_checking.assert_is_integer(num_rows_to_keep)
    error_checking.assert_is_greater(num_rows_to_keep, 0)
    assert numpy.mod(num_rows_to_keep, 2) == 0

    error_checking.assert_is_integer(num_columns_to_keep)
    error_checking.assert_is_greater(num_columns_to_keep, 0)
    assert numpy.mod(num_columns_to_keep, 2) == 0

    error_checking.assert_is_boolean(for_high_res)

    t = satellite_table_xarray
    if (
            for_high_res and t[BIDIRECTIONAL_REFLECTANCE_KEY].values.size == 0
    ):
        return satellite_table_xarray

    if (
            not for_high_res and t[BRIGHTNESS_TEMPERATURE_KEY].values.size == 0
    ):
        return satellite_table_xarray

    row_dim = HIGH_RES_ROW_DIM if for_high_res else LOW_RES_ROW_DIM
    column_dim = HIGH_RES_COLUMN_DIM if for_high_res else LOW_RES_COLUMN_DIM

    num_rows_orig = len(satellite_table_xarray.coords[row_dim].values)
    num_columns_orig = len(satellite_table_xarray.coords[column_dim].values)

    assert numpy.mod(num_rows_orig, 2) == 0
    assert numpy.mod(num_columns_orig, 2) == 0

    error_checking.assert_is_less_than(num_rows_to_keep, num_rows_orig)
    error_checking.assert_is_less_than(num_columns_to_keep, num_columns_orig)

    # Do actual stuff.
    first_index = int(numpy.round(
        num_rows_orig / 2 - num_rows_to_keep / 2
    ))
    last_index = int(numpy.round(
        num_rows_orig / 2 + num_rows_to_keep / 2
    ))
    good_indices = numpy.linspace(
        first_index, last_index - 1, num=num_rows_to_keep, dtype=int
    )
    satellite_table_xarray = satellite_table_xarray.isel(
        indexers={row_dim: good_indices}
    )

    first_index = int(numpy.round(
        num_columns_orig / 2 - num_columns_to_keep / 2
    ))
    last_index = int(numpy.round(
        num_columns_orig / 2 + num_columns_to_keep / 2
    ))
    good_indices = numpy.linspace(
        first_index, last_index - 1, num=num_columns_to_keep, dtype=int
    )
    return satellite_table_xarray.isel(
        indexers={column_dim: good_indices}
    )


def subset_wavelengths(satellite_table_xarray, wavelengths_to_keep_microns,
                       for_high_res):
    """Subsets wavelengths.

    :param satellite_table_xarray: xarray table in format returned by
        `read_file`.
    :param wavelengths_to_keep_microns: 1-D numpy array of wavelengths to keep.
    :param for_high_res: Boolean flag.  If True (False), will subset wavelengths
        for high- (low-)resolution data.
    :return: new_table_xarray: Same as input but maybe with fewer wavelengths.
    """

    error_checking.assert_is_numpy_array(
        wavelengths_to_keep_microns, num_dimensions=1
    )
    error_checking.assert_is_greater_numpy_array(
        wavelengths_to_keep_microns, 0.
    )
    error_checking.assert_equals(
        len(wavelengths_to_keep_microns),
        len(numpy.unique(wavelengths_to_keep_microns))
    )
    error_checking.assert_is_boolean(for_high_res)

    t = satellite_table_xarray
    if (
            for_high_res and t[BIDIRECTIONAL_REFLECTANCE_KEY].values.size == 0
    ):
        return satellite_table_xarray

    if (
            not for_high_res and t[BRIGHTNESS_TEMPERATURE_KEY].values.size == 0
    ):
        return satellite_table_xarray

    wavelength_dim = (
        HIGH_RES_WAVELENGTH_DIM if for_high_res else LOW_RES_WAVELENGTH_DIM
    )
    orig_wavelengths_microns = (
        METRES_TO_MICRONS * satellite_table_xarray.coords[wavelength_dim].values
    )

    good_indices = numpy.array([
        numpy.where(
            numpy.absolute(w - orig_wavelengths_microns) < TOLERANCE
        )[0][0]
        for w in wavelengths_to_keep_microns
    ], dtype=int)

    if len(good_indices) == 0:
        if for_high_res:
            satellite_table_xarray = satellite_table_xarray.drop_vars(
                names=BIDIRECTIONAL_REFLECTANCE_KEY
            )
            # satellite_table_xarray = satellite_table_xarray.drop_dims(
            #     HIGH_RES_WAVELENGTH_DIM
            # )
        else:
            satellite_table_xarray = satellite_table_xarray.drop_vars(
                names=BRIGHTNESS_TEMPERATURE_KEY
            )
            # satellite_table_xarray = satellite_table_xarray.drop_dims(
            #     LOW_RES_WAVELENGTH_DIM
            # )

    return satellite_table_xarray.isel(
        indexers={wavelength_dim: good_indices}
    )


def subset_to_multiple_time_windows(
        satellite_table_xarray, start_times_unix_sec, end_times_unix_sec):
    """Subsets to multiple time windows.

    W = number of time windows

    :param satellite_table_xarray: xarray table in format returned by
        `read_file`.
    :param start_times_unix_sec: length-W numpy array with start of each time
        window.
    :param end_times_unix_sec: length-W numpy array with end of each time
        window.
    :return: new_table_xarray: Same as input but maybe with fewer times.
    """

    t = satellite_table_xarray
    if (
            BIDIRECTIONAL_REFLECTANCE_KEY in t
            and t[BIDIRECTIONAL_REFLECTANCE_KEY].values.size == 0
    ):
        return satellite_table_xarray

    if t[BRIGHTNESS_TEMPERATURE_KEY].values.size == 0:
        return satellite_table_xarray

    # Check input args.
    error_checking.assert_is_numpy_array(start_times_unix_sec, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(start_times_unix_sec)
    num_time_windows = len(start_times_unix_sec)

    error_checking.assert_is_numpy_array(
        end_times_unix_sec,
        exact_dimensions=numpy.array([num_time_windows], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(end_times_unix_sec)
    error_checking.assert_is_geq_numpy_array(
        end_times_unix_sec - start_times_unix_sec, 0
    )

    # Do actual stuff.
    orig_times_unix_sec = satellite_table_xarray.coords[TIME_DIM].values

    num_orig_times = len(orig_times_unix_sec)
    keep_orig_time_flags = numpy.full(num_orig_times, 0, dtype=bool)

    for j in range(num_time_windows):
        these_good_indices = numpy.where(numpy.logical_and(
            orig_times_unix_sec >= start_times_unix_sec[j],
            orig_times_unix_sec <= end_times_unix_sec[j]
        ))[0]

        keep_orig_time_flags[these_good_indices] = True

    good_indices = numpy.where(keep_orig_time_flags)[0]
    return satellite_table_xarray.isel(
        indexers={TIME_DIM: good_indices}
    )


def _find_times_with_all_nan_maps(satellite_table_xarray):
    """Finds every time with at least one all-NaN map.

    An "all-NaN map" is a spatial map, for one wavelength, that contains only
    NaN -- no real values.

    T = number of times in table

    :param satellite_table_xarray: xarray table in format returned by
        `read_file`.
    :return: bad_time_flags: length-T numpy array of Boolean flags.
    """

    t = satellite_table_xarray

    if BIDIRECTIONAL_REFLECTANCE_KEY in t:
        bad_time_wavelength_flag_matrix = numpy.all(
            numpy.isnan(t[BIDIRECTIONAL_REFLECTANCE_KEY].values), axis=(1, 2)
        )
        bad_time_flags = numpy.any(bad_time_wavelength_flag_matrix, axis=1)
    else:
        num_times = len(t.coords[TIME_DIM].values)
        bad_time_flags = numpy.full(num_times, 0, dtype=bool)

    if BRIGHTNESS_TEMPERATURE_KEY in t:
        bad_time_wavelength_flag_matrix = numpy.all(
            numpy.isnan(t[BRIGHTNESS_TEMPERATURE_KEY].values), axis=(1, 2)
        )
        new_bad_time_flags = numpy.any(bad_time_wavelength_flag_matrix, axis=1)
        bad_time_flags = numpy.logical_or(bad_time_flags, new_bad_time_flags)

    return bad_time_flags


def subset_times(satellite_table_xarray, desired_times_unix_sec,
                 tolerances_sec, max_num_missing_times, max_interp_gaps_sec):
    """Subsets time steps.

    T = number of desired times

    :param satellite_table_xarray: xarray table in format returned by
        `read_file`.
    :param desired_times_unix_sec: length-T numpy array of desired times.
    :param tolerances_sec: length-T numpy array of tolerances, one for each
        desired time.
    :param max_num_missing_times: Max number of missing times.
    :param max_interp_gaps_sec: length-T numpy array of maximum interpolation
        gaps.  Will not interpolate over a longer gap.
    :return: new_table_xarray: Same as input but maybe with fewer times.
    :raises: ValueError: if number of missing times > `max_num_missing_times`.
    :raises: ValueError: if all brightness temperatures are NaN at the end.
    """

    t = satellite_table_xarray
    if (
            BIDIRECTIONAL_REFLECTANCE_KEY in t
            and t[BIDIRECTIONAL_REFLECTANCE_KEY].values.size == 0
    ):
        return satellite_table_xarray

    if t[BRIGHTNESS_TEMPERATURE_KEY].values.size == 0:
        return satellite_table_xarray

    # Check input args.
    error_checking.assert_is_numpy_array(
        desired_times_unix_sec, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_times_unix_sec)
    error_checking.assert_equals(
        len(desired_times_unix_sec),
        len(numpy.unique(desired_times_unix_sec))
    )

    num_desired_times = len(desired_times_unix_sec)

    error_checking.assert_is_numpy_array(
        tolerances_sec,
        exact_dimensions=numpy.array([num_desired_times], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(tolerances_sec)
    error_checking.assert_is_geq_numpy_array(tolerances_sec, 0)

    error_checking.assert_is_integer(max_num_missing_times)
    error_checking.assert_is_geq(max_num_missing_times, 0)
    error_checking.assert_is_leq(max_num_missing_times, num_desired_times)

    error_checking.assert_is_numpy_array(
        max_interp_gaps_sec,
        exact_dimensions=numpy.array([num_desired_times], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(max_interp_gaps_sec)
    error_checking.assert_is_geq_numpy_array(max_interp_gaps_sec, 0)

    # Do actual stuff.
    orig_times_with_dummy_unix_sec = (
        satellite_table_xarray.coords[TIME_DIM].values + 0
    )
    print(satellite_table_xarray)
    bad_time_flags = _find_times_with_all_nan_maps(satellite_table_xarray)
    orig_times_with_dummy_unix_sec[bad_time_flags] = DUMMY_TIME_UNIX_SEC

    desired_indices = numpy.full(num_desired_times, -1, dtype=int)

    for i in range(num_desired_times):
        these_diffs_sec = numpy.absolute(
            orig_times_with_dummy_unix_sec - desired_times_unix_sec[i]
        )
        if numpy.min(these_diffs_sec) > tolerances_sec[i]:
            continue

        desired_indices[i] = numpy.argmin(these_diffs_sec)

    num_missing_times = numpy.sum(desired_indices == -1)

    if num_missing_times > max_num_missing_times:
        missing_time_strings = [
            time_conversion.unix_sec_to_string(t, TIME_FORMAT_FOR_LOG_MESSAGES)
            for t in desired_times_unix_sec[desired_indices == -1]
        ]

        error_string = (
            'Could not find satellite data at the following times:\n{0:s}'
        ).format(str(missing_time_strings))

        warning_string = 'POTENTIAL ERROR: {0:s}'.format(error_string)
        warnings.warn(warning_string)

        raise ValueError(error_string)

    if num_missing_times == 0:
        new_table_xarray = satellite_table_xarray.isel(
            indexers={TIME_DIM: desired_indices}
        )

        try:
            return new_table_xarray.assign_coords({
                TIME_DIM: desired_times_unix_sec
            })
        except:
            return new_table_xarray.assign_coords(
                TIME_DIM=desired_times_unix_sec
            )

    new_table_xarray = copy.deepcopy(satellite_table_xarray)
    new_table_xarray = new_table_xarray.isel(
        indexers={TIME_DIM: desired_indices}
    )

    try:
        new_table_xarray = new_table_xarray.assign_coords({
            TIME_DIM: desired_times_unix_sec
        })
    except:
        new_table_xarray = new_table_xarray.assign_coords(
            TIME_DIM=desired_times_unix_sec
        )

    low_res_wavelengths_microns = (
        METRES_TO_MICRONS *
        satellite_table_xarray.coords[LOW_RES_WAVELENGTH_DIM].values
    )
    high_res_wavelengths_microns = (
        METRES_TO_MICRONS *
        satellite_table_xarray.coords[HIGH_RES_WAVELENGTH_DIM].values
    )

    failed_to_interp = False

    new_brightness_temp_matrix_kelvins = (
        new_table_xarray[BRIGHTNESS_TEMPERATURE_KEY].values
    )
    if len(high_res_wavelengths_microns) > 0:
        new_bidirectional_reflectance_matrix = (
            new_table_xarray[BIDIRECTIONAL_REFLECTANCE_KEY].values
        )

    for i in range(num_desired_times):
        if desired_indices[i] != -1:
            continue

        for j in range(len(low_res_wavelengths_microns)):
            source_table_xarray = subset_wavelengths(
                satellite_table_xarray=copy.deepcopy(satellite_table_xarray),
                wavelengths_to_keep_microns=low_res_wavelengths_microns[[j]],
                for_high_res=False
            )

            bad_time_flags = _find_times_with_all_nan_maps(source_table_xarray)
            good_time_indices = numpy.where(numpy.invert(bad_time_flags))[0]

            if len(good_time_indices) == 0:
                new_brightness_temp_matrix_kelvins[i, ..., j] = numpy.nan
                failed_to_interp = True
                break

            source_table_xarray = source_table_xarray.isel(
                indexers={TIME_DIM: good_time_indices}
            )

            # TODO(thunderhoser): I could make this code more efficient by
            # calling isel once instead of twice.
            sort_indices = numpy.argsort(
                source_table_xarray.coords[TIME_DIM].values
            )
            source_table_xarray = source_table_xarray.isel(
                indexers={TIME_DIM: sort_indices}
            )

            interp_gap_sec = _compute_interpolation_gap(
                source_times_unix_sec=
                source_table_xarray.coords[TIME_DIM].values,
                target_time_unix_sec=desired_times_unix_sec[i]
            )

            if interp_gap_sec > max_interp_gaps_sec[i]:
                new_brightness_temp_matrix_kelvins[i, ..., j] = numpy.nan
                failed_to_interp = True
                break

            # st = source_table_xarray
            # fill_value_arg = (
            #     st[BRIGHTNESS_TEMPERATURE_KEY].values[0, ..., 0],
            #     st[BRIGHTNESS_TEMPERATURE_KEY].values[-1, ..., 0]
            # )

            interp_object = interp1d(
                x=source_table_xarray.coords[TIME_DIM].values,
                y=source_table_xarray[BRIGHTNESS_TEMPERATURE_KEY].values[
                    ..., 0
                ],
                kind='linear', axis=0, bounds_error=False,
                fill_value='extrapolate', assume_sorted=True
            )

            new_brightness_temp_matrix_kelvins[i, ..., j] = interp_object(
                new_table_xarray.coords[TIME_DIM].values[i]
            )

        if failed_to_interp:
            break

        for j in range(len(high_res_wavelengths_microns)):
            source_table_xarray = subset_wavelengths(
                satellite_table_xarray=copy.deepcopy(satellite_table_xarray),
                wavelengths_to_keep_microns=high_res_wavelengths_microns[[j]],
                for_high_res=True
            )

            bad_time_flags = _find_times_with_all_nan_maps(source_table_xarray)
            good_time_indices = numpy.where(numpy.invert(bad_time_flags))[0]

            if len(good_time_indices) == 0:
                new_bidirectional_reflectance_matrix[i, ..., j] = numpy.nan
                failed_to_interp = True
                break

            source_table_xarray = source_table_xarray.isel(
                indexers={TIME_DIM: good_time_indices}
            )

            # TODO(thunderhoser): I could make this code more efficient by
            # calling isel once instead of twice.
            sort_indices = numpy.argsort(
                source_table_xarray.coords[TIME_DIM].values
            )
            source_table_xarray = source_table_xarray.isel(
                indexers={TIME_DIM: sort_indices}
            )

            interp_gap_sec = _compute_interpolation_gap(
                source_times_unix_sec=
                source_table_xarray.coords[TIME_DIM].values,
                target_time_unix_sec=desired_times_unix_sec[i]
            )

            if interp_gap_sec > max_interp_gaps_sec[i]:
                new_bidirectional_reflectance_matrix[i, ..., j] = numpy.nan
                failed_to_interp = True
                break

            # st = source_table_xarray
            # fill_value_arg = (
            #     st[BRIGHTNESS_TEMPERATURE_KEY].values[0, ..., 0],
            #     st[BRIGHTNESS_TEMPERATURE_KEY].values[-1, ..., 0]
            # )

            interp_object = interp1d(
                x=source_table_xarray.coords[TIME_DIM].values,
                y=source_table_xarray[BIDIRECTIONAL_REFLECTANCE_KEY].values[
                    ..., 0
                ],
                kind='linear', axis=0, bounds_error=False,
                fill_value='extrapolate', assume_sorted=True
            )

            new_bidirectional_reflectance_matrix[i, ..., j] = interp_object(
                new_table_xarray.coords[TIME_DIM].values[i]
            )

        if failed_to_interp:
            break

    new_table_xarray = new_table_xarray.assign({
        BRIGHTNESS_TEMPERATURE_KEY: (
            new_table_xarray[BRIGHTNESS_TEMPERATURE_KEY].dims,
            new_brightness_temp_matrix_kelvins
        )
    })

    if len(high_res_wavelengths_microns) > 0:
        new_table_xarray = new_table_xarray.assign({
            BIDIRECTIONAL_REFLECTANCE_KEY: (
                new_table_xarray[BIDIRECTIONAL_REFLECTANCE_KEY].dims,
                new_bidirectional_reflectance_matrix
            )
        })

    bad_time_flags = _find_times_with_all_nan_maps(new_table_xarray)
    if not numpy.any(bad_time_flags):
        return new_table_xarray

    bad_times_unix_sec = (
        new_table_xarray.coords[TIME_DIM].values[bad_time_flags]
    )
    bad_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT_FOR_LOG_MESSAGES)
        for t in bad_times_unix_sec
    ]

    error_string = (
        'After interpolation, all-NaN maps still exist for the following '
        'times:\n{0:s}'
    ).format(str(bad_time_strings))

    warning_string = 'POTENTIAL ERROR: {0:s}'.format(error_string)
    warnings.warn(warning_string)

    raise ValueError(error_string)


def subset_times_exact(satellite_table_xarray, desired_times_unix_sec):
    """Subsets time steps exactly (no error tolerance).

    T = number of desired times

    :param satellite_table_xarray: xarray table in format returned by
        `read_file`.
    :param desired_times_unix_sec: length-T numpy array of desired times.
    :return: new_table_xarray: Same as input but maybe with fewer times.
    """

    t = satellite_table_xarray
    if (
            BIDIRECTIONAL_REFLECTANCE_KEY in t
            and t[BIDIRECTIONAL_REFLECTANCE_KEY].values.size == 0
    ):
        return satellite_table_xarray

    if t[BRIGHTNESS_TEMPERATURE_KEY].values.size == 0:
        return satellite_table_xarray

    # Check input args.
    error_checking.assert_is_numpy_array(
        desired_times_unix_sec, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_times_unix_sec)
    error_checking.assert_equals(
        len(desired_times_unix_sec),
        len(numpy.unique(desired_times_unix_sec))
    )

    num_desired_times = len(desired_times_unix_sec)

    # Do actual stuff.
    orig_times_unix_sec = satellite_table_xarray.coords[TIME_DIM].values
    desired_indices = numpy.full(num_desired_times, -1, dtype=int)

    for i in range(num_desired_times):
        these_diffs_sec = numpy.absolute(
            orig_times_unix_sec - desired_times_unix_sec[i]
        )
        if numpy.min(these_diffs_sec) > 0:
            continue

        desired_indices[i] = numpy.argmin(these_diffs_sec)

    num_missing_times = numpy.sum(desired_indices == -1)

    if num_missing_times > 0:
        missing_time_strings = [
            time_conversion.unix_sec_to_string(t, TIME_FORMAT_FOR_LOG_MESSAGES)
            for t in desired_times_unix_sec[desired_indices == -1]
        ]

        error_string = (
            'Could not find satellite data at the following times:\n{0:s}'
        ).format(str(missing_time_strings))

        warning_string = 'POTENTIAL ERROR: {0:s}'.format(error_string)
        warnings.warn(warning_string)

        raise ValueError(error_string)

    new_table_xarray = satellite_table_xarray.isel(
        indexers={TIME_DIM: desired_indices}
    )
    bad_time_flags = _find_times_with_all_nan_maps(new_table_xarray)

    if not numpy.any(bad_time_flags):
        return new_table_xarray

    desired_indices[bad_time_flags] = -1
    missing_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT_FOR_LOG_MESSAGES)
        for t in desired_times_unix_sec[desired_indices == -1]
    ]

    error_string = (
        'Could not find satellite data at the following times:\n{0:s}'
    ).format(str(missing_time_strings))

    warning_string = 'POTENTIAL ERROR: {0:s}'.format(error_string)
    warnings.warn(warning_string)

    raise ValueError(error_string)
