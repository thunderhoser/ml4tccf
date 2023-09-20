"""Methods for reading and converting raw XBT (extended best track) data."""

import os
import sys
import numpy
import pandas
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import longitude_conversion as lng_conversion
import error_checking
import misc_utils
import extended_best_track_utils as xbt_utils

TIME_FORMAT = '%Y%m%d%H'
SENTINEL_VALUE = -99.

KT_TO_METRES_PER_SECOND = 1.852 / 3.6
NAUTICAL_MILES_TO_METRES = 1852.
KM_TO_METRES = 1000.
MB_TO_PASCALS = 100.
SECONDS_TO_HOURS = 1. / 3600

STORM_ID_KEY = xbt_utils.STORM_ID_KEY
STORM_NAME_KEY = xbt_utils.STORM_NAME_KEY
VALID_TIME_NO_YEAR_KEY = 'valid_time_string_no_year'
VALID_YEAR_KEY = 'valid_year'
VALID_TIME_KEY = xbt_utils.VALID_TIME_KEY
CENTER_LATITUDE_KEY = xbt_utils.CENTER_LATITUDE_KEY
CENTER_LONGITUDE_KEY = xbt_utils.CENTER_LONGITUDE_KEY
MAX_SUSTAINED_WIND_KEY = xbt_utils.MAX_SUSTAINED_WIND_KEY
MIN_PRESSURE_KEY = xbt_utils.MIN_PRESSURE_KEY
MAX_WIND_RADIUS_KEY = xbt_utils.MAX_WIND_RADIUS_KEY
EYE_DIAMETER_KEY = xbt_utils.EYE_DIAMETER_KEY
OUTERMOST_PRESSURE_KEY = xbt_utils.OUTERMOST_PRESSURE_KEY
OUTERMOST_RADIUS_KEY = xbt_utils.OUTERMOST_RADIUS_KEY
RADIUS_OF_34KT_WIND_NE_KEY = 'radius_of_34kt_wind_ne_metres'
RADIUS_OF_34KT_WIND_SE_KEY = 'radius_of_34kt_wind_se_metres'
RADIUS_OF_34KT_WIND_SW_KEY = 'radius_of_34kt_wind_sw_metres'
RADIUS_OF_34KT_WIND_NW_KEY = 'radius_of_34kt_wind_nw_metres'
RADII_OF_34KT_WIND_KEY = xbt_utils.RADII_OF_34KT_WIND_KEY
RADIUS_OF_50KT_WIND_NE_KEY = 'radius_of_50kt_wind_ne_metres'
RADIUS_OF_50KT_WIND_SE_KEY = 'radius_of_50kt_wind_se_metres'
RADIUS_OF_50KT_WIND_SW_KEY = 'radius_of_50kt_wind_sw_metres'
RADIUS_OF_50KT_WIND_NW_KEY = 'radius_of_50kt_wind_nw_metres'
RADII_OF_50KT_WIND_KEY = xbt_utils.RADII_OF_50KT_WIND_KEY
RADIUS_OF_64KT_WIND_NE_KEY = 'radius_of_64kt_wind_ne_metres'
RADIUS_OF_64KT_WIND_SE_KEY = 'radius_of_64kt_wind_se_metres'
RADIUS_OF_64KT_WIND_SW_KEY = 'radius_of_64kt_wind_sw_metres'
RADIUS_OF_64KT_WIND_NW_KEY = 'radius_of_64kt_wind_nw_metres'
RADII_OF_64KT_WIND_KEY = xbt_utils.RADII_OF_64KT_WIND_KEY
STORM_TYPE_KEY = xbt_utils.STORM_TYPE_KEY
DISTANCE_TO_LAND_KEY = xbt_utils.DISTANCE_TO_LAND_KEY
DATA_SOURCE_KEY = 'data_source_string'
MAX_WIND_RADIUS_SOURCE_KEY = xbt_utils.MAX_WIND_RADIUS_SOURCE_KEY
EYE_DIAMETER_SOURCE_KEY = xbt_utils.EYE_DIAMETER_SOURCE_KEY
OUTERMOST_PRESSURE_SOURCE_KEY = xbt_utils.OUTERMOST_PRESSURE_SOURCE_KEY
OUTERMOST_RADIUS_SOURCE_KEY = xbt_utils.OUTERMOST_RADIUS_SOURCE_KEY

COLUMN_NAMES = [
    STORM_ID_KEY, STORM_NAME_KEY, VALID_TIME_NO_YEAR_KEY, VALID_YEAR_KEY,
    CENTER_LATITUDE_KEY, CENTER_LONGITUDE_KEY, MAX_SUSTAINED_WIND_KEY,
    MIN_PRESSURE_KEY, MAX_WIND_RADIUS_KEY, EYE_DIAMETER_KEY,
    OUTERMOST_PRESSURE_KEY, OUTERMOST_RADIUS_KEY,
    RADIUS_OF_34KT_WIND_NE_KEY, RADIUS_OF_34KT_WIND_SE_KEY,
    RADIUS_OF_34KT_WIND_SW_KEY, RADIUS_OF_34KT_WIND_NW_KEY,
    RADIUS_OF_50KT_WIND_NE_KEY, RADIUS_OF_50KT_WIND_SE_KEY,
    RADIUS_OF_50KT_WIND_SW_KEY, RADIUS_OF_50KT_WIND_NW_KEY,
    RADIUS_OF_64KT_WIND_NE_KEY, RADIUS_OF_64KT_WIND_SE_KEY,
    RADIUS_OF_64KT_WIND_SW_KEY, RADIUS_OF_64KT_WIND_NW_KEY,
    STORM_TYPE_KEY, DISTANCE_TO_LAND_KEY, DATA_SOURCE_KEY
]

COLUMN_LIMITS = [
    (0, 8), (8, 20), (20, 27), (27, 32), (32, 38), (38, 45), (45, 49), (49, 54),
    (54, 58), (58, 62), (62, 67), (67, 71), (71, 75), (75, 78), (78, 81),
    (81, 84), (84, 88), (88, 91), (91, 94), (94, 97), (97, 101), (101, 104),
    (104, 107), (107, 110), (110, 112), (112, 118), (118, 124)
]

STORM_TYPE_CONVERSION_DICT = {
    '*': xbt_utils.TROPICAL_TYPE_STRING,
    'W': xbt_utils.WAVE_TYPE_STRING,
    'D': xbt_utils.DISTURBANCE_TYPE_STRING,
    'S': xbt_utils.SUBTROPICAL_TYPE_STRING,
    'E': xbt_utils.EXTRATROPICAL_TYPE_STRING,
    'L': xbt_utils.REMNANT_LOW_TYPE_STRING,
    'X': xbt_utils.UNKNOWN_TYPE_STRING,
    'A': xbt_utils.UNKNOWN_TYPE_STRING,
    'M': xbt_utils.UNKNOWN_TYPE_STRING,
    'P': xbt_utils.UNKNOWN_TYPE_STRING
}

DATA_SOURCE_CONVERSION_DICT = {
    '0': xbt_utils.B_DECK_SOURCE_STRING,
    '1': xbt_utils.A_DECK_SOURCE_STRING,
    '2': xbt_utils.OFFICIAL_0H_FCST_SOURCE_STRING,
    '3': xbt_utils.OFFICIAL_3H_FCST_SOURCE_STRING,
    '4': xbt_utils.PREVIOUS_XBT_VERSION_SOURCE_STRING,
    '9': xbt_utils.MISSING_SOURCE_STRING
}


def _cyclone_id_orig_to_new(orig_cyclone_id_string):
    """Converts cyclone ID from original format to new format.

    :param orig_cyclone_id_string: Original ID (format BBYYYYNN), where bb is
        the basin; YYYY is the year; and NN is the ordinal number.
    :return: cyclone_id_string: Proper ID (format YYYYBBNN).
    """

    cyclone_id_string = '{0:s}{1:s}'.format(
        orig_cyclone_id_string[-4:], orig_cyclone_id_string[:4].upper()
    )
    _ = misc_utils.parse_cyclone_id(cyclone_id_string)

    return cyclone_id_string


def _remove_sentinel_values(xbt_table_pandas, variable_name):
    """Removes sentinel values of one variable, by replacing with NaN.

    :param xbt_table_pandas: pandas DataFrame with XBT data.
    :param variable_name: Variable name.
    :return: xbt_table_pandas: Same as input but with sentinel values converted
        to NaN.
    """

    data_values = xbt_table_pandas[variable_name].values.astype(float)
    data_values[data_values < SENTINEL_VALUE + 0.1] = numpy.nan
    return xbt_table_pandas.assign(**{
        variable_name: data_values
    })


def read_file(ascii_file_name):
    """Reads XBT (extended best track) data from ASCII file.

    :param ascii_file_name: Path to input file.
    :return: xbt_table_xarray: xarray table with XBT data.
    """

    xbt_table_pandas = pandas.read_fwf(
        ascii_file_name, colspecs=COLUMN_LIMITS, names=COLUMN_NAMES
    )

    xbt_table_strings_only = xbt_table_pandas.select_dtypes(['object'])
    xbt_table_pandas[xbt_table_strings_only.columns] = (
        xbt_table_strings_only.apply(lambda x: x.str.strip())
    )

    # Convert valid time to Unix format.
    valid_time_strings = [
        '{0:04d}{1:06d}'.format(y, n) for y, n in zip(
            xbt_table_pandas[VALID_YEAR_KEY].values,
            xbt_table_pandas[VALID_TIME_NO_YEAR_KEY].values
        )
    ]

    num_storm_objects = len(valid_time_strings)
    valid_times_unix_sec = numpy.full(num_storm_objects, -1, dtype=int)

    for i in range(num_storm_objects):
        try:
            valid_times_unix_sec[i] = time_conversion.string_to_unix_sec(
                valid_time_strings[i], TIME_FORMAT
            )
        except:
            pass

    bad_indices = numpy.where(valid_times_unix_sec == -1)[0]
    valid_times_unix_sec = valid_times_unix_sec[valid_times_unix_sec != -1]
    xbt_table_pandas.drop(
        xbt_table_pandas.index[bad_indices], axis=0, inplace=True
    )

    valid_times_unix_hours = numpy.round(
        valid_times_unix_sec * SECONDS_TO_HOURS
    ).astype(int)

    storm_id_strings = numpy.array([
        _cyclone_id_orig_to_new(c)
        for c in xbt_table_pandas[STORM_ID_KEY].values
    ])

    these_dim = (xbt_utils.STORM_OBJECT_DIM,)
    main_data_dict = {
        STORM_ID_KEY: (these_dim, storm_id_strings),
        STORM_NAME_KEY: (these_dim, xbt_table_pandas[STORM_NAME_KEY].values),
        VALID_TIME_KEY: (these_dim, valid_times_unix_hours)
    }

    xbt_table_pandas.drop(
        [VALID_TIME_NO_YEAR_KEY, VALID_YEAR_KEY], axis=1, inplace=True
    )

    # Verify latitude.
    center_latitudes_deg_n = xbt_table_pandas[CENTER_LATITUDE_KEY].values
    error_checking.assert_is_valid_lat_numpy_array(
        latitudes_deg=center_latitudes_deg_n, allow_nan=False
    )
    main_data_dict[CENTER_LATITUDE_KEY] = (
        (xbt_utils.STORM_OBJECT_DIM,), center_latitudes_deg_n
    )

    # Convert longitude to deg E, positive in western hemisphere.
    center_longitudes_deg_e = xbt_table_pandas[CENTER_LONGITUDE_KEY].values
    pathless_file_name = os.path.split(ascii_file_name)[1]

    if (
            'EBTRK_AL' in pathless_file_name or
            'EBTRK_CP' in pathless_file_name or
            'EBTRK_EP' in pathless_file_name
    ):
        center_longitudes_deg_e = -1 * center_longitudes_deg_e
        center_longitudes_deg_e[center_longitudes_deg_e <= -180] += 360

    center_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        longitudes_deg=center_longitudes_deg_e, allow_nan=False
    )
    main_data_dict[CENTER_LONGITUDE_KEY] = (
        (xbt_utils.STORM_OBJECT_DIM,), center_longitudes_deg_e
    )

    # Convert max sustained wind to m/s.
    xbt_table_pandas = _remove_sentinel_values(
        xbt_table_pandas=xbt_table_pandas, variable_name=MAX_SUSTAINED_WIND_KEY
    )
    max_sustained_winds_m_s01 = (
        KT_TO_METRES_PER_SECOND *
        xbt_table_pandas[MAX_SUSTAINED_WIND_KEY].values
    )

    error_checking.assert_is_greater_numpy_array(
        max_sustained_winds_m_s01, 0., allow_nan=True
    )
    error_checking.assert_is_leq_numpy_array(
        max_sustained_winds_m_s01, 100., allow_nan=True
    )
    main_data_dict[MAX_SUSTAINED_WIND_KEY] = (
        (xbt_utils.STORM_OBJECT_DIM,), max_sustained_winds_m_s01
    )

    # Convert minimum pressure to Pascals.
    xbt_table_pandas = _remove_sentinel_values(
        xbt_table_pandas=xbt_table_pandas, variable_name=MIN_PRESSURE_KEY
    )
    min_pressures_pa = MB_TO_PASCALS * xbt_table_pandas[MIN_PRESSURE_KEY].values
    min_pressures_pa[min_pressures_pa > 900000] = numpy.nan

    error_checking.assert_is_geq_numpy_array(
        min_pressures_pa, 85000., allow_nan=True
    )
    error_checking.assert_is_leq_numpy_array(
        min_pressures_pa, 105000., allow_nan=True
    )
    main_data_dict[MIN_PRESSURE_KEY] = (
        (xbt_utils.STORM_OBJECT_DIM,), min_pressures_pa
    )

    # Convert radius of max wind to metres.
    xbt_table_pandas = _remove_sentinel_values(
        xbt_table_pandas=xbt_table_pandas, variable_name=MAX_WIND_RADIUS_KEY
    )
    max_wind_radii_metres = (
        NAUTICAL_MILES_TO_METRES * xbt_table_pandas[MAX_WIND_RADIUS_KEY].values
    )

    error_checking.assert_is_geq_numpy_array(
        max_wind_radii_metres, 0., allow_nan=True
    )
    main_data_dict[MAX_WIND_RADIUS_KEY] = (
        (xbt_utils.STORM_OBJECT_DIM,), max_wind_radii_metres
    )

    # Convert eye diameter to metres.
    xbt_table_pandas = _remove_sentinel_values(
        xbt_table_pandas=xbt_table_pandas, variable_name=EYE_DIAMETER_KEY
    )
    eye_diameters_metres = (
        NAUTICAL_MILES_TO_METRES * xbt_table_pandas[EYE_DIAMETER_KEY].values
    )

    error_checking.assert_is_geq_numpy_array(
        eye_diameters_metres, 0., allow_nan=True
    )
    error_checking.assert_is_leq_numpy_array(
        eye_diameters_metres, 200000., allow_nan=True
    )
    main_data_dict[EYE_DIAMETER_KEY] = (
        (xbt_utils.STORM_OBJECT_DIM,), eye_diameters_metres
    )

    # Convert pressure of outermost closed isobar to Pascals.
    xbt_table_pandas = _remove_sentinel_values(
        xbt_table_pandas=xbt_table_pandas, variable_name=OUTERMOST_PRESSURE_KEY
    )
    outermost_pressures_pa = (
        MB_TO_PASCALS * xbt_table_pandas[OUTERMOST_PRESSURE_KEY].values
    )
    outermost_pressures_pa[outermost_pressures_pa < 85000] = numpy.nan

    error_checking.assert_is_geq_numpy_array(
        outermost_pressures_pa, 85000., allow_nan=True
    )
    error_checking.assert_is_leq_numpy_array(
        outermost_pressures_pa, 105000., allow_nan=True
    )
    main_data_dict[OUTERMOST_PRESSURE_KEY] = (
        (xbt_utils.STORM_OBJECT_DIM,), outermost_pressures_pa
    )

    # Convert radius of outermost closed isobar to metres.
    xbt_table_pandas = _remove_sentinel_values(
        xbt_table_pandas=xbt_table_pandas, variable_name=OUTERMOST_RADIUS_KEY
    )
    outermost_radii_metres = (
        NAUTICAL_MILES_TO_METRES *
        xbt_table_pandas[OUTERMOST_RADIUS_KEY].values
    )

    error_checking.assert_is_geq_numpy_array(
        outermost_radii_metres, 0., allow_nan=True
    )
    main_data_dict[OUTERMOST_RADIUS_KEY] = (
        (xbt_utils.STORM_OBJECT_DIM,), outermost_radii_metres
    )

    # Convert radius of 34-kt wind to metres.
    var_name_by_quadrant = [
        RADIUS_OF_34KT_WIND_NE_KEY, RADIUS_OF_34KT_WIND_SE_KEY,
        RADIUS_OF_34KT_WIND_SW_KEY, RADIUS_OF_34KT_WIND_NW_KEY
    ]
    num_storm_objects = len(xbt_table_pandas.index)
    num_quadrants = len(var_name_by_quadrant)

    radius_of_34kt_wind_matrix_metres = numpy.full(
        (num_storm_objects, num_quadrants), numpy.nan
    )

    for j in range(num_quadrants):
        xbt_table_pandas = _remove_sentinel_values(
            xbt_table_pandas=xbt_table_pandas,
            variable_name=var_name_by_quadrant[j]
        )
        radius_of_34kt_wind_matrix_metres[:, j] = (
            NAUTICAL_MILES_TO_METRES *
            xbt_table_pandas[var_name_by_quadrant[j]].values
        )

    error_checking.assert_is_geq_numpy_array(
        radius_of_34kt_wind_matrix_metres, 0., allow_nan=True
    )

    # Convert radius of 50-kt wind to metres.
    var_name_by_quadrant = [
        RADIUS_OF_50KT_WIND_NE_KEY, RADIUS_OF_50KT_WIND_SE_KEY,
        RADIUS_OF_50KT_WIND_SW_KEY, RADIUS_OF_50KT_WIND_NW_KEY
    ]
    radius_of_50kt_wind_matrix_metres = numpy.full(
        (num_storm_objects, num_quadrants), numpy.nan
    )

    for j in range(num_quadrants):
        xbt_table_pandas = _remove_sentinel_values(
            xbt_table_pandas=xbt_table_pandas,
            variable_name=var_name_by_quadrant[j]
        )
        radius_of_50kt_wind_matrix_metres[:, j] = (
            NAUTICAL_MILES_TO_METRES *
            xbt_table_pandas[var_name_by_quadrant[j]].values
        )

    error_checking.assert_is_geq_numpy_array(
        radius_of_50kt_wind_matrix_metres, 0., allow_nan=True
    )

    # Convert radius of 64-kt wind to metres.
    var_name_by_quadrant = [
        RADIUS_OF_64KT_WIND_NE_KEY, RADIUS_OF_64KT_WIND_SE_KEY,
        RADIUS_OF_64KT_WIND_SW_KEY, RADIUS_OF_64KT_WIND_NW_KEY
    ]
    radius_of_64kt_wind_matrix_metres = numpy.full(
        (num_storm_objects, num_quadrants), numpy.nan
    )

    for j in range(num_quadrants):
        xbt_table_pandas = _remove_sentinel_values(
            xbt_table_pandas=xbt_table_pandas,
            variable_name=var_name_by_quadrant[j]
        )
        radius_of_64kt_wind_matrix_metres[:, j] = (
            NAUTICAL_MILES_TO_METRES *
            xbt_table_pandas[var_name_by_quadrant[j]].values
        )

    error_checking.assert_is_geq_numpy_array(
        radius_of_64kt_wind_matrix_metres, 0., allow_nan=True
    )

    bad_indices = numpy.where(
        radius_of_50kt_wind_matrix_metres > radius_of_34kt_wind_matrix_metres
    )
    radius_of_50kt_wind_matrix_metres[bad_indices] = (
        radius_of_34kt_wind_matrix_metres[bad_indices]
    )

    print((
        'Replaced {0:d} of {1:d} 50-kt wind radii because they were '
        'inconsistent with 34-kt wind radii.'
    ).format(
        len(bad_indices[0]), num_storm_objects * num_quadrants
    ))

    bad_indices = numpy.where(
        radius_of_64kt_wind_matrix_metres > radius_of_50kt_wind_matrix_metres
    )
    radius_of_64kt_wind_matrix_metres[bad_indices] = (
        radius_of_50kt_wind_matrix_metres[bad_indices]
    )

    print((
        'Replaced {0:d} of {1:d} 64-kt wind radii because they were '
        'inconsistent with 50-kt wind radii.'
    ).format(
        len(bad_indices[0]), num_storm_objects * num_quadrants
    ))

    these_dim = (xbt_utils.STORM_OBJECT_DIM, xbt_utils.QUADRANT_DIM)

    main_data_dict.update({
        RADII_OF_34KT_WIND_KEY: (these_dim, radius_of_34kt_wind_matrix_metres),
        RADII_OF_50KT_WIND_KEY: (these_dim, radius_of_50kt_wind_matrix_metres),
        RADII_OF_64KT_WIND_KEY: (these_dim, radius_of_64kt_wind_matrix_metres)
    })

    # Convert storm types to more descriptive strings.
    storm_type_strings = [
        STORM_TYPE_CONVERSION_DICT[t] for t in
        xbt_table_pandas[STORM_TYPE_KEY].values
    ]
    main_data_dict[STORM_TYPE_KEY] = (
        (xbt_utils.STORM_OBJECT_DIM,), storm_type_strings
    )

    # Convert distance to land to metres.
    xbt_table_pandas = _remove_sentinel_values(
        xbt_table_pandas=xbt_table_pandas, variable_name=DISTANCE_TO_LAND_KEY
    )
    distances_to_land_metres = (
        KM_TO_METRES * xbt_table_pandas[DISTANCE_TO_LAND_KEY].values
    )

    main_data_dict[DISTANCE_TO_LAND_KEY] = (
        (xbt_utils.STORM_OBJECT_DIM,), distances_to_land_metres
    )

    # Convert data sources to more descriptive strings.
    data_source_strings = [
        '{0:04d}'.format(s) for s in xbt_table_pandas[DATA_SOURCE_KEY].values
    ]

    max_wind_radius_source_strings = [
        DATA_SOURCE_CONVERSION_DICT[t[0]] for t in data_source_strings
    ]
    main_data_dict[MAX_WIND_RADIUS_SOURCE_KEY] = (
        (xbt_utils.STORM_OBJECT_DIM,), max_wind_radius_source_strings
    )

    eye_diameter_source_strings = [
        DATA_SOURCE_CONVERSION_DICT[t[1]] for t in data_source_strings
    ]
    main_data_dict[EYE_DIAMETER_SOURCE_KEY] = (
        (xbt_utils.STORM_OBJECT_DIM,), eye_diameter_source_strings
    )

    outermost_pressure_source_strings = [
        DATA_SOURCE_CONVERSION_DICT[t[2]] for t in data_source_strings
    ]
    main_data_dict[OUTERMOST_PRESSURE_SOURCE_KEY] = (
        (xbt_utils.STORM_OBJECT_DIM,), outermost_pressure_source_strings
    )

    outermost_radius_source_strings = [
        DATA_SOURCE_CONVERSION_DICT[t[3]] for t in data_source_strings
    ]
    main_data_dict[OUTERMOST_RADIUS_SOURCE_KEY] = (
        (xbt_utils.STORM_OBJECT_DIM,), outermost_radius_source_strings
    )

    metadata_dict = {
        xbt_utils.STORM_OBJECT_DIM: numpy.linspace(
            0, num_storm_objects - 1, num=num_storm_objects, dtype=int
        ),
        xbt_utils.QUADRANT_DIM: xbt_utils.QUADRANT_STRINGS
    }

    return xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )
