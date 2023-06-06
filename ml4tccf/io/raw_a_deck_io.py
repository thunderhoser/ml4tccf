"""IO methods for raw A-deck files from the ATCF.

ATCF = Automated Tropical-cyclone-forecasting System
"""

import os
import glob
import gzip
import shutil
import numpy
import xarray
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from ml4tccf.io import a_deck_io
from ml4tccf.utils import misc_utils
from atcf import ABRead

# TODO(thunderhoser): These files are garbage and need way more quality control.

# TODO(thunderhoser): Not putting this file in the stand-alone repository,
# since it depends on Chris Slocum's ATCF library.

TOLERANCE = 1e-6
GZIP_FILE_EXTENSION = '.gz'
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

NO_RADIUS_STRINGS = ['NAN', '']
CIRCLE_STRING = 'AAA'
QUADRANTS_STRING = 'NEQ'
VALID_RADIUS_TYPE_STRINGS = (
    NO_RADIUS_STRINGS + [CIRCLE_STRING, QUADRANTS_STRING]
)

MAX_NUM_WIND_THRESHOLDS = 3
MAX_NUM_WAVE_HEIGHT_THRESHOLDS = 10

KT_TO_METRES_PER_SECOND = 1.852 / 3.6
NAUTICAL_MILES_TO_METRES = 1852.
FEET_TO_METRES = 1. / 3.2808
MB_TO_PASCALS = 100.

VALID_TIME_KEY = 'DTG'
TECHNIQUE_KEY = 'TECH'
LEAD_TIME_HOURS_KEY = 'TAU'
LATITUDE_KEY = 'LAT'
LONGITUDE_KEY = 'LON'
INTENSITY_KEY = 'VMAX'
SEA_LEVEL_PRESSURE_KEY = 'MSLP'
STORM_TYPE_KEY = 'TY'
LAST_ISOBAR_PRESSURE_KEY = 'POUTER'
LAST_ISOBAR_RADIUS_KEY = 'ROUTER'
MAX_WIND_RADIUS_KEY = 'RWM'
GUST_SPEED_KEY = 'GUSTS'
EYE_DIAMETER_KEY = 'EYE'
MAX_SEA_HEIGHT_KEY = 'MAXSEAS'
MOTION_HEADING_KEY = 'DIR'
MOTION_SPEED_KEY = 'SPEED'
SYSTEM_DEPTH_KEY = 'DEPTH'
WIND_THRESHOLD_KEY = 'RAD'
WIND_RADIUS_TYPE_KEY = 'WINDCODE'
WIND_RADIUS_CIRCULAR_KEY = 'RAD1'
WIND_RADIUS_NE_QUADRANT_KEY = 'RAD1'
WIND_RADIUS_SE_QUADRANT_KEY = 'RAD2'
WIND_RADIUS_SW_QUADRANT_KEY = 'RAD3'
WIND_RADIUS_NW_QUADRANT_KEY = 'RAD4'
WAVE_HEIGHT_THRESHOLD_KEY = 'SEAS'
WAVE_HEIGHT_RADIUS_TYPE_KEY = 'SEACODE'
WAVE_HEIGHT_RADIUS_CIRCULAR_KEY = 'SEAS1'
WAVE_HEIGHT_RADIUS_NE_QUADRANT_KEY = 'SEAS1'
WAVE_HEIGHT_RADIUS_SE_QUADRANT_KEY = 'SEAS2'
WAVE_HEIGHT_RADIUS_SW_QUADRANT_KEY = 'SEAS3'
WAVE_HEIGHT_RADIUS_NW_QUADRANT_KEY = 'SEAS4'

FIELD_RENAMING_DICT = {
    LATITUDE_KEY: a_deck_io.LATITUDE_KEY,
    LONGITUDE_KEY: a_deck_io.LONGITUDE_KEY,
    INTENSITY_KEY: a_deck_io.INTENSITY_KEY,
    SEA_LEVEL_PRESSURE_KEY: a_deck_io.SEA_LEVEL_PRESSURE_KEY,
    LAST_ISOBAR_PRESSURE_KEY: a_deck_io.LAST_ISOBAR_PRESSURE_KEY,
    LAST_ISOBAR_RADIUS_KEY: a_deck_io.LAST_ISOBAR_RADIUS_KEY,
    MAX_WIND_RADIUS_KEY: a_deck_io.MAX_WIND_RADIUS_KEY,
    GUST_SPEED_KEY: a_deck_io.GUST_SPEED_KEY,
    EYE_DIAMETER_KEY: a_deck_io.EYE_DIAMETER_KEY,
    MAX_SEA_HEIGHT_KEY: a_deck_io.MAX_SEA_HEIGHT_KEY,
    MOTION_HEADING_KEY: a_deck_io.MOTION_HEADING_KEY,
    MOTION_SPEED_KEY: a_deck_io.MOTION_SPEED_KEY
}

RAW_FIELD_TO_CONV_FACTOR = {
    LATITUDE_KEY: 1.,
    LONGITUDE_KEY: 1.,
    INTENSITY_KEY: KT_TO_METRES_PER_SECOND,
    SEA_LEVEL_PRESSURE_KEY: MB_TO_PASCALS,
    LAST_ISOBAR_PRESSURE_KEY: MB_TO_PASCALS,
    LAST_ISOBAR_RADIUS_KEY: NAUTICAL_MILES_TO_METRES,
    MAX_WIND_RADIUS_KEY: NAUTICAL_MILES_TO_METRES,
    GUST_SPEED_KEY: KT_TO_METRES_PER_SECOND,
    EYE_DIAMETER_KEY: NAUTICAL_MILES_TO_METRES,
    MAX_SEA_HEIGHT_KEY: FEET_TO_METRES,
    MOTION_HEADING_KEY: 1.,
    MOTION_SPEED_KEY: KT_TO_METRES_PER_SECOND
}

STORM_TYPE_RENAMING_DICT = {
    'DB': a_deck_io.TROPICAL_DISTURBANCE_TYPE_STRING,
    'TD': a_deck_io.TROPICAL_DEPRESSION_TYPE_STRING,
    'TS': a_deck_io.TROPICAL_STORM_TYPE_STRING,
    'TY': a_deck_io.TROPICAL_TYPHOON_TYPE_STRING,
    'ST': a_deck_io.TROPICAL_SUPER_TYPHOON_TYPE_STRING,
    'TC': a_deck_io.TROPICAL_CYCLONE_TYPE_STRING,
    'HU': a_deck_io.TROPICAL_HURRICANE_TYPE_STRING,
    'SD': a_deck_io.SUBTROPICAL_DEPRESSION_TYPE_STRING,
    'SS': a_deck_io.SUBTROPICAL_STORM_TYPE_STRING,
    'EX': a_deck_io.EXTRATROPICAL_TYPE_STRING,
    'PT': a_deck_io.POSTTROPICAL_TYPE_STRING,
    'IN': a_deck_io.INLAND_TYPE_STRING,
    'DS': a_deck_io.DISSIPATING_TYPE_STRING,
    'LO': a_deck_io.LOW_TYPE_STRING,
    'WV': a_deck_io.WAVE_TYPE_STRING,
    'ET': a_deck_io.EXTRAPOLATED_TYPE_STRING,
    'MD': a_deck_io.MONSOON_DEPRESSION_TYPE_STRING,
    'XX': a_deck_io.UNKNOWN_TYPE_STRING
}

WIND_THRESHOLDS_KEY = 'wind_thresholds_kt'
WIND_RADII_CIRCULAR_KEY = 'wind_radii_circular_nm'
WIND_RADII_NE_QUADRANT_KEY = 'wind_radii_ne_quadrant_nm'
WIND_RADII_SE_QUADRANT_KEY = 'wind_radii_se_quadrant_nm'
WIND_RADII_SW_QUADRANT_KEY = 'wind_radii_sw_quadrant_nm'
WIND_RADII_NW_QUADRANT_KEY = 'wind_radii_nw_quadrant_nm'

WAVE_HEIGHT_THRESHOLDS_KEY = 'height_thresholds_feet'
WAVE_HEIGHT_RADII_CIRCULAR_KEY = 'height_radii_circular_nm'
WAVE_HEIGHT_RADII_NE_QUADRANT_KEY = 'height_radii_ne_quadrant_nm'
WAVE_HEIGHT_RADII_SE_QUADRANT_KEY = 'height_radii_se_quadrant_nm'
WAVE_HEIGHT_RADII_SW_QUADRANT_KEY = 'height_radii_sw_quadrant_nm'
WAVE_HEIGHT_RADII_NW_QUADRANT_KEY = 'height_radii_nw_quadrant_nm'


def _convert_to_numpy_floats(input_array):
    """Converts input array to numpy array of floats.

    :param input_array: Input array.
    :return: output_array: numpy array of floats.
    """

    try:
        _ = 0.1 * input_array
        return input_array.astype(float)
    except:
        return numpy.array([
            numpy.nan if s is None or s.strip() == ''
            else float(s)
            for s in input_array
        ])


def _convert_to_float(input_value):
    """Converts input value to float.

    :param input_value: Input value.
    :return: output_value: Float value.
    """

    try:
        _ = 0.1 * input_value
        return float(input_value)
    except:
        return (
            numpy.nan if input_value is None or input_value.strip() == ''
            else float(input_value)
        )


def _convert_to_string(input_value):
    """Converts input value to string.

    :param input_value: Input value.
    :return: output_value: Float value.
    """

    try:
        return str(input_value).strip().upper()
    except:
        return ''


def _convert_to_numpy_strings(input_array):
    """Converts input array to numpy array of strings.

    :param input_array: Input array.
    :return: output_array: numpy array of strings.
    """

    return numpy.array([_convert_to_string(s) for s in input_array])


def _read_wind_radii(atcf_table_pandas, row_indices):
    """Reads wind radii for one valid time.

    T = max number of possible wind thresholds

    :param atcf_table_pandas: pandas table returned by the method `ABRead`.
    :param row_indices: 1-D numpy array of row indices for a single valid time.
    :return: wind_radius_dict: Dictionary with the following keys.  Many values
        in the numpy arrays will likely be NaN.
    wind_radius_dict['wind_thresholds_kt']: length-T numpy array of wind
        thresholds (knots).
    wind_radius_dict['wind_radii_circular_nm']: length-T numpy array of circular
        wind radii (nautical miles).
    wind_radius_dict['wind_radii_ne_quadrant_nm']: length-T numpy array of wind
        radii for northeast quadrant (nautical miles).
    wind_radius_dict['wind_radii_se_quadrant_nm']: Same but for southeast
        quadrant.
    wind_radius_dict['wind_radii_sw_quadrant_nm']: Same but for southwest
        quadrant.
    wind_radius_dict['wind_radii_nw_quadrant_nm']: Same but for northwest
        quadrant.

    :raises: ValueError: if radius type is not recognized.
    """

    wind_thresholds_kt = _convert_to_numpy_floats(
        atcf_table_pandas[WIND_THRESHOLD_KEY].values[row_indices]
    )
    radius_type_strings = _convert_to_numpy_strings(
        atcf_table_pandas[WIND_RADIUS_TYPE_KEY].values[row_indices]
    )
    threshold_and_type_strings = numpy.array([
        '{0:s}_{1:.6f}'.format(a, b)
        for a, b in zip(radius_type_strings, wind_thresholds_kt)
    ])

    unique_indices = numpy.unique(
        threshold_and_type_strings, return_index=True
    )[1]

    unique_indices = numpy.array([
        k for k in unique_indices if wind_thresholds_kt[k] > TOLERANCE
    ], dtype=int)

    row_indices = row_indices[unique_indices]

    wind_thresholds_kt = numpy.full(MAX_NUM_WIND_THRESHOLDS, numpy.nan)
    wind_radii_circular_nm = numpy.full(MAX_NUM_WIND_THRESHOLDS, numpy.nan)
    wind_radii_ne_quadrant_nm = numpy.full(MAX_NUM_WIND_THRESHOLDS, numpy.nan)
    wind_radii_se_quadrant_nm = numpy.full(MAX_NUM_WIND_THRESHOLDS, numpy.nan)
    wind_radii_sw_quadrant_nm = numpy.full(MAX_NUM_WIND_THRESHOLDS, numpy.nan)
    wind_radii_nw_quadrant_nm = numpy.full(MAX_NUM_WIND_THRESHOLDS, numpy.nan)

    for i, r in enumerate(row_indices):
        wind_thresholds_kt[i] = _convert_to_float(
            atcf_table_pandas[WIND_THRESHOLD_KEY].values[r]
        )
        this_radius_type_string = _convert_to_string(
            atcf_table_pandas[WIND_RADIUS_TYPE_KEY].values[r]
        )

        if this_radius_type_string not in VALID_RADIUS_TYPE_STRINGS:
            error_string = (
                'Radius type ("{0:s}") is not in the following list:\n{1:s}'
            ).format(
                this_radius_type_string, str(VALID_RADIUS_TYPE_STRINGS)
            )

            raise ValueError(error_string)

        if this_radius_type_string in NO_RADIUS_STRINGS:
            continue

        if this_radius_type_string == CIRCLE_STRING:
            wind_radii_circular_nm[i] = _convert_to_float(
                atcf_table_pandas[WIND_RADIUS_CIRCULAR_KEY].values[r]
            )
            continue

        if this_radius_type_string == QUADRANTS_STRING:
            wind_radii_ne_quadrant_nm[i] = _convert_to_float(
                atcf_table_pandas[WIND_RADIUS_NE_QUADRANT_KEY].values[r]
            )
            wind_radii_se_quadrant_nm[i] = _convert_to_float(
                atcf_table_pandas[WIND_RADIUS_SE_QUADRANT_KEY].values[r]
            )
            wind_radii_sw_quadrant_nm[i] = _convert_to_float(
                atcf_table_pandas[WIND_RADIUS_SW_QUADRANT_KEY].values[r]
            )
            wind_radii_nw_quadrant_nm[i] = _convert_to_float(
                atcf_table_pandas[WIND_RADIUS_NW_QUADRANT_KEY].values[r]
            )

    return {
        WIND_THRESHOLDS_KEY: wind_thresholds_kt,
        WIND_RADII_CIRCULAR_KEY: wind_radii_circular_nm,
        WIND_RADII_NE_QUADRANT_KEY: wind_radii_ne_quadrant_nm,
        WIND_RADII_SE_QUADRANT_KEY: wind_radii_se_quadrant_nm,
        WIND_RADII_SW_QUADRANT_KEY: wind_radii_sw_quadrant_nm,
        WIND_RADII_NW_QUADRANT_KEY: wind_radii_nw_quadrant_nm
    }


def _read_wave_height_radii(atcf_table_pandas, row_indices):
    """Reads wave-height radii for one valid time.

    T = max number of possible wave-height thresholds

    :param atcf_table_pandas: Object (kind of like a pandas table) returned by
        the method `ABRead`.
    :param row_indices: 1-D numpy array of row indices for a single valid time.
    :return: wave_height_radius_dict: Dictionary with the following keys.  Many
        values in the numpy arrays will likely be NaN.
    wave_height_radius_dict['height_thresholds_feet']: length-T numpy array
        of height thresholds.
    wave_height_radius_dict['height_radii_circular_nm']: length-T numpy array of
        circular height radii (nautical miles).
    wave_height_radius_dict['height_radii_ne_quadrant_nm']: length-T numpy array
        of height radii for northeast quadrant (nautical miles).
    wave_height_radius_dict['height_radii_se_quadrant_nm']: Same but for
        southeast quadrant.
    wave_height_radius_dict['height_radii_sw_quadrant_nm']: Same but for
        southwest quadrant.
    wave_height_radius_dict['height_radii_nw_quadrant_nm']: Same but for
        northwest quadrant.

    :raises: ValueError: if radius type is not recognized.
    """

    height_thresholds_feet = _convert_to_numpy_floats(
        atcf_table_pandas[WAVE_HEIGHT_THRESHOLD_KEY].values[row_indices]
    )
    radius_type_strings = _convert_to_numpy_strings(
        atcf_table_pandas[WAVE_HEIGHT_RADIUS_TYPE_KEY].values[row_indices]
    )
    threshold_and_type_strings = numpy.array([
        '{0:s}_{1:.6f}'.format(a, b)
        for a, b in zip(radius_type_strings, height_thresholds_feet)
    ])

    unique_indices = numpy.unique(
        threshold_and_type_strings, return_index=True
    )[1]

    unique_indices = numpy.array([
        k for k in unique_indices if height_thresholds_feet[k] > TOLERANCE
    ], dtype=int)

    row_indices = row_indices[unique_indices]

    height_thresholds_feet = numpy.full(
        MAX_NUM_WAVE_HEIGHT_THRESHOLDS, numpy.nan
    )
    height_radii_circular_nm = numpy.full(
        MAX_NUM_WAVE_HEIGHT_THRESHOLDS, numpy.nan
    )
    height_radii_ne_quadrant_nm = numpy.full(
        MAX_NUM_WAVE_HEIGHT_THRESHOLDS, numpy.nan
    )
    height_radii_se_quadrant_nm = numpy.full(
        MAX_NUM_WAVE_HEIGHT_THRESHOLDS, numpy.nan
    )
    height_radii_sw_quadrant_nm = numpy.full(
        MAX_NUM_WAVE_HEIGHT_THRESHOLDS, numpy.nan
    )
    height_radii_nw_quadrant_nm = numpy.full(
        MAX_NUM_WAVE_HEIGHT_THRESHOLDS, numpy.nan
    )

    for i, r in enumerate(row_indices):
        height_thresholds_feet[i] = _convert_to_float(
            atcf_table_pandas[WAVE_HEIGHT_THRESHOLD_KEY].values[r]
        )
        this_radius_type_string = _convert_to_string(
            atcf_table_pandas[WAVE_HEIGHT_RADIUS_TYPE_KEY].values[r]
        )

        if this_radius_type_string not in VALID_RADIUS_TYPE_STRINGS:
            error_string = (
                'Radius type ("{0:s}") is not in the following list:\n{1:s}'
            ).format(
                this_radius_type_string, str(VALID_RADIUS_TYPE_STRINGS)
            )

            raise ValueError(error_string)

        if this_radius_type_string in NO_RADIUS_STRINGS:
            continue

        if this_radius_type_string == CIRCLE_STRING:
            height_radii_circular_nm[i] = _convert_to_float(
                atcf_table_pandas[WAVE_HEIGHT_RADIUS_CIRCULAR_KEY].values[r]
            )
            continue

        if this_radius_type_string == QUADRANTS_STRING:
            height_radii_ne_quadrant_nm[i] = _convert_to_float(
                atcf_table_pandas[WAVE_HEIGHT_RADIUS_NE_QUADRANT_KEY].values[r]
            )
            height_radii_se_quadrant_nm[i] = _convert_to_float(
                atcf_table_pandas[WAVE_HEIGHT_RADIUS_SE_QUADRANT_KEY].values[r]
            )
            height_radii_sw_quadrant_nm[i] = _convert_to_float(
                atcf_table_pandas[WAVE_HEIGHT_RADIUS_SW_QUADRANT_KEY].values[r]
            )
            height_radii_nw_quadrant_nm[i] = _convert_to_float(
                atcf_table_pandas[WAVE_HEIGHT_RADIUS_NW_QUADRANT_KEY].values[r]
            )

    return {
        WAVE_HEIGHT_THRESHOLDS_KEY: height_thresholds_feet,
        WAVE_HEIGHT_RADII_CIRCULAR_KEY: height_radii_circular_nm,
        WAVE_HEIGHT_RADII_NE_QUADRANT_KEY: height_radii_ne_quadrant_nm,
        WAVE_HEIGHT_RADII_SE_QUADRANT_KEY: height_radii_se_quadrant_nm,
        WAVE_HEIGHT_RADII_SW_QUADRANT_KEY: height_radii_sw_quadrant_nm,
        WAVE_HEIGHT_RADII_NW_QUADRANT_KEY: height_radii_nw_quadrant_nm
    }


def find_file(directory_name, cyclone_id_string, raise_error_if_missing=True):
    """Finds ASCII file with A-deck data.

    :param directory_name: Name of directory with A-deck data.
    :param cyclone_id_string: Cyclone ID (must be accepted by
        `misc_utils.parse_cyclone_id`).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: a_deck_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    misc_utils.parse_cyclone_id(cyclone_id_string)
    error_checking.assert_is_boolean(raise_error_if_missing)

    extensionless_file_name = '{0:s}/a{1:s}{2:s}'.format(
        directory_name, cyclone_id_string[4:].lower(), cyclone_id_string[:4]
    )
    a_deck_file_name = '{0:s}.dat'.format(extensionless_file_name)
    if os.path.isfile(a_deck_file_name):
        return a_deck_file_name

    a_deck_file_name = '{0:s}.dat.gz'.format(extensionless_file_name)
    if os.path.isfile(a_deck_file_name):
        return a_deck_file_name

    a_deck_file_name = '{0:s}.txt'.format(extensionless_file_name)
    if os.path.isfile(a_deck_file_name):
        return a_deck_file_name

    a_deck_file_name = '{0:s}.dat'.format(extensionless_file_name)
    if not raise_error_if_missing:
        return a_deck_file_name

    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        a_deck_file_name
    )
    raise ValueError(error_string)


def file_name_to_cyclone_id(a_deck_file_name):
    """Parses cyclone ID from file.

    :param a_deck_file_name: Path to raw A-deck file.
    :return: cyclone_id_string: Cyclone ID.
    """

    pathless_file_name = os.path.split(a_deck_file_name)[1]
    extensionless_file_name = pathless_file_name.split('.')[0]

    return misc_utils.get_cyclone_id(
        year=int(extensionless_file_name[-4:]),
        basin_id_string=extensionless_file_name[-8:-6].upper(),
        cyclone_number=int(extensionless_file_name[-6:-4])
    )


def find_cyclones_one_year(directory_name, year,
                           raise_error_if_all_missing=True):
    """Finds all cyclones in one year.

    :param directory_name: Name of directory with A-deck data.
    :param year: Year (integer).
    :param raise_error_if_all_missing: Boolean flag.  If no cyclones are found
        and `raise_error_if_all_missing == True`, will throw error.  If no
        cyclones are found and `raise_error_if_all_missing == False`, will
        return empty list.
    :return: cyclone_id_strings: List of cyclone IDs.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_integer(year)
    error_checking.assert_is_boolean(raise_error_if_all_missing)

    extensionless_file_pattern = '{0:s}/a[a-z][a-z][0-9][0-9]{1:04d}'.format(
        directory_name, year
    )
    a_deck_file_names = glob.glob(
        '{0:s}.dat'.format(extensionless_file_pattern)
    )
    a_deck_file_names += glob.glob(
        '{0:s}.dat.gz'.format(extensionless_file_pattern)
    )
    a_deck_file_names += glob.glob(
        '{0:s}.txt'.format(extensionless_file_pattern)
    )

    cyclone_id_strings = [file_name_to_cyclone_id(f) for f in a_deck_file_names]
    cyclone_id_strings.sort()

    if raise_error_if_all_missing and len(cyclone_id_strings) == 0:
        error_string = (
            'Could not find any cyclone IDs in directory: "{0:s}"'
        ).format(directory_name)

        raise ValueError(error_string)

    cyclone_id_strings = list(set(cyclone_id_strings))
    cyclone_id_strings.sort()

    return cyclone_id_strings


def read_file(ascii_file_name):
    """Reads A-deck data from ASCII file.

    :param ascii_file_name: Path to input file.
    :return: a_deck_table_xarray: xarray table.  Documentation in the xarray
        table should make values self-explanatory.
    """

    if ascii_file_name.endswith(GZIP_FILE_EXTENSION):
        temp_file_name = ascii_file_name[:-len(GZIP_FILE_EXTENSION)]

        with gzip.open(ascii_file_name, 'rb') as ascii_file_handle:
            with open(temp_file_name, 'wb') as temp_file_handle:
                shutil.copyfileobj(ascii_file_handle, temp_file_handle)

        atcf_table_pandas = ABRead(temp_file_name).deck
        os.remove(temp_file_name)
    else:
        atcf_table_pandas = ABRead(filename=ascii_file_name).deck

    # Remove non-CARQ forecasts.
    technique_strings = _convert_to_numpy_strings(
        atcf_table_pandas[TECHNIQUE_KEY].values
    )
    carq_flags = numpy.array(
        [t.upper() == 'CARQ' for t in technique_strings], dtype=bool
    )
    del technique_strings

    bad_indices = numpy.where(numpy.invert(carq_flags))[0]
    atcf_table_pandas.drop(
        atcf_table_pandas.index[bad_indices], axis=0, inplace=True
    )

    # Remove all lead times except 0 hours (analyses).
    lead_times_hours = _convert_to_numpy_floats(
        atcf_table_pandas[LEAD_TIME_HOURS_KEY].values
    )
    lead_times_hours = numpy.round(lead_times_hours).astype(int)
    bad_indices = numpy.where(lead_times_hours != 0)[0]
    del lead_times_hours

    atcf_table_pandas.drop(
        atcf_table_pandas.index[bad_indices], axis=0, inplace=True
    )

    # Find unique times.
    valid_times_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in atcf_table_pandas[VALID_TIME_KEY].values
    ], dtype=int)

    unique_times_unix_sec, unique_time_indices = numpy.unique(
        valid_times_unix_sec, return_index=True
    )

    technique_strings = _convert_to_numpy_strings(
        atcf_table_pandas[TECHNIQUE_KEY].values[unique_time_indices]
    )
    system_depth_strings = _convert_to_numpy_strings(
        atcf_table_pandas[SYSTEM_DEPTH_KEY].values[unique_time_indices]
    )
    storm_type_strings = _convert_to_numpy_strings(
        atcf_table_pandas[STORM_TYPE_KEY].values[unique_time_indices]
    )
    storm_type_strings = numpy.array([
        STORM_TYPE_RENAMING_DICT[s] for s in storm_type_strings
    ])

    # Process metadata.
    cyclone_id_string = file_name_to_cyclone_id(ascii_file_name)

    num_entries = len(unique_times_unix_sec)
    cyclone_id_strings = [cyclone_id_string] * num_entries
    storm_object_indices = numpy.linspace(
        0, num_entries - 1, num=num_entries, dtype=int
    )
    wind_threshold_indices = numpy.linspace(
        0, MAX_NUM_WIND_THRESHOLDS - 1, num=MAX_NUM_WIND_THRESHOLDS, dtype=int
    )
    wave_height_threshold_indices = numpy.linspace(
        0, MAX_NUM_WAVE_HEIGHT_THRESHOLDS - 1,
        num=MAX_NUM_WAVE_HEIGHT_THRESHOLDS, dtype=int
    )

    metadata_dict = {
        a_deck_io.STORM_OBJECT_DIM: storm_object_indices,
        a_deck_io.WIND_THRESHOLD_DIM: wind_threshold_indices,
        a_deck_io.WAVE_HEIGHT_THRESHOLD_DIM: wave_height_threshold_indices
    }

    # Process actual data.
    these_dim = (a_deck_io.STORM_OBJECT_DIM,)
    main_data_dict = {
        a_deck_io.CYCLONE_ID_KEY: (these_dim, cyclone_id_strings),
        a_deck_io.VALID_TIME_KEY: (these_dim, unique_times_unix_sec),
        a_deck_io.TECHNIQUE_KEY: (these_dim, technique_strings),
        a_deck_io.SYSTEM_DEPTH_KEY: (these_dim, system_depth_strings),
        a_deck_io.STORM_TYPE_KEY: (these_dim, storm_type_strings)
    }

    for raw_field_name in FIELD_RENAMING_DICT:
        processed_field_name = FIELD_RENAMING_DICT[raw_field_name]
        these_values = _convert_to_numpy_floats(
            atcf_table_pandas[raw_field_name].values[unique_time_indices]
        )

        if raw_field_name in RAW_FIELD_TO_CONV_FACTOR:
            these_values *= RAW_FIELD_TO_CONV_FACTOR[raw_field_name]

        main_data_dict[processed_field_name] = (these_dim, these_values)

    these_dim = (num_entries, MAX_NUM_WIND_THRESHOLDS)
    wind_threshold_matrix_m_s01 = numpy.full(these_dim, numpy.nan)
    wind_radius_matrix_circular_metres = numpy.full(these_dim, numpy.nan)
    wind_radius_matrix_ne_metres = numpy.full(these_dim, numpy.nan)
    wind_radius_matrix_se_metres = numpy.full(these_dim, numpy.nan)
    wind_radius_matrix_sw_metres = numpy.full(these_dim, numpy.nan)
    wind_radius_matrix_nw_metres = numpy.full(these_dim, numpy.nan)

    these_dim = (num_entries, MAX_NUM_WAVE_HEIGHT_THRESHOLDS)
    wave_height_threshold_matrix_metres = numpy.full(these_dim, numpy.nan)
    wave_height_radius_matrix_circular_metres = numpy.full(these_dim, numpy.nan)
    wave_height_radius_matrix_ne_metres = numpy.full(these_dim, numpy.nan)
    wave_height_radius_matrix_nw_metres = numpy.full(these_dim, numpy.nan)
    wave_height_radius_matrix_sw_metres = numpy.full(these_dim, numpy.nan)
    wave_height_radius_matrix_se_metres = numpy.full(these_dim, numpy.nan)

    for i in range(num_entries):
        row_indices = numpy.where(
            valid_times_unix_sec == unique_times_unix_sec[i]
        )[0]
        this_dict = _read_wind_radii(
            atcf_table_pandas=atcf_table_pandas, row_indices=row_indices
        )

        wind_threshold_matrix_m_s01[i, :] = this_dict[WIND_THRESHOLDS_KEY]
        wind_radius_matrix_circular_metres[i, :] = (
            this_dict[WIND_RADII_CIRCULAR_KEY]
        )
        wind_radius_matrix_ne_metres[i, :] = (
            this_dict[WIND_RADII_NE_QUADRANT_KEY]
        )
        wind_radius_matrix_se_metres[i, :] = (
            this_dict[WIND_RADII_SE_QUADRANT_KEY]
        )
        wind_radius_matrix_sw_metres[i, :] = (
            this_dict[WIND_RADII_SW_QUADRANT_KEY]
        )
        wind_radius_matrix_nw_metres[i, :] = (
            this_dict[WIND_RADII_NW_QUADRANT_KEY]
        )

        this_dict = _read_wave_height_radii(
            atcf_table_pandas=atcf_table_pandas, row_indices=row_indices
        )
        wave_height_threshold_matrix_metres[i, :] = (
            this_dict[WAVE_HEIGHT_THRESHOLDS_KEY]
        )
        wave_height_radius_matrix_circular_metres[i, :] = (
            this_dict[WAVE_HEIGHT_RADII_CIRCULAR_KEY]
        )
        wave_height_radius_matrix_ne_metres[i, :] = (
            this_dict[WAVE_HEIGHT_RADII_NE_QUADRANT_KEY]
        )
        wave_height_radius_matrix_se_metres[i, :] = (
            this_dict[WAVE_HEIGHT_RADII_SE_QUADRANT_KEY]
        )
        wave_height_radius_matrix_sw_metres[i, :] = (
            this_dict[WAVE_HEIGHT_RADII_SW_QUADRANT_KEY]
        )
        wave_height_radius_matrix_nw_metres[i, :] = (
            this_dict[WAVE_HEIGHT_RADII_NW_QUADRANT_KEY]
        )

    wind_threshold_matrix_m_s01 *= KT_TO_METRES_PER_SECOND
    wind_radius_matrix_circular_metres *= NAUTICAL_MILES_TO_METRES
    wind_radius_matrix_ne_metres *= NAUTICAL_MILES_TO_METRES
    wind_radius_matrix_nw_metres *= NAUTICAL_MILES_TO_METRES
    wind_radius_matrix_sw_metres *= NAUTICAL_MILES_TO_METRES
    wind_radius_matrix_se_metres *= NAUTICAL_MILES_TO_METRES

    wave_height_threshold_matrix_metres *= FEET_TO_METRES
    wave_height_radius_matrix_circular_metres *= NAUTICAL_MILES_TO_METRES
    wave_height_radius_matrix_ne_metres *= NAUTICAL_MILES_TO_METRES
    wave_height_radius_matrix_nw_metres *= NAUTICAL_MILES_TO_METRES
    wave_height_radius_matrix_sw_metres *= NAUTICAL_MILES_TO_METRES
    wave_height_radius_matrix_se_metres *= NAUTICAL_MILES_TO_METRES

    these_dim = (a_deck_io.STORM_OBJECT_DIM, a_deck_io.WIND_THRESHOLD_DIM)
    main_data_dict.update({
        a_deck_io.WIND_THRESHOLD_KEY: (these_dim, wind_threshold_matrix_m_s01),
        a_deck_io.WIND_RADIUS_CIRCULAR_KEY:
            (these_dim, wind_radius_matrix_circular_metres),
        a_deck_io.WIND_RADIUS_NE_QUADRANT_KEY:
            (these_dim, wind_radius_matrix_ne_metres),
        a_deck_io.WIND_RADIUS_SE_QUADRANT_KEY:
            (these_dim, wind_radius_matrix_se_metres),
        a_deck_io.WIND_RADIUS_SW_QUADRANT_KEY:
            (these_dim, wind_radius_matrix_sw_metres),
        a_deck_io.WIND_RADIUS_NW_QUADRANT_KEY:
            (these_dim, wind_radius_matrix_nw_metres)
    })

    these_dim = (
        a_deck_io.STORM_OBJECT_DIM, a_deck_io.WAVE_HEIGHT_THRESHOLD_DIM
    )
    main_data_dict.update({
        a_deck_io.WAVE_HEIGHT_THRESHOLD_KEY:
            (these_dim, wave_height_threshold_matrix_metres),
        a_deck_io.WAVE_HEIGHT_RADIUS_CIRCULAR_KEY:
            (these_dim, wave_height_radius_matrix_circular_metres),
        a_deck_io.WAVE_HEIGHT_RADIUS_NE_QUADRANT_KEY:
            (these_dim, wave_height_radius_matrix_ne_metres),
        a_deck_io.WAVE_HEIGHT_RADIUS_SE_QUADRANT_KEY:
            (these_dim, wave_height_radius_matrix_se_metres),
        a_deck_io.WAVE_HEIGHT_RADIUS_SW_QUADRANT_KEY:
            (these_dim, wave_height_radius_matrix_sw_metres),
        a_deck_io.WAVE_HEIGHT_RADIUS_NW_QUADRANT_KEY:
            (these_dim, wave_height_radius_matrix_nw_metres)
    })

    return xarray.Dataset(data_vars=main_data_dict, coords=metadata_dict)
