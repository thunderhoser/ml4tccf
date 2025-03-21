"""Input and output methods for raw ARCHER-2 files."""

import os
import sys
import requests
import numpy
import pandas
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import number_rounding
import time_conversion
import file_system_utils
import error_checking
import extended_best_track_io as ebtrk_io
import extended_best_track_utils as ebtrk_utils
import misc_utils

TOLERANCE = 1e-6
HOURS_TO_SECONDS = 3600

BASIN_ID_TO_ARCHER_FORMAT = {
    misc_utils.NORTH_ATLANTIC_ID_STRING: 'L',
    misc_utils.NORTHEAST_PACIFIC_ID_STRING: 'E',
    misc_utils.NORTHWEST_PACIFIC_ID_STRING: 'W'
}

BASE_URL_STRING = 'https://tropic.ssec.wisc.edu/real-time/archerOnline/cyclones'

COLUMN_SPECS = [
    (0, 18),
    (31, 40),
    (53, 62),
    (64, 72)
]

TIME_DIM = 'valid_time_unix_sec'
VALID_TIME_KEY = 'valid_time_unix_sec'
SENSOR_KEY = 'sensor_string'
LATITUDE_KEY = 'archer_latitude_deg_n'
LONGITUDE_KEY = 'archer_longitude_deg_e'
COLUMN_NAMES = [
    VALID_TIME_KEY, SENSOR_KEY, LATITUDE_KEY, LONGITUDE_KEY
]

EBTRK_LATITUDE_KEY = 'ebtrk_latitude_deg_n'
EBTRK_LONGITUDE_KEY = 'ebtrk_longitude_deg_e'

TIME_FORMAT = '%Y%m%d %H:%M:%S'


def find_url(cyclone_id_string):
    """Finds URL with raw ARCHER-2 data for a given cyclone.

    :param cyclone_id_string: Cyclone ID.
    :return: url_string: URL.
    """

    year, basin_id_string, cyclone_number = misc_utils.parse_cyclone_id(
        cyclone_id_string
    )
    archer_cyclone_id_string = '{0:04d}_{1:02d}{2:s}'.format(
        year, cyclone_number, BASIN_ID_TO_ARCHER_FORMAT[basin_id_string]
    )
    return '{0:s}/{1:s}/web/summaryTable.txt'.format(
        BASE_URL_STRING, archer_cyclone_id_string
    )


def download_file(url_string, local_file_name):
    """Downloads file with raw ARCHER-2 data to local machine.

    :param url_string: URL.
    :param local_file_name: Path to local file (will download here).
    """

    response_object = requests.get(url_string)
    response_object.raise_for_status()

    file_system_utils.mkdir_recursive_if_necessary(file_name=local_file_name)

    print('Downloading file from "{0:s}" to "{1:s}"...'.format(
        url_string, local_file_name
    ))
    with open(local_file_name, 'wb') as local_file_handle:
        local_file_handle.write(response_object.content)


def read_file(ascii_file_name, cyclone_id_string, ebtrk_file_name,
              time_tolerance_sec, match_to_synoptic_times_only):
    """Reads raw ARCHER-2 data from file.

    :param ascii_file_name: Path to ASCII file with raw ARCHER-2 data.
    :param cyclone_id_string: Cyclone ID.  This method assumes that all data in
        `ascii_file_name` are for the same cyclone.
    :param ebtrk_file_name: Path to file with extended best-track
        (EBTRK) data, which are needed to process the raw ARCHER-2 data.
    :param time_tolerance_sec: Time tolerance for matching an ARCHER-2 center
        fix with an EBTRK center fix.
    :param match_to_synoptic_times_only: Boolean flag.  If True, will match to
        EBTRK center fixes only at synoptic times.  If False, will match to any
        EBTRK center fix.
    :return: archer_table_xarray: xarray table.
    """

    misc_utils.parse_cyclone_id(cyclone_id_string)
    error_checking.assert_is_integer(time_tolerance_sec)
    error_checking.assert_is_greater(time_tolerance_sec, 0)
    error_checking.assert_is_boolean(match_to_synoptic_times_only)

    print('Reading data from: "{0:s}"...'.format(ascii_file_name))
    archer_table_pandas = pandas.read_fwf(
        ascii_file_name,
        colspecs=COLUMN_SPECS,
        names=COLUMN_NAMES,
        skiprows=1
    )
    atp = archer_table_pandas

    atp = atp[atp[LATITUDE_KEY] != '***']
    atp[LATITUDE_KEY] = pandas.to_numeric(atp[LATITUDE_KEY], errors='coerce')
    atp[LONGITUDE_KEY] = pandas.to_numeric(atp[LONGITUDE_KEY], errors='coerce')

    atp[SENSOR_KEY] = atp[SENSOR_KEY].str.strip()

    atp[VALID_TIME_KEY] = atp[VALID_TIME_KEY].str.strip()
    atp[VALID_TIME_KEY] = [t.replace('*', '') for t in atp[VALID_TIME_KEY]]
    atp[VALID_TIME_KEY] = atp[VALID_TIME_KEY].str.strip()

    atp[VALID_TIME_KEY] = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in atp[VALID_TIME_KEY]
    ], dtype=int)

    atp = atp.dropna()

    archer_table_xarray = xarray.Dataset(
        data_vars={
            SENSOR_KEY: (TIME_DIM, atp[SENSOR_KEY].values),
            LATITUDE_KEY: (TIME_DIM, atp[LATITUDE_KEY].values),
            LONGITUDE_KEY: (TIME_DIM, atp[LONGITUDE_KEY].values)
        },
        coords={
            TIME_DIM: atp[VALID_TIME_KEY].values
        }
    )

    print('Reading data from: "{0:s}"...'.format(ebtrk_file_name))
    ebtrk_table_xarray = ebtrk_io.read_file(ebtrk_file_name)

    if match_to_synoptic_times_only:
        ebtrk_times_unix_hours = (
            ebtrk_table_xarray[ebtrk_utils.VALID_TIME_KEY].values
        )
        rounded_ebtrk_times_unix_hours = number_rounding.round_to_nearest(
            ebtrk_times_unix_hours, 6.
        )
        rounding_diffs_hours = numpy.absolute(
            ebtrk_times_unix_hours - rounded_ebtrk_times_unix_hours
        )

        good_indices = numpy.where(rounding_diffs_hours <= TOLERANCE)[0]
        ebtrk_table_xarray = ebtrk_table_xarray.isel(
            {ebtrk_utils.STORM_OBJECT_DIM: good_indices}
        )

    ebtrk_cyclone_id_strings = (
        ebtrk_table_xarray[ebtrk_utils.STORM_ID_KEY].values
    )
    try:
        ebtrk_cyclone_id_strings = numpy.array([
            c.decode('utf-8') for c in ebtrk_cyclone_id_strings
        ])
    except:
        pass

    good_indices = numpy.where(ebtrk_cyclone_id_strings == cyclone_id_string)[0]
    ebtrk_table_xarray = ebtrk_table_xarray.isel(
        {ebtrk_utils.STORM_OBJECT_DIM: good_indices}
    )
    ebtrk_times_unix_sec = (
        HOURS_TO_SECONDS * ebtrk_table_xarray[ebtrk_utils.VALID_TIME_KEY].values
    ).astype(int)

    num_archer_examples = len(archer_table_xarray.coords[TIME_DIM].values)
    archer_to_ebtrk_indices = numpy.full(num_archer_examples, -1, dtype=int)

    for i in range(num_archer_examples):
        time_diffs_sec = numpy.absolute(
            archer_table_xarray.coords[TIME_DIM].values[i] -
            ebtrk_times_unix_sec
        )
        min_diff_index = numpy.argmin(time_diffs_sec)
        if time_diffs_sec[min_diff_index] > time_tolerance_sec:
            continue

        archer_to_ebtrk_indices[i] = min_diff_index + 0

    good_archer_indices = numpy.where(archer_to_ebtrk_indices > -1)[0]
    archer_table_xarray = archer_table_xarray.isel(
        {TIME_DIM: good_archer_indices}
    )

    print((
        'With time tolerance of {0:d} seconds, managed to assign '
        '{1:d} of {2:d} ARCHER-2 center fixes to an EBTRK center fix.'
    ).format(
        time_tolerance_sec,
        len(good_archer_indices),
        len(archer_to_ebtrk_indices)
    ))

    ebtrk_indices = archer_to_ebtrk_indices[good_archer_indices]
    ebtrk_latitudes_deg_n = ebtrk_table_xarray[
        ebtrk_utils.CENTER_LATITUDE_KEY
    ].values[ebtrk_indices]
    ebtrk_longitudes_deg_e = ebtrk_table_xarray[
        ebtrk_utils.CENTER_LONGITUDE_KEY
    ].values[ebtrk_indices]

    return archer_table_xarray.assign({
        EBTRK_LATITUDE_KEY: (
            (TIME_DIM,), ebtrk_latitudes_deg_n
        ),
        EBTRK_LONGITUDE_KEY: (
            (TIME_DIM,), ebtrk_longitudes_deg_e
        )
    })


def subset_by_sensor_type(archer_table_xarray, get_microwave=False,
                          get_scatterometer=False, get_infrared=False):
    """Subsets ARCHER-2 data by sensor type.

    :param archer_table_xarray: xarray table in format returned by `read_file`.
    :param get_microwave: Boolean flag.  If True, will return only center fixes
        based on microwave data.
    :param get_scatterometer: Boolean flag.  If True, will return only center
        fixes based on scatterometer data.
    :param get_infrared: Boolean flag.  If True, will return only center fixes
        based on infrared data.
    :return: archer_table_xarray: Same as input but maybe with fewer examples.
    """

    error_checking.assert_is_boolean(get_microwave)
    error_checking.assert_is_boolean(get_scatterometer)
    error_checking.assert_is_boolean(get_infrared)
    error_checking.assert_equals(
        int(get_microwave) + int(get_scatterometer) + int(get_infrared),
        1
    )

    if get_microwave:
        good_indices = numpy.where(numpy.logical_or(
            archer_table_xarray[SENSOR_KEY].values == '37GHz',
            archer_table_xarray[SENSOR_KEY].values == '85-92GHz'
        ))[0]
    elif get_scatterometer:
        good_indices = numpy.where(
            archer_table_xarray[SENSOR_KEY].values == 'ASCAT'
        )[0]
    else:
        good_indices = numpy.where(numpy.logical_or(
            archer_table_xarray[SENSOR_KEY].values == 'IR',
            archer_table_xarray[SENSOR_KEY].values == 'NearIR'
        ))[0]

    return archer_table_xarray.isel({TIME_DIM: good_indices})
