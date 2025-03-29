"""Merges ARCHER-2 data with best-track data."""

import os
import re
import csv
import glob
import pickle
import argparse
import warnings
import numpy
import pandas
import xarray
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4tccf.io import prediction_io as geocenter_io
from ml4tccf.utils import misc_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

ARCHER_GEOCENTER_TIME_TOLERANCE_SEC = 1800

ASSUMED_YEAR = 2024
TIME_FORMAT = '%Y%m%d%H%M%S'
GEOCENTER_DATE_FORMAT = '%Y%m%d'

BASIN_ID_KEY = 'basin_id_string'
CYCLONE_NUMBER_KEY = 'cyclone_number'
VALID_TIME_STRING_KEY = 'valid_time_string'
ARCHER_PLATFORM_KEY = 'archer_platform_string'
ARCHER_PRED_LATITUDE_KEY = 'archer_predicted_latitude_deg_n'
ARCHER_PRED_LONGITUDE_KEY = 'archer_predicted_longitude_deg_e'
ARCHER_TARGET_LATITUDE_KEY = 'archer_target_latitude_deg_n'
ARCHER_TARGET_LONGITUDE_KEY = 'archer_target_longitude_deg_e'
GEOCENTER_PRED_LATITUDE_KEY = 'geocenter_predicted_latitude_deg_n'
GEOCENTER_PRED_LONGITUDE_KEY = 'geocenter_predicted_longitude_deg_e'
GEOCENTER_TARGET_LATITUDE_KEY = 'geocenter_target_latitude_deg_n'
GEOCENTER_TARGET_LONGITUDE_KEY = 'geocenter_target_longitude_deg_e'
GEOCENTER_VALID_TIME_KEY = 'geocenter_valid_time_unix_sec'

ARCHER_COLUMN_NAMES = [
    BASIN_ID_KEY, CYCLONE_NUMBER_KEY, VALID_TIME_STRING_KEY,
    'modifier', ARCHER_PLATFORM_KEY, 'instrument', 'event',
    ARCHER_PRED_LATITUDE_KEY, ARCHER_PRED_LONGITUDE_KEY,
    'minimum_pressure', 'quality', 'vmax', 'number_2',
    'central_pressure', 'number_3', 'category', 'wind', 'quadrant', 'radii_1',
    'radii_2', 'radii_3', 'radii_4', 'c25___', 'c26___', 'c27___', 'c28___',
    'c29___', 'c30___', 'c31___', 'c32___', 'c33___', 'c34___', 'c35___', 'c36',
    'unknown', 'unknown', 'unknown', 'unknown', 'unknown'
]

STORM_OBJECT_DIM = 'storm_object'
CYCLONE_ID_KEY = 'cyclone_id_string'
VALID_TIME_KEY = 'valid_time_unix_sec'

ARCHER_FILE_PATTERN_ARG_NAME = 'input_raw_archer_file_pattern'
BEST_TRACK_DIR_ARG_NAME = 'input_raw_best_track_dir_name'
GEOCENTER_DIR_ARG_NAME = 'input_geocenter_dir_name'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

ARCHER_FILE_PATTERN_HELP_STRING = (
    'Glob pattern for raw ARCHER-2 files (F-decks).'
)
BEST_TRACK_DIR_HELP_STRING = (
    'Path to directory with raw best-track data.  Files therein will be found '
    'by `_find_best_track_file` and read by `_read_best_track_file`.'
)
GEOCENTER_DIR_HELP_STRING = (
    'Path to directory with GeoCenter predictions.  Files therein will be '
    'found by `_find_geocenter_file` and read by `_read_geocenter_file`.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file (NetCDF).  Merged data will be written here by this '
    'script.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + ARCHER_FILE_PATTERN_ARG_NAME, type=str, required=True,
    help=ARCHER_FILE_PATTERN_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BEST_TRACK_DIR_ARG_NAME, type=str, required=True,
    help=BEST_TRACK_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + GEOCENTER_DIR_ARG_NAME, type=str, required=True,
    help=GEOCENTER_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _parse_archer_latitude(latitude_string):
    """Parses latitude from F-deck file with ARCHER-2 data.

    :param latitude_string: String formatted like "1559N" (for 15.59 deg north)
        or "2967S" (for 29.67 deg south).
    :return: latitude_deg_n: Latitude in actual deg north (a scalar float).
    """

    if pandas.isna(latitude_string):
        return numpy.nan

    match_object = re.match(r'(\d+)([NS])', latitude_string)
    if not match_object:
        return numpy.nan

    value, direction = match_object.groups()
    latitude_deg_n = 0.01 * float(value)
    if direction == 'S':
        latitude_deg_n *= -1

    return latitude_deg_n


def _parse_archer_longitude(longitude_string):
    """Parses longitude from F-deck file with ARCHER-2 data.

    :param longitude_string: String formatted like "1559E" (for 15.59 deg east)
        or "13967W" (for 139.67 deg west).
    :return: longitude_deg_e: Longitude in actual deg east (a scalar float).
    """

    if pandas.isna(longitude_string):
        return numpy.nan

    match_object = re.match(r'(\d+)([EW])', longitude_string)
    if not match_object:
        return numpy.nan

    value, direction = match_object.groups()
    longitude_deg_e = 0.01 * float(value)
    if direction == 'W':
        longitude_deg_e *= -1

    return longitude_deg_e % 360


def _find_best_track_file(directory_name, cyclone_id_string):
    """Finds Pickle file containing best-track data.

    :param directory_name: Path to directory.
    :param cyclone_id_string: Cyclone ID.
    :return: pickle_file_name: Path to best-track file.
    """

    fake_cyclone_id_string = '{0:s}{1:s}'.format(
        cyclone_id_string[4:].lower(),
        cyclone_id_string[:4]
    )

    pickle_file_pattern = '{0:s}/b{1:s}/storm_track_interp_b{1:s}.pkl'.format(
        directory_name, fake_cyclone_id_string
    )
    pickle_file_names = glob.glob(pickle_file_pattern)

    if len(pickle_file_names) != 1:
        warning_string = (
            'POTENTIAL ERROR: Cannot find file with pattern: "{0:s}"'
        ).format(pickle_file_pattern)

        warnings.warn(warning_string)
        return None

    return pickle_file_names[0]


def _read_best_track_file(pickle_file_name, valid_time_unix_sec,
                          best_track_dict=None):
    """Reads TC-center location from best-track file.

    :param pickle_file_name: Path to input file.
    :param valid_time_unix_sec: Desired time step.
    :param best_track_dict: Dictionary contained in file.  If this is None, the
        file will be read from scratch.
    :return: best_track_dict: Dictionary contained in file.
    :return: latitude_deg_n: Latitude of TC center.
    :return: longitude_deg_e: Longitude of TC center.
    """

    if best_track_dict is None:
        pickle_file_handle = open(pickle_file_name, 'rb')
        best_track_dict = pickle.load(pickle_file_handle)
        pickle_file_handle.close()

        valid_time_strings = [
            t.strftime('%Y-%m-%d-%H%M%S')
            for t in best_track_dict['st_one_sec_time']
        ]
        best_track_dict['st_one_sec_time'] = numpy.array([
            time_conversion.string_to_unix_sec(t, '%Y-%m-%d-%H%M%S')
            for t in valid_time_strings
        ], dtype=int)

    good_indices = numpy.where(
        best_track_dict['st_one_sec_time'] == valid_time_unix_sec
    )[0]

    if len(good_indices) == 0:
        warning_string = (
            'POTENTIAL ERROR: Cannot find valid time {0:s} in file "{1:s}".'
        ).format(
            time_conversion.unix_sec_to_string(
                valid_time_unix_sec, '%Y-%m-%d-%H%M%S'
            ),
            pickle_file_name
        )

        warnings.warn(warning_string)
        return best_track_dict, numpy.nan, numpy.nan

    if len(good_indices) > 1:
        warning_string = (
            'POTENTIAL ERROR: Found {0:d} entries with valid time {1:s} in '
            'file "{2:s}".'
        ).format(
            len(good_indices),
            time_conversion.unix_sec_to_string(
                valid_time_unix_sec, '%Y-%m-%d-%H%M%S'
            ),
            pickle_file_name
        )

        warnings.warn(warning_string)

        for k in good_indices:
            print('{0:.4f} deg N, {1:.4f} deg E'.format(
                best_track_dict['st_one_sec_lats'][k],
                best_track_dict['st_one_sec_lond'][k]
            ))
    else:
        info_string = (
            'Found a single entry with valid time {0:s} in file "{1:s}".  Yay!'
        ).format(
            time_conversion.unix_sec_to_string(
                valid_time_unix_sec, '%Y-%m-%d-%H%M%S'
            ),
            pickle_file_name
        )

        print(info_string)

    good_index = good_indices[0]

    latitude_deg_n = best_track_dict['st_one_sec_lats'][good_index]
    error_checking.assert_is_valid_latitude(latitude_deg_n, allow_nan=False)

    longitude_deg_e = best_track_dict['st_one_sec_lond'][good_index]
    longitude_deg_e = lng_conversion.convert_lng_positive_in_west(
        longitude_deg_e, allow_nan=False
    )

    return best_track_dict, latitude_deg_n, longitude_deg_e


def _find_geocenter_file(directory_name, cyclone_id_string, valid_time_unix_sec,
                         raise_error_if_missing=True):
    """Finds file with GeoCenter predictions.

    :param directory_name: Path to directory.
    :param cyclone_id_string: Cyclone ID.
    :param valid_time_unix_sec: Valid time.
    :param raise_error_if_missing: Boolean flag.  If file is not found and
        `raise_error_if_missing == True`, this method will raise an error.  If
        file is not found and `raise_error_if_missing == False`, this method
        will return the expected file path.
    :return: prediction_file_name: Actual or expected file path.
    :raises: ValueError: if file is not found and
        `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    _ = misc_utils.parse_cyclone_id(cyclone_id_string)
    error_checking.assert_is_integer(valid_time_unix_sec)
    error_checking.assert_is_boolean(raise_error_if_missing)

    fake_cyclone_id_string = '{0:s}{1:s}'.format(
        cyclone_id_string[4:].lower(), cyclone_id_string[:4]
    )

    prediction_file_name = '{0:s}/{1:s}/{1:s}_{2:s}.txt'.format(
        directory_name,
        fake_cyclone_id_string,
        time_conversion.unix_sec_to_string(valid_time_unix_sec, '%Y%m%d_%H%M')
    )

    if os.path.isfile(prediction_file_name):
        return prediction_file_name

    if not raise_error_if_missing:
        return prediction_file_name

    error_string = 'Cannot find prediction file.  Expected at: "{0:s}"'.format(
        prediction_file_name
    )
    raise ValueError(error_string)


def _read_geocenter_file(ascii_file_name):
    """Reads prediction from GeoCenter file.

    :param ascii_file_name: Path to input file.
    :return: latitude_deg_n: Latitude (deg north).
    :return: longitude_deg_e: Longitude (deg east).
    :return: valid_time_unix_sec: Valid time.
    """

    with open(ascii_file_name, 'r') as file_handle:
        csv_reader_object = csv.reader(file_handle)
        these_words = next(csv_reader_object)
        latitude_deg_n = float(these_words[-2])
        longitude_deg_e = float(these_words[-1])

        this_time_string = '{0:s}{1:s}'.format(
            these_words[-4].strip(), these_words[-3].strip()
        )
        valid_time_unix_sec = time_conversion.string_to_unix_sec(
            this_time_string, '%Y%m%d%H%M'
        )

    error_checking.assert_is_valid_latitude(latitude_deg_n, allow_nan=False)
    longitude_deg_e = lng_conversion.convert_lng_positive_in_west(
        longitude_deg_e, allow_nan=False
    )

    return latitude_deg_n, longitude_deg_e, valid_time_unix_sec


def _run(archer_file_pattern, raw_best_track_dir_name, geocenter_dir_name,
         output_file_name):
    """Merges ARCHER-2 data with best-track data.

    This is effectively the main method.

    :param archer_file_pattern: See documentation at top of this script.
    :param raw_best_track_dir_name: Same.
    :param geocenter_dir_name: Same.
    :param output_file_name: Same.
    """

    archer_file_names = glob.glob(archer_file_pattern)
    archer_file_names.sort()

    num_archer_files = len(archer_file_names)
    archer_tables_pandas = [None] * num_archer_files

    for i in range(num_archer_files):
        print('Reading data from: "{0:s}"...'.format(archer_file_names[i]))
        archer_tables_pandas[i] = pandas.read_csv(
            archer_file_names[i], skipinitialspace=True, header=None
        )

    archer_table_pandas = pandas.concat(archer_tables_pandas, axis=0)
    archer_table_pandas.columns = ARCHER_COLUMN_NAMES
    del archer_tables_pandas

    cyclone_id_strings = [
        misc_utils.get_cyclone_id(
            year=ASSUMED_YEAR, basin_id_string=b, cyclone_number=c
        )
        for b, c in zip(
            archer_table_pandas[BASIN_ID_KEY],
            archer_table_pandas[CYCLONE_NUMBER_KEY]
        )
    ]

    valid_times_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec('{0:012d}'.format(t), TIME_FORMAT)
        for t in archer_table_pandas[VALID_TIME_STRING_KEY]
    ], dtype=int)

    archer_table_pandas[ARCHER_PRED_LATITUDE_KEY] = (
        archer_table_pandas[ARCHER_PRED_LATITUDE_KEY].apply(
            _parse_archer_latitude
        )
    )
    archer_latitudes_deg_n = numpy.array(
        archer_table_pandas[ARCHER_PRED_LATITUDE_KEY]
    )
    error_checking.assert_is_valid_lat_numpy_array(
        archer_latitudes_deg_n, allow_nan=True
    )

    archer_table_pandas[ARCHER_PRED_LONGITUDE_KEY] = (
        archer_table_pandas[ARCHER_PRED_LONGITUDE_KEY].apply(
            _parse_archer_longitude
        )
    )
    archer_longitudes_deg_e = numpy.array(
        archer_table_pandas[ARCHER_PRED_LONGITUDE_KEY]
    )
    archer_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        archer_longitudes_deg_e, allow_nan=True
    )

    num_storm_objects = len(cyclone_id_strings)
    coord_dict = {
        STORM_OBJECT_DIM: numpy.linspace(
            0, num_storm_objects - 1, num=num_storm_objects, dtype=int
        )
    }

    main_data_dict = {
        CYCLONE_ID_KEY: (
            (STORM_OBJECT_DIM,), cyclone_id_strings
        ),
        VALID_TIME_KEY: (
            (STORM_OBJECT_DIM,), valid_times_unix_sec
        ),
        ARCHER_PRED_LATITUDE_KEY: (
            (STORM_OBJECT_DIM,), archer_latitudes_deg_n
        ),
        ARCHER_PRED_LONGITUDE_KEY: (
            (STORM_OBJECT_DIM,), archer_longitudes_deg_e
        ),
        ARCHER_PLATFORM_KEY: (
            (STORM_OBJECT_DIM,), archer_table_pandas[ARCHER_PLATFORM_KEY]
        )
    }

    archer_table_xarray = xarray.Dataset(
        coords=coord_dict, data_vars=main_data_dict
    )
    del cyclone_id_strings
    del valid_times_unix_sec
    del archer_latitudes_deg_n
    del archer_longitudes_deg_e

    sort_indices = numpy.argsort(archer_table_xarray[CYCLONE_ID_KEY].values)
    archer_table_xarray = archer_table_xarray.isel(
        {STORM_OBJECT_DIM: sort_indices}
    )

    orig_num_storm_objects = len(
        archer_table_xarray[ARCHER_PRED_LATITUDE_KEY].values
    )
    good_indices = numpy.where(numpy.invert(numpy.logical_or(
        numpy.isnan(archer_table_xarray[ARCHER_PRED_LATITUDE_KEY].values),
        numpy.isnan(archer_table_xarray[ARCHER_PRED_LONGITUDE_KEY].values)
    )))[0]
    archer_table_xarray = archer_table_xarray.isel(
        {STORM_OBJECT_DIM: good_indices}
    )
    num_storm_objects = len(
        archer_table_xarray[ARCHER_PRED_LATITUDE_KEY].values
    )

    print((
        'Removed {0:d} of {1:d} ARCHER-2 storm objects due to missing lat/long.'
    ).format(
        orig_num_storm_objects - num_storm_objects, orig_num_storm_objects
    ))

    orig_num_storm_objects = len(
        archer_table_xarray[ARCHER_PRED_LATITUDE_KEY].values
    )
    good_indices = numpy.where(
        archer_table_xarray[ARCHER_PLATFORM_KEY].values != 'UNKN'
    )[0]
    archer_table_xarray = archer_table_xarray.isel(
        {STORM_OBJECT_DIM: good_indices}
    )
    num_storm_objects = len(
        archer_table_xarray[ARCHER_PRED_LATITUDE_KEY].values
    )

    print((
        'Removed {0:d} of {1:d} ARCHER-2 storm objects due to unknown source.'
    ).format(
        orig_num_storm_objects - num_storm_objects, orig_num_storm_objects
    ))

    archer_target_latitudes_deg_n = numpy.full(num_storm_objects, numpy.nan)
    archer_target_longitudes_deg_e = numpy.full(num_storm_objects, numpy.nan)
    best_track_dict = None
    best_track_file_name = None
    atx = archer_table_xarray

    for i in range(num_storm_objects):
        if (
                i == 0 or
                atx[CYCLONE_ID_KEY].values[i] !=
                atx[CYCLONE_ID_KEY].values[i - 1]
        ):
            print(SEPARATOR_STRING)

            best_track_dict = None
            best_track_file_name = _find_best_track_file(
                directory_name=raw_best_track_dir_name,
                cyclone_id_string=atx[CYCLONE_ID_KEY].values[i]
            )

            if best_track_file_name is not None:
                print('Reading data from: "{0:s}"...'.format(
                    best_track_file_name
                ))

        if best_track_file_name is None:
            continue

        (
            best_track_dict,
            archer_target_latitudes_deg_n[i],
            archer_target_longitudes_deg_e[i]
        ) = _read_best_track_file(
                pickle_file_name=best_track_file_name,
                valid_time_unix_sec=atx[VALID_TIME_KEY].values[i],
                best_track_dict=best_track_dict
            )

    print(SEPARATOR_STRING)
    archer_table_xarray = archer_table_xarray.assign({
        ARCHER_TARGET_LATITUDE_KEY: (
            (STORM_OBJECT_DIM,), archer_target_latitudes_deg_n
        ),
        ARCHER_TARGET_LONGITUDE_KEY: (
            (STORM_OBJECT_DIM,), archer_target_longitudes_deg_e
        )
    })

    orig_num_storm_objects = len(
        archer_table_xarray[ARCHER_TARGET_LATITUDE_KEY].values
    )
    good_indices = numpy.where(numpy.invert(numpy.logical_or(
        numpy.isnan(archer_table_xarray[ARCHER_TARGET_LATITUDE_KEY].values),
        numpy.isnan(archer_table_xarray[ARCHER_TARGET_LONGITUDE_KEY].values)
    )))[0]
    archer_table_xarray = archer_table_xarray.isel(
        {STORM_OBJECT_DIM: good_indices}
    )
    num_storm_objects = len(
        archer_table_xarray[ARCHER_TARGET_LATITUDE_KEY].values
    )

    print((
        'Removed {0:d} of {1:d} ARCHER-2 storm objects due to missing '
        'best-track lat/long.'
    ).format(
        orig_num_storm_objects - num_storm_objects, orig_num_storm_objects
    ))

    geocenter_pred_latitudes_deg_n = numpy.full(num_storm_objects, numpy.nan)
    geocenter_pred_longitudes_deg_e = numpy.full(num_storm_objects, numpy.nan)
    geocenter_valid_times_unix_sec = numpy.full(
        num_storm_objects, -1, dtype=int
    )
    atx = archer_table_xarray
    print(SEPARATOR_STRING)

    for i in range(num_storm_objects):
        this_start_time_unix_sec = (
            atx[VALID_TIME_KEY].values[i] - ARCHER_GEOCENTER_TIME_TOLERANCE_SEC
        )
        this_end_time_unix_sec = (
            atx[VALID_TIME_KEY].values[i] + ARCHER_GEOCENTER_TIME_TOLERANCE_SEC
        )
        possible_geocenter_times_unix_sec = (
            time_periods.range_and_interval_to_list(
                start_time_unix_sec=this_start_time_unix_sec,
                end_time_unix_sec=this_end_time_unix_sec,
                time_interval_sec=600,
                include_endpoint=True
            )
        )

        time_diffs_sec = numpy.absolute(
            atx[VALID_TIME_KEY].values[i] - possible_geocenter_times_unix_sec
        )
        possible_geocenter_times_unix_sec = possible_geocenter_times_unix_sec[
            time_diffs_sec <= ARCHER_GEOCENTER_TIME_TOLERANCE_SEC
        ]

        time_diffs_sec = numpy.absolute(
            atx[VALID_TIME_KEY].values[i] - possible_geocenter_times_unix_sec
        )
        possible_geocenter_times_unix_sec = possible_geocenter_times_unix_sec[
            numpy.argsort(time_diffs_sec)
        ]
        del time_diffs_sec

        for this_geocenter_time_unix_sec in possible_geocenter_times_unix_sec:
            geocenter_file_name = _find_geocenter_file(
                directory_name=geocenter_dir_name,
                cyclone_id_string=atx[CYCLONE_ID_KEY].values[i],
                valid_time_unix_sec=this_geocenter_time_unix_sec,
                raise_error_if_missing=False
            )

            if not os.path.isfile(geocenter_file_name):
                continue

            print('Reading data from: "{0:s}"...'.format(geocenter_file_name))
            (
                geocenter_pred_latitudes_deg_n[i],
                geocenter_pred_longitudes_deg_e[i],
                geocenter_valid_times_unix_sec[i]
            ) = _read_geocenter_file(geocenter_file_name)

            break

    print(SEPARATOR_STRING)
    archer_table_xarray = archer_table_xarray.assign({
        GEOCENTER_PRED_LATITUDE_KEY: (
            (STORM_OBJECT_DIM,), geocenter_pred_latitudes_deg_n
        ),
        GEOCENTER_PRED_LONGITUDE_KEY: (
            (STORM_OBJECT_DIM,), geocenter_pred_longitudes_deg_e
        ),
        GEOCENTER_VALID_TIME_KEY: (
            (STORM_OBJECT_DIM,), geocenter_valid_times_unix_sec
        )
    })

    orig_num_storm_objects = len(
        archer_table_xarray[GEOCENTER_PRED_LATITUDE_KEY].values
    )
    good_indices = numpy.where(numpy.invert(numpy.logical_or(
        numpy.isnan(archer_table_xarray[GEOCENTER_PRED_LATITUDE_KEY].values),
        numpy.isnan(archer_table_xarray[GEOCENTER_PRED_LONGITUDE_KEY].values)
    )))[0]
    archer_table_xarray = archer_table_xarray.isel(
        {STORM_OBJECT_DIM: good_indices}
    )
    num_storm_objects = len(
        archer_table_xarray[GEOCENTER_PRED_LATITUDE_KEY].values
    )

    print((
        'Removed {0:d} of {1:d} ARCHER-2 storm objects due to missing '
        'GeoCenter lat/long.'
    ).format(
        orig_num_storm_objects - num_storm_objects, orig_num_storm_objects
    ))

    geocenter_target_latitudes_deg_n = numpy.full(num_storm_objects, numpy.nan)
    geocenter_target_longitudes_deg_e = numpy.full(num_storm_objects, numpy.nan)
    best_track_dict = None
    best_track_file_name = None
    atx = archer_table_xarray
    print(SEPARATOR_STRING)

    for i in range(num_storm_objects):
        if (
                i == 0 or
                atx[CYCLONE_ID_KEY].values[i] !=
                atx[CYCLONE_ID_KEY].values[i - 1]
        ):
            print(SEPARATOR_STRING)

            best_track_dict = None
            best_track_file_name = _find_best_track_file(
                directory_name=raw_best_track_dir_name,
                cyclone_id_string=atx[CYCLONE_ID_KEY].values[i]
            )

            if best_track_file_name is not None:
                print('Reading data from: "{0:s}"...'.format(
                    best_track_file_name
                ))

        if best_track_file_name is None:
            continue

        (
            best_track_dict,
            geocenter_target_latitudes_deg_n[i],
            geocenter_target_longitudes_deg_e[i]
        ) = _read_best_track_file(
            pickle_file_name=best_track_file_name,
            valid_time_unix_sec=atx[GEOCENTER_VALID_TIME_KEY].values[i],
            best_track_dict=best_track_dict
        )

    print(SEPARATOR_STRING)
    archer_table_xarray = archer_table_xarray.assign({
        GEOCENTER_TARGET_LATITUDE_KEY: (
            (STORM_OBJECT_DIM,), geocenter_target_latitudes_deg_n
        ),
        GEOCENTER_TARGET_LONGITUDE_KEY: (
            (STORM_OBJECT_DIM,), geocenter_target_longitudes_deg_e
        )
    })

    orig_num_storm_objects = len(
        archer_table_xarray[GEOCENTER_TARGET_LATITUDE_KEY].values
    )
    good_indices = numpy.where(numpy.invert(numpy.logical_or(
        numpy.isnan(archer_table_xarray[GEOCENTER_TARGET_LATITUDE_KEY].values),
        numpy.isnan(archer_table_xarray[GEOCENTER_TARGET_LONGITUDE_KEY].values)
    )))[0]
    archer_table_xarray = archer_table_xarray.isel(
        {STORM_OBJECT_DIM: good_indices}
    )
    num_storm_objects = len(
        archer_table_xarray[GEOCENTER_TARGET_LATITUDE_KEY].values
    )

    print((
        'Removed {0:d} of {1:d} GeoCenter storm objects due to missing '
        'best-track lat/long.'
    ).format(
        orig_num_storm_objects - num_storm_objects, orig_num_storm_objects
    ))

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    print('Writing merged data to: "{0:s}"...'.format(output_file_name))
    archer_table_xarray.to_netcdf(
        path=output_file_name, mode='w', format='NETCDF3_64BIT'
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        archer_file_pattern=getattr(
            INPUT_ARG_OBJECT, ARCHER_FILE_PATTERN_ARG_NAME
        ),
        raw_best_track_dir_name=getattr(
            INPUT_ARG_OBJECT, BEST_TRACK_DIR_ARG_NAME
        ),
        geocenter_dir_name=getattr(INPUT_ARG_OBJECT, GEOCENTER_DIR_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
