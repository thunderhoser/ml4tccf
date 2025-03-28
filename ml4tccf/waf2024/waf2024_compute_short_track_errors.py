"""Computes error summary for short-track data."""

import os
import glob
import pickle
import argparse
import warnings
import numpy
import xarray
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4tccf.io import short_track_io

MINUTES_TO_SECONDS = 60
SECONDS_TO_MINUTES = 1. / 60
SYNOPTIC_TIME_INTERVAL_SEC = 6 * 3600
ALLOWED_SHORT_TRACK_LEAD_TIMES_SEC = numpy.array([3 * 3600], dtype=int)

TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H%M%S'

# CYCLONE_ID_STRINGS = [
#     '2024AL01', '2024AL02', '2024AL03', '2024AL04', '2024AL05', '2024AL06',
#     '2024AL07', '2024AL08', '2024AL09', '2024AL10', '2024AL11', '2024AL12',
#     '2024AL13', '2024AL14', '2024AL15', '2024AL16', '2024AL17', '2024AL18',
#     '2024AL19', '2024AL90', '2024AL91', '2024AL92', '2024AL93', '2024AL94',
#     '2024AL95', '2024AL96', '2024AL97', '2024AL98', '2024AL99', '2024CP01',
#     '2024CP90', '2024CP91', '2024EP01', '2024EP02', '2024EP03', '2024EP04',
#     '2024EP05', '2024EP06', '2024EP07', '2024EP08', '2024EP09', '2024EP10',
#     '2024EP11', '2024EP12', '2024EP13', '2024EP14', '2024EP90', '2024EP91',
#     '2024EP92', '2024EP93', '2024EP94', '2024EP95', '2024EP96', '2024EP97',
#     '2024EP98', '2024EP99', '2024WP01', '2024WP02', '2024WP03', '2024WP04',
#     '2024WP05', '2024WP06', '2024WP07', '2024WP08', '2024WP09', '2024WP10',
#     '2024WP11', '2024WP12', '2024WP13', '2024WP14', '2024WP15', '2024WP16',
#     '2024WP17', '2024WP18', '2024WP19', '2024WP20', '2024WP21', '2024WP22',
#     '2024WP23', '2024WP24', '2024WP25', '2024WP26', '2024WP27', '2024WP28',
#     '2024WP90', '2024WP91', '2024WP92', '2024WP93', '2024WP94', '2024WP95',
#     '2024WP96', '2024WP98', '2024WP99'
# ]

CYCLONE_ID_STRINGS = [
    '2024AL01', '2024AL02', '2024AL03', '2024AL04', '2024AL05', '2024AL06',
    '2024AL07', '2024AL08', '2024AL09', '2024AL10', '2024AL11', '2024AL12',
    '2024AL13', '2024AL14', '2024AL15', '2024AL16', '2024AL17', '2024AL18',
    '2024AL19', '2024AL90', '2024AL91', '2024AL92', '2024AL93', '2024AL94',
    '2024AL95', '2024AL96', '2024AL97', '2024AL98', '2024AL99'
]

STORM_OBJECT_DIM = 'storm_object'
CYCLONE_ID_KEY = 'cyclone_id_string'
VALID_TIME_KEY = 'valid_time_unix_sec'
SHORT_TRACK_LATITUDE_KEY = 'short_track_latitude_deg_n'
SHORT_TRACK_LONGITUDE_KEY = 'short_track_longitude_deg_e'
BEST_TRACK_LATITUDE_KEY = 'best_track_latitude_deg_n'
BEST_TRACK_LONGITUDE_KEY = 'best_track_longitude_deg_e'

SHORT_TRACK_DIR_ARG_NAME = 'input_processed_short_track_dir_name'
BEST_TRACK_DIR_ARG_NAME = 'input_raw_best_track_dir_name'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

SHORT_TRACK_DIR_HELP_STRING = (
    'Path to directory with processed short-track data.  Files therein will be '
    'found by `short_track_io.find_file` and read by '
    '`short_track_io.read_file`.'
)
BEST_TRACK_DIR_HELP_STRING = (
    'Path to directory with raw best-track data.  Files therein will be found '
    'by `_find_best_track_file` and read by `_read_best_track_file`.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file (NetCDF).  Merged data will be written here by this '
    'script.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SHORT_TRACK_DIR_ARG_NAME, type=str, required=True,
    help=SHORT_TRACK_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BEST_TRACK_DIR_ARG_NAME, type=str, required=True,
    help=BEST_TRACK_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


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


def _find_short_track_forecast(short_track_table_xarray, valid_time_unix_sec):
    """Finds short-track forecast for one valid time.

    :param short_track_table_xarray: xarray table in format returned by
        `short_track_io.read_file`.
    :param valid_time_unix_sec: Valid time.
    :return: short_track_latitude_deg_n: Latitude (deg north).
    :return: short_track_longitude_deg_e: Longitude (deg east).
    """

    init_time_index = -1
    lead_time_index = -1
    sttx = short_track_table_xarray

    for this_lead_time_sec in ALLOWED_SHORT_TRACK_LEAD_TIMES_SEC:
        short_track_init_time_unix_sec = (
            valid_time_unix_sec - this_lead_time_sec
        )
        these_indices = numpy.where(
            sttx.coords[short_track_io.INIT_TIME_DIM].values ==
            short_track_init_time_unix_sec
        )[0]

        if len(these_indices) == 0:
            continue

        init_time_index = these_indices[0]

        first_flags = (
            sttx.coords[short_track_io.LEAD_TIME_DIM].values ==
            this_lead_time_sec
        )

        second_flags = numpy.invert(numpy.logical_or(
            numpy.isnan(
                sttx[short_track_io.LATITUDE_KEY].values[init_time_index, :]
            ),
            numpy.isnan(
                sttx[short_track_io.LONGITUDE_KEY].values[init_time_index, :]
            )
        ))

        these_indices = numpy.where(
            numpy.logical_and(first_flags, second_flags)
        )[0]

        if len(these_indices) == 0:
            init_time_index = -1
            continue

        lead_time_index = these_indices[0]
        break

    if init_time_index == -1:
        warning_string = (
            'POTENTIAL ERROR: Cannot find short-track forecasts '
            'initialized between {0:s} and {1:s}.'
        ).format(
            time_conversion.unix_sec_to_string(
                valid_time_unix_sec -
                numpy.max(ALLOWED_SHORT_TRACK_LEAD_TIMES_SEC),
                TIME_FORMAT_FOR_LOG_MESSAGES
            ),
            time_conversion.unix_sec_to_string(
                valid_time_unix_sec -
                numpy.min(ALLOWED_SHORT_TRACK_LEAD_TIMES_SEC),
                TIME_FORMAT_FOR_LOG_MESSAGES
            )
        )

        warnings.warn(warning_string)
        return numpy.nan, numpy.nan

    info_string = (
        'Found short-track forecast for {0:s} with lead time of {1:.0f} '
        'minutes!'
    ).format(
        time_conversion.unix_sec_to_string(
            valid_time_unix_sec, TIME_FORMAT_FOR_LOG_MESSAGES
        ),
        SECONDS_TO_MINUTES *
        sttx.coords[short_track_io.LEAD_TIME_DIM].values[lead_time_index]
    )

    print(info_string)

    short_track_latitude_deg_n = sttx[short_track_io.LATITUDE_KEY].values[
        init_time_index, lead_time_index
    ]
    short_track_longitude_deg_e = sttx[short_track_io.LONGITUDE_KEY].values[
        init_time_index, lead_time_index
    ]
    short_track_longitude_deg_e = lng_conversion.convert_lng_positive_in_west(
        short_track_longitude_deg_e
    )
    return short_track_latitude_deg_n, short_track_longitude_deg_e


def _run(processed_short_track_dir_name, raw_best_track_dir_name,
         output_file_name):
    """Computes error summary for short-track data.

    This is effectively the main method.

    :param processed_short_track_dir_name: See documentation at top of this
        script.
    :param raw_best_track_dir_name: Same.
    :param output_file_name: Same.
    """

    merged_tables_xarray = []

    for this_cyclone_id_string in CYCLONE_ID_STRINGS:
        best_track_file_name = _find_best_track_file(
            directory_name=raw_best_track_dir_name,
            cyclone_id_string=this_cyclone_id_string
        )
        if best_track_file_name is None:
            continue

        short_track_file_name = short_track_io.find_file(
            directory_name=processed_short_track_dir_name,
            cyclone_id_string=this_cyclone_id_string,
            raise_error_if_missing=False
        )
        if not os.path.isfile(short_track_file_name):
            continue

        print('Reading data from: "{0:s}"...'.format(short_track_file_name))
        short_track_table_xarray = short_track_io.read_file(
            short_track_file_name
        )

        print('Reading data from: "{0:s}"...'.format(best_track_file_name))
        best_track_dict = _read_best_track_file(
            pickle_file_name=best_track_file_name,
            valid_time_unix_sec=0,
            best_track_dict=None
        )[0]

        valid_times_unix_sec = best_track_dict['st_one_sec_time']
        valid_times_unix_sec = valid_times_unix_sec[
            numpy.mod(valid_times_unix_sec, SYNOPTIC_TIME_INTERVAL_SEC) == 0
        ]

        best_track_latitudes_deg_n = numpy.full(
            len(valid_times_unix_sec), numpy.nan
        )
        best_track_longitudes_deg_e = numpy.full(
            len(valid_times_unix_sec), numpy.nan
        )
        short_track_latitudes_deg_n = numpy.full(
            len(valid_times_unix_sec), numpy.nan
        )
        short_track_longitudes_deg_e = numpy.full(
            len(valid_times_unix_sec), numpy.nan
        )

        for i in range(len(valid_times_unix_sec)):
            _, best_track_latitudes_deg_n[i], best_track_longitudes_deg_e[i] = (
                _read_best_track_file(
                    pickle_file_name=best_track_file_name,
                    valid_time_unix_sec=valid_times_unix_sec[i],
                    best_track_dict=best_track_dict
                )
            )

            short_track_latitudes_deg_n[i], short_track_longitudes_deg_e[i] = (
                _find_short_track_forecast(
                    short_track_table_xarray=short_track_table_xarray,
                    valid_time_unix_sec=valid_times_unix_sec[i]
                )
            )

        num_storm_objects = len(valid_times_unix_sec)
        coord_dict = {
            STORM_OBJECT_DIM: numpy.linspace(
                0, num_storm_objects - 1, num=num_storm_objects, dtype=int
            )
        }

        main_data_dict = {
            CYCLONE_ID_KEY: (
                (STORM_OBJECT_DIM,),
                [this_cyclone_id_string] * num_storm_objects
            ),
            VALID_TIME_KEY: (
                (STORM_OBJECT_DIM,), valid_times_unix_sec
            ),
            BEST_TRACK_LATITUDE_KEY: (
                (STORM_OBJECT_DIM,), best_track_latitudes_deg_n
            ),
            BEST_TRACK_LONGITUDE_KEY: (
                (STORM_OBJECT_DIM,), best_track_longitudes_deg_e
            ),
            SHORT_TRACK_LATITUDE_KEY: (
                (STORM_OBJECT_DIM,), short_track_latitudes_deg_n
            ),
            SHORT_TRACK_LONGITUDE_KEY: (
                (STORM_OBJECT_DIM,), short_track_longitudes_deg_e
            )
        }

        merged_tables_xarray.append(
            xarray.Dataset(coords=coord_dict, data_vars=main_data_dict)
        )

    merged_table_xarray = xarray.concat(
        merged_tables_xarray, dim=STORM_OBJECT_DIM, data_vars='all',
        coords='minimal', compat='identical', join='exact'
    )
    del merged_tables_xarray

    orig_num_storm_objects = len(
        merged_table_xarray[SHORT_TRACK_LATITUDE_KEY].values
    )
    good_indices = numpy.where(numpy.invert(numpy.logical_or(
        numpy.isnan(merged_table_xarray[SHORT_TRACK_LATITUDE_KEY].values),
        numpy.isnan(merged_table_xarray[SHORT_TRACK_LONGITUDE_KEY].values)
    )))[0]
    merged_table_xarray = merged_table_xarray.isel(
        {STORM_OBJECT_DIM: good_indices}
    )
    num_storm_objects = len(
        merged_table_xarray[SHORT_TRACK_LATITUDE_KEY].values
    )

    print((
        'Removed {0:d} of {1:d} storm objects due to missing short-track data.'
    ).format(
        orig_num_storm_objects - num_storm_objects, orig_num_storm_objects
    ))

    orig_num_storm_objects = len(
        merged_table_xarray[BEST_TRACK_LATITUDE_KEY].values
    )
    good_indices = numpy.where(numpy.invert(numpy.logical_or(
        numpy.isnan(merged_table_xarray[BEST_TRACK_LATITUDE_KEY].values),
        numpy.isnan(merged_table_xarray[BEST_TRACK_LONGITUDE_KEY].values)
    )))[0]
    merged_table_xarray = merged_table_xarray.isel(
        {STORM_OBJECT_DIM: good_indices}
    )
    num_storm_objects = len(
        merged_table_xarray[BEST_TRACK_LATITUDE_KEY].values
    )

    print((
        'Removed {0:d} of {1:d} storm objects due to missing best-track data.'
    ).format(
        orig_num_storm_objects - num_storm_objects, orig_num_storm_objects
    ))

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    print('Writing merged data to: "{0:s}"...'.format(output_file_name))
    merged_table_xarray.to_netcdf(
        path=output_file_name, mode='w', format='NETCDF3_64BIT'
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        processed_short_track_dir_name=getattr(
            INPUT_ARG_OBJECT, SHORT_TRACK_DIR_ARG_NAME
        ),
        raw_best_track_dir_name=getattr(
            INPUT_ARG_OBJECT, BEST_TRACK_DIR_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
