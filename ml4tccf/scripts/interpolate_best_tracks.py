"""Interpolates tropical-cyclone best tracks from 6-h to 30-min intervals."""

import os
import re
import errno
import argparse
import numpy
import xarray
import pandas
from scipy.interpolate import splprep, splev

HOURS_TO_SECONDS = 3600
INTERP_TIME_INTERVAL_SEC = 1800

CYCLONE_SAMPLE_DIM = 'storm_object'

CYCLONE_ID_KEY = 'storm_id_string'
VALID_TIME_KEY = 'valid_time_unix_hours'
CENTER_LATITUDE_KEY = 'center_latitude_deg_n'
CENTER_LONGITUDE_KEY = 'center_longitude_deg_e'

INPUT_FILE_ARG_NAME = 'input_file_name'
CYCLONE_ID_PATTERN_ARG_NAME = 'cyclone_id_pattern'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  This must be a NetCDF file, and it will be read by '
    '`xarray.open_dataset`.  File must contain keys "storm_id_string", '
    '"valid_time_unix_hours", "center_latitude_deg_n", and '
    '"center_longitude_deg_e" -- where "storm_id_string" is the ATCF cyclone '
    'ID, e.g., 2005AL12 for Hurricane Katrina.  Every one of these keys must '
    'be a length-S vector, where S = number of TC samples, where one TC '
    'sample = one tropical cyclone at one time step.'
)
CYCLONE_ID_PATTERN_HELP_STRING = (
    'Glob pattern (a regular expression).  This script will process every '
    'tropical cyclone whose ATCF cyclone ID matches the given glob pattern.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  This will be formatted the same way as the input '
    'file, except that [1] it will contain 30-min, rather than 6-h, time '
    'intervals; [2] it will contain data only for the desired tropical '
    'cyclones.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CYCLONE_ID_PATTERN_ARG_NAME, type=str, required=True,
    help=CYCLONE_ID_PATTERN_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def __mkdir_recursive_if_necessary(directory_name=None, file_name=None):
    """Creates directory if necessary (i.e., doesn't already exist).

    This method checks for the argument `directory_name` first.  If
    `directory_name` is None, this method checks for `file_name` and extracts
    the directory.

    :param directory_name: Path to local directory.
    :param file_name: Path to local file.
    """

    if directory_name is None:
        assert isinstance(file_name, str)
        directory_name = os.path.dirname(file_name)
    else:
        assert isinstance(directory_name, str)

    if directory_name == '':
        return

    try:
        os.makedirs(directory_name)
    except OSError as this_error:
        if this_error.errno == errno.EEXIST and os.path.isdir(directory_name):
            pass
        else:
            raise


def __make_longitude_negative_in_west(longitudes_deg_e):
    """Converts longitudes so that all values in western hemi are negative.

    This means that all values in the western hemisphere are from -180...0
    deg E.

    :param longitudes_deg_e: Longitudes (either scalar or numpy array) in deg
        east.
    :return: new_longitudes_deg_e: Same as input but with new format for western
        hemisphere.
    """

    was_input_array = isinstance(longitudes_deg_e, numpy.ndarray)
    if not was_input_array:
        longitudes_deg_e = numpy.full(1, longitudes_deg_e)

    new_longitudes_deg_e = longitudes_deg_e + 0.
    new_longitudes_deg_e[new_longitudes_deg_e > 180.] -= 360.

    if was_input_array:
        return new_longitudes_deg_e

    return new_longitudes_deg_e[0]


def __make_longitude_positive_in_west(longitudes_deg_e):
    """Converts longitudes so that all values in western hemi are positive.

    This means that all values in the western hemisphere are from 180...360
    deg E.

    :param longitudes_deg_e: Longitudes (either scalar or numpy array) in deg
        east.
    :return: new_longitudes_deg_e: Same as input but with new format for western
        hemisphere.
    """

    was_input_array = isinstance(longitudes_deg_e, numpy.ndarray)
    if not was_input_array:
        longitudes_deg_e = numpy.full(1, longitudes_deg_e)

    new_longitudes_deg_e = longitudes_deg_e + 0.
    new_longitudes_deg_e[new_longitudes_deg_e < 0.] += 360.

    if was_input_array:
        return new_longitudes_deg_e

    return new_longitudes_deg_e[0]


def _interp_one_track(best_track_table_xarray, cyclone_id_string):
    """Interpolates best track for one tropical cyclone.

    T = number of interpolated time steps

    :param best_track_table_xarray: xarray table described at top of this
        script.
    :param cyclone_id_string: Will interpolate data for this cyclone.
    :return: interp_times_unix_sec: length-T numpy array of valid times.
    :return: interp_latitudes_deg_n: length-T numpy array of latitudes (deg
        north).
    :return: interp_longitudes_deg_e: length-T numpy array of longitudes (deg
        east).
    """

    bttx = best_track_table_xarray
    idxs = numpy.where(
        bttx[CYCLONE_ID_KEY].values == cyclone_id_string
    )[0]

    if len(idxs) <= 3:
        return numpy.array([]), numpy.array([]), numpy.array([])

    orig_times_unix_hours = bttx[VALID_TIME_KEY].values[idxs]
    orig_latitudes_deg_n = bttx[CENTER_LATITUDE_KEY].values[idxs]
    orig_longitudes_deg_e = bttx[CENTER_LONGITUDE_KEY].values[idxs]

    assert not numpy.any(orig_longitudes_deg_e < -180.)
    assert not numpy.any(orig_longitudes_deg_e > 360.)

    orig_longitudes_deg_e = __make_longitude_positive_in_west(
        orig_longitudes_deg_e
    )

    absolute_longitude_diffs_deg = numpy.absolute(
        numpy.diff(orig_longitudes_deg_e)
    )
    if numpy.any(absolute_longitude_diffs_deg > 180.):
        orig_longitudes_deg_e = __make_longitude_negative_in_west(
            orig_longitudes_deg_e
        )

    absolute_longitude_diffs_deg = numpy.absolute(
        numpy.diff(orig_longitudes_deg_e)
    )
    assert not numpy.any(absolute_longitude_diffs_deg > 180.)

    sort_indices = numpy.argsort(orig_times_unix_hours)
    orig_times_unix_hours = orig_times_unix_hours[sort_indices]
    orig_latitudes_deg_n = orig_latitudes_deg_n[sort_indices]
    orig_longitudes_deg_e = orig_longitudes_deg_e[sort_indices]

    orig_times_unix_sec = HOURS_TO_SECONDS * orig_times_unix_hours

    num_interp_times = 1 + int(numpy.round(
        float(orig_times_unix_sec[-1] - orig_times_unix_sec[0]) /
        INTERP_TIME_INTERVAL_SEC
    ))
    interp_times_unix_sec = numpy.linspace(
        orig_times_unix_sec[0], orig_times_unix_sec[-1],
        num=num_interp_times, dtype=int
    )

    spline_params_tuple = splprep(
        x=[orig_longitudes_deg_e, orig_latitudes_deg_n],
        u=orig_times_unix_sec, k=3, s=0
    )[0]

    interp_points = splev(x=interp_times_unix_sec, tck=spline_params_tuple)
    interp_longitudes_deg_e = interp_points[0]
    interp_latitudes_deg_n = interp_points[1]
    interp_longitudes_deg_e = __make_longitude_positive_in_west(
        interp_longitudes_deg_e
    )

    return (
        interp_times_unix_sec, interp_latitudes_deg_n, interp_longitudes_deg_e
    )


def _run(input_file_name, cyclone_id_pattern, output_file_name):
    """Interpolates tropical-cyclone best tracks from 6-h to 30-min intervals.

    This is effectively the main method.

    :param input_file_name: See documentation at top of this script.
    :param cyclone_id_pattern: Same.
    :param output_file_name: Same.
    """

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    best_track_table_xarray = xarray.open_dataset(input_file_name)
    bttx = best_track_table_xarray

    all_cyclone_id_strings = numpy.unique(bttx[CYCLONE_ID_KEY].values)
    cyclone_id_strings = [
        s for s in all_cyclone_id_strings if re.match(cyclone_id_pattern, s)
    ]

    interp_times_unix_sec = numpy.array([], dtype=int)
    interp_latitudes_deg_n = numpy.array([], dtype=float)
    interp_longitudes_deg_e = numpy.array([], dtype=float)
    interp_cyclone_id_strings = []

    for this_cyclone_id_string in cyclone_id_strings:
        print('Interpolating best track for cyclone {0:s}...'.format(
            this_cyclone_id_string
        ))
        (
            these_times_unix_sec,
            these_latitudes_deg_n,
            these_longitudes_deg_e
        ) = _interp_one_track(
            best_track_table_xarray=best_track_table_xarray,
            cyclone_id_string=this_cyclone_id_string
        )

        interp_cyclone_id_strings += (
            [this_cyclone_id_string] * len(these_times_unix_sec)
        )
        interp_times_unix_sec = numpy.concatenate(
            [interp_times_unix_sec, these_times_unix_sec]
        )
        interp_latitudes_deg_n = numpy.concatenate(
            [interp_latitudes_deg_n, these_latitudes_deg_n]
        )
        interp_longitudes_deg_e = numpy.concatenate(
            [interp_longitudes_deg_e, these_longitudes_deg_e]
        )

    interp_times_unix_hours = (
        interp_times_unix_sec.astype(float) / HOURS_TO_SECONDS
    )

    these_dim = (CYCLONE_SAMPLE_DIM,)
    main_data_dict = {
        CYCLONE_ID_KEY: (these_dim, interp_cyclone_id_strings),
        VALID_TIME_KEY: (these_dim, interp_times_unix_hours),
        CENTER_LATITUDE_KEY: (these_dim, interp_latitudes_deg_n),
        CENTER_LONGITUDE_KEY: (these_dim, interp_longitudes_deg_e)
    }

    num_tc_samples = len(interp_cyclone_id_strings)
    coord_dict = {
        CYCLONE_SAMPLE_DIM: numpy.linspace(
            0, num_tc_samples - 1, num=num_tc_samples, dtype=int
        )
    }

    best_track_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=coord_dict
    )

    __mkdir_recursive_if_necessary(file_name=output_file_name)
    print('Writing interpolated best tracks to: "{0:s}"...'.format(
        output_file_name
    ))
    best_track_table_xarray.to_netcdf(
        path=output_file_name, mode='w', format='NETCDF4'
    )

    if output_file_name.endswith('.nc'):
        output_csv_file_name = output_file_name[:-3] + '.csv'
    else:
        output_csv_file_name = output_file_name + '.csv'

    best_track_table_pandas = pandas.DataFrame({
        CYCLONE_ID_KEY: interp_cyclone_id_strings,
        VALID_TIME_KEY: interp_times_unix_hours,
        CENTER_LATITUDE_KEY: interp_latitudes_deg_n,
        CENTER_LONGITUDE_KEY: interp_longitudes_deg_e
    })

    print('Writing interpolated best tracks to: "{0:s}"...'.format(
        output_csv_file_name
    ))
    best_track_table_pandas.to_csv(output_csv_file_name, index=False)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        cyclone_id_pattern=getattr(
            INPUT_ARG_OBJECT, CYCLONE_ID_PATTERN_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
