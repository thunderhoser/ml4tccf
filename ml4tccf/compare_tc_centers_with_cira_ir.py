"""Compares TC centers in Robert/Galina dataset with CIRA IR dataset."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import number_rounding
import longitude_conversion as lng_conversion
import cira_ir_satellite_io
import cira_ir_satellite_utils
import satellite_io as robert_satellite_io
import satellite_utils as robert_satellite_utils
import misc_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SYNOPTIC_TIME_INTERVAL_SEC = 6 * 3600
PERCENTILES_TO_REPORT = numpy.array([
    25, 50, 75, 80, 85, 90, 95, 96, 97, 98, 99, 99.5, 99.9, 100
])

ROBERT_SATELLITE_DIR_ARG_NAME = 'input_robert_satellite_dir_name'
CIRA_IR_SATELLITE_DIR_ARG_NAME = 'input_cira_ir_satellite_dir_name'
YEARS_ARG_NAME = 'years'

ROBERT_SATELLITE_DIR_HELP_STRING = (
    'Name of directory with Robert/Galina dataset.  Files therein will be '
    'found by `satellite_io.find_file` and read by `satellite_io.read_file`.'
)
CIRA_IR_SATELLITE_DIR_HELP_STRING = (
    'Name of directory with CIRA IR dataset.  Files therein will be '
    'found by `ml4tc.io.satellite_io.find_file` and read by '
    '`ml4tc.io.satellite_io.read_file`.'
)
YEARS_HELP_STRING = (
    'List of years.  Will compare centers for matching TC objects (one TC '
    'object = one TC at one time) in these years.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + ROBERT_SATELLITE_DIR_ARG_NAME, type=str, required=True,
    help=ROBERT_SATELLITE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CIRA_IR_SATELLITE_DIR_ARG_NAME, type=str, required=True,
    help=CIRA_IR_SATELLITE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + YEARS_ARG_NAME, type=int, nargs='+', required=True,
    help=YEARS_HELP_STRING
)


def _find_centers_one_tc(cira_ir_satellite_dir_name, robert_satellite_dir_name,
                         cyclone_id_string):
    """Finds centers in both datasets for one TC.

    T = number of matching time steps

    :param cira_ir_satellite_dir_name: See documentation at top of file.
    :param robert_satellite_dir_name: Same.
    :param cyclone_id_string: Same.
    :return: cira_ir_latitudes_deg_n: length-T numpy array of latitudes for CIRA
        IR centers (deg north).
    :return: cira_ir_longitudes_deg_e: length-T numpy array of longitudes for
        CIRA IR centers (deg east).
    :return: robert_latitudes_deg_n: length-T numpy array of latitudes for
        Robert/Galina centers (deg north).
    :return: robert_longitudes_deg_e: length-T numpy array of longitudes for
        Robert/Galina centers (deg east).
    """

    cira_ir_file_name = cira_ir_satellite_io.find_file(
        directory_name=cira_ir_satellite_dir_name,
        cyclone_id_string=cyclone_id_string,
        prefer_zipped=False, allow_other_format=True,
        raise_error_if_missing=True
    )

    print('Reading data from: "{0:s}"...'.format(cira_ir_file_name))
    cira_ir_table_xarray = cira_ir_satellite_io.read_file(cira_ir_file_name)

    ct = cira_ir_table_xarray
    cira_ir_times_unix_sec = (
        ct.coords[cira_ir_satellite_utils.TIME_DIM].values
    )

    # TODO(thunderhoser): I might want some tolerance here.  CIRA IR contains
    # very few synoptic times.
    good_indices = numpy.where(
        numpy.mod(cira_ir_times_unix_sec, SYNOPTIC_TIME_INTERVAL_SEC) == 0
    )[0]

    if len(good_indices) == 0:
        return (
            numpy.array([]), numpy.array([]), numpy.array([]), numpy.array([])
        )

    cira_ir_times_unix_sec = cira_ir_times_unix_sec[good_indices]
    cira_ir_latitudes_deg_n = (
        ct[cira_ir_satellite_utils.STORM_LATITUDE_KEY].values[good_indices]
    )
    cira_ir_longitudes_deg_e = (
        ct[cira_ir_satellite_utils.STORM_LONGITUDE_KEY].values[good_indices]
    )

    robert_file_names = robert_satellite_io.find_files_one_cyclone(
        directory_name=robert_satellite_dir_name,
        cyclone_id_string=cyclone_id_string,
        raise_error_if_all_missing=True
    )

    robert_times_unix_sec = numpy.array([], dtype=int)
    robert_latitudes_deg_n = numpy.array([], dtype=float)
    robert_longitudes_deg_e = numpy.array([], dtype=float)

    for this_robert_file_name in robert_file_names:
        print('Reading data from: "{0:s}"...'.format(this_robert_file_name))
        this_robert_table_xarray = robert_satellite_io.read_file(
            this_robert_file_name
        )
        rt = this_robert_table_xarray

        robert_times_unix_sec = numpy.concatenate((
            robert_times_unix_sec,
            rt.coords[robert_satellite_utils.TIME_DIM].values
        ))

        num_rows = len(
            rt.coords[robert_satellite_utils.LOW_RES_ROW_DIM].values
        )
        first_row_index = int(numpy.round(0.5 * num_rows - 1))
        last_row_index = int(numpy.round(0.5 * num_rows + 1))
        these_latitudes_deg_n = numpy.mean(
            rt[robert_satellite_utils.LATITUDE_LOW_RES_KEY].values[
                :, first_row_index:last_row_index
            ], axis=1
        )
        robert_latitudes_deg_n = numpy.concatenate((
            robert_latitudes_deg_n, these_latitudes_deg_n
        ))

        num_columns = len(
            rt.coords[robert_satellite_utils.LOW_RES_COLUMN_DIM].values
        )
        first_column_index = int(numpy.round(0.5 * num_columns - 1))
        last_column_index = int(numpy.round(0.5 * num_columns + 1))
        these_longitudes_deg_e = numpy.mean(
            rt[robert_satellite_utils.LONGITUDE_LOW_RES_KEY].values[
                :, first_column_index:last_column_index
            ], axis=1
        )
        robert_longitudes_deg_e = numpy.concatenate((
            robert_longitudes_deg_e, these_longitudes_deg_e
        ))

    cira_ir_to_robert_indices = [
        numpy.where(robert_times_unix_sec == t)[0]
        for t in cira_ir_times_unix_sec
    ]
    cira_ir_to_robert_indices = numpy.array([
        idxs[0] if len(idxs) > 0 else -1
        for idxs in cira_ir_to_robert_indices
    ], dtype=int)

    del robert_times_unix_sec
    del cira_ir_times_unix_sec

    if numpy.all(cira_ir_to_robert_indices == -1):
        return (
            numpy.array([]), numpy.array([]), numpy.array([]), numpy.array([])
        )

    good_cira_ir_indices = numpy.where(cira_ir_to_robert_indices > -1)[0]
    # cira_ir_times_unix_sec = cira_ir_times_unix_sec[good_cira_ir_indices]
    cira_ir_latitudes_deg_n = cira_ir_latitudes_deg_n[good_cira_ir_indices]
    cira_ir_longitudes_deg_e = cira_ir_longitudes_deg_e[
        good_cira_ir_indices
    ]

    good_robert_indices = cira_ir_to_robert_indices[good_cira_ir_indices]
    # robert_times_unix_sec = robert_times_unix_sec[good_robert_indices]
    robert_latitudes_deg_n = robert_latitudes_deg_n[good_robert_indices]
    robert_longitudes_deg_e = robert_longitudes_deg_e[good_robert_indices]

    return (
        cira_ir_latitudes_deg_n, cira_ir_longitudes_deg_e,
        robert_latitudes_deg_n, robert_longitudes_deg_e
    )


def _run(robert_satellite_dir_name, cira_ir_satellite_dir_name, years):
    """Compares TC centers in Robert/Galina dataset with CIRA IR dataset.

    This is effectively the main method.

    :param robert_satellite_dir_name: See documentation at top of file.
    :param cira_ir_satellite_dir_name: Same.
    :param years: Same.
    """

    robert_cyclone_id_strings = robert_satellite_io.find_cyclones(
        directory_name=robert_satellite_dir_name,
        raise_error_if_all_missing=True
    )
    robert_cyclone_id_strings = set([
        s for s in robert_cyclone_id_strings
        if misc_utils.parse_cyclone_id(s)[0] in years
    ])

    cira_ir_cyclone_id_strings = cira_ir_satellite_io.find_cyclones(
        directory_name=cira_ir_satellite_dir_name,
        raise_error_if_all_missing=True
    )
    cira_ir_cyclone_id_strings = set([
        s for s in cira_ir_cyclone_id_strings
        if cira_ir_satellite_utils.parse_cyclone_id(s)[0] in years
    ])
    cyclone_id_strings = list(
        robert_cyclone_id_strings.intersection(cira_ir_cyclone_id_strings)
    )

    cira_ir_latitudes_deg_n = numpy.array([], dtype=float)
    cira_ir_longitudes_deg_e = numpy.array([], dtype=float)
    robert_latitudes_deg_n = numpy.array([], dtype=float)
    robert_longitudes_deg_e = numpy.array([], dtype=float)

    for this_cyclone_id_string in cyclone_id_strings:
        (
            these_cira_ir_latitudes_deg_n, these_cira_ir_longitudes_deg_e,
            these_robert_latitudes_deg_n, these_robert_longitudes_deg_e
        ) = _find_centers_one_tc(
            cira_ir_satellite_dir_name=cira_ir_satellite_dir_name,
            robert_satellite_dir_name=robert_satellite_dir_name,
            cyclone_id_string=this_cyclone_id_string
        )

        cira_ir_latitudes_deg_n = numpy.concatenate((
            cira_ir_latitudes_deg_n, these_cira_ir_latitudes_deg_n
        ))
        cira_ir_longitudes_deg_e = numpy.concatenate((
            cira_ir_longitudes_deg_e, these_cira_ir_longitudes_deg_e
        ))
        robert_latitudes_deg_n = numpy.concatenate((
            robert_latitudes_deg_n, these_robert_latitudes_deg_n
        ))
        robert_longitudes_deg_e = numpy.concatenate((
            robert_longitudes_deg_e, these_robert_longitudes_deg_e
        ))

    print(SEPARATOR_STRING)

    cira_ir_latitudes_deg_n = number_rounding.round_to_nearest(
        cira_ir_latitudes_deg_n, 0.01
    )
    cira_ir_longitudes_deg_e = number_rounding.round_to_nearest(
        cira_ir_longitudes_deg_e, 0.01
    )
    robert_latitudes_deg_n = number_rounding.round_to_nearest(
        robert_latitudes_deg_n, 0.01
    )
    robert_longitudes_deg_e = number_rounding.round_to_nearest(
        robert_longitudes_deg_e, 0.01
    )

    absolute_latitude_diffs_deg = numpy.absolute(
        cira_ir_latitudes_deg_n - robert_latitudes_deg_n
    )

    cira_ir_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        cira_ir_longitudes_deg_e, allow_nan=False
    )
    robert_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        robert_longitudes_deg_e, allow_nan=False
    )
    first_longitude_diffs_deg = numpy.absolute(
        cira_ir_longitudes_deg_e - robert_longitudes_deg_e
    )

    cira_ir_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
        cira_ir_longitudes_deg_e, allow_nan=False
    )
    robert_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
        robert_longitudes_deg_e, allow_nan=False
    )
    second_longitude_diffs_deg = numpy.absolute(
        cira_ir_longitudes_deg_e - robert_longitudes_deg_e
    )

    absolute_longitude_diffs_deg = numpy.minimum(
        first_longitude_diffs_deg, second_longitude_diffs_deg
    )
    euclidean_distances_deg = numpy.sqrt(
        absolute_latitude_diffs_deg ** 2 + absolute_longitude_diffs_deg ** 2
    )

    for this_percentile in PERCENTILES_TO_REPORT:
        print((
            '{0:.1f}th-percentile Euclidean distance between TC centers '
            '(over {1:d} examples) = {2:.4g} deg'
        ).format(
            this_percentile,
            len(euclidean_distances_deg),
            numpy.percentile(euclidean_distances_deg, this_percentile)
        ))

    print(SEPARATOR_STRING)

    for i in range(len(euclidean_distances_deg)):
        print((
            'Robert/Galina coords = {0:.2f} deg N, {1:.2f} deg E ... '
            'CIRA IR = {2:.2f} deg N, {3:.2f} deg E'
        ).format(
            robert_latitudes_deg_n[i],
            robert_longitudes_deg_e[i],
            cira_ir_latitudes_deg_n[i],
            cira_ir_longitudes_deg_e[i]
        ))


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        robert_satellite_dir_name=getattr(
            INPUT_ARG_OBJECT, ROBERT_SATELLITE_DIR_ARG_NAME
        ),
        cira_ir_satellite_dir_name=getattr(
            INPUT_ARG_OBJECT, CIRA_IR_SATELLITE_DIR_ARG_NAME
        ),
        years=numpy.array(getattr(INPUT_ARG_OBJECT, YEARS_ARG_NAME), dtype=int)
    )
