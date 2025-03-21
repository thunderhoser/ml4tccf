"""Creates fake extended best-track (EBTRK) file.

This 'fake' EBTRK file will contain the TC centers in training images.  It is
'fake' because these centers are interpolated from 6-hour best-track data to
30-min intervals.
"""

import os
import sys
import re
import glob
import argparse
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import longitude_conversion as lng_conversion
import error_checking
import satellite_io
import extended_best_track_io as ebtrk_io
import satellite_utils
import extended_best_track_utils as ebtrk_utils

SECONDS_TO_HOURS = 1. / 3600

INPUT_FILE_PATTERN_ARG_NAME = 'input_satellite_file_pattern'
OUTPUT_FILE_ARG_NAME = 'output_fake_ebtrk_file_name'

INPUT_FILE_PATTERN_HELP_STRING = (
    'Glob pattern for input files.  Each input file will be read by '
    '`extended_best_track_io.read_file`.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file (will be written here by '
    '`extended_best_track_io.write_file`).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_PATTERN_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_PATTERN_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(satellite_file_pattern, fake_ebtrk_file_name):
    """Creates fake extended best-track (EBTRK) file.

    This is effectively the main method.

    :param satellite_file_pattern: See documentation at top of this script.
    :param fake_ebtrk_file_name: Same.
    """

    satellite_file_names = glob.glob(satellite_file_pattern)
    satellite_file_names.sort()

    cyclone_id_strings = numpy.array([])
    valid_times_unix_sec = numpy.array([], dtype=int)
    center_latitudes_deg_n = numpy.array([], dtype=float)
    center_longitudes_deg_e = numpy.array([], dtype=float)

    for this_file_name in satellite_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_satellite_table_xarray = satellite_io.read_file(this_file_name)
        this_stx = this_satellite_table_xarray

        cyclone_id_strings = numpy.concatenate([
            cyclone_id_strings,
            this_stx[satellite_utils.CYCLONE_ID_KEY].values
        ])
        valid_times_unix_sec = numpy.concatenate([
            valid_times_unix_sec,
            this_stx.coords[satellite_utils.TIME_DIM].values
        ])

        these_projection_strings = (
            this_stx[satellite_utils.PYPROJ_STRING_KEY].values
        )
        try:
            these_projection_strings = numpy.array([
                p.decode('utf-8') for p in these_projection_strings
            ])
        except:
            pass

        these_latitudes_deg_n = numpy.full(
            len(these_projection_strings), numpy.nan
        )

        for i in range(len(these_projection_strings)):
            match_object = re.search(
                r'\+lat_ts=([\d.]+)',
                these_projection_strings[i]
            )
            assert match_object is not None

            these_latitudes_deg_n[i] = float(match_object.group(1))

        error_checking.assert_is_valid_lat_numpy_array(these_latitudes_deg_n)

        num_grid_columns = len(
            this_stx.coords[satellite_utils.LOW_RES_COLUMN_DIM].values
        )
        half_num_grid_columns = num_grid_columns // 2
        j = half_num_grid_columns - 1

        print((
            'Extracting longitudes from {0:d}th and {1:d}th grid columns...'
        ).format(
            j - 1, j
        ))

        this_longitude_matrix_deg_e = this_stx[
            satellite_utils.LONGITUDE_LOW_RES_KEY
        ].values[:, (j - 1):(j + 1)]

        abs_diff_matrix = numpy.absolute(
            numpy.diff(this_longitude_matrix_deg_e, axis=1)
        )
        assert numpy.all(abs_diff_matrix < 0.05)

        these_longitudes_deg_e = numpy.mean(
            this_longitude_matrix_deg_e, axis=1
        )[:, 0]
        these_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
            these_longitudes_deg_e, allow_nan=False
        )

        center_latitudes_deg_n = numpy.concatenate([
            center_latitudes_deg_n, these_latitudes_deg_n
        ])
        center_longitudes_deg_e = numpy.concatenate([
            center_longitudes_deg_e, these_longitudes_deg_e
        ])

    try:
        cyclone_id_strings = numpy.array([
            c.decode('utf-8') for c in cyclone_id_strings
        ])
    except:
        pass

    valid_times_unix_hours = (
        valid_times_unix_sec.astype(float) * SECONDS_TO_HOURS
    )

    num_storm_objects = len(valid_times_unix_hours)
    coord_dict = {
        ebtrk_utils.STORM_OBJECT_DIM: numpy.linspace(
            0, num_storm_objects - 1, num=num_storm_objects, dtype=int
        )
    }

    main_data_dict = {
        ebtrk_utils.STORM_ID_KEY: (
            (ebtrk_utils.STORM_OBJECT_DIM,), cyclone_id_strings
        ),
        ebtrk_utils.VALID_TIME_KEY: (
            (ebtrk_utils.STORM_OBJECT_DIM,), valid_times_unix_hours
        ),
        ebtrk_utils.CENTER_LATITUDE_KEY: (
            (ebtrk_utils.STORM_OBJECT_DIM,), center_latitudes_deg_n
        ),
        ebtrk_utils.CENTER_LONGITUDE_KEY: (
            (ebtrk_utils.STORM_OBJECT_DIM,), center_longitudes_deg_e
        )
    }

    ebtrk_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=coord_dict
    )
    print(ebtrk_table_xarray)

    print('Writing fake EBTRK data to: "{0:s}"...'.format(fake_ebtrk_file_name))
    ebtrk_io.write_file(
        xbt_table_xarray=ebtrk_table_xarray,
        netcdf_file_name=fake_ebtrk_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        satellite_file_pattern=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_PATTERN_ARG_NAME
        ),
        fake_ebtrk_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
