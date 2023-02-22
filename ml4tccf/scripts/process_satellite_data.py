"""Processes satellite data.

Specifically, converts from two files per TC per time step to one file per TC
per day, with more verbose metadata and variable names.
"""

import shutil
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from ml4tccf.io import raw_satellite_io
from ml4tccf.io import satellite_io
from ml4tccf.utils import misc_utils
from ml4tccf.utils import satellite_utils

INPUT_DIR_ARG_NAME = 'input_dir_name'
CYCLONE_ID_ARG_NAME = 'cyclone_id_string'
MAX_BAD_PIXELS_LOW_RES_ARG_NAME = 'max_bad_pixels_low_res'
MAX_BAD_PIXELS_HIGH_RES_ARG_NAME = 'max_bad_pixels_high_res'
TEMPORARY_DIR_ARG_NAME = 'temporary_dir_name'
MIN_VISIBLE_FRACTION_ARG_NAME = 'min_visible_pixel_fraction'
ALTITUDE_ANGLE_EXE_ARG_NAME = 'altitude_angle_exe_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing raw satellite data.  This script will '
    'use `raw_satellite_io.find_files_one_tc` to find files therein.'
)
CYCLONE_ID_HELP_STRING = 'Cyclone ID in format YYYYBBNN.'
MAX_BAD_PIXELS_LOW_RES_HELP_STRING = (
    'Max number of bad pixels per time/channel for low-resolution satellite '
    'data.  For any time/channel pair with more bad pixels, all pixels will be '
    'replaced with NaN.  For any time/channel pair with fewer or equal bad '
    'pixels, bad pixels will replaced by inpainting.'
)
MAX_BAD_PIXELS_HIGH_RES_HELP_STRING = (
    'Same as {0:s} but for high-resolution data.'
).format(MAX_BAD_PIXELS_LOW_RES_ARG_NAME)

TEMPORARY_DIR_HELP_STRING = (
    'Path to directory for temporary files with solar altitude angles.'
)
MIN_VISIBLE_FRACTION_HELP_STRING = (
    'Minimum fraction of visible pixels (solar altitude angle > 0 deg) at a '
    'given time.  For any time with fewer visible pixels, all visible '
    '(high-resolution) data will be replaced with NaN.'
)
ALTITUDE_ANGLE_EXE_HELP_STRING = (
    'Path to Fortran executable (pathless file name should probably be '
    '"solarpos") that computes solar altitude angles.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Processed files (one per day) will be saved '
    'here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CYCLONE_ID_ARG_NAME, type=str, required=True,
    help=CYCLONE_ID_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_BAD_PIXELS_LOW_RES_ARG_NAME, type=int, required=False,
    default=satellite_utils.DEFAULT_MAX_BAD_PIXELS_LOW_RES,
    help=MAX_BAD_PIXELS_LOW_RES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_BAD_PIXELS_HIGH_RES_ARG_NAME, type=int, required=False,
    default=satellite_utils.DEFAULT_MAX_BAD_PIXELS_HIGH_RES,
    help=MAX_BAD_PIXELS_HIGH_RES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TEMPORARY_DIR_ARG_NAME, type=str, required=True,
    help=TEMPORARY_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_VISIBLE_FRACTION_ARG_NAME, type=float, required=False,
    default=satellite_utils.DEFAULT_MIN_VISIBLE_PIXEL_FRACTION,
    help=MIN_VISIBLE_FRACTION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ALTITUDE_ANGLE_EXE_ARG_NAME, type=str, required=False,
    default=misc_utils.DEFAULT_EXE_NAME_FOR_ALTITUDE_ANGLE,
    help=ALTITUDE_ANGLE_EXE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_dir_name, cyclone_id_string, max_bad_pixels_low_res,
         max_bad_pixels_high_res, temporary_dir_name,
         min_visible_pixel_fraction, altitude_angle_exe_name, output_dir_name):
    """Processes satellite data.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param cyclone_id_string: Same.
    :param max_bad_pixels_low_res: Same.
    :param max_bad_pixels_high_res: Same.
    :param temporary_dir_name: Same.
    :param min_visible_pixel_fraction: Same.
    :param altitude_angle_exe_name: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=temporary_dir_name
    )

    low_res_file_names = raw_satellite_io.find_files_one_tc(
        directory_name=input_dir_name, cyclone_id_string=cyclone_id_string,
        look_for_high_res=False, raise_error_if_all_missing=True
    )
    valid_times_unix_sec = numpy.array(
        [raw_satellite_io.file_name_to_time(f) for f in low_res_file_names],
        dtype=int
    )

    sort_indices = numpy.argsort(valid_times_unix_sec)
    valid_times_unix_sec = valid_times_unix_sec[sort_indices]
    low_res_file_names = [low_res_file_names[k] for k in sort_indices]

    high_res_file_names = [
        raw_satellite_io.find_file(
            directory_name=input_dir_name, cyclone_id_string=cyclone_id_string,
            valid_time_unix_sec=t, look_for_high_res=True,
            raise_error_if_missing=True
        ) for t in valid_times_unix_sec
    ]

    valid_date_strings = [
        time_conversion.unix_sec_to_string(t, satellite_io.DATE_FORMAT)
        for t in valid_times_unix_sec
    ]
    valid_date_strings.append('None')

    satellite_tables_xarray = []

    for i in range(len(valid_date_strings)):
        if (
                len(satellite_tables_xarray) > 0 and
                valid_date_strings[i] != valid_date_strings[i - 1]
        ):
            satellite_table_xarray = satellite_utils.concat_over_time(
                satellite_tables_xarray
            )
            satellite_tables_xarray = []

            satellite_table_xarray = satellite_utils.quality_control_low_res(
                satellite_table_xarray=satellite_table_xarray,
                max_bad_pixels_per_time_channel=max_bad_pixels_low_res
            )
            satellite_table_xarray = satellite_utils.quality_control_high_res(
                satellite_table_xarray=satellite_table_xarray,
                max_bad_pixels_per_time_channel=max_bad_pixels_high_res
            )
            satellite_utils.mask_visible_data_at_night(
                satellite_table_xarray=satellite_table_xarray,
                temporary_dir_name=temporary_dir_name,
                min_visible_pixel_fraction=min_visible_pixel_fraction,
                altitude_angle_exe_name=altitude_angle_exe_name
            )

            this_output_file_name = satellite_io.find_file(
                directory_name=output_dir_name,
                cyclone_id_string=cyclone_id_string,
                valid_date_string=valid_date_strings[i - 1],
                raise_error_if_missing=False
            )

            print('Writing data to: "{0:s}"...'.format(this_output_file_name))
            satellite_io.write_file(
                satellite_table_xarray=satellite_table_xarray,
                zarr_file_name=this_output_file_name
            )

            del satellite_table_xarray

        if i == len(valid_date_strings) - 1:
            break

        print('Reading data from: "{0:s}"...'.format(high_res_file_names[i]))
        this_high_res_table_xarray = raw_satellite_io.read_file(
            satellite_file_name=high_res_file_names[i], is_high_res=True
        )

        print('Reading data from: "{0:s}"...'.format(low_res_file_names[i]))
        this_low_res_table_xarray = raw_satellite_io.read_file(
            satellite_file_name=low_res_file_names[i], is_high_res=False
        )

        satellite_tables_xarray.append(
            raw_satellite_io.merge_low_and_high_res(
                low_res_satellite_table_xarray=this_low_res_table_xarray,
                high_res_satellite_table_xarray=this_high_res_table_xarray
            )
        )

    shutil.rmtree(temporary_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        cyclone_id_string=getattr(INPUT_ARG_OBJECT, CYCLONE_ID_ARG_NAME),
        max_bad_pixels_low_res=getattr(
            INPUT_ARG_OBJECT, MAX_BAD_PIXELS_LOW_RES_ARG_NAME
        ),
        max_bad_pixels_high_res=getattr(
            INPUT_ARG_OBJECT, MAX_BAD_PIXELS_HIGH_RES_ARG_NAME
        ),
        temporary_dir_name=getattr(INPUT_ARG_OBJECT, TEMPORARY_DIR_ARG_NAME),
        min_visible_pixel_fraction=getattr(
            INPUT_ARG_OBJECT, MIN_VISIBLE_FRACTION_ARG_NAME
        ),
        altitude_angle_exe_name=getattr(
            INPUT_ARG_OBJECT, ALTITUDE_ANGLE_EXE_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
