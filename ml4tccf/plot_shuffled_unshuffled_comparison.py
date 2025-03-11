"""Plots comparison between shuffled and unshuffled satellite data."""

import os
import sys
import copy
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import number_rounding
import time_conversion
import file_system_utils
import satellite_io
import border_io
import satellite_utils
import normalization
import plot_satellite

DAYS_TO_SECONDS = 86400

SHUFFLED_FILE_ARG_NAME = 'input_shuffled_file_name'
UNSHUFFLED_DIR_ARG_NAME = 'input_unshuffled_dir_name'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
LOW_RES_WAVELENGTHS_ARG_NAME = 'low_res_wavelengths_microns'
NUM_GRID_ROWS_ARG_NAME = 'num_grid_rows_low_res'
NUM_GRID_COLUMNS_ARG_NAME = 'num_grid_columns_low_res'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

SHUFFLED_FILE_HELP_STRING = (
    'Path to file with shuffled satellite data.  For every TC object (one TC '
    'object = one TC at one time) in this file, we will plot a comparison '
    'between the image in the shuffled file and the original image in the '
    'unshuffled directory.'
)
UNSHUFFLED_DIR_HELP_STRING = (
    'Path to directory with unshuffled satellite data.  Files therein will be '
    'found by `satellite_io.find_file` and read by `satellite_io.read_file`.'
)

# TODO(thunderhoser): Be careful, because we could have both normalized and
# unnormalized data coming from the input file/directory.
NORMALIZATION_FILE_HELP_STRING = (
    'Path to file with normalization parameters.  If the satellite data are '
    'normalized, this file will be used to convert back to physical units '
    'before plotting.  You can also leave this empty, if you just want to plot '
    'in z-score units.'
)
LOW_RES_WAVELENGTHS_HELP_STRING = (
    'Will plot low-resolution data (brightness temperature) at these '
    'wavelengths.'
)
NUM_GRID_ROWS_HELP_STRING = (
    'Number of rows in low-resolution (brightness-temperature) grid.'
)
NUM_GRID_COLUMNS_HELP_STRING = (
    'Number of columns in low-resolution (brightness-temperature) grid.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SHUFFLED_FILE_ARG_NAME, type=str, required=True,
    help=SHUFFLED_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + UNSHUFFLED_DIR_ARG_NAME, type=str, required=True,
    help=UNSHUFFLED_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=False, default='',
    help=NORMALIZATION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LOW_RES_WAVELENGTHS_ARG_NAME, type=float, nargs='+',
    required=False, default=[8.5, 9.61, 12.3],
    help=LOW_RES_WAVELENGTHS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_GRID_ROWS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_GRID_ROWS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_GRID_COLUMNS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_GRID_COLUMNS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(shuffled_file_name, unshuffled_dir_name, normalization_file_name,
         low_res_wavelengths_microns, num_grid_rows_low_res,
         num_grid_columns_low_res, output_dir_name):
    """Plots comparison between shuffled and unshuffled satellite data.

    This is effectively the main method.

    :param shuffled_file_name: See documentation at top of this script.
    :param unshuffled_dir_name: Same.
    :param normalization_file_name: Same.
    :param low_res_wavelengths_microns: Same.
    :param num_grid_rows_low_res: Same.
    :param num_grid_columns_low_res: Same.
    :param output_dir_name: Same.
    """

    if normalization_file_name == '':
        normalization_file_name = None

    shuffled_output_dir_name = '{0:s}/shuffled'.format(output_dir_name)
    unshuffled_output_dir_name = '{0:s}/unshuffled'.format(output_dir_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=shuffled_output_dir_name
    )
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=unshuffled_output_dir_name
    )

    if num_grid_rows_low_res <= 0 or num_grid_columns_low_res <= 0:
        num_grid_rows_low_res = None
        num_grid_columns_low_res = None

    print('Reading shuffled data from: "{0:s}"...'.format(shuffled_file_name))
    shuffled_satellite_table_xarray = satellite_io.read_file(shuffled_file_name)
    shuffled_satellite_table_xarray = satellite_utils.subset_wavelengths(
        satellite_table_xarray=shuffled_satellite_table_xarray,
        wavelengths_to_keep_microns=low_res_wavelengths_microns,
        for_high_res=False
    )
    shuffled_stx = shuffled_satellite_table_xarray

    are_shuffled_data_norm = not numpy.any(
        shuffled_stx[satellite_utils.BRIGHTNESS_TEMPERATURE_KEY].values > 20.
    )

    if are_shuffled_data_norm and normalization_file_name is not None:
        print('Reading data from: "{0:s}"...'.format(normalization_file_name))
        norm_param_table_xarray = normalization.read_file(
            normalization_file_name
        )

        shuffled_satellite_table_xarray = normalization.denormalize_data(
            satellite_table_xarray=shuffled_satellite_table_xarray,
            normalization_param_table_xarray=norm_param_table_xarray
        )
        are_shuffled_data_norm = False
        shuffled_stx = shuffled_satellite_table_xarray
    else:
        norm_param_table_xarray = None

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()
    num_cyclone_objects = len(
        shuffled_stx[satellite_utils.CYCLONE_ID_KEY].values
    )

    last_unshuffled_file_name = ''
    unshuffled_satellite_table_xarray = None
    are_unshuffled_data_norm = False

    for i in range(num_cyclone_objects):
        plot_satellite.plot_data_one_time(
            satellite_table_xarray=shuffled_satellite_table_xarray,
            time_index=i,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            are_data_normalized=are_shuffled_data_norm,
            output_dir_name=shuffled_output_dir_name
        )

        valid_time_unix_sec = (
            shuffled_stx.coords[satellite_utils.TIME_DIM].values[i]
        )
        valid_date_unix_sec = int(number_rounding.floor_to_nearest(
            valid_time_unix_sec, DAYS_TO_SECONDS
        ))
        valid_date_string = time_conversion.unix_sec_to_string(
            valid_date_unix_sec, satellite_io.DATE_FORMAT
        )

        unshuffled_file_name = satellite_io.find_file(
            directory_name=unshuffled_dir_name,
            cyclone_id_string=
            shuffled_stx[satellite_utils.CYCLONE_ID_KEY].values[i],
            valid_date_string=valid_date_string,
            raise_error_if_missing=True
        )

        if unshuffled_file_name != last_unshuffled_file_name:
            print('Reading unshuffled data from: "{0:s}"...'.format(
                unshuffled_file_name
            ))
            unshuffled_satellite_table_xarray = satellite_io.read_file(
                unshuffled_file_name
            )
            last_unshuffled_file_name = copy.deepcopy(unshuffled_file_name)

            are_unshuffled_data_norm = not numpy.any(
                shuffled_stx[satellite_utils.BRIGHTNESS_TEMPERATURE_KEY].values
                > 20.
            )

            if are_unshuffled_data_norm and norm_param_table_xarray is not None:
                unshuffled_satellite_table_xarray = normalization.denormalize_data(
                    satellite_table_xarray=unshuffled_satellite_table_xarray,
                    normalization_param_table_xarray=norm_param_table_xarray
                )
                are_unshuffled_data_norm = False

        unshuffled_stx = unshuffled_satellite_table_xarray
        i_new = numpy.where(
            unshuffled_stx.coords[satellite_utils.TIME_DIM].values ==
            valid_time_unix_sec
        )[0][0]

        plot_satellite.plot_data_one_time(
            satellite_table_xarray=unshuffled_satellite_table_xarray,
            time_index=i_new,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            are_data_normalized=are_unshuffled_data_norm,
            output_dir_name=unshuffled_output_dir_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        shuffled_file_name=getattr(INPUT_ARG_OBJECT, SHUFFLED_FILE_ARG_NAME),
        unshuffled_dir_name=getattr(INPUT_ARG_OBJECT, UNSHUFFLED_DIR_ARG_NAME),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        low_res_wavelengths_microns=numpy.array(
            getattr(INPUT_ARG_OBJECT, LOW_RES_WAVELENGTHS_ARG_NAME), dtype=float
        ),
        num_grid_rows_low_res=getattr(INPUT_ARG_OBJECT, NUM_GRID_ROWS_ARG_NAME),
        num_grid_columns_low_res=getattr(
            INPUT_ARG_OBJECT, NUM_GRID_COLUMNS_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
