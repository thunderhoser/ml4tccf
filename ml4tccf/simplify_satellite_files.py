"""Simplifies satellite files by removing most wavelengths and pixels."""

import os
import sys
import argparse
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import satellite_io
import satellite_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
TOLERANCE = 1e-6

NUM_GRID_ROWS_TO_KEEP = 760
NUM_GRID_COLUMNS_TO_KEEP = 1080

GOES_WAVELENGTHS_METRES = 1e-6 * numpy.array([
    3.9, 6.185, 6.95, 7.34, 8.5, 9.61, 10.35, 11.2, 12.3, 13.3
])
HIMAWARI_WAVELENGTHS_METRES = 1e-6 * numpy.array([
    3.9, 6.200, 6.90, 7.30, 8.6, 9.60, 10.40, 11.2, 12.4, 13.3
])

INPUT_DIR_ARG_NAME = 'input_dir_name'
CYCLONE_ID_ARG_NAME = 'cyclone_id_string'
SHUFFLED_FILE_NUMBER_ARG_NAME = 'shuffled_file_number'
WAVELENGTHS_TO_KEEP_ARG_NAME = 'wavelengths_to_keep_microns'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein (containing ALL satellite data) '
    'will be found by `satellite_io.find_file` and read by '
    '`satellite_io.read_file`.'
)
CYCLONE_ID_HELP_STRING = (
    'Will simplify satellite data for this cyclone (format "yyyyBBnn").  If '
    'you want to simplify shuffled data instead, leave this argument alone.'
)
SHUFFLED_FILE_NUMBER_HELP_STRING = (
    'Will simplify shuffled satellite data with this file number (integer).  '
    'If you want to simplify organized data (one file per cyclone-day) '
    'instead, leave this argument alone.'
)
WAVELENGTHS_TO_KEEP_HELP_STRING = (
    'List of wavelengths to keep (for infrared brightness temperature only).'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Files with subset satellite data will be '
    'written here by `satellite_io.write_file`, to exact locations determined '
    'by `satellite_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CYCLONE_ID_ARG_NAME, type=str, required=False, default='',
    help=CYCLONE_ID_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SHUFFLED_FILE_NUMBER_ARG_NAME, type=int, required=False, default=-1,
    help=SHUFFLED_FILE_NUMBER_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + WAVELENGTHS_TO_KEEP_ARG_NAME, type=float, nargs='+', required=True,
    help=WAVELENGTHS_TO_KEEP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_dir_name, cyclone_id_string, shuffled_file_number,
         wavelengths_to_keep_microns, output_dir_name):
    """Simplifies satellite files by removing most wavelengths and pixels.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param cyclone_id_string: Same.
    :param shuffled_file_number: Same.
    :param wavelengths_to_keep_microns: Same.
    :param output_dir_name: Same.
    """

    if cyclone_id_string == '':
        input_file_name = satellite_io.find_shuffled_file(
            directory_name=input_dir_name, file_number=shuffled_file_number,
            raise_error_if_missing=True
        )
        input_file_names = [input_file_name]
    else:
        input_file_names = satellite_io.find_files_one_cyclone(
            directory_name=input_dir_name,
            cyclone_id_string=cyclone_id_string,
            raise_error_if_all_missing=True
        )

    for this_input_file_name in input_file_names:
        print('Reading data from: "{0:s}"...'.format(this_input_file_name))
        satellite_table_xarray = satellite_io.read_file(this_input_file_name)
        tst = satellite_table_xarray

        if numpy.allclose(
                tst.coords[satellite_utils.LOW_RES_WAVELENGTH_DIM].values,
                HIMAWARI_WAVELENGTHS_METRES,
                atol=TOLERANCE
        ):
            tst = tst.assign_coords({
                satellite_utils.LOW_RES_WAVELENGTH_DIM: GOES_WAVELENGTHS_METRES
            })

        satellite_table_xarray = tst
        satellite_table_xarray = satellite_utils.subset_wavelengths(
            satellite_table_xarray=satellite_table_xarray,
            wavelengths_to_keep_microns=wavelengths_to_keep_microns,
            for_high_res=False
        )

        if (
                satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY in
                satellite_table_xarray.variables
        ):
            satellite_table_xarray = satellite_utils.subset_wavelengths(
                satellite_table_xarray=satellite_table_xarray,
                wavelengths_to_keep_microns=numpy.array([]),
                for_high_res=True
            )

        satellite_table_xarray = satellite_utils.subset_grid(
            satellite_table_xarray=satellite_table_xarray,
            num_rows_to_keep=NUM_GRID_ROWS_TO_KEEP,
            num_columns_to_keep=NUM_GRID_COLUMNS_TO_KEEP,
            for_high_res=False
        )

        bad_time_flags = satellite_utils._find_times_with_all_nan_maps(
            satellite_table_xarray
        )
        good_time_indices = numpy.where(numpy.invert(bad_time_flags))[0]

        if numpy.any(bad_time_flags):
            print((
                'REMOVING {0:d} of {1:d} time steps, due to all-NaN maps!'
            ).format(
                numpy.sum(bad_time_flags), len(bad_time_flags)
            ))

        if len(good_time_indices) == 0:
            del satellite_table_xarray
            continue

        satellite_table_xarray = satellite_table_xarray.isel(
            indexers={satellite_utils.TIME_DIM: good_time_indices}
        )

        main_data_dict = {}
        for var_name in satellite_table_xarray.data_vars:
            main_data_dict[var_name] = (
                satellite_table_xarray[var_name].dims,
                satellite_table_xarray[var_name].values
            )

        metadata_dict = {}
        for coord_name in satellite_table_xarray.coords:
            metadata_dict[coord_name] = (
                satellite_table_xarray.coords[coord_name].values
            )

        satellite_table_xarray = xarray.Dataset(
            data_vars=main_data_dict, coords=metadata_dict
        )

        if cyclone_id_string == '':
            this_output_file_name = satellite_io.find_shuffled_file(
                directory_name=output_dir_name,
                file_number=shuffled_file_number,
                raise_error_if_missing=False
            )
        else:
            this_output_file_name = satellite_io.find_file(
                directory_name=output_dir_name,
                cyclone_id_string=cyclone_id_string,
                valid_date_string=
                satellite_io.file_name_to_date(this_input_file_name),
                raise_error_if_missing=False
            )

        print('Writing subset data to: "{0:s}"...\n\n'.format(
            this_output_file_name
        ))
        print(satellite_table_xarray)
        satellite_io.write_file(
            satellite_table_xarray=satellite_table_xarray,
            zarr_file_name=this_output_file_name
        )

        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        cyclone_id_string=getattr(INPUT_ARG_OBJECT, CYCLONE_ID_ARG_NAME),
        shuffled_file_number=getattr(
            INPUT_ARG_OBJECT, SHUFFLED_FILE_NUMBER_ARG_NAME
        ),
        wavelengths_to_keep_microns=numpy.array(
            getattr(INPUT_ARG_OBJECT, WAVELENGTHS_TO_KEEP_ARG_NAME), dtype=float
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
