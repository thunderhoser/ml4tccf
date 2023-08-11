"""Subsets shuffled files, keeping only 2-h chunks ending at synoptic time."""

import copy
import argparse
import numpy
import xarray
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from ml4tccf.io import satellite_io
from ml4tccf.utils import satellite_utils

TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H%M%S'

DAYS_TO_SECONDS = 86400
SYNOPTIC_TIMES_SEC_INTO_DAY = numpy.array([0, 21600, 43200, 64800], dtype=int)
NONZERO_LAG_TIMES_SEC = numpy.array([1800, 3600, 5400, 7200], dtype=int)

INPUT_DIR_ARG_NAME = 'input_satellite_dir_name'
FIRST_IN_FILE_NUM_ARG_NAME = 'first_input_file_num'
LAST_IN_FILE_NUM_ARG_NAME = 'last_input_file_num'
NUM_CHUNKS_PER_OUTPUT_ARG_NAME = 'num_chunks_per_output_file'
FIRST_OUT_FILE_NUM_ARG_NAME = 'first_output_file_num'
OUTPUT_DIR_ARG_NAME = 'output_satellite_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing original shuffled satellite files.  '
    'Files therein will be found by `satellite_io.find_shuffled_file` and read '
    'by `satellite_io.read_file`.'
)
FIRST_IN_FILE_NUM_HELP_STRING = 'Number of first input file to process.'
LAST_IN_FILE_NUM_HELP_STRING = 'Number of last input file to process.'
NUM_CHUNKS_PER_OUTPUT_HELP_STRING = 'Number of 2-hour chunks per output file.'
FIRST_OUT_FILE_NUM_HELP_STRING = (
    'Number used to name first output file produced by this script.  For each '
    'successive output file, the number will increment by 1.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Output files will be written here by '
    '`satellite_io.write_file`, to exact locations determined by '
    '`satellite_io.find_shuffled_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_IN_FILE_NUM_ARG_NAME, type=int, required=True,
    help=FIRST_IN_FILE_NUM_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_IN_FILE_NUM_ARG_NAME, type=int, required=True,
    help=LAST_IN_FILE_NUM_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_CHUNKS_PER_OUTPUT_ARG_NAME, type=int, required=True,
    help=NUM_CHUNKS_PER_OUTPUT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_OUT_FILE_NUM_ARG_NAME, type=int, required=True,
    help=FIRST_OUT_FILE_NUM_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_dir_name, first_input_file_num, last_input_file_num,
         num_chunks_per_output_file, first_output_file_num, output_dir_name):
    """Subsets shuffled files, keeping only 2-h chunks ending at synoptic time.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param first_input_file_num: Same.
    :param last_input_file_num: Same.
    :param num_chunks_per_output_file: Same.
    :param first_output_file_num: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    error_checking.assert_is_geq(first_input_file_num, 0)
    error_checking.assert_is_geq(last_input_file_num, first_input_file_num)
    error_checking.assert_is_geq(num_chunks_per_output_file, 10)
    error_checking.assert_is_geq(first_output_file_num, 0)

    # Do actual stuff.
    input_file_nums = numpy.linspace(
        first_input_file_num, last_input_file_num,
        num=last_input_file_num - first_input_file_num + 1,
        dtype=int
    )

    input_file_names = [
        satellite_io.find_shuffled_file(
            directory_name=input_dir_name, file_number=n,
            raise_error_if_missing=True
        ) for n in input_file_nums
    ]

    num_input_files = len(input_file_names)
    num_output_files_written = 0
    output_satellite_table_xarray = None
    num_chunks_in_output_table = 0

    for i in range(num_input_files):
        print('Reading data from: "{0:s}"...'.format(input_file_names[i]))
        this_satellite_table_xarray = satellite_io.read_file(
            input_file_names[i]
        )

        valid_times_unix_sec = (
            this_satellite_table_xarray.coords[satellite_utils.TIME_DIM].values
        )
        valid_time_strings = [
            time_conversion.unix_sec_to_string(t, TIME_FORMAT_FOR_LOG_MESSAGES)
            for t in valid_times_unix_sec
        ]

        valid_times_sec_into_day = numpy.mod(
            valid_times_unix_sec, DAYS_TO_SECONDS
        )
        synoptic_time_indices = numpy.where(numpy.isin(
            element=SYNOPTIC_TIMES_SEC_INTO_DAY,
            test_elements=valid_times_sec_into_day
        ))[0]
        print('Synoptic times in file:\n{0:s}'.format(
            str([valid_time_strings[k] for k in synoptic_time_indices])
        ))

        good_time_indices = []

        for j in synoptic_time_indices:
            predictor_times_needed_unix_sec = (
                valid_times_unix_sec[j] - NONZERO_LAG_TIMES_SEC
            )
            found_flags = numpy.isin(
                element=valid_times_unix_sec,
                test_elements=predictor_times_needed_unix_sec
            )
            if not numpy.all(found_flags):
                continue

            good_time_indices.append(j)

            for this_time_unix_sec in predictor_times_needed_unix_sec:
                this_index = numpy.where(
                    valid_times_unix_sec == this_time_unix_sec
                )[0][0]
                good_time_indices.append(this_index)

        good_time_indices = numpy.array(good_time_indices, dtype=int)
        this_satellite_table_xarray = this_satellite_table_xarray.isel(
            indexers={satellite_utils.TIME_DIM: good_time_indices}
        )

        print('There are {0:d} time steps left in the file:\n{1:s}'.format(
            len(good_time_indices),
            str([valid_time_strings[k] for k in good_time_indices])
        ))

        if output_satellite_table_xarray is None:
            output_satellite_table_xarray = copy.deepcopy(
                this_satellite_table_xarray
            )
        else:
            output_satellite_table_xarray = satellite_utils.concat_over_time(
                satellite_tables_xarray=[
                    output_satellite_table_xarray, this_satellite_table_xarray
                ],
                allow_different_cyclones=True
            )

        num_chunks_in_output_table += 1
        if (
                num_chunks_in_output_table < num_chunks_per_output_file
                and not i == num_input_files - 1
        ):
            continue

        main_data_dict = {}
        for var_name in output_satellite_table_xarray.data_vars:
            main_data_dict[var_name] = (
                output_satellite_table_xarray[var_name].dims,
                output_satellite_table_xarray[var_name].values
            )

        metadata_dict = {}
        for coord_name in output_satellite_table_xarray.coords:
            metadata_dict[coord_name] = (
                output_satellite_table_xarray.coords[coord_name].values
            )

        output_satellite_table_xarray = xarray.Dataset(
            data_vars=main_data_dict, coords=metadata_dict
        )

        output_file_name = satellite_io.find_shuffled_file(
            directory_name=output_dir_name,
            file_number=num_output_files_written + first_output_file_num,
            raise_error_if_missing=False
        )

        print('Writing subset data to: "{0:s}"...'.format(output_file_name))
        satellite_io.write_file(
            satellite_table_xarray=output_satellite_table_xarray,
            zarr_file_name=output_file_name
        )

        num_output_files_written += 1
        output_satellite_table_xarray = None
        num_chunks_in_output_table = 0


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_input_file_num=getattr(
            INPUT_ARG_OBJECT, FIRST_IN_FILE_NUM_ARG_NAME
        ),
        last_input_file_num=getattr(
            INPUT_ARG_OBJECT, LAST_IN_FILE_NUM_ARG_NAME
        ),
        num_chunks_per_output_file=getattr(
            INPUT_ARG_OBJECT, NUM_CHUNKS_PER_OUTPUT_ARG_NAME
        ),
        first_output_file_num=getattr(
            INPUT_ARG_OBJECT, FIRST_OUT_FILE_NUM_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
