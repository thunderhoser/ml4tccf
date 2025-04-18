"""Shuffles satellite files to make training more efficient."""

import os
import copy
import argparse
import warnings
import numpy
import xarray
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import error_checking
from ml4tccf.io import satellite_io
from ml4tccf.utils import misc_utils
from ml4tccf.utils import satellite_utils

TOLERANCE = 1e-6
TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H%M%S'

INPUT_TIME_INTERVAL_MINUTES = 30
DAYS_TO_MINUTES = 1440
DAYS_TO_SECONDS = 86400
MINUTES_TO_SECONDS = 60

GOES_WAVELENGTHS_METRES = 1e-6 * numpy.array([
    3.9, 6.185, 6.95, 7.34, 8.5, 9.61, 10.35, 11.2, 12.3, 13.3
])
HIMAWARI_WAVELENGTHS_METRES = 1e-6 * numpy.array([
    3.9, 6.200, 6.90, 7.30, 8.6, 9.60, 10.40, 11.2, 12.4, 13.3
])

INPUT_DIR_ARG_NAME = 'input_satellite_dir_name'
NUM_CHUNKS_PER_INPUT_ARG_NAME = 'num_chunks_per_input_file'
NUM_CHUNKS_PER_OUTPUT_ARG_NAME = 'num_chunks_per_output_file'
CHUNK_PREBUFFER_ARG_NAME = 'chunk_prebuffer_minutes'
TIME_INTERVAL_ARG_NAME = 'output_time_interval_minutes'
YEARS_ARG_NAME = 'years'
CYCLONE_IDS_ARG_NAME = 'cyclone_id_strings'
FIRST_OUT_FILE_NUM_ARG_NAME = 'first_output_file_num'
OUTPUT_DIR_ARG_NAME = 'output_satellite_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing unshuffled satellite files (one file '
    'per cyclone-day).  Files therein will be found by '
    '`satellite_io.find_file` and read by `satellite_io.read_file`.'
)
NUM_CHUNKS_PER_INPUT_HELP_STRING = (
    'Each input file (containing one cyclone-day) will be split into this many '
    'chunks, with each chunk being written to a different output file.'
)
NUM_CHUNKS_PER_OUTPUT_HELP_STRING = 'Number of chunks per output file.'
CHUNK_PREBUFFER_HELP_STRING = (
    'Chunk prebuffer.  The beginning of every chunk will get this many '
    'additional time steps at the beginning, so that in the shuffled dataset, '
    'if we want predictors for target time t, the lagged predictors are always '
    'available in the same file as t.  Hence, {0:s} should be equivalent to '
    'the longest lag time in the model you wish to train.'
).format(
    CHUNK_PREBUFFER_ARG_NAME
)
TIME_INTERVAL_HELP_STRING = (
    'Time interval for output files.  This must be a multiple of 30 minutes.'
)
YEARS_HELP_STRING = (
    'Will shuffle cyclones in these years (list).  If you would rather specify '
    'cyclone IDs, leave this argument alone.'
)
CYCLONE_IDS_HELP_STRING = (
    'Will shuffle cyclones in this list (format "yyyyBBnn").  If you would '
    'rather specify years, leave this argument alone.'
)
FIRST_OUT_FILE_NUM_HELP_STRING = (
    'Number used to name first output file produced by this script.  For each '
    'successive output file, the number will increment by 1.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Shuffled files will be written here by '
    '`satellite_io.write_file`, to exact locations determined by '
    '`satellite_io.find_shuffled_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_CHUNKS_PER_INPUT_ARG_NAME, type=int, required=True,
    help=NUM_CHUNKS_PER_INPUT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_CHUNKS_PER_OUTPUT_ARG_NAME, type=int, required=True,
    help=NUM_CHUNKS_PER_OUTPUT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CHUNK_PREBUFFER_ARG_NAME, type=int, required=True,
    help=CHUNK_PREBUFFER_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TIME_INTERVAL_ARG_NAME, type=int, required=True,
    help=TIME_INTERVAL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + YEARS_ARG_NAME, type=int, nargs='+', required=False, default=[-1],
    help=YEARS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CYCLONE_IDS_ARG_NAME, type=str, nargs='+', required=False,
    default=[''], help=CYCLONE_IDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_OUT_FILE_NUM_ARG_NAME, type=int, required=True,
    help=FIRST_OUT_FILE_NUM_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _get_time_range_by_chunk(num_chunks_per_input_file,
                             output_time_interval_minutes):
    """Returns time range for each chunk.

    N = number of chunks per input file

    :param num_chunks_per_input_file: See documentation at top of file.
    :param output_time_interval_minutes: Same.
    :return: start_time_by_chunk_sec_into_day: length-N numpy array of start
        times (unit = seconds into day).
    :return: end_time_by_chunk_sec_into_day: length-N numpy array of end
        times (unit = seconds into day).
    """

    start_time_by_chunk_sec_into_day = numpy.linspace(
        0, DAYS_TO_SECONDS, num=num_chunks_per_input_file + 1, dtype=float
    )[:-1]
    end_time_by_chunk_sec_into_day = numpy.linspace(
        0, DAYS_TO_SECONDS, num=num_chunks_per_input_file + 1, dtype=float
    )[1:]

    output_time_interval_sec = MINUTES_TO_SECONDS * output_time_interval_minutes
    start_time_by_chunk_sec_into_day = number_rounding.floor_to_nearest(
        start_time_by_chunk_sec_into_day, output_time_interval_sec
    )
    start_time_by_chunk_sec_into_day = numpy.round(
        start_time_by_chunk_sec_into_day
    ).astype(int)

    end_time_by_chunk_sec_into_day = number_rounding.ceiling_to_nearest(
        end_time_by_chunk_sec_into_day, output_time_interval_sec
    )
    end_time_by_chunk_sec_into_day = numpy.round(
        end_time_by_chunk_sec_into_day
    ).astype(int)

    start_time_by_chunk_sec_into_day = numpy.minimum(
        start_time_by_chunk_sec_into_day, DAYS_TO_SECONDS
    )
    end_time_by_chunk_sec_into_day = numpy.minimum(
        end_time_by_chunk_sec_into_day, DAYS_TO_SECONDS
    )
    start_time_by_chunk_sec_into_day[1:] = numpy.maximum(
        start_time_by_chunk_sec_into_day[1:],
        end_time_by_chunk_sec_into_day[:-1]
    )

    good_indices = numpy.where(
        end_time_by_chunk_sec_into_day > start_time_by_chunk_sec_into_day
    )[0]
    start_time_by_chunk_sec_into_day = start_time_by_chunk_sec_into_day[
        good_indices
    ]
    end_time_by_chunk_sec_into_day = end_time_by_chunk_sec_into_day[
        good_indices
    ]

    return start_time_by_chunk_sec_into_day, end_time_by_chunk_sec_into_day


def _run(input_dir_name, num_chunks_per_input_file, num_chunks_per_output_file,
         chunk_prebuffer_minutes, output_time_interval_minutes,
         years, cyclone_id_strings, first_output_file_num, output_dir_name):
    """Shuffles satellite files to make training more efficient.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param num_chunks_per_input_file: Same.
    :param num_chunks_per_output_file: Same.
    :param chunk_prebuffer_minutes: Same.
    :param output_time_interval_minutes: Same.
    :param years: Same.
    :param cyclone_id_strings: Same.
    :param first_output_file_num: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    if len(years) == 1 and years[0] < 1900:
        years = None
    if len(cyclone_id_strings) == 1 and cyclone_id_strings[0] == '':
        cyclone_id_strings = None

    assert not (years is None and cyclone_id_strings is None)

    error_checking.assert_is_geq(num_chunks_per_input_file, 2)
    error_checking.assert_is_geq(num_chunks_per_output_file, 2)
    error_checking.assert_is_geq(first_output_file_num, 0)

    error_checking.assert_is_geq(
        output_time_interval_minutes, INPUT_TIME_INTERVAL_MINUTES
    )
    error_checking.assert_equals(
        numpy.mod(output_time_interval_minutes, INPUT_TIME_INTERVAL_MINUTES),
        0
    )

    error_checking.assert_is_geq(chunk_prebuffer_minutes, 0)
    error_checking.assert_equals(
        numpy.mod(chunk_prebuffer_minutes, INPUT_TIME_INTERVAL_MINUTES),
        0
    )

    (
        start_time_by_chunk_sec_into_day, end_time_by_chunk_sec_into_day
    ) = _get_time_range_by_chunk(
        num_chunks_per_input_file=num_chunks_per_input_file,
        output_time_interval_minutes=output_time_interval_minutes
    )

    num_chunks_per_input_file = len(start_time_by_chunk_sec_into_day)
    error_checking.assert_is_geq(num_chunks_per_input_file, 2)

    num_minutes_per_chunk = (
        float(DAYS_TO_MINUTES - INPUT_TIME_INTERVAL_MINUTES) /
        num_chunks_per_input_file
    )
    num_time_steps_per_output_chunk = 1 + int(numpy.ceil(
        num_minutes_per_chunk / output_time_interval_minutes
    ))
    num_time_steps_per_output_file = (
        num_time_steps_per_output_chunk * num_chunks_per_output_file
    )
    error_checking.assert_is_leq(num_time_steps_per_output_file, 200)

    # Do actual stuff.
    if cyclone_id_strings is None:
        cyclone_id_strings = satellite_io.find_cyclones(
            directory_name=input_dir_name, raise_error_if_all_missing=True
        )
        cyclone_years = numpy.array(
            [misc_utils.parse_cyclone_id(c)[0] for c in cyclone_id_strings],
            dtype=int
        )

        good_flags = numpy.array(
            [c in years for c in cyclone_years], dtype=float
        )
        good_indices = numpy.where(good_flags)[0]
        cyclone_id_strings = [cyclone_id_strings[k] for k in good_indices]
    else:
        for this_cyclone_id_string in cyclone_id_strings:
            these_file_names = satellite_io.find_files_one_cyclone(
                directory_name=input_dir_name,
                cyclone_id_string=this_cyclone_id_string,
                raise_error_if_all_missing=False
            )

            if len(these_file_names) == 0:
                cyclone_id_strings.remove(this_cyclone_id_string)

    cyclone_id_strings.sort()

    input_file_names = []
    for this_id_string in cyclone_id_strings:
        input_file_names += satellite_io.find_files_one_cyclone(
            directory_name=input_dir_name, cyclone_id_string=this_id_string,
            raise_error_if_all_missing=True
        )

    cyclone_id_string_by_input_file = [
        satellite_io.file_name_to_cyclone_id(f) for f in input_file_names
    ]
    valid_date_string_by_input_file = [
        satellite_io.file_name_to_date(f) for f in input_file_names
    ]

    num_input_files = len(input_file_names)
    is_input_chunk_processed_matrix = numpy.full(
        (num_input_files, num_chunks_per_input_file), False, dtype=bool
    )

    num_output_files_written = 0
    output_satellite_table_xarray = None
    num_chunks_in_output_table = 0
    chunk_prebuffer_seconds = chunk_prebuffer_minutes * MINUTES_TO_SECONDS

    while not numpy.all(is_input_chunk_processed_matrix):
        unprocessed_indices_2d = numpy.where(
            is_input_chunk_processed_matrix == False
        )
        unprocessed_indices_linear = numpy.ravel_multi_index(
            multi_index=unprocessed_indices_2d,
            dims=is_input_chunk_processed_matrix.shape
        )

        working_index_linear = numpy.random.choice(
            unprocessed_indices_linear, size=1, replace=False
        )
        working_indices_2d = numpy.unravel_index(
            indices=working_index_linear,
            shape=is_input_chunk_processed_matrix.shape
        )
        is_input_chunk_processed_matrix[working_indices_2d] = True

        print('Reading data from: "{0:s}"...'.format(
            input_file_names[working_indices_2d[0][0]]
        ))
        this_satellite_table_xarray = satellite_io.read_file(
            input_file_names[working_indices_2d[0][0]]
        )
        this_satellite_table_xarray = satellite_utils.subset_wavelengths(
            satellite_table_xarray=this_satellite_table_xarray,
            wavelengths_to_keep_microns=numpy.array([]),
            for_high_res=True
        )

        this_date_unix_sec = time_conversion.string_to_unix_sec(
            valid_date_string_by_input_file[working_indices_2d[0][0]],
            satellite_io.DATE_FORMAT
        )
        this_start_time_unix_sec = (
            this_date_unix_sec +
            start_time_by_chunk_sec_into_day[working_indices_2d[1][0]]
        )
        this_end_time_unix_sec = (
            this_date_unix_sec +
            end_time_by_chunk_sec_into_day[working_indices_2d[1][0]]
        )

        this_start_time_unix_sec -= chunk_prebuffer_seconds

        if this_start_time_unix_sec < this_date_unix_sec:
            previous_date_file_name = satellite_io.find_file(
                directory_name=input_dir_name,
                cyclone_id_string=
                cyclone_id_string_by_input_file[working_indices_2d[0][0]],
                valid_date_string=time_conversion.unix_sec_to_string(
                    this_date_unix_sec - DAYS_TO_SECONDS,
                    satellite_io.DATE_FORMAT
                ),
                raise_error_if_missing=False
            )

            if os.path.isdir(previous_date_file_name):
                print('Reading data from previous date: "{0:s}"...'.format(
                    previous_date_file_name
                ))
                prev_satellite_table_xarray = satellite_io.read_file(
                    previous_date_file_name
                )
                prev_satellite_table_xarray = (
                    satellite_utils.subset_wavelengths(
                        satellite_table_xarray=prev_satellite_table_xarray,
                        wavelengths_to_keep_microns=numpy.array([]),
                        for_high_res=True
                    )
                )
                this_satellite_table_xarray = satellite_utils.concat_over_time(
                    satellite_tables_xarray=
                    [prev_satellite_table_xarray, this_satellite_table_xarray],
                    allow_different_cyclones=False
                )

        print('Subsetting chunk from {0:s} to {1:s}...'.format(
            time_conversion.unix_sec_to_string(
                this_start_time_unix_sec, TIME_FORMAT_FOR_LOG_MESSAGES
            ),
            time_conversion.unix_sec_to_string(
                this_end_time_unix_sec, TIME_FORMAT_FOR_LOG_MESSAGES
            )
        ))
        this_satellite_table_xarray = (
            satellite_utils.subset_to_multiple_time_windows(
                satellite_table_xarray=this_satellite_table_xarray,
                start_times_unix_sec=
                numpy.array([this_start_time_unix_sec], dtype=int),
                end_times_unix_sec=
                numpy.array([this_end_time_unix_sec], dtype=int)
            )
        )

        tst = this_satellite_table_xarray
        print('There are {0:d} time steps left in the chunk.'.format(
            len(tst.coords[satellite_utils.TIME_DIM].values)
        ))

        print('Subsetting chunk to {0:d}-min time intervals...'.format(
            output_time_interval_minutes
        ))
        these_desired_times_unix_sec = time_periods.range_and_interval_to_list(
            start_time_unix_sec=this_start_time_unix_sec,
            end_time_unix_sec=this_end_time_unix_sec,
            time_interval_sec=MINUTES_TO_SECONDS * output_time_interval_minutes,
            include_endpoint=True
        )
        found_flags = numpy.isin(
            element=these_desired_times_unix_sec,
            test_elements=this_satellite_table_xarray.coords[
                satellite_utils.TIME_DIM
            ].values
        )

        if not numpy.any(found_flags):
            warnings.warn(
                'POTENTIAL ERROR: There are 0 time steps left in the chunk.'
            )
            continue

        these_desired_times_unix_sec = these_desired_times_unix_sec[found_flags]
        this_satellite_table_xarray = this_satellite_table_xarray.sel(
            indexers={satellite_utils.TIME_DIM: these_desired_times_unix_sec}
        )

        tst = this_satellite_table_xarray
        print('There are {0:d} time steps left in the chunk.'.format(
            len(tst.coords[satellite_utils.TIME_DIM].values)
        ))

        if numpy.allclose(
                tst.coords[satellite_utils.LOW_RES_WAVELENGTH_DIM].values,
                HIMAWARI_WAVELENGTHS_METRES,
                atol=TOLERANCE
        ):
            tst = tst.assign_coords({
                satellite_utils.LOW_RES_WAVELENGTH_DIM: GOES_WAVELENGTHS_METRES
            })

        this_satellite_table_xarray = tst

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
                and not numpy.all(is_input_chunk_processed_matrix)
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

        print('Writing shuffled data to: "{0:s}"...'.format(output_file_name))
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
        num_chunks_per_input_file=getattr(
            INPUT_ARG_OBJECT, NUM_CHUNKS_PER_INPUT_ARG_NAME
        ),
        num_chunks_per_output_file=getattr(
            INPUT_ARG_OBJECT, NUM_CHUNKS_PER_OUTPUT_ARG_NAME
        ),
        chunk_prebuffer_minutes=getattr(
            INPUT_ARG_OBJECT, CHUNK_PREBUFFER_ARG_NAME
        ),
        output_time_interval_minutes=getattr(
            INPUT_ARG_OBJECT, TIME_INTERVAL_ARG_NAME
        ),
        years=numpy.array(getattr(INPUT_ARG_OBJECT, YEARS_ARG_NAME), dtype=int),
        cyclone_id_strings=getattr(INPUT_ARG_OBJECT, CYCLONE_IDS_ARG_NAME),
        first_output_file_num=getattr(
            INPUT_ARG_OBJECT, FIRST_OUT_FILE_NUM_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
