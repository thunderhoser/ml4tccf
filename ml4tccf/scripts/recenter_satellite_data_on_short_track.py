"""Recenters satellite data, using estimated TC centers from short-track."""

import glob
import pickle
import argparse
import warnings
import numpy
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking
from ml4tccf.io import satellite_io
from ml4tccf.utils import satellite_utils
from ml4tccf.machine_learning import data_augmentation

MINUTES_TO_SECONDS = 60

INPUT_SATELLITE_DIR_ARG_NAME = 'input_satellite_dir_name'
SHORT_TRACK_DIR_ARG_NAME = 'input_short_track_dir_name'
CYCLONE_ID_ARG_NAME = 'cyclone_id_string'
OUTPUT_SATELLITE_DIR_ARG_NAME = 'output_satellite_dir_name'

INPUT_SATELLITE_DIR_HELP_STRING = (
    'Path to input directory, containing satellite data with bad centers based '
    'on linear extrapolation.  Files therein will be found by '
    '`satellite_io.find_file` and read by `satellite_io.read_file`.'
)
SHORT_TRACK_DIR_HELP_STRING = (
    'Path to directory with short-track files.  Files therein will be found by '
    '`_find_short_track_file`, and read by `_read_short_track_file`, both '
    'found in this script.'
)
CYCLONE_ID_HELP_STRING = 'Will recenter satellite data only for this cyclone.'
OUTPUT_SATELLITE_DIR_HELP_STRING = (
    'Path to output directory.  Recentered data will be written by '
    '`satellite_io.write_file`, to exact locations in this directory '
    'determined by `satellite_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_SATELLITE_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_SATELLITE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SHORT_TRACK_DIR_ARG_NAME, type=str, required=True,
    help=SHORT_TRACK_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CYCLONE_ID_ARG_NAME, type=str, required=True,
    help=CYCLONE_ID_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_SATELLITE_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_SATELLITE_DIR_HELP_STRING
)


def _find_short_track_file(directory_name, cyclone_id_string,
                           target_time_unix_sec):
    """Finds Pickle file containing short-track data.

    :param directory_name: Path to directory.
    :param cyclone_id_string: Cyclone ID.
    :param target_time_unix_sec: Target time, i.e., time at which short-track
        center is desired.
    :return: pickle_file_name: Path to short-track file.
    """

    fake_cyclone_id_string = '{0:s}{1:s}'.format(
        cyclone_id_string[4:].lower(),
        cyclone_id_string[:4]
    )

    file_time_unix_sec = number_rounding.floor_to_nearest(
        target_time_unix_sec - 25 * MINUTES_TO_SECONDS,
        10 * MINUTES_TO_SECONDS
    )
    file_time_unix_sec = int(file_time_unix_sec)
    file_time_string = time_conversion.unix_sec_to_string(
        file_time_unix_sec, '%Y-%m-%d-%H-%M-%S'
    )

    pickle_file_pattern = (
        '{0:s}/a{1:s}/storm_track_interp_a{1:s}_*_{2:s}.pkl'
    ).format(
        directory_name,
        fake_cyclone_id_string,
        file_time_string
    )
    pickle_file_names = glob.glob(pickle_file_pattern)

    if len(pickle_file_names) == 0:
        pickle_file_pattern = (
            '{0:s}/a{1:s}/storm_track_interp_a{1:s}_{2:s}.pkl'
        ).format(
            directory_name,
            fake_cyclone_id_string,
            file_time_string
        )
        pickle_file_names = glob.glob(pickle_file_pattern)

    if len(pickle_file_names) == 0:
        pickle_file_pattern = (
            '{0:s}/{1:s}/storm_track_interp_a{1:s}_*_{2:s}.pkl'
        ).format(
            directory_name,
            fake_cyclone_id_string,
            file_time_string
        )
        pickle_file_names = glob.glob(pickle_file_pattern)

    if len(pickle_file_names) != 1:
        warning_string = (
            'POTENTIAL ERROR: Cannot find file with pattern: "{0:s}"'
        ).format(pickle_file_pattern)

        warnings.warn(warning_string)
        return None

    return pickle_file_names[0]


def _read_short_track_file(pickle_file_name, target_time_unix_sec):
    """Reads TC-center location from short-track file.

    :param pickle_file_name: Path to input file.
    :param target_time_unix_sec: Desired time step.
    :return: latitude_deg_n: Latitude of TC center.
    :return: longitude_deg_e: Longitude of TC center.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    short_track_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    valid_time_strings = [
        t.strftime('%Y-%m-%d-%H%M%S')
        for t in short_track_dict['st_one_sec_time']
    ]
    valid_times_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, '%Y-%m-%d-%H%M%S')
        for t in valid_time_strings
    ], dtype=int)

    good_indices = numpy.where(valid_times_unix_sec == target_time_unix_sec)[0]
    if len(good_indices) == 0:
        warning_string = (
            'POTENTIAL ERROR: Cannot find valid time {0:s} in file "{1:s}".'
        ).format(
            time_conversion.unix_sec_to_string(
                target_time_unix_sec, '%Y-%m-%d-%H%M'
            ),
            pickle_file_name
        )

        warnings.warn(warning_string)
        return None, None

    if len(good_indices) > 1:
        warning_string = (
            'POTENTIAL ERROR: Found {0:d} entries with valid time {1:s} in '
            'file "{2:s}".'
        ).format(
            len(good_indices),
            time_conversion.unix_sec_to_string(
                target_time_unix_sec, '%Y-%m-%d-%H%M'
            ),
            pickle_file_name
        )

        warnings.warn(warning_string)

        for k in good_indices:
            print('{0:.4f} deg N, {1:.4f} deg E'.format(
                short_track_dict['st_one_sec_lats'][k],
                short_track_dict['st_one_sec_lond'][k]
            ))

    good_index = good_indices[0]

    latitude_deg_n = short_track_dict['st_one_sec_lats'][good_index]
    error_checking.assert_is_valid_latitude(latitude_deg_n, allow_nan=False)

    longitude_deg_e = short_track_dict['st_one_sec_lond'][good_index]
    longitude_deg_e = lng_conversion.convert_lng_positive_in_west(
        longitude_deg_e, allow_nan=False
    )

    return latitude_deg_n, longitude_deg_e


def _run(input_satellite_dir_name, short_track_dir_name, cyclone_id_string,
         output_satellite_dir_name):
    """Recenters satellite data, using estimated TC centers from short-track.

    This is effectively the main method.

    :param input_satellite_dir_name: See documentation at top of this script.
    :param short_track_dir_name: Same.
    :param cyclone_id_string: Same.
    :param output_satellite_dir_name: Same.
    """

    input_satellite_file_names = satellite_io.find_files_one_cyclone(
        directory_name=input_satellite_dir_name,
        cyclone_id_string=cyclone_id_string,
        raise_error_if_all_missing=True
    )

    date_strings = [
        satellite_io.file_name_to_date(f) for f in input_satellite_file_names
    ]

    if cyclone_id_string == '2024AL12':
        input_satellite_file_names = [
            input_satellite_file_names[date_strings.index('2024-10-06')]
        ]
    if cyclone_id_string == '2024AL14':
        input_satellite_file_names = [
            input_satellite_file_names[date_strings.index('2024-10-11')]
        ]

    # if cyclone_id_string == '2024AL09':
    #     input_satellite_file_names = [input_satellite_file_names[-1]]
    # elif cyclone_id_string == '2024AL11':
    #     input_satellite_file_names = input_satellite_file_names[-2:]
    # elif cyclone_id_string == '2024AL12':
    #     input_satellite_file_names = [input_satellite_file_names[-1]]
    # elif cyclone_id_string == '2024AL13':
    #     input_satellite_file_names = [input_satellite_file_names[-1]]
    # elif cyclone_id_string == '2024AL15':
    #     input_satellite_file_names = [input_satellite_file_names[-1]]
    # else:
    #     input_satellite_file_names = []

    num_files = len(input_satellite_file_names)

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(
            input_satellite_file_names[i]
        ))
        satellite_table_xarray = satellite_io.read_file(
            input_satellite_file_names[i]
        )
        stx = satellite_table_xarray

        target_times_unix_sec = stx.coords[satellite_utils.TIME_DIM].values
        num_times = len(target_times_unix_sec)

        half_num_grid_rows = (
            len(stx.coords[satellite_utils.LOW_RES_ROW_DIM].values) // 2
        )
        half_num_grid_columns = (
            len(stx.coords[satellite_utils.LOW_RES_COLUMN_DIM].values) // 2
        )

        brightness_temp_matrix_kelvins = (
            stx[satellite_utils.BRIGHTNESS_TEMPERATURE_KEY].values
        )
        grid_latitude_matrix_deg_n = (
            stx[satellite_utils.LATITUDE_LOW_RES_KEY].values
        )
        grid_longitude_matrix_deg_e = (
            stx[satellite_utils.LONGITUDE_LOW_RES_KEY].values
        )

        keep_time_flags = numpy.full(num_times, False, dtype=bool)

        for j in range(num_times):
            short_track_file_name = _find_short_track_file(
                directory_name=short_track_dir_name,
                cyclone_id_string=cyclone_id_string,
                target_time_unix_sec=target_times_unix_sec[j]
            )
            if short_track_file_name is None:
                continue

            print('Reading data from: "{0:s}"...'.format(short_track_file_name))
            short_track_latitude_deg_n, short_track_longitude_deg_e = (
                _read_short_track_file(
                    pickle_file_name=short_track_file_name,
                    target_time_unix_sec=target_times_unix_sec[j]
                )
            )

            if short_track_latitude_deg_n is None:
                continue

            keep_time_flags[j] = True

            # TODO: Relevant code starts here.
            grid_latitudes_deg_n = (
                stx[satellite_utils.LATITUDE_LOW_RES_KEY].values[j, :]
            )
            grid_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
                stx[satellite_utils.LONGITUDE_LOW_RES_KEY].values[j, :]
            )

            orig_latitude_deg_n = numpy.mean(
                grid_latitudes_deg_n[
                    (half_num_grid_rows - 1):(half_num_grid_rows + 1)
                ]
            )
            orig_longitude_deg_e = numpy.mean(
                grid_longitudes_deg_e[
                    (half_num_grid_columns - 1):(half_num_grid_columns + 1)
                ]
            )

            print((
                'Recentering from ({0:.4f} deg N, {1:.4f} deg E) to '
                '({2:.4f} deg N, {3:.4f} deg E)...'
            ).format(
                orig_latitude_deg_n,
                orig_longitude_deg_e,
                short_track_latitude_deg_n,
                short_track_longitude_deg_e
            ))

            latitude_diff_deg = numpy.absolute(
                short_track_latitude_deg_n - orig_latitude_deg_n
            )
            if latitude_diff_deg > 5:
                error_string = (
                    'Difference between original ({0:.4f} deg N) and short-'
                    'track ({1:.4f} deg N) latitudes is too big.'
                ).format(
                    orig_latitude_deg_n,
                    short_track_latitude_deg_n
                )

                raise ValueError(error_string)

            longitude_diff_deg = numpy.absolute(
                short_track_longitude_deg_e - orig_longitude_deg_e
            )
            if longitude_diff_deg > 7.5:
                error_string = (
                    'Difference between original ({0:.4f} deg E) and short-'
                    'track ({1:.4f} deg E) longitudes is too big.'
                ).format(
                    orig_longitude_deg_e,
                    short_track_longitude_deg_e
                )

                raise ValueError(error_string)

            latitude_diffs_deg = numpy.absolute(
                grid_latitudes_deg_n - short_track_latitude_deg_n
            )
            assert numpy.min(latitude_diffs_deg) < 0.03
            new_center_row = numpy.argmin(latitude_diffs_deg)

            longitude_diffs_deg = numpy.absolute(
                grid_longitudes_deg_e - short_track_longitude_deg_e
            )
            assert numpy.min(longitude_diffs_deg) < 0.05
            new_center_column = numpy.argmin(longitude_diffs_deg)

            row_translation = half_num_grid_rows - new_center_row
            column_translation = half_num_grid_columns - new_center_column

            this_bt_matrix = brightness_temp_matrix_kelvins[[j], ...]
            _, this_bt_matrix = data_augmentation.augment_data_specific_trans(
                bidirectional_reflectance_matrix=None,
                brightness_temp_matrix_kelvins=this_bt_matrix,
                row_translations_low_res_px=
                numpy.array([row_translation], dtype=int),
                column_translations_low_res_px=
                numpy.array([column_translation], dtype=int),
                sentinel_value=-10.
            )
            brightness_temp_matrix_kelvins[j, ...] = this_bt_matrix[0, ...]

        stx = stx.assign({
            satellite_utils.BRIGHTNESS_TEMPERATURE_KEY: (
                stx[satellite_utils.BRIGHTNESS_TEMPERATURE_KEY].dims,
                brightness_temp_matrix_kelvins
            ),
            satellite_utils.LATITUDE_LOW_RES_KEY: (
                stx[satellite_utils.LATITUDE_LOW_RES_KEY].dims,
                grid_latitude_matrix_deg_n
            ),
            satellite_utils.LONGITUDE_LOW_RES_KEY: (
                stx[satellite_utils.LONGITUDE_LOW_RES_KEY].dims,
                grid_longitude_matrix_deg_e
            )
        })

        stx = stx.isel({
            satellite_utils.TIME_DIM: numpy.where(keep_time_flags)[0]
        })

        this_output_file_name = satellite_io.find_file(
            directory_name=output_satellite_dir_name,
            cyclone_id_string=cyclone_id_string,
            valid_date_string=
            satellite_io.file_name_to_date(input_satellite_file_names[i]),
            raise_error_if_missing=False
        )

        print('Writing new data to: "{0:s}"...'.format(this_output_file_name))
        satellite_io.write_file(
            satellite_table_xarray=stx,
            zarr_file_name=this_output_file_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_satellite_dir_name=getattr(
            INPUT_ARG_OBJECT, INPUT_SATELLITE_DIR_ARG_NAME
        ),
        short_track_dir_name=getattr(
            INPUT_ARG_OBJECT, SHORT_TRACK_DIR_ARG_NAME
        ),
        cyclone_id_string=getattr(INPUT_ARG_OBJECT, CYCLONE_ID_ARG_NAME),
        output_satellite_dir_name=getattr(
            INPUT_ARG_OBJECT, OUTPUT_SATELLITE_DIR_ARG_NAME
        )
    )
