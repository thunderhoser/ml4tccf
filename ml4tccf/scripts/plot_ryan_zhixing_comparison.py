"""Plots comparison of estimated tracks from Ryan and Zhixing."""

import csv
import glob
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from ml4tccf.io import border_io
from ml4tccf.utils import misc_utils
from ml4tccf.plotting import plotting_utils

HOURS_TO_SECONDS = 3600
COLOUR_BAR_TIME_FORMAT = '%HZ %b %-d'

DATE_PATTERN_STRING = '[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]'
HOUR_MINUTE_PATTERN_STRING = '[0-2][0-9][0-5][0-9]'

RYAN_MARKER_SIZE = 36
RYAN_MARKER_TYPE = 's'
RYAN_MARKER_EDGE_WIDTH = 1.5
RYAN_MARKER_EDGE_COLOUR = numpy.full(3, 0.)

ZHIXING_MARKER_SIZE = 36
ZHIXING_MARKER_TYPE = 'o'
ZHIXING_MARKER_EDGE_WIDTH = 1.5
ZHIXING_MARKER_EDGE_COLOUR = numpy.full(3, 152. / 255)

BEST_TRACK_MARKER_SIZE = 60
BEST_TRACK_MARKER_TYPE = '*'
BEST_TRACK_MARKER_EDGE_WIDTH = 0.
BEST_TRACK_MARKER_EDGE_COLOUR = numpy.full(3, 152. / 255)

TICK_LABEL_FONT_SIZE = 40
COLOUR_MAP_OBJECT = pyplot.get_cmap('gist_ncar')

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

RYAN_DIR_ARG_NAME = 'input_ryan_dir_name'
ZHIXING_DIR_ARG_NAME = 'input_zhixing_dir_name'
RAW_BT_FILE_ARG_NAME = 'input_raw_best_track_file_name'
CYCLONE_ID_ARG_NAME = 'cyclone_id_string'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

RYAN_DIR_HELP_STRING = (
    'Path to directory with estimates from Ryan -- stored as CSV files with '
    'names like "<bbnnyyyy>_<yyyymmdd>_<HHMM>.txt", where <bbnnyyyy> is the '
    'cyclone ID.'
)
ZHIXING_DIR_HELP_STRING = (
    'Path to directory with estimates from Zhixing -- stored as CSV files with '
    'names like "<bbnnyyyy>_<yyyymmdd>_<HHMM>.txt", where <bbnnyyyy> is the '
    'cyclone ID.'
)
RAW_BT_FILE_HELP_STRING = (
    'Path to file with raw best tracks for the same cyclone.  Should be named '
    'like "b<bbnnyyyy>.dat".'
)
CYCLONE_ID_HELP_STRING = 'Cyclone ID in the usual format (yyyynnBB).'
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Comparison figure will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + RYAN_DIR_ARG_NAME, type=str, required=True,
    help=RYAN_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ZHIXING_DIR_ARG_NAME, type=str, required=True,
    help=ZHIXING_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + RAW_BT_FILE_ARG_NAME, type=str, required=True,
    help=RAW_BT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CYCLONE_ID_ARG_NAME, type=str, required=True,
    help=CYCLONE_ID_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _read_raw_best_track_file(csv_file_name, cyclone_id_string):
    """Reads raw best tracks from CSV file.

    P = number of points in track

    :param csv_file_name: Path to input file.
    :param cyclone_id_string: ID of expected cyclone.
    :return: bt_latitudes_deg_n: length-P numpy array of latitudes (deg north).
    :return: bt_longitudes_deg_e: length-P numpy array of longitudes (deg east).
    :return: bt_times_unix_sec: length-P numpy array of times.
    """

    year, basin_id_string, cyclone_number = misc_utils.parse_cyclone_id(
        cyclone_id_string
    )

    csv_file_handle = open(csv_file_name, 'r')
    bt_latitudes_deg_n = []
    bt_longitudes_deg_e = []
    bt_times_unix_sec = []

    for this_line in csv_file_handle.readlines():
        these_words = this_line.split(',')
        these_words = [w.strip() for w in these_words]

        assert these_words[0].upper() == basin_id_string
        assert int(these_words[1]) == cyclone_number
        assert these_words[4].upper() == 'BEST'

        this_time_unix_sec = time_conversion.string_to_unix_sec(
            these_words[2], '%Y%m%d%H'
        )

        this_latitude_string = these_words[6]
        assert (
            this_latitude_string.endswith('N') or
            this_latitude_string.endswith('S')
        )
        this_latitude_deg_n = int(this_latitude_string[:-1])
        if this_latitude_string.endswith('S'):
            this_latitude_deg_n *= -1.

        this_longitude_string = these_words[7]
        assert (
            this_longitude_string.endswith('E') or
            this_longitude_string.endswith('W')
        )
        this_longitude_deg_e = int(this_longitude_string[:-1])
        if this_longitude_string.endswith('W'):
            this_longitude_deg_e *= -1.

        bt_latitudes_deg_n.append(this_latitude_deg_n)
        bt_longitudes_deg_e.append(this_longitude_deg_e)
        bt_times_unix_sec.append(this_time_unix_sec)

    bt_latitudes_deg_n = numpy.array(bt_latitudes_deg_n, dtype=float)
    error_checking.assert_is_valid_lat_numpy_array(
        bt_latitudes_deg_n, allow_nan=False
    )

    bt_longitudes_deg_e = numpy.array(bt_longitudes_deg_e, dtype=float)
    bt_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
        bt_longitudes_deg_e, allow_nan=False
    )

    bt_times_unix_sec = numpy.array(bt_times_unix_sec, dtype=int)

    return bt_latitudes_deg_n, bt_longitudes_deg_e, bt_times_unix_sec


def _run(ryan_dir_name, zhixing_dir_name, raw_best_track_file_name,
         cyclone_id_string, output_file_name):
    """Plots comparison of estimated tracks from Ryan and Zhixing.

    This is effectively the main method.

    :param ryan_dir_name: See documentation at top of this script.
    :param zhixing_dir_name: Same.
    :param raw_best_track_file_name: Same.
    :param cyclone_id_string: Same.
    :param output_file_name: Same.
    """

    year, basin_id_string, cyclone_number = misc_utils.parse_cyclone_id(
        cyclone_id_string
    )
    fake_cyclone_id_string = '{0:s}{1:02d}{2:04d}'.format(
        basin_id_string.lower(),
        cyclone_number,
        year
    )

    ryan_file_pattern = '{0:s}/{1:s}_{2:s}_{3:s}.txt'.format(
        ryan_dir_name,
        fake_cyclone_id_string,
        DATE_PATTERN_STRING,
        HOUR_MINUTE_PATTERN_STRING
    )
    ryan_file_names = glob.glob(ryan_file_pattern)

    if len(ryan_file_names) == 0:
        error_string = (
            'Could not find any files with the following pattern:\n{0:s}'
        ).format(ryan_file_pattern)

        raise ValueError(error_string)

    zhixing_file_pattern = '{0:s}/{1:s}_{2:s}_{3:s}.txt'.format(
        zhixing_dir_name,
        fake_cyclone_id_string,
        DATE_PATTERN_STRING,
        HOUR_MINUTE_PATTERN_STRING
    )
    zhixing_file_names = glob.glob(zhixing_file_pattern)

    if len(zhixing_file_names) == 0:
        error_string = (
            'Could not find any files with the following pattern:\n{0:s}'
        ).format(zhixing_file_pattern)

        raise ValueError(error_string)

    ryan_file_names.sort()
    zhixing_file_names.sort()

    num_files_ryan = len(ryan_file_names)
    ryan_latitudes_deg_n = numpy.full(num_files_ryan, numpy.nan)
    ryan_longitudes_deg_e = numpy.full(num_files_ryan, numpy.nan)
    ryan_times_unix_sec = numpy.full(num_files_ryan, -1, dtype=int)

    for i in range(num_files_ryan):
        print('Reading data from: "{0:s}"...'.format(ryan_file_names[i]))

        with open(ryan_file_names[i], 'r') as file_handle:
            csv_reader_object = csv.reader(file_handle)
            these_words = next(csv_reader_object)
            ryan_latitudes_deg_n[i] = float(these_words[-2])
            ryan_longitudes_deg_e[i] = float(these_words[-1])

            this_time_string = '{0:s}{1:s}'.format(
                these_words[-4].strip(), these_words[-3].strip()
            )
            ryan_times_unix_sec[i] = time_conversion.string_to_unix_sec(
                this_time_string, '%Y%m%d%H%M'
            )

    sort_indices = numpy.argsort(ryan_times_unix_sec)
    ryan_times_unix_sec = ryan_times_unix_sec[sort_indices]
    ryan_latitudes_deg_n = ryan_latitudes_deg_n[sort_indices]
    ryan_longitudes_deg_e = ryan_longitudes_deg_e[sort_indices]

    error_checking.assert_is_valid_lat_numpy_array(
        ryan_latitudes_deg_n, allow_nan=False
    )
    ryan_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
        ryan_longitudes_deg_e, allow_nan=False
    )

    num_files_zhixing = len(zhixing_file_names)
    zhixing_latitudes_deg_n = numpy.full(num_files_zhixing, numpy.nan)
    zhixing_longitudes_deg_e = numpy.full(num_files_zhixing, numpy.nan)
    zhixing_times_unix_sec = numpy.full(num_files_zhixing, -1, dtype=int)

    for i in range(num_files_zhixing):
        print('Reading data from: "{0:s}"...'.format(zhixing_file_names[i]))

        with open(zhixing_file_names[i], 'r') as file_handle:
            csv_reader_object = csv.reader(file_handle)
            these_words = next(csv_reader_object)
            zhixing_latitudes_deg_n[i] = float(these_words[-3])
            zhixing_longitudes_deg_e[i] = float(these_words[-2])

            this_time_string = '{0:s}{1:s}'.format(
                these_words[-5].strip(), these_words[-4].strip()
            )
            zhixing_times_unix_sec[i] = time_conversion.string_to_unix_sec(
                this_time_string, '%Y%m%d%H%M'
            )

    sort_indices = numpy.argsort(zhixing_times_unix_sec)
    zhixing_times_unix_sec = zhixing_times_unix_sec[sort_indices]
    zhixing_latitudes_deg_n = zhixing_latitudes_deg_n[sort_indices]
    zhixing_longitudes_deg_e = zhixing_longitudes_deg_e[sort_indices]

    error_checking.assert_is_valid_lat_numpy_array(
        zhixing_latitudes_deg_n, allow_nan=False
    )
    zhixing_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
        zhixing_longitudes_deg_e, allow_nan=False
    )

    print('Reading data from: "{0:s}"...'.format(raw_best_track_file_name))
    bt_latitudes_deg_n, bt_longitudes_deg_e, bt_times_unix_sec = (
        _read_raw_best_track_file(
            csv_file_name=raw_best_track_file_name,
            cyclone_id_string=cyclone_id_string
        )
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()
    border_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
        border_longitudes_deg_e
    )
    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object
    )

    min_time_unix_sec = min([
        numpy.min(ryan_times_unix_sec),
        numpy.min(zhixing_times_unix_sec),
        numpy.min(bt_times_unix_sec)
    ])
    max_time_unix_sec = max([
        numpy.max(ryan_times_unix_sec),
        numpy.max(zhixing_times_unix_sec),
        numpy.max(bt_times_unix_sec)
    ])
    colour_norm_object = pyplot.Normalize(
        vmin=min_time_unix_sec, vmax=max_time_unix_sec
    )

    axes_object.scatter(
        x=ryan_longitudes_deg_e, y=ryan_latitudes_deg_n,
        s=RYAN_MARKER_SIZE, marker=RYAN_MARKER_TYPE,
        linewidths=RYAN_MARKER_EDGE_WIDTH, edgecolors=RYAN_MARKER_EDGE_COLOUR,
        c=ryan_times_unix_sec, cmap=COLOUR_MAP_OBJECT, norm=colour_norm_object
    )
    axes_object.scatter(
        x=zhixing_longitudes_deg_e, y=zhixing_latitudes_deg_n,
        s=ZHIXING_MARKER_SIZE, marker=ZHIXING_MARKER_TYPE,
        linewidths=ZHIXING_MARKER_EDGE_WIDTH,
        edgecolors=ZHIXING_MARKER_EDGE_COLOUR,
        c=zhixing_times_unix_sec,
        cmap=COLOUR_MAP_OBJECT,
        norm=colour_norm_object
    )
    axes_object.scatter(
        x=bt_longitudes_deg_e, y=bt_latitudes_deg_n,
        s=BEST_TRACK_MARKER_SIZE, marker=BEST_TRACK_MARKER_TYPE,
        linewidths=BEST_TRACK_MARKER_EDGE_WIDTH,
        edgecolors=BEST_TRACK_MARKER_EDGE_COLOUR,
        c=bt_times_unix_sec, cmap=COLOUR_MAP_OBJECT, norm=colour_norm_object
    )

    all_latitudes_deg_n = numpy.concatenate(
        [ryan_latitudes_deg_n, zhixing_latitudes_deg_n, bt_latitudes_deg_n]
    )
    all_longitudes_deg_e = numpy.concatenate(
        [ryan_longitudes_deg_e, zhixing_longitudes_deg_e, bt_longitudes_deg_e]
    )
    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=all_latitudes_deg_n,
        plot_longitudes_deg_e=all_longitudes_deg_e,
        axes_object=axes_object,
        parallel_spacing_deg=1.,
        meridian_spacing_deg=1.
    )

    axes_object.set_xlim([
        numpy.min(all_longitudes_deg_e) - 0.5,
        numpy.max(all_longitudes_deg_e) + 0.5
    ])
    axes_object.set_ylim([
        numpy.min(all_latitudes_deg_n) - 0.5,
        numpy.max(all_latitudes_deg_n) + 0.5
    ])

    axes_object.set_xticklabels(
        axes_object.get_xticklabels(),
        fontsize=TICK_LABEL_FONT_SIZE, rotation=90.
    )
    axes_object.set_yticklabels(
        axes_object.get_yticklabels(), fontsize=TICK_LABEL_FONT_SIZE
    )
    axes_object.set_title(
        'Ryan (black) and Zhixing (grey) tracks for {0:s}'.format(
            cyclone_id_string
        )
    )

    colour_bar_object = gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=ryan_times_unix_sec.astype(float),
        colour_map_object=COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical',
        extend_min=False,
        extend_max=False
    )

    all_times_unix_sec = numpy.concatenate([
        ryan_times_unix_sec, zhixing_times_unix_sec, bt_times_unix_sec
    ])
    tick_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=numpy.min(all_times_unix_sec),
        end_time_unix_sec=numpy.max(all_times_unix_sec),
        time_interval_sec=12 * HOURS_TO_SECONDS,
        include_endpoint=True
    )
    tick_time_strings = [
        time_conversion.unix_sec_to_string(t, COLOUR_BAR_TIME_FORMAT)
        for t in tick_times_unix_sec
    ]
    colour_bar_object.set_ticks(tick_times_unix_sec)
    colour_bar_object.set_ticklabels(tick_time_strings)

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name,
        dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        ryan_dir_name=getattr(INPUT_ARG_OBJECT, RYAN_DIR_ARG_NAME),
        zhixing_dir_name=getattr(INPUT_ARG_OBJECT, ZHIXING_DIR_ARG_NAME),
        raw_best_track_file_name=getattr(
            INPUT_ARG_OBJECT, RAW_BT_FILE_ARG_NAME
        ),
        cyclone_id_string=getattr(INPUT_ARG_OBJECT, CYCLONE_ID_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
