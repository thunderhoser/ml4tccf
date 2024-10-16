"""Analyzes sensitivity experiment.

Specifically, we are looking at sensitivity to translation distance.
"""

import csv
import glob
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from geopy.distance import geodesic
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4tccf.utils import misc_utils

CYCLONE_ID_STRING = '2024AL14'
TRANSLATION_DISTANCES_PX = numpy.linspace(1, 98, num=98, dtype=int)

DATE_PATTERN_STRING = '[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]'
HOUR_MINUTE_PATTERN_STRING = '[0-2][0-9][0-5][0-9]'

DEFAULT_FONT_SIZE = 30
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

LINE_WIDTH = 3.
MARKER_TYPE = 'o'
MARKER_SIZE = 12
MEAN_ERROR_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
MEDIAN_ERROR_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_DIR_ARG_NAME = 'input_dir_name'
RAW_BT_FILE_ARG_NAME = 'input_raw_best_track_file_name'
OUTPUT_FILE_ARG_NAME = 'output_figure_file_name'

INPUT_DIR_HELP_STRING = (
    'Path to directory with predictions.  These will be found at locations '
    '"mean_translation_distance_px=<TTT>/isotonic_regression/for_zhixing/'
    '<bbnnyyyy>/<bbnnyyyy>_<yyyymmdd>_<HHMM>.txt", where <TTT> is the '
    'translation distance and <bbnnyyyy> is the fake cyclone ID.'
)
RAW_BT_FILE_HELP_STRING = (
    'Path to file with raw best tracks for the same cyclone.  Should be named '
    'like "b<bbnnyyyy>.dat".'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  A graph, showing mean and median Euclidean error '
    'vs. translation distance, will be plotted and saved here as an image.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + RAW_BT_FILE_ARG_NAME, type=str, required=True,
    help=RAW_BT_FILE_HELP_STRING
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
        this_latitude_deg_n = 0.1 * int(this_latitude_string[:-1])
        if this_latitude_string.endswith('S'):
            this_latitude_deg_n *= -1.

        this_longitude_string = these_words[7]
        assert (
            this_longitude_string.endswith('E') or
            this_longitude_string.endswith('W')
        )
        this_longitude_deg_e = 0.1 * int(this_longitude_string[:-1])
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


def _compute_errors(
        bt_latitudes_deg_n, bt_longitudes_deg_e, bt_times_unix_sec,
        pred_latitudes_deg_n, pred_longitudes_deg_e, pred_times_unix_sec):
    """Computes errors between best track and GeoCenter.

    The GeoCenter predictions may come from either Ryan or Zhixing here.

    B = number of best-track points
    G = number of predicted GeoCenter points

    :param bt_latitudes_deg_n: length-B numpy array of latitudes (deg north).
    :param bt_longitudes_deg_e: length-B numpy array of longitudes (deg east).
    :param bt_times_unix_sec: length-B numpy array of times.
    :param pred_latitudes_deg_n: length-G numpy array of latitudes (deg north).
    :param pred_longitudes_deg_e: length-G numpy array of longitudes (deg east).
    :param pred_times_unix_sec: length-G numpy array of times.
    :return: mean_error_km: Mean error.
    :return: median_error_km: Median error.
    :return: num_samples: Number of samples in estimate.
    """

    num_best_track_points = len(bt_times_unix_sec)
    distance_errors_km = numpy.full(num_best_track_points, numpy.nan)

    for i in range(num_best_track_points):
        match_indices = numpy.where(
            pred_times_unix_sec == bt_times_unix_sec[i]
        )[0]
        if len(match_indices) == 0:
            continue

        assert len(match_indices) == 1
        match_idx = match_indices[0]

        distance_errors_km[i] = geodesic(
            (bt_latitudes_deg_n[i], bt_longitudes_deg_e[i]),
            (pred_latitudes_deg_n[match_idx], pred_longitudes_deg_e[match_idx])
        ).kilometers

    good_indices = numpy.where(
        numpy.invert(numpy.isnan(distance_errors_km))
    )[0]
    num_samples = len(numpy.unique(bt_times_unix_sec[good_indices]))

    return (
        numpy.nanmean(distance_errors_km),
        numpy.nanmedian(distance_errors_km),
        num_samples
    )


def _run(input_dir_name, raw_best_track_file_name, output_figure_file_name):
    """Analyzes sensitivity experiment.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of this script.
    :param output_figure_file_name: Same.
    """

    year, basin_id_string, cyclone_number = misc_utils.parse_cyclone_id(
        CYCLONE_ID_STRING
    )
    fake_cyclone_id_string = '{0:s}{1:02d}{2:04d}'.format(
        basin_id_string.lower(),
        cyclone_number,
        year
    )

    print('Reading data from: "{0:s}"...'.format(raw_best_track_file_name))
    bt_latitudes_deg_n, bt_longitudes_deg_e, bt_times_unix_sec = (
        _read_raw_best_track_file(
            csv_file_name=raw_best_track_file_name,
            cyclone_id_string=CYCLONE_ID_STRING
        )
    )

    num_translation_distances = len(TRANSLATION_DISTANCES_PX)
    mean_errors_km = numpy.full(num_translation_distances, numpy.nan)
    median_errors_km = numpy.full(num_translation_distances, numpy.nan)

    for i in range(num_translation_distances):
        prediction_file_pattern = (
            '{0:s}/mean_translation_distance_px={1:03d}/isotonic_regression/'
            'for_zhixing/{2:s}/{2:s}_{3:s}_{4:s}.txt'
        ).format(
            input_dir_name,
            TRANSLATION_DISTANCES_PX[i],
            fake_cyclone_id_string,
            DATE_PATTERN_STRING,
            HOUR_MINUTE_PATTERN_STRING
        )
        print(prediction_file_pattern)
        prediction_file_names = glob.glob(prediction_file_pattern)
        assert len(prediction_file_names) > 0

        prediction_file_names.sort()
        num_files = len(prediction_file_names)
        pred_latitudes_deg_n = numpy.full(num_files, numpy.nan)
        pred_longitudes_deg_e = numpy.full(num_files, numpy.nan)
        pred_times_unix_sec = numpy.full(num_files, -1, dtype=int)

        for j in range(num_files):
            print('Reading data from: "{0:s}"...'.format(
                prediction_file_names[j]
            ))

            with open(prediction_file_names[j], 'r') as file_handle:
                csv_reader_object = csv.reader(file_handle)
                these_words = next(csv_reader_object)
                pred_latitudes_deg_n[j] = float(these_words[-2])
                pred_longitudes_deg_e[j] = float(these_words[-1])

                this_time_string = '{0:s}{1:s}'.format(
                    these_words[-4].strip(), these_words[-3].strip()
                )
                pred_times_unix_sec[j] = time_conversion.string_to_unix_sec(
                    this_time_string, '%Y%m%d%H%M'
                )

        sort_indices = numpy.argsort(pred_times_unix_sec)
        pred_times_unix_sec = pred_times_unix_sec[sort_indices]
        pred_latitudes_deg_n = pred_latitudes_deg_n[sort_indices]
        pred_longitudes_deg_e = pred_longitudes_deg_e[sort_indices]

        error_checking.assert_is_valid_lat_numpy_array(
            pred_latitudes_deg_n, allow_nan=False
        )
        pred_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
            pred_longitudes_deg_e, allow_nan=False
        )

        mean_errors_km[i], median_errors_km[i], _ = _compute_errors(
            bt_latitudes_deg_n=bt_latitudes_deg_n,
            bt_longitudes_deg_e=bt_longitudes_deg_e,
            bt_times_unix_sec=bt_times_unix_sec,
            pred_latitudes_deg_n=pred_latitudes_deg_n,
            pred_longitudes_deg_e=pred_longitudes_deg_e,
            pred_times_unix_sec=pred_times_unix_sec
        )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.plot(
        TRANSLATION_DISTANCES_PX * 2,
        mean_errors_km,
        linestyle='solid',
        linewidth=LINE_WIDTH,
        color=MEAN_ERROR_COLOUR,
        marker=MARKER_TYPE,
        markersize=MARKER_SIZE,
        markerfacecolor=MEAN_ERROR_COLOUR,
        markeredgewidth=0
    )
    axes_object.plot(
        TRANSLATION_DISTANCES_PX * 2,
        median_errors_km,
        linestyle='solid',
        linewidth=LINE_WIDTH,
        color=MEDIAN_ERROR_COLOUR,
        marker=MARKER_TYPE,
        markersize=MARKER_SIZE,
        markerfacecolor=MEDIAN_ERROR_COLOUR,
        markeredgewidth=0
    )

    axes_object.set_xlabel('Translation distance (km)')
    axes_object.set_ylabel('Mean/median error (km)')
    axes_object.set_title(
        'Mean (orange) and median (purple) error vs. translation distance'
    )

    file_system_utils.mkdir_recursive_if_necessary(
        file_name=output_figure_file_name
    )
    print('Saving figure to file: "{0:s}"...'.format(output_figure_file_name))
    figure_object.savefig(
        output_figure_file_name,
        dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        raw_best_track_file_name=getattr(
            INPUT_ARG_OBJECT, RAW_BT_FILE_ARG_NAME
        ),
        output_figure_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
