"""Plots full predicted track along with 6-hourly best-track points."""

import os
import sys
import csv
import glob
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from geopy.distance import geodesic

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import time_periods
import longitude_conversion as lng_conversion
import file_system_utils
import error_checking
import gg_plotting_utils
import border_io
import extended_best_track_io as ebtrk_io
import extended_best_track_utils as ebtrk_utils
import misc_utils
import plotting_utils

HOURS_TO_SECONDS = 3600
COLOUR_BAR_TIME_FORMAT = '%HZ %b %-d'

DATE_PATTERN_STRING = '[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]'
HOUR_MINUTE_PATTERN_STRING = '[0-2][0-9][0-5][0-9]'

PREDICTION_MARKER_SIZE = 24
PREDICTION_MARKER_TYPE = 's'
PREDICTION_MARKER_EDGE_WIDTH = 0.75
PREDICTION_MARKER_EDGE_COLOUR = numpy.full(3, 0.)

BEST_TRACK_MARKER_SIZE = 300
BEST_TRACK_MARKER_TYPE = '*'
BEST_TRACK_MARKER_EDGE_WIDTH = 1.5
BEST_TRACK_MARKER_EDGE_COLOUR = numpy.full(3, 0.)

TITLE_FONT_SIZE = 20
TICK_LABEL_FONT_SIZE = 40
COLOUR_MAP_OBJECT = pyplot.get_cmap('gist_ncar')

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_DIR_ARG_NAME = 'input_ascii_prediction_dir_name'
EBTRK_FILE_ARG_NAME = 'input_ebtrk_file_name'
CYCLONE_ID_ARG_NAME = 'cyclone_id_string'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_DIR_HELP_STRING = (
    'Path to input directory, containing prediction files in ASCII format, '
    'produced by convert_prediction_file_for_zhixing.py'
)
EBTRK_FILE_HELP_STRING = (
    'Path to extended best-track file, which will be read by '
    '`extended_best_track_io.read_file`.'
)
CYCLONE_ID_HELP_STRING = (
    'Cyclone ID (will plot the full track for this one cyclone).'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output image file (figure will be saved here).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EBTRK_FILE_ARG_NAME, type=str, required=True,
    help=EBTRK_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CYCLONE_ID_ARG_NAME, type=str, required=True,
    help=CYCLONE_ID_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _compute_errors(
        ebtrk_latitudes_deg_n, ebtrk_longitudes_deg_e, ebtrk_times_unix_sec,
        pred_latitudes_deg_n, pred_longitudes_deg_e, pred_times_unix_sec):
    """Computes errors between GeoCenter and extended best-track.

    B = number of extended best-track points
    G = number of predicted GeoCenter points

    :param ebtrk_latitudes_deg_n: length-B numpy array of latitudes (deg north).
    :param ebtrk_longitudes_deg_e: length-B numpy array of longitudes (deg
        east).
    :param ebtrk_times_unix_sec: length-B numpy array of times.
    :param pred_latitudes_deg_n: length-G numpy array of latitudes (deg north).
    :param pred_longitudes_deg_e: length-G numpy array of longitudes (deg east).
    :param pred_times_unix_sec: length-G numpy array of times.
    :return: mean_error_km: Mean error.
    :return: median_error_km: Median error.
    :return: num_samples: Number of samples in estimate.
    """

    num_best_track_points = len(ebtrk_times_unix_sec)
    distance_errors_km = numpy.full(num_best_track_points, numpy.nan)

    for i in range(num_best_track_points):
        match_indices = numpy.where(
            pred_times_unix_sec == ebtrk_times_unix_sec[i]
        )[0]
        if len(match_indices) == 0:
            continue

        assert len(match_indices) == 1
        match_idx = match_indices[0]

        distance_errors_km[i] = geodesic(
            (ebtrk_latitudes_deg_n[i], ebtrk_longitudes_deg_e[i]),
            (pred_latitudes_deg_n[match_idx], pred_longitudes_deg_e[match_idx])
        ).kilometers

    good_indices = numpy.where(
        numpy.invert(numpy.isnan(distance_errors_km))
    )[0]
    num_samples = len(numpy.unique(ebtrk_times_unix_sec[good_indices]))

    array_string = ', '.join([
        '{0:.10f}'.format(d) for d in
        distance_errors_km[numpy.isfinite(distance_errors_km)]
    ])
    print(array_string)

    return (
        numpy.nanmean(distance_errors_km),
        numpy.nanmedian(distance_errors_km),
        num_samples
    )


def _run(ascii_prediction_dir_name, ebtrk_file_name, cyclone_id_string,
         output_file_name):
    """Plots full predicted track along with 6-hourly best-track points.

    This is effectively the main method.

    :param ascii_prediction_dir_name: See documentation at top of this script.
    :param ebtrk_file_name: Same.
    :param cyclone_id_string: Same.
    :param output_file_name: Same.
    """

    # Find prediction files.
    year, basin_id_string, cyclone_number = misc_utils.parse_cyclone_id(
        cyclone_id_string
    )
    fake_cyclone_id_string = '{0:s}{1:02d}{2:04d}'.format(
        basin_id_string.lower(),
        cyclone_number,
        year
    )

    prediction_file_pattern = '{0:s}/{1:s}/{1:s}_{2:s}_{3:s}.txt'.format(
        ascii_prediction_dir_name,
        fake_cyclone_id_string,
        DATE_PATTERN_STRING,
        HOUR_MINUTE_PATTERN_STRING
    )
    prediction_file_names = glob.glob(prediction_file_pattern)

    if len(prediction_file_names) == 0:
        error_string = (
            'Could not find any files with the following pattern:\n{0:s}'
        ).format(prediction_file_pattern)

        raise ValueError(error_string)

    # Read prediction files.
    prediction_file_names.sort()

    num_files = len(prediction_file_names)
    predicted_latitudes_deg_n = numpy.full(num_files, numpy.nan)
    predicted_longitudes_deg_e = numpy.full(num_files, numpy.nan)
    predicted_times_unix_sec = numpy.full(num_files, -1, dtype=int)

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(prediction_file_names[i]))

        with open(prediction_file_names[i], 'r') as file_handle:
            csv_reader_object = csv.reader(file_handle)
            these_words = next(csv_reader_object)
            predicted_latitudes_deg_n[i] = float(these_words[-2])
            predicted_longitudes_deg_e[i] = float(these_words[-1])

            this_time_string = '{0:s}{1:s}'.format(
                these_words[-4].strip(), these_words[-3].strip()
            )
            predicted_times_unix_sec[i] = time_conversion.string_to_unix_sec(
                this_time_string, '%Y%m%d%H%M'
            )

    sort_indices = numpy.argsort(predicted_times_unix_sec)
    predicted_times_unix_sec = predicted_times_unix_sec[sort_indices]
    predicted_latitudes_deg_n = predicted_latitudes_deg_n[sort_indices]
    predicted_longitudes_deg_e = predicted_longitudes_deg_e[sort_indices]

    error_checking.assert_is_valid_lat_numpy_array(
        predicted_latitudes_deg_n, allow_nan=False
    )
    predicted_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
        predicted_longitudes_deg_e, allow_nan=False
    )

    # Read best-track file.
    print('Reading data from: "{0:s}"...'.format(ebtrk_file_name))
    ebtrk_table_xarray = ebtrk_io.read_file(ebtrk_file_name)
    etx = ebtrk_table_xarray

    good_indices = numpy.where(
        etx[ebtrk_utils.STORM_ID_KEY].values == cyclone_id_string
    )[0]
    ebtrk_latitudes_deg_n = (
        etx[ebtrk_utils.CENTER_LATITUDE_KEY].values[good_indices]
    )
    ebtrk_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
        etx[ebtrk_utils.CENTER_LONGITUDE_KEY].values[good_indices],
        allow_nan=False
    )
    ebtrk_times_unix_sec = (
        etx[ebtrk_utils.VALID_TIME_KEY].values[good_indices]
        * HOURS_TO_SECONDS
    )
    ebtrk_times_unix_sec = numpy.round(ebtrk_times_unix_sec).astype(int)

    min_time_unix_sec = numpy.min(predicted_times_unix_sec)
    max_time_unix_sec = numpy.max(predicted_times_unix_sec)

    good_indices = numpy.where(numpy.logical_and(
        ebtrk_times_unix_sec >= min_time_unix_sec - 3 * HOURS_TO_SECONDS,
        ebtrk_times_unix_sec <= max_time_unix_sec + 3 * HOURS_TO_SECONDS
    ))
    ebtrk_times_unix_sec = ebtrk_times_unix_sec[good_indices]
    ebtrk_latitudes_deg_n = ebtrk_latitudes_deg_n[good_indices]
    ebtrk_longitudes_deg_e = ebtrk_longitudes_deg_e[good_indices]

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
        axes_object=axes_object,
        line_colour=numpy.full(3, 152. / 255)
    )

    colour_norm_object = pyplot.Normalize(
        vmin=min_time_unix_sec, vmax=max_time_unix_sec
    )

    axes_object.scatter(
        x=predicted_longitudes_deg_e, y=predicted_latitudes_deg_n,
        s=PREDICTION_MARKER_SIZE,
        marker=PREDICTION_MARKER_TYPE,
        # linewidths=PREDICTION_MARKER_EDGE_WIDTH,
        # edgecolors=PREDICTION_MARKER_EDGE_COLOUR,
        # c=PREDICTION_MARKER_EDGE_COLOUR,
        linewidths=PREDICTION_MARKER_EDGE_WIDTH,
        edgecolors=PREDICTION_MARKER_EDGE_COLOUR,
        c=predicted_times_unix_sec,
        cmap=COLOUR_MAP_OBJECT,
        norm=colour_norm_object
    )

    axes_object.scatter(
        x=ebtrk_longitudes_deg_e, y=ebtrk_latitudes_deg_n,
        s=BEST_TRACK_MARKER_SIZE,
        marker=BEST_TRACK_MARKER_TYPE,
        linewidths=BEST_TRACK_MARKER_EDGE_WIDTH,
        edgecolors=BEST_TRACK_MARKER_EDGE_COLOUR,
        c=ebtrk_times_unix_sec,
        cmap=COLOUR_MAP_OBJECT,
        norm=colour_norm_object
    )

    all_latitudes_deg_n = numpy.concatenate(
        [ebtrk_latitudes_deg_n, predicted_latitudes_deg_n]
    )
    all_longitudes_deg_e = numpy.concatenate(
        [ebtrk_longitudes_deg_e, predicted_longitudes_deg_e]
    )
    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=all_latitudes_deg_n,
        plot_longitudes_deg_e=all_longitudes_deg_e,
        axes_object=axes_object,
        parallel_spacing_deg=5.,
        meridian_spacing_deg=5.
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

    title_string = (
        'Track comparison for {0:s}\n'
        'Best track in coloured stars; GeoCenter in squares'
    ).format(
        cyclone_id_string
    )

    mean_error_km, median_error_km, num_samples = (
        _compute_errors(
            ebtrk_latitudes_deg_n=ebtrk_latitudes_deg_n,
            ebtrk_longitudes_deg_e=ebtrk_longitudes_deg_e,
            ebtrk_times_unix_sec=ebtrk_times_unix_sec,
            pred_latitudes_deg_n=predicted_latitudes_deg_n,
            pred_longitudes_deg_e=predicted_longitudes_deg_e,
            pred_times_unix_sec=predicted_times_unix_sec
        )
    )

    title_string += (
        '\nMean/median errors: {0:.1f}/{1:.1f} km; sample size = {2:d}'
    ).format(
        mean_error_km, median_error_km, num_samples
    )
    axes_object.set_title(title_string, fontsize=TITLE_FONT_SIZE)

    all_times_unix_sec = numpy.concatenate(
        [predicted_times_unix_sec, ebtrk_times_unix_sec]
    )
    colour_bar_object = gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=all_times_unix_sec.astype(float),
        colour_map_object=COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical',
        extend_min=False,
        extend_max=False
    )

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
        ascii_prediction_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        ebtrk_file_name=getattr(INPUT_ARG_OBJECT, EBTRK_FILE_ARG_NAME),
        cyclone_id_string=getattr(INPUT_ARG_OBJECT, CYCLONE_ID_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
