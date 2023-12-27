"""Plots interpolation case study for 2024 WAF paper.

WAF = Weather and Forecasting

The 'interpolation' in this case is the B-spline interpolation done by Robert,
to estimate actual TC centers in between best-track times.
"""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4tccf.io import border_io
from ml4tccf.io import satellite_io
from ml4tccf.utils import satellite_utils
from ml4tccf.plotting import plotting_utils

TIME_FORMAT = '%Y-%m-%d-%H%M'
SYNOPTIC_TIME_INTERVAL_SEC = 21600

MARKER_EDGE_COLOUR = numpy.full(3, 0.)

CORRECT_MARKER_TYPE = '*'
CORRECT_MARKER_SIZE = 30
CORRECT_MARKER_FACE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
CORRECT_MARKER_EDGE_WIDTH = 0

SYNOPTIC_MARKER_TYPE = 's'
SYNOPTIC_MARKER_SIZE = 18
SYNOPTIC_MARKER_FACE_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
SYNOPTIC_MARKER_EDGE_WIDTH = 2

GENERIC_MARKER_TYPE = 'o'
GENERIC_MARKER_SIZE = 14
GENERIC_MARKER_FACE_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
GENERIC_MARKER_EDGE_WIDTH = 0

INTERP_MARKER_TYPE = '*'
INTERP_MARKER_SIZE = 30
INTERP_MARKER_FACE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
INTERP_MARKER_EDGE_WIDTH = 0

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

SATELLITE_DIR_ARG_NAME = 'input_satellite_dir_name'
CYCLONE_ID_ARG_NAME = 'cyclone_id_string'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
SPECIAL_TIMES_ARG_NAME = 'special_time_strings'
SPECIAL_LATITUDES_ARG_NAME = 'special_interp_latitudes_deg_n'
SPECIAL_LONGITUDES_ARG_NAME = 'special_interp_longitudes_deg_e'
FIGURE_FILE_ARG_NAME = 'output_figure_file_name'

SATELLITE_DIR_HELP_STRING = (
    'Name of directory with satellite data.  Files therein will be found by '
    '`satellite_io.find_file` and read by `satellite_io.read_file`.'
)
CYCLONE_ID_HELP_STRING = 'Cyclone ID.'
FIRST_TIME_HELP_STRING = (
    'First time shown in figure (format "yyyy-mm-dd-HHMM").'
)
LAST_TIME_HELP_STRING = 'Last time shown in figure (format "yyyy-mm-dd-HHMM").'
SPECIAL_TIMES_HELP_STRING = (
    'List of special times (format "yyyy-mm-dd-HHMM"), i.e., non-synoptic '
    'times with a best-track center fix.'
)
SPECIAL_LATITUDES_HELP_STRING = (
    'List of interpolated latitudes (deg north; same length as {0:s}).'
).format(SPECIAL_TIMES_ARG_NAME)

SPECIAL_LONGITUDES_HELP_STRING = (
    'List of interpolated longitudes (deg east; same length as {0:s}).'
).format(SPECIAL_TIMES_ARG_NAME)

FIGURE_FILE_HELP_STRING = 'Path to output file.  Figure will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SATELLITE_DIR_ARG_NAME, type=str, required=True,
    help=SATELLITE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CYCLONE_ID_ARG_NAME, type=str, required=True,
    help=CYCLONE_ID_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True,
    help=FIRST_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True,
    help=LAST_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SPECIAL_TIMES_ARG_NAME, type=str, nargs='+', required=True,
    help=SPECIAL_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SPECIAL_LATITUDES_ARG_NAME, type=float, nargs='+', required=True,
    help=SPECIAL_LATITUDES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SPECIAL_LONGITUDES_ARG_NAME, type=float, nargs='+', required=True,
    help=SPECIAL_LONGITUDES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIGURE_FILE_ARG_NAME, type=str, required=True,
    help=FIGURE_FILE_HELP_STRING
)


def _run(satellite_dir_name, cyclone_id_string,
         first_time_string, last_time_string,
         special_time_strings, special_latitudes_deg_n,
         special_longitudes_deg_e, figure_file_name):
    """Plots interpolation case study for 2024 WAF paper.

    This is effectively the main method.

    :param satellite_dir_name: See documentation at the top of this script.
    :param cyclone_id_string: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param special_time_strings: Same.
    :param special_latitudes_deg_n: Same.
    :param special_longitudes_deg_e: Same.
    :param figure_file_name: Same.
    """

    # Check input args.
    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, TIME_FORMAT
    )
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, TIME_FORMAT
    )
    error_checking.assert_is_greater(last_time_unix_sec, first_time_unix_sec)

    special_times_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in special_time_strings
    ], dtype=int)

    error_checking.assert_is_greater_numpy_array(
        special_times_unix_sec, first_time_unix_sec
    )
    error_checking.assert_is_less_than_numpy_array(
        special_times_unix_sec, last_time_unix_sec
    )

    num_special_times = len(special_time_strings)
    error_checking.assert_is_numpy_array(
        special_latitudes_deg_n,
        exact_dimensions=numpy.array([num_special_times], dtype=int)
    )
    error_checking.assert_is_valid_lat_numpy_array(
        special_latitudes_deg_n, allow_nan=False
    )

    error_checking.assert_is_numpy_array(
        special_longitudes_deg_e,
        exact_dimensions=numpy.array([num_special_times], dtype=int)
    )
    special_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
        special_longitudes_deg_e, allow_nan=False
    )

    file_system_utils.mkdir_recursive_if_necessary(file_name=figure_file_name)

    # Do actual stuff.
    first_date_string = time_conversion.unix_sec_to_string(
        first_time_unix_sec, '%Y%m%d'
    )
    last_date_string = time_conversion.unix_sec_to_string(
        last_time_unix_sec, '%Y%m%d'
    )
    date_strings = time_conversion.get_spc_dates_in_range(
        first_date_string, last_date_string
    )
    date_strings = [
        '{0:s}-{1:s}-{2:s}'.format(d[:4], d[4:6], d[6:]) for d in date_strings
    ]

    satellite_file_names = [
        satellite_io.find_file(
            directory_name=satellite_dir_name,
            cyclone_id_string=cyclone_id_string,
            valid_date_string=d,
            raise_error_if_missing=True
        )
        for d in date_strings
    ]

    satellite_tables_xarray = []

    for this_file_name in satellite_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_table_xarray = satellite_io.read_file(this_file_name)

        try:
            this_table_xarray = satellite_utils.subset_grid(
                satellite_table_xarray=this_table_xarray,
                num_rows_to_keep=2,
                num_columns_to_keep=2,
                for_high_res=True
            )
        except KeyError:
            pass

        this_table_xarray = satellite_utils.subset_grid(
            satellite_table_xarray=this_table_xarray,
            num_rows_to_keep=2,
            num_columns_to_keep=2,
            for_high_res=False
        )

        satellite_tables_xarray.append(this_table_xarray)

    satellite_table_xarray = satellite_utils.concat_over_time(
        satellite_tables_xarray
    )
    del satellite_tables_xarray

    satellite_table_xarray = satellite_utils.subset_to_multiple_time_windows(
        satellite_table_xarray=satellite_table_xarray,
        start_times_unix_sec=numpy.array([first_time_unix_sec], dtype=int),
        end_times_unix_sec=numpy.array([last_time_unix_sec], dtype=int)
    )
    stx = satellite_table_xarray

    valid_times_unix_sec = stx.coords[satellite_utils.TIME_DIM].values
    _ = numpy.array([
        numpy.where(valid_times_unix_sec == t)[0][0]
        for t in special_times_unix_sec
    ], dtype=int)

    # TODO(thunderhoser): This wouldn't work for a TC that crosses the
    # International Date Line.
    center_longitudes_deg_e = numpy.mean(
        stx[satellite_utils.LONGITUDE_LOW_RES_KEY].values, axis=1
    )
    center_latitudes_deg_n = numpy.mean(
        stx[satellite_utils.LATITUDE_LOW_RES_KEY].values, axis=1
    )
    center_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
        center_longitudes_deg_e
    )

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object
    )

    num_times = len(valid_times_unix_sec)
    correct_legend_handle = None
    synoptic_legend_handle = None
    generic_legend_handle = None
    interp_legend_handle = None

    for i in range(num_times):
        if valid_times_unix_sec[i] in special_times_unix_sec:
            marker_type = CORRECT_MARKER_TYPE
            marker_size = CORRECT_MARKER_SIZE
            marker_face_colour = CORRECT_MARKER_FACE_COLOUR
            marker_edge_width = CORRECT_MARKER_EDGE_WIDTH
        elif numpy.mod(valid_times_unix_sec[i], SYNOPTIC_TIME_INTERVAL_SEC) == 0:
            marker_type = SYNOPTIC_MARKER_TYPE
            marker_size = SYNOPTIC_MARKER_SIZE
            marker_face_colour = SYNOPTIC_MARKER_FACE_COLOUR
            marker_edge_width = SYNOPTIC_MARKER_EDGE_WIDTH
        else:
            marker_type = GENERIC_MARKER_TYPE
            marker_size = GENERIC_MARKER_SIZE
            marker_face_colour = GENERIC_MARKER_FACE_COLOUR
            marker_edge_width = GENERIC_MARKER_EDGE_WIDTH

        this_handle = axes_object.plot(
            center_longitudes_deg_e[i], center_latitudes_deg_n[i],
            linestyle='None', marker=marker_type, markersize=marker_size,
            markerfacecolor=marker_face_colour,
            markeredgecolor=MARKER_EDGE_COLOUR,
            markeredgewidth=marker_edge_width,
            zorder=1e10
        )[0]

        if valid_times_unix_sec[i] in special_times_unix_sec:
            correct_legend_handle = this_handle
        elif numpy.mod(valid_times_unix_sec[i], SYNOPTIC_TIME_INTERVAL_SEC) == 0:
            synoptic_legend_handle = this_handle

            axes_object.text(
                center_longitudes_deg_e[i] - 0.1,
                center_latitudes_deg_n[i],
                time_conversion.unix_sec_to_string(
                    valid_times_unix_sec[i], '%HZ %b %-d'
                ),
                horizontalalignment='right', verticalalignment='center',
                fontsize=20, fontweight='bold'
            )
        else:
            generic_legend_handle = this_handle

        if valid_times_unix_sec[i] not in special_times_unix_sec:
            continue

        j = numpy.where(special_times_unix_sec == valid_times_unix_sec[i])[0][0]

        interp_legend_handle = axes_object.plot(
            special_longitudes_deg_e[j], special_latitudes_deg_n[j],
            linestyle='None',
            marker=INTERP_MARKER_TYPE, markersize=INTERP_MARKER_SIZE,
            markerfacecolor=INTERP_MARKER_FACE_COLOUR,
            markeredgecolor=INTERP_MARKER_FACE_COLOUR,
            markeredgewidth=INTERP_MARKER_EDGE_WIDTH,
            zorder=1e10
        )[0]

    legend_strings = []
    legend_handles = []

    if generic_legend_handle is not None:
        legend_handles.append(generic_legend_handle)
        legend_strings.append('Estimated center at\ntime w/out BT center')
    if synoptic_legend_handle is not None:
        legend_handles.append(synoptic_legend_handle)
        legend_strings.append('BT center')
    if correct_legend_handle is not None:
        legend_handles.append(correct_legend_handle)
        legend_strings.append('BT center at special time')
    if interp_legend_handle is not None:
        legend_handles.append(interp_legend_handle)
        legend_strings.append('Estimated center\nat special time')

    all_latitudes_deg_n = numpy.concatenate((
        center_latitudes_deg_n, special_latitudes_deg_n
    ))
    all_longitudes_deg_e = numpy.concatenate((
        center_longitudes_deg_e, special_longitudes_deg_e
    ))

    axes_object.set_xlim(left=numpy.min(all_longitudes_deg_e) - 0.5)
    axes_object.set_xlim(right=numpy.max(all_longitudes_deg_e) + 0.5)
    axes_object.set_ylim(bottom=numpy.min(all_latitudes_deg_n) - 0.5)
    axes_object.set_ylim(top=numpy.max(all_latitudes_deg_n) + 0.5)

    x_tick_values = axes_object.get_xticks()
    x_tick_labels = ['{0:.1f}'.format(v) + r'$^{\circ}$' for v in x_tick_values]
    axes_object.set_xticklabels(x_tick_labels, rotation=90.)

    y_tick_values = axes_object.get_yticks()
    y_tick_labels = ['{0:.1f}'.format(v) + r'$^{\circ}$' for v in y_tick_values]
    axes_object.set_yticklabels(y_tick_labels)

    axes_object.legend(
        legend_handles, legend_strings, loc='center left',
        bbox_to_anchor=(0, 0.5), fancybox=True, shadow=False,
        facecolor='white', edgecolor='k', framealpha=0.5, ncol=1
    )

    title_string = 'Case study for best-track interpolation: {0:s}'.format(
        cyclone_id_string
    )
    axes_object.set_title(title_string)

    print('Saving figure to file: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        satellite_dir_name=getattr(INPUT_ARG_OBJECT, SATELLITE_DIR_ARG_NAME),
        cyclone_id_string=getattr(INPUT_ARG_OBJECT, CYCLONE_ID_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        special_time_strings=getattr(INPUT_ARG_OBJECT, SPECIAL_TIMES_ARG_NAME),
        special_latitudes_deg_n=numpy.array(
            getattr(INPUT_ARG_OBJECT, SPECIAL_LATITUDES_ARG_NAME), dtype=float
        ),
        special_longitudes_deg_e=numpy.array(
            getattr(INPUT_ARG_OBJECT, SPECIAL_LONGITUDES_ARG_NAME), dtype=float
        ),
        figure_file_name=getattr(INPUT_ARG_OBJECT, FIGURE_FILE_ARG_NAME)
    )
