"""Plots histogram of interpolation errors for 2024 WAF paper.

WAF = Weather and Forecasting

The 'interpolation' in this case is the B-spline interpolation done by Robert,
to estimate actual TC centers in between best-track times.
"""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import histograms
from gewittergefahr.gg_utils import file_system_utils

FONT_SIZE = 30
HISTOGRAM_FACE_COLOUR = numpy.array([31, 120, 180], dtype=float) / 255
HISTOGRAM_EDGE_COLOUR = numpy.full(3, 0.)
HISTOGRAM_EDGE_WIDTH = 2.

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

LOWER_BIN_EDGE_KM = 0.
UPPER_BIN_EDGE_KM = 50.
NUM_BINS = 50

ERROR_FILE_ARG_NAME = 'input_error_file_name'
FIGURE_FILE_ARG_NAME = 'output_figure_file_name'

ERROR_FILE_HELP_STRING = (
    'Path to error file provided by Robert.  The pathless file name should '
    'probably be "b_spline_errors.csv".'
)
FIGURE_FILE_HELP_STRING = 'Path to output file.  The figure will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + ERROR_FILE_ARG_NAME, type=str, required=True,
    help=ERROR_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIGURE_FILE_ARG_NAME, type=str, required=True,
    help=FIGURE_FILE_HELP_STRING
)


def _run(error_file_name, figure_file_name):
    """Plots histogram of interpolation errors for 2024 WAF paper.

    This is effectively the main method.

    :param error_file_name: See documentation at top of file.
    :param figure_file_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=figure_file_name)

    print('Reading data from: "{0:s}"...'.format(error_file_name))
    euclidean_errors_km = numpy.loadtxt(
        error_file_name, delimiter=',', skiprows=1, dtype=str
    )[:, -1]

    euclidean_errors_km = numpy.array([float(a) for a in euclidean_errors_km])

    bin_counts = histograms.create_histogram(
        input_values=euclidean_errors_km, num_bins=NUM_BINS,
        min_value=LOWER_BIN_EDGE_KM, max_value=UPPER_BIN_EDGE_KM
    )[1]
    bin_frequencies = bin_counts.astype(float) / numpy.sum(bin_counts)

    bin_edges_km = numpy.linspace(
        LOWER_BIN_EDGE_KM, UPPER_BIN_EDGE_KM, num=NUM_BINS + 1, dtype=float
    )
    bin_centers_km = numpy.mean(
        numpy.vstack([bin_edges_km[:-1], bin_edges_km[1:]]),
        axis=0
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    axes_object.bar(
        x=bin_centers_km, height=bin_frequencies,
        width=numpy.diff(bin_centers_km[:2])[0],
        color=HISTOGRAM_FACE_COLOUR,
        edgecolor=HISTOGRAM_EDGE_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH
    )
    axes_object.set_ylim(bottom=0.)
    axes_object.set_xlim(left=LOWER_BIN_EDGE_KM)
    axes_object.set_xlim(right=UPPER_BIN_EDGE_KM)

    x_tick_values = axes_object.get_xticks()
    x_tick_labels = ['{0:.0f}'.format(v) for v in x_tick_values]
    x_tick_labels[-1] += '+'

    axes_object.set_xticks(x_tick_values)
    axes_object.set_xticklabels(x_tick_labels)

    axes_object.set_xlabel('Euclidean error (km)')
    axes_object.set_ylabel('Frequency')
    axes_object.set_title('FBT interpolation errors')

    print('Saving figure to file: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        error_file_name=getattr(INPUT_ARG_OBJECT, ERROR_FILE_ARG_NAME),
        figure_file_name=getattr(INPUT_ARG_OBJECT, FIGURE_FILE_ARG_NAME)
    )
