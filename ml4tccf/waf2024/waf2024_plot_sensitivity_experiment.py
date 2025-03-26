"""Plots results of sensitivity experiment.

Specifically, creates one figure with two curves:

- Mean Euclidean error vs. translation distance
- Median Euclidean error vs. translation distance
"""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from ml4tccf.utils import scalar_evaluation as evaluation

METRES_TO_KM = 0.001
TRANSLATION_DISTANCES_PX = numpy.linspace(1, 60, num=60, dtype=int)

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
OUTPUT_FILE_ARG_NAME = 'output_figure_file_name'

INPUT_DIR_HELP_STRING = (
    'Path to top-level input directory.  Within this directory, evaluation '
    'results for each translation distance d will be found at '
    '"translation_distance_px=<ddd>/evaluation_no_bootstrap.nc", where <ddd> '
    'is the 3-digit representation of d with leading zeros.'
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
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_dir_name, output_figure_file_name):
    """Plots results of sensitivity experiment.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of this script.
    :param output_figure_file_name: Same.
    """

    evaluation_file_names = [
        (
            '{0:s}/translation_distance_px={1:03d}/evaluation_no_bootstrap.nc'
        ).format(input_dir_name, d)
        for d in TRANSLATION_DISTANCES_PX
    ]

    num_translation_distances = len(TRANSLATION_DISTANCES_PX)
    mean_errors_km = numpy.full(num_translation_distances, numpy.nan)
    median_errors_km = numpy.full(num_translation_distances, numpy.nan)

    for i in range(num_translation_distances):
        print('Reading data from: "{0:s}"...'.format(evaluation_file_names[i]))
        this_eval_table_xarray = evaluation.read_file(evaluation_file_names[i])
        mean_errors_km[i] = METRES_TO_KM * numpy.mean(
            this_eval_table_xarray[evaluation.MEAN_DISTANCE_KEY].values
        )
        median_errors_km[i] = METRES_TO_KM * numpy.mean(
            this_eval_table_xarray[evaluation.MEDIAN_DISTANCE_KEY].values
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

    axes_object.set_xlabel('Whole-track error in first guess (km)')
    axes_object.set_ylabel('Mean/median GeoCenter error (km)')
    axes_object.set_title(
        'Mean (orange) and median (purple) GeoCenter error\n'
        'vs. whole-track error in first guess'
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
        output_figure_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
