"""Plots results of discard test (error vs. discard fraction)."""

import os
import sys
import argparse
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import discard_test_utils as dt_utils
import uq_evaluation_plotting as uq_eval_plotting

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `dt_utils.read_results`.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_file_name, output_dir_name):
    """Plots results of discard test (error vs. discard fraction).

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    result_table_xarray = dt_utils.read_results(input_file_name)
    t = result_table_xarray

    for this_var_name in t.coords[dt_utils.TARGET_FIELD_DIM].values:
        figure_object, _ = uq_eval_plotting.plot_discard_test(
            result_table_xarray=result_table_xarray,
            target_var_name=this_var_name
        )

        this_figure_file_name = '{0:s}/discard_test_{1:s}.jpg'.format(
            output_dir_name, this_var_name.replace('_', '-')
        )
        print('Saving figure to file: "{0:s}"...'.format(this_figure_file_name))
        figure_object.savefig(
            this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )