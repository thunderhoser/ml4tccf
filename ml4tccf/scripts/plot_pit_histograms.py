"""Plots PIT (prob integral transform) histogram for each target variable."""

import argparse
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from ml4tccf.utils import pit_utils
from ml4tccf.plotting import uq_evaluation_plotting as uq_eval_plotting

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_file_name'
ENSEMBLE_SIZE_ARG_NAME = 'ensemble_size_for_rank_hist'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `pit_utils.read_results`.'
)
ENSEMBLE_SIZE_HELP_STRING = (
    'Ensemble size.  If you specify this value, the script will plot a rank '
    'histogram instead of a PIT histogram.'
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
    '--' + ENSEMBLE_SIZE_ARG_NAME, type=int, required=False, default=-1,
    help=ENSEMBLE_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_file_name, ensemble_size_for_rank_hist, output_dir_name):
    """Plots PIT (prob integral transform) histogram for each target variable.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param ensemble_size_for_rank_hist: Same.
    :param output_dir_name: Same.
    """

    if ensemble_size_for_rank_hist < 2:
        ensemble_size_for_rank_hist = None

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    result_table_xarray = pit_utils.read_results(input_file_name)
    t = result_table_xarray

    for this_var_name in t.coords[pit_utils.TARGET_FIELD_DIM].values:
        if ensemble_size_for_rank_hist is None:
            figure_object, _ = uq_eval_plotting.plot_pit_histogram(
                result_table_xarray=result_table_xarray,
                target_var_name=this_var_name
            )
        else:
            figure_object, _ = uq_eval_plotting.plot_rank_histogram(
                result_table_xarray=result_table_xarray,
                target_var_name=this_var_name,
                ensemble_size=ensemble_size_for_rank_hist
            )

        this_figure_file_name = '{0:s}/pit_histogram_{1:s}.jpg'.format(
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
        ensemble_size_for_rank_hist=getattr(
            INPUT_ARG_OBJECT, ENSEMBLE_SIZE_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
