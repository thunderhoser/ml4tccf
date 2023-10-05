"""Plots evaluation metrics as a function of ocean basin."""

import os
import sys
import argparse
import numpy
import xarray
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import prediction_io
import misc_utils
import scalar_evaluation
import scalar_evaluation_plotting as scalar_eval_plotting

BIAS_COLOUR_MAP_NAME = 'seismic'
MAIN_COLOUR_MAP_NAME = 'viridis'
MIN_COLOUR_PERCENTILE = 1.
MAX_COLOUR_PERCENTILE = 99.

FIGURE_RESOLUTION_DPI = 300

INPUT_FILES_ARG_NAME = 'input_evaluation_file_names'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILES_HELP_STRING = (
    'List of paths to basin-specific evaluation files (each will be read by '
    '`scalar_evaluation.read_file` or `gridded_evaluation.read_file`).'
)
CONFIDENCE_LEVEL_HELP_STRING = (
    'Confidence level for uncertainty intervals.  Must be in range 0...1.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(evaluation_file_names, confidence_level, output_dir_name):
    """Plots evaluation metrics as a function of ocean basin.

    This is effectively the main method.

    :param evaluation_file_names: See documentation at top of file.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    error_checking.assert_is_geq(confidence_level, 0.8)
    error_checking.assert_is_less_than(confidence_level, 1.)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Do actual stuff.
    num_basins = len(evaluation_file_names)
    basin_id_strings = [''] * num_basins
    eval_table_by_basin = [xarray.Dataset()] * num_basins
    num_bootstrap_reps = -1

    for i in range(num_basins):
        print('Reading data from: "{0:s}"...'.format(evaluation_file_names[i]))

        try:
            eval_table_by_basin[i] = scalar_evaluation.read_file(
                evaluation_file_names[i]
            )
            assert (
                scalar_evaluation.TARGET_FIELD_DIM
                in eval_table_by_basin[i].coords
            )
        except:
            raise ValueError(
                'This script does not yet work for gridded predictions.'
            )

        etbb = eval_table_by_basin

        num_bootstrap_reps = len(
            etbb[0].coords[scalar_evaluation.BOOTSTRAP_REP_DIM].values
        )
        this_num_bootstrap_reps = len(
            etbb[i].coords[scalar_evaluation.BOOTSTRAP_REP_DIM].values
        )
        assert num_bootstrap_reps == this_num_bootstrap_reps

        these_prediction_file_names = (
            etbb[i].attrs[scalar_evaluation.PREDICTION_FILES_KEY]
        )
        this_cyclone_id_string = prediction_io.file_name_to_cyclone_id(
            these_prediction_file_names[0]
        )
        basin_id_strings[i] = misc_utils.parse_cyclone_id(
            this_cyclone_id_string
        )[1]

    etbb = eval_table_by_basin

    for metric_name in scalar_eval_plotting.BASIC_METRIC_NAMES:
        for target_field_name in scalar_eval_plotting.BASIC_TARGET_FIELD_NAMES:
            metric_matrix = numpy.full(
                (num_basins, num_bootstrap_reps), numpy.nan
            )

            for i in range(num_basins):
                j = numpy.where(
                    etbb[i].coords[scalar_evaluation.TARGET_FIELD_DIM].values
                    == target_field_name
                )[0][0]

                metric_matrix[i, :] = etbb[i][metric_name].values[j, :]

            figure_object, axes_object = (
                scalar_eval_plotting.plot_metric_by_category(
                    metric_matrix=metric_matrix,
                    metric_name=metric_name,
                    target_field_name=target_field_name,
                    category_description_strings=basin_id_strings,
                    x_label_string='Basin',
                    confidence_level=confidence_level
                )
            )
            axes_object.set_xticklabels(basin_id_strings)

            output_file_name = '{0:s}/{1:s}_{2:s}_by-basin.jpg'.format(
                output_dir_name,
                target_field_name.replace('_', '-'),
                metric_name.replace('_', '-')
            )

            print('Saving figure to: "{0:s}"...'.format(output_file_name))
            figure_object.savefig(
                output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                bbox_inches='tight'
            )
            pyplot.close(figure_object)

    for metric_name in scalar_eval_plotting.ADVANCED_METRIC_NAMES:
        metric_matrix = numpy.full(
            (num_basins, num_bootstrap_reps), numpy.nan
        )

        for i in range(num_basins):
            if metric_name == scalar_eval_plotting.NUM_EXAMPLES_KEY:
                metric_matrix[i, :] = numpy.sum(
                    etbb[i][scalar_evaluation.OFFSET_DIST_BIN_COUNT_KEY].values
                )
            else:
                metric_matrix[i, :] = etbb[i][metric_name].values[:]

        figure_object, axes_object = (
            scalar_eval_plotting.plot_metric_by_category(
                metric_matrix=metric_matrix,
                metric_name=metric_name,
                target_field_name=scalar_evaluation.OFFSET_DISTANCE_NAME,
                category_description_strings=basin_id_strings,
                x_label_string='Basin',
                confidence_level=confidence_level
            )
        )
        axes_object.set_xticklabels(basin_id_strings)

        output_file_name = '{0:s}/{1:s}_by-basin.jpg'.format(
            output_dir_name,
            metric_name.replace('_', '-')
        )

        print('Saving figure to: "{0:s}"...'.format(output_file_name))
        figure_object.savefig(
            output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
            bbox_inches='tight'
        )
        pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        evaluation_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
