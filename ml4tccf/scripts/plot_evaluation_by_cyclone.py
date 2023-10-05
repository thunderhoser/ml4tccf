"""Plots evaluation metrics as a function of individual TC."""

import glob
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import xarray
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4tccf.io import prediction_io
from ml4tccf.utils import scalar_evaluation
from ml4tccf.plotting import scalar_evaluation_plotting as scalar_eval_plotting

FIGURE_RESOLUTION_DPI = 300

INPUT_FILES_ARG_NAME = 'input_file_names_or_pattern'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
CYCLONE_ID_FONT_SIZE_ARG_NAME = 'cyclone_id_font_size'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILES_HELP_STRING = (
    'This argument may be formatted in two ways: (1) a space-separated list of '
    'paths to evaluation files or (2) a glob pattern for evaluation files.  '
    'Either way, there must be one evaluation file per TC, and they will be '
    'read by `scalar_evaluation.read_file` or `gridded_evaluation.read_file`.'
)
CONFIDENCE_LEVEL_HELP_STRING = (
    'Confidence level for uncertainty intervals.  Must be in range 0...1.'
)
CYCLONE_ID_FONT_SIZE_HELP_STRING = 'Font size for cyclone-ID labels on x-axis.'
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
    '--' + CYCLONE_ID_FONT_SIZE_ARG_NAME, type=float, required=False,
    default=36, help=CYCLONE_ID_FONT_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_file_names_or_pattern, confidence_level, cyclone_id_font_size,
         output_dir_name):
    """Plots evaluation metrics as a function of individual TC.

    This is effectively the main method.

    :param input_file_names_or_pattern: See documentation at top of file.
    :param confidence_level: Same.
    :param cyclone_id_font_size: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    if len(input_file_names_or_pattern) == 1:
        input_file_names = glob.glob(input_file_names_or_pattern[0])
        input_file_names.sort()
    else:
        input_file_names = input_file_names_or_pattern

    assert len(input_file_names) > 0

    error_checking.assert_is_geq(confidence_level, 0.8)
    error_checking.assert_is_less_than(confidence_level, 1.)
    error_checking.assert_is_greater(cyclone_id_font_size, 0.)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Do actual stuff.
    num_cyclones = len(input_file_names)
    eval_table_by_cyclone = [xarray.Dataset()] * num_cyclones
    cyclone_id_strings = [''] * num_cyclones
    num_bootstrap_reps = -1

    for i in range(num_cyclones):
        print('Reading data from: "{0:s}"...'.format(input_file_names[i]))

        try:
            eval_table_by_cyclone[i] = scalar_evaluation.read_file(
                input_file_names[i]
            )
            assert (
                scalar_evaluation.TARGET_FIELD_DIM
                in eval_table_by_cyclone[i].coords
            )
        except:
            raise ValueError(
                'This script does not yet work for gridded predictions.'
            )

        etbc = eval_table_by_cyclone

        num_bootstrap_reps = len(
            etbc[0].coords[scalar_evaluation.BOOTSTRAP_REP_DIM].values
        )
        this_num_bootstrap_reps = len(
            etbc[i].coords[scalar_evaluation.BOOTSTRAP_REP_DIM].values
        )
        assert num_bootstrap_reps == this_num_bootstrap_reps

        these_prediction_file_names = (
            etbc[i].attrs[scalar_evaluation.PREDICTION_FILES_KEY]
        )
        assert len(these_prediction_file_names) == 1

        cyclone_id_strings[i] = prediction_io.file_name_to_cyclone_id(
            these_prediction_file_names[0]
        )

    etbc = eval_table_by_cyclone

    for metric_name in scalar_eval_plotting.BASIC_METRIC_NAMES:
        for target_field_name in scalar_eval_plotting.BASIC_TARGET_FIELD_NAMES:
            metric_matrix = numpy.full(
                (num_cyclones, num_bootstrap_reps), numpy.nan
            )

            for i in range(num_cyclones):
                j = numpy.where(
                    etbc[i].coords[scalar_evaluation.TARGET_FIELD_DIM].values
                    == target_field_name
                )[0][0]

                metric_matrix[i, :] = etbc[i][metric_name].values[j, :]

            figure_object, axes_object = (
                scalar_eval_plotting.plot_metric_by_category(
                    metric_matrix=metric_matrix,
                    metric_name=metric_name,
                    target_field_name=target_field_name,
                    category_description_strings=cyclone_id_strings,
                    x_label_string='Cyclone ID',
                    confidence_level=confidence_level
                )
            )
            axes_object.set_xticklabels(
                cyclone_id_strings, fontdict={'fontsize': cyclone_id_font_size},
                rotation=90.
            )

            output_file_name = '{0:s}/{1:s}_{2:s}_by-cyclone.jpg'.format(
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
            (num_cyclones, num_bootstrap_reps), numpy.nan
        )

        for i in range(num_cyclones):
            if metric_name == scalar_eval_plotting.NUM_EXAMPLES_KEY:
                metric_matrix[i, :] = numpy.sum(
                    etbc[i][scalar_evaluation.OFFSET_DIST_BIN_COUNT_KEY].values
                )
            else:
                metric_matrix[i, :] = etbc[i][metric_name].values[:]

        figure_object, axes_object = (
            scalar_eval_plotting.plot_metric_by_category(
                metric_matrix=metric_matrix,
                metric_name=metric_name,
                target_field_name=scalar_evaluation.OFFSET_DISTANCE_NAME,
                category_description_strings=cyclone_id_strings,
                x_label_string='Cyclone ID',
                confidence_level=confidence_level
            )
        )
        axes_object.set_xticklabels(
            cyclone_id_strings, fontdict={'fontsize': cyclone_id_font_size},
            rotation=90.
        )

        output_file_name = '{0:s}/{1:s}_by-cyclone.jpg'.format(
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
        input_file_names_or_pattern=getattr(
            INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME
        ),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        cyclone_id_font_size=getattr(
            INPUT_ARG_OBJECT, CYCLONE_ID_FONT_SIZE_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
