"""Plots output of generator (predictors and target) for one batch.

NOTE: This script works only for semantic segmentation (i.e., gridded
prediction) -- not for scalar prediction (of just the x- and y-coords).
"""

import os
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from ml4tccf.io import border_io
from ml4tccf.utils import misc_utils
from ml4tccf.machine_learning import neural_net
from ml4tccf.scripts import plot_predictions

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DUMMY_CYCLONE_ID_STRING = '1900AL01'
DUMMY_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '1900-01-01-00', '%Y-%m-%d-%H'
)
DUMMY_END_LATITUDES_DEG_N = numpy.array([32, 33], dtype=float)
DUMMY_END_LONGITUDES_DEG_E = numpy.array([-43, -42], dtype=float)

MODEL_FILE_ARG_NAME = 'input_model_file_name'
SATELLITE_DIR_ARG_NAME = 'input_satellite_dir_name'
ARE_DATA_NORMALIZED_ARG_NAME = 'are_data_normalized'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
MAX_EXAMPLES_PER_CYCLONE_ARG_NAME = 'max_examples_per_cyclone'
YEARS_ARG_NAME = 'years'
MIN_TARGET_ARG_NAME = 'min_target_value'
MAX_TARGET_ARG_NAME = 'max_target_value'
MIN_TARGET_PERCENTILE_ARG_NAME = 'min_target_percentile'
MAX_TARGET_PERCENTILE_ARG_NAME = 'max_target_percentile'
TARGET_COLOUR_MAP_ARG_NAME = 'target_colour_map_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model (will be read by `neural_net.read_model`).'
)
SATELLITE_DIR_HELP_STRING = (
    'Name of directory with satellite data (predictors).  Files therein will '
    'be found by `satellite_io.find_file` and read by `satellite_io.read_file`.'
)
ARE_DATA_NORMALIZED_HELP_STRING = (
    'Boolean flag.  If 1 (0), plotting code will assume that satellite data '
    'are (un)normalized.'
)
NUM_EXAMPLES_HELP_STRING = 'Number of examples to generate.'
MAX_EXAMPLES_PER_CYCLONE_HELP_STRING = (
    'Max number of examples to generate for one cyclone.'
)
YEARS_HELP_STRING = 'List of years from which to generate examples.'
MIN_TARGET_HELP_STRING = (
    'Minimum target value to show in colour scheme.  If you want to specify '
    'colour limits with percentiles instead, leave this argument alone.'
)
MAX_TARGET_HELP_STRING = 'Same as `{0:s}` but for max.'.format(
    MIN_TARGET_HELP_STRING
)
MIN_TARGET_PERCENTILE_HELP_STRING = (
    'Minimum target value to show in colour scheme, stated as a percentile '
    '(from 0...100) over all values in the grid.  If you want to specify '
    'colour limits with raw values instead, leave this argument alone.'
)
MAX_TARGET_PERCENTILE_HELP_STRING = 'Same as `{0:s}` but for max.'.format(
    MIN_TARGET_PERCENTILE_HELP_STRING
)
TARGET_COLOUR_MAP_HELP_STRING = (
    'Name of colour scheme used for target values.  Must be accepted by '
    '`pyplot.get_cmap`.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SATELLITE_DIR_ARG_NAME, type=str, required=True,
    help=SATELLITE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ARE_DATA_NORMALIZED_ARG_NAME, type=int, required=True,
    help=ARE_DATA_NORMALIZED_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=True,
    help=NUM_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_EXAMPLES_PER_CYCLONE_ARG_NAME, type=int, required=True,
    help=MAX_EXAMPLES_PER_CYCLONE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + YEARS_ARG_NAME, type=int, nargs='+', required=True,
    help=YEARS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_TARGET_ARG_NAME, type=float, required=False, default=1,
    help=MIN_TARGET_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_TARGET_ARG_NAME, type=float, required=False, default=0,
    help=MAX_TARGET_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_TARGET_PERCENTILE_ARG_NAME, type=float, required=False,
    default=100, help=MIN_TARGET_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_TARGET_PERCENTILE_ARG_NAME, type=float, required=False,
    default=0, help=MAX_TARGET_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_COLOUR_MAP_ARG_NAME, type=str, required=False,
    default='BuGn', help=TARGET_COLOUR_MAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(model_file_name, satellite_dir_name, are_data_normalized,
         num_examples, max_examples_per_cyclone, years,
         min_target_value, max_target_value,
         min_target_value_percentile, max_target_value_percentile,
         target_colour_map_name, output_dir_name):
    """Plots output of generator (predictors and target) for one batch.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param satellite_dir_name: Same.
    :param are_data_normalized: Same.
    :param num_examples: Same.
    :param max_examples_per_cyclone: Same.
    :param years: Same.
    :param min_target_value: Same.
    :param max_target_value: Same.
    :param min_target_value_percentile: Same.
    :param max_target_value_percentile: Same.
    :param target_colour_map_name: Same.
    :param output_dir_name: Same.
    """

    if min_target_value >= max_target_value:
        min_target_value = None
        max_target_value = None

        error_checking.assert_is_geq(min_target_value_percentile, 50.)
        error_checking.assert_is_leq(max_target_value_percentile, 100.)
        error_checking.assert_is_greater(
            max_target_value_percentile, min_target_value_percentile
        )
    else:
        error_checking.assert_is_greater(min_target_value, 0.)
        error_checking.assert_is_greater(max_target_value, 0.)

    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    assert training_option_dict[neural_net.SEMANTIC_SEG_FLAG_KEY]
    training_option_dict[neural_net.SATELLITE_DIRECTORY_KEY] = (
        satellite_dir_name
    )
    training_option_dict[neural_net.YEARS_KEY] = years
    training_option_dict[neural_net.BATCH_SIZE_KEY] = num_examples
    training_option_dict[neural_net.MAX_EXAMPLES_PER_CYCLONE_KEY] = (
        max_examples_per_cyclone
    )
    training_option_dict[neural_net.DATA_AUG_NUM_TRANS_KEY] = 1

    # TODO(thunderhoser): Remove these two lines later.
    training_option_dict[neural_net.DATA_AUG_MEAN_TRANS_KEY] = 100.
    training_option_dict[neural_net.DATA_AUG_STDEV_TRANS_KEY] = 25.

    model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY] = training_option_dict

    print(SEPARATOR_STRING)
    generator_handle = neural_net.data_generator(training_option_dict)
    predictor_matrices, target_matrix = next(generator_handle)
    target_matrix = target_matrix[..., 0]
    print(SEPARATOR_STRING)

    num_examples = predictor_matrices[0].shape[0]
    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    num_grid_rows = target_matrix.shape[1]
    num_grid_columns = target_matrix.shape[2]

    dummy_low_res_grid_latitudes_deg_n = numpy.linspace(
        DUMMY_END_LATITUDES_DEG_N[0], DUMMY_END_LATITUDES_DEG_N[1],
        num=num_grid_rows, dtype=float
    )
    dummy_low_res_grid_longitudes_deg_e = numpy.linspace(
        DUMMY_END_LONGITUDES_DEG_E[0], DUMMY_END_LONGITUDES_DEG_E[1],
        num=num_grid_columns, dtype=float
    )
    dummy_high_res_grid_latitudes_deg_n = numpy.linspace(
        DUMMY_END_LATITUDES_DEG_N[0], DUMMY_END_LATITUDES_DEG_N[1],
        num=num_grid_rows * 4, dtype=float
    )
    dummy_high_res_grid_longitudes_deg_e = numpy.linspace(
        DUMMY_END_LONGITUDES_DEG_E[0], DUMMY_END_LONGITUDES_DEG_E[1],
        num=num_grid_columns * 4, dtype=float
    )

    for i in range(num_examples):
        row_offset_px, column_offset_px = misc_utils.target_matrix_to_centroid(
            target_matrix[i, ...]
        )

        output_file_name = '{0:s}/example{1:06d}.png'.format(output_dir_name, i)

        if min_target_value is None:
            this_min_target_value = numpy.percentile(
                target_matrix[i, ...], min_target_value_percentile
            )
            this_max_target_value = numpy.percentile(
                target_matrix[i, ...], max_target_value_percentile
            )

            if this_max_target_value - this_min_target_value < 0.01:
                new_max = this_min_target_value + 0.01
                if new_max > 1:
                    this_min_target_value = this_max_target_value - 0.01
                else:
                    this_max_target_value = this_min_target_value + 0.01
        else:
            this_min_target_value = min_target_value + 0.
            this_max_target_value = max_target_value + 0.

        print('SUM OF TARGETS OVER GRID = {0:f}'.format(
            numpy.sum(target_matrix[i, ...])
        ))

        plot_predictions._plot_data_one_example(
            predictor_matrices=[p[i, ...] for p in predictor_matrices],
            scalar_target_values=numpy.array([row_offset_px, column_offset_px]),
            prediction_matrix=target_matrix[i, ...],
            model_metadata_dict=model_metadata_dict,
            cyclone_id_string=DUMMY_CYCLONE_ID_STRING,
            target_time_unix_sec=DUMMY_TIME_UNIX_SEC,
            low_res_latitudes_deg_n=dummy_low_res_grid_latitudes_deg_n,
            low_res_longitudes_deg_e=dummy_low_res_grid_longitudes_deg_e,
            high_res_latitudes_deg_n=dummy_high_res_grid_latitudes_deg_n,
            high_res_longitudes_deg_e=dummy_high_res_grid_longitudes_deg_e,
            are_data_normalized=are_data_normalized,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            output_file_name=output_file_name,
            min_gridded_prob=this_min_target_value,
            max_gridded_prob=this_max_target_value,
            prob_colour_map_name=target_colour_map_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        satellite_dir_name=getattr(INPUT_ARG_OBJECT, SATELLITE_DIR_ARG_NAME),
        are_data_normalized=bool(getattr(
            INPUT_ARG_OBJECT, ARE_DATA_NORMALIZED_ARG_NAME
        )),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        max_examples_per_cyclone=getattr(
            INPUT_ARG_OBJECT, MAX_EXAMPLES_PER_CYCLONE_ARG_NAME
        ),
        years=numpy.array(getattr(INPUT_ARG_OBJECT, YEARS_ARG_NAME), dtype=int),
        min_target_value=getattr(INPUT_ARG_OBJECT, MIN_TARGET_ARG_NAME),
        max_target_value=getattr(INPUT_ARG_OBJECT, MAX_TARGET_ARG_NAME),
        min_target_value_percentile=getattr(
            INPUT_ARG_OBJECT, MIN_TARGET_PERCENTILE_ARG_NAME
        ),
        max_target_value_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_TARGET_PERCENTILE_ARG_NAME
        ),
        target_colour_map_name=getattr(
            INPUT_ARG_OBJECT, TARGET_COLOUR_MAP_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
