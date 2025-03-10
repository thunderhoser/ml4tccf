"""Plots output of generator (predictors and target) for one batch."""

import os
import argparse
import numpy
from ml4tccf.io import border_io
from ml4tccf.machine_learning import neural_net_utils as nn_utils
from ml4tccf.machine_learning import \
    neural_net_training_cira_ir as nn_training_cira_ir
from ml4tccf.machine_learning import \
    neural_net_training_simple as nn_training_simple
from ml4tccf.machine_learning import \
    neural_net_training_fancy as nn_training_fancy
from ml4tccf.scripts import plot_predictions

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DUMMY_CYCLONE_ID_STRING = '1900AL01'
DUMMY_END_LATITUDES_DEG_N = numpy.array([32, 33], dtype=float)
DUMMY_END_LONGITUDES_DEG_E = numpy.array([-43, -42], dtype=float)

MODEL_FILE_ARG_NAME = 'input_model_file_name'
SATELLITE_DIR_ARG_NAME = 'input_satellite_dir_name'
ARE_DATA_NORMALIZED_ARG_NAME = 'are_data_normalized'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
MAX_EXAMPLES_PER_CYCLONE_ARG_NAME = 'max_examples_per_cyclone'
YEARS_ARG_NAME = 'years'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model (will be read by `nn_utils.read_model`).'
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
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(model_file_name, satellite_dir_name, are_data_normalized,
         num_examples, max_examples_per_cyclone, years, output_dir_name):
    """Plots output of generator (predictors and target) for one batch.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param satellite_dir_name: Same.
    :param are_data_normalized: Same.
    :param num_examples: Same.
    :param max_examples_per_cyclone: Same.
    :param years: Same.
    :param output_dir_name: Same.
    """

    model_metafile_name = nn_utils.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = nn_utils.read_metafile(model_metafile_name)
    training_option_dict = model_metadata_dict[
        nn_utils.TRAINING_OPTIONS_KEY
    ]
    assert not training_option_dict[nn_utils.SEMANTIC_SEG_FLAG_KEY]

    training_option_dict[nn_utils.SATELLITE_DIRECTORY_KEY] = satellite_dir_name
    training_option_dict[nn_utils.YEARS_KEY] = years
    training_option_dict[nn_utils.MAX_EXAMPLES_PER_CYCLONE_KEY] = (
        max_examples_per_cyclone
    )
    training_option_dict[nn_utils.DATA_AUG_NUM_TRANS_KEY] = 1
    model_metadata_dict[nn_utils.TRAINING_OPTIONS_KEY] = training_option_dict

    data_type_string = model_metadata_dict[nn_utils.DATA_TYPE_KEY]

    if data_type_string == nn_utils.CIRA_IR_DATA_TYPE_STRING:
        generator_handle = nn_training_cira_ir.data_generator(
            training_option_dict
        )
    elif data_type_string == nn_utils.RG_SIMPLE_DATA_TYPE_STRING:
        generator_handle = nn_training_simple.data_generator(
            training_option_dict
        )
    else:
        generator_handle = nn_training_fancy.data_generator(
            training_option_dict
        )

    print(SEPARATOR_STRING)

    num_examples_read = 0
    num_examples_plotted = 0
    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    while num_examples_read < num_examples:
        predictor_matrices, target_matrix = next(generator_handle)
        print(SEPARATOR_STRING)

        this_num_examples = predictor_matrices[0].shape[0]
        num_grid_rows = predictor_matrices[0].shape[1]
        num_grid_columns = predictor_matrices[0].shape[2]

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

        for i in range(this_num_examples):
            if num_examples_plotted >= num_examples:
                break

            output_file_name = '{0:s}/example{1:06d}.png'.format(
                output_dir_name, i + num_examples_read
            )
            num_examples_plotted += 1

            dummy_prediction_matrix = numpy.expand_dims(
                numpy.transpose(target_matrix[i, :2]),
                axis=-1
            )

            title_string = 'Row/column trans = {0:.1f}, {1:.1f}'.format(
                target_matrix[i, 0],
                target_matrix[i, 1]
            )

            plot_predictions._plot_data_one_example(
                predictor_matrices=[p[i, ...] for p in predictor_matrices],
                scalar_target_values=target_matrix[i, :2],
                prediction_matrix=dummy_prediction_matrix,
                model_metadata_dict=model_metadata_dict,
                cyclone_id_string=title_string,
                low_res_latitudes_deg_n=dummy_low_res_grid_latitudes_deg_n,
                low_res_longitudes_deg_e=dummy_low_res_grid_longitudes_deg_e,
                high_res_latitudes_deg_n=dummy_high_res_grid_latitudes_deg_n,
                high_res_longitudes_deg_e=dummy_high_res_grid_longitudes_deg_e,
                are_data_normalized=are_data_normalized,
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                output_file_name=output_file_name
            )

        num_examples_read += this_num_examples


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
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
