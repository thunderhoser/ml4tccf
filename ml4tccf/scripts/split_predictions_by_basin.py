"""Splits predictions by ocean basin."""

import glob
import argparse
import numpy
from gewittergefahr.gg_utils import file_system_utils
from ml4tccf.io import prediction_io
from ml4tccf.io import scalar_prediction_io
from ml4tccf.utils import misc_utils
from ml4tccf.utils import scalar_prediction_utils
from ml4tccf.machine_learning import neural_net_training_cira_ir as nn_training

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
SYNOPTIC_TIME_TOLERANCE_SEC = nn_training.SYNOPTIC_TIME_TOLERANCE_SEC

HOURS_TO_SECONDS = 3600
TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H%M'

INPUT_PREDICTION_FILES_ARG_NAME = 'input_prediction_file_pattern'
OUTPUT_DIR_ARG_NAME = 'output_prediction_dir_name'

INPUT_PREDICTION_FILES_HELP_STRING = (
    'Glob pattern for input files.  Each will be read by '
    '`prediction_io.read_file`.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Within this directory, one '
    'subdirectory will be created for every ocean basin.  Then subset '
    'predictions will be written to these subdirectories by '
    '`scalar_prediction_io.write_file` or `gridded_prediction_io.write_file`, '
    'to exact locations determined by `prediction_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_PREDICTION_FILES_ARG_NAME, type=str, required=True,
    help=INPUT_PREDICTION_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _write_scalar_predictions_1basin(prediction_table_1basin_xarray,
                                     output_dir_name_1basin):
    """Writes scalar predictions for one ocean basin.

    :param prediction_table_1basin_xarray: xarray table with predictions for the
        one basin.
    :param output_dir_name_1basin: Name of output directory for the one basin.
    """

    all_cyclone_id_strings = numpy.array([
        s.decode('utf-8') for s in
        prediction_table_1basin_xarray[
            scalar_prediction_utils.CYCLONE_ID_KEY
        ].values
    ])

    unique_cyclone_id_strings = numpy.unique(all_cyclone_id_strings)

    for cyclone_id_string in unique_cyclone_id_strings:
        these_indices = numpy.where(
            all_cyclone_id_strings == cyclone_id_string
        )[0]
        prediction_table_xarray_1cyclone = prediction_table_1basin_xarray.isel(
            indexers={scalar_prediction_utils.EXAMPLE_DIM_KEY: these_indices}
        )

        output_file_name = prediction_io.find_file(
            directory_name=output_dir_name_1basin,
            cyclone_id_string=cyclone_id_string,
            raise_error_if_missing=False
        )

        pt1cyc = prediction_table_xarray_1cyclone

        target_matrix = numpy.transpose(numpy.vstack((
            pt1cyc[scalar_prediction_utils.ACTUAL_ROW_OFFSET_KEY].values,
            pt1cyc[scalar_prediction_utils.ACTUAL_COLUMN_OFFSET_KEY].values,
            pt1cyc[scalar_prediction_utils.GRID_SPACING_KEY].values,
            pt1cyc[scalar_prediction_utils.ACTUAL_CENTER_LATITUDE_KEY].values
        )))

        prediction_matrix = numpy.stack((
            pt1cyc[scalar_prediction_utils.PREDICTED_ROW_OFFSET_KEY].values,
            pt1cyc[scalar_prediction_utils.PREDICTED_COLUMN_OFFSET_KEY].values
        ), axis=-2)

        print('Writing data to: "{0:s}"...'.format(output_file_name))
        scalar_prediction_io.write_file(
            netcdf_file_name=output_file_name,
            target_matrix=target_matrix,
            prediction_matrix=prediction_matrix,
            cyclone_id_string=cyclone_id_string,
            target_times_unix_sec=
            pt1cyc[scalar_prediction_utils.TARGET_TIME_KEY].values,
            model_file_name=
            pt1cyc.attrs[scalar_prediction_utils.MODEL_FILE_KEY],
            isotonic_model_file_name=pt1cyc.attrs[
                scalar_prediction_utils.ISOTONIC_MODEL_FILE_KEY
            ],
            uncertainty_calib_model_file_name=pt1cyc.attrs[
                scalar_prediction_utils.UNCERTAINTY_CALIB_MODEL_FILE_KEY
            ]
        )


def _run(input_prediction_file_pattern, top_output_prediction_dir_name):
    """Splits predictions by ocean basin.

    This is effectively the main method.

    :param input_prediction_file_pattern: See documentation at top of file.
    :param top_output_prediction_dir_name: Same.
    :raises: ValueError: if no prediction files could be found.
    """

    # Read files.

    # TODO(thunderhoser): I should probably modularize the globbing, reading of
    # multiple files, and concatenating -- all into 1-2 methods.
    input_prediction_file_names = glob.glob(input_prediction_file_pattern)
    if len(input_prediction_file_names) == 0:
        error_string = (
            'Could not find any prediction file with the following pattern: '
            '"{0:s}"'
        ).format(input_prediction_file_pattern)

        raise ValueError(error_string)

    input_prediction_file_names.sort()

    num_files = len(input_prediction_file_names)
    prediction_tables_xarray = [None] * num_files
    are_predictions_gridded = False

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(
            input_prediction_file_names[i]
        ))
        prediction_tables_xarray[i] = prediction_io.read_file(
            input_prediction_file_names[i]
        )

        are_predictions_gridded = (
            scalar_prediction_utils.PREDICTED_ROW_OFFSET_KEY
            not in prediction_tables_xarray[i]
        )

        if are_predictions_gridded:
            raise ValueError(
                'This script does not yet work for gridded predictions.'
            )

    prediction_table_xarray = scalar_prediction_utils.concat_over_examples(
        prediction_tables_xarray
    )
    pt = prediction_table_xarray

    cyclone_id_strings = pt[scalar_prediction_utils.CYCLONE_ID_KEY].values
    basin_id_strings = numpy.array([
        misc_utils.parse_cyclone_id(c.decode('utf-8'))[1]
        for c in cyclone_id_strings
    ])
    unique_basin_id_strings = numpy.unique(basin_id_strings)

    for this_basin_id_string in unique_basin_id_strings:
        this_output_dir_name = '{0:s}/basin={1:s}'.format(
            top_output_prediction_dir_name, this_basin_id_string
        )
        file_system_utils.mkdir_recursive_if_necessary(
            directory_name=this_output_dir_name
        )

        these_indices = numpy.where(basin_id_strings == this_basin_id_string)[0]

        _write_scalar_predictions_1basin(
            prediction_table_1basin_xarray=prediction_table_xarray.isel(
                indexers=
                {scalar_prediction_utils.EXAMPLE_DIM_KEY: these_indices}
            ),
            output_dir_name_1basin=this_output_dir_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_file_pattern=getattr(
            INPUT_ARG_OBJECT, INPUT_PREDICTION_FILES_ARG_NAME
        ),
        top_output_prediction_dir_name=getattr(
            INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME
        )
    )
