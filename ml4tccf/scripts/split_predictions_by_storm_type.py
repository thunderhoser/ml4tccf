"""Splits predictions by storm type."""

import glob
import argparse
import warnings
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from ml4tccf.io import a_deck_io
from ml4tccf.io import prediction_io
from ml4tccf.utils import scalar_prediction_utils
from ml4tccf.machine_learning import neural_net_training_cira_ir as nn_training
from ml4tccf.scripts import split_predictions_by_basin as split_by_basin

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H%M'

SYNOPTIC_TIME_TOLERANCE_SEC = nn_training.SYNOPTIC_TIME_TOLERANCE_SEC

TROPICAL_TYPE_STRING = 'tropical'
SUBTROPICAL_TYPE_STRING = 'subtropical'
EXTRATROPICAL_TYPE_STRING = 'extratropical'
DISTURBANCE_TYPE_STRING = 'disturbance'
MISC_TYPE_STRING = 'miscellaneous'

NON_TROPICAL_TYPE_STRING = 'non_tropical'

INPUT_PREDICTION_FILES_ARG_NAME = 'input_prediction_file_pattern'
A_DECK_FILE_ARG_NAME = 'input_a_deck_file_name'
MAKE_STORM_TYPES_BINARY_ARG_NAME = 'make_storm_types_binary'
OUTPUT_DIR_ARG_NAME = 'output_prediction_dir_name'

INPUT_PREDICTION_FILES_HELP_STRING = (
    'Glob pattern for input files.  Each will be read by '
    '`prediction_io.read_file`.'
)
A_DECK_FILE_HELP_STRING = (
    'Path to A-deck file, containing storm types.  Will be read by '
    '`a_deck_io.read_file`.'
)
MAKE_STORM_TYPES_BINARY_HELP_STRING = (
    'Boolean flag.  If 1, storm types will be binary (tropical vs. '
    'non-tropical).  If 0, there will be five storm types.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Within this directory, one '
    'subdirectory will be created for every storm type.  Then subset '
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
    '--' + A_DECK_FILE_ARG_NAME, type=str, required=True,
    help=A_DECK_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAKE_STORM_TYPES_BINARY_ARG_NAME, type=int, required=True,
    help=MAKE_STORM_TYPES_BINARY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_prediction_file_pattern, a_deck_file_name,
         make_storm_types_binary, top_output_prediction_dir_name):
    """Splits predictions by storm type.

    This is effectively the main method.

    :param input_prediction_file_pattern: See documentation at top of file.
    :param a_deck_file_name: Same.
    :param make_storm_types_binary: Same.
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

    print('Reading data from: "{0:s}"...'.format(a_deck_file_name))
    a_deck_table_xarray = a_deck_io.read_file(a_deck_file_name)
    a_deck_table_xarray = a_deck_io.storm_types_to_1hot_encoding(
        a_deck_table_xarray
    )
    a_deck_times_unix_sec = a_deck_table_xarray[a_deck_io.VALID_TIME_KEY]

    try:
        a_deck_cyclone_id_strings = numpy.array([
            s.decode('utf-8')
            for s in a_deck_table_xarray[a_deck_io.CYCLONE_ID_KEY].values
        ])
    except AttributeError:
        a_deck_cyclone_id_strings = (
            a_deck_table_xarray[a_deck_io.CYCLONE_ID_KEY].values
        )

    # Find storm type corresponding to each prediction.
    num_examples = len(pt[scalar_prediction_utils.TARGET_TIME_KEY].values)
    prediction_storm_type_strings = numpy.full(num_examples, '', dtype=object)
    print(SEPARATOR_STRING)

    for i in range(num_examples):
        if numpy.mod(i, 100) == 0:
            print((
                'Have sought storm type for {0:d} of {1:d} examples...'
            ).format(
                i, num_examples
            ))

        this_cyclone_id_string = (
            pt[scalar_prediction_utils.CYCLONE_ID_KEY].values[i].decode('utf-8')
        )
        this_time_unix_sec = (
            pt[scalar_prediction_utils.TARGET_TIME_KEY].values[i]
        )

        these_indices = numpy.where(numpy.logical_and(
            a_deck_cyclone_id_strings == this_cyclone_id_string,
            numpy.absolute(a_deck_times_unix_sec - this_time_unix_sec) <=
            SYNOPTIC_TIME_TOLERANCE_SEC
        ))[0]

        if len(these_indices) == 0:
            warning_string = (
                'POTENTIAL ERROR: Cannot find cyclone {0:s} within {1:d} '
                'seconds of {2:s} in A-deck data.'
            ).format(
                this_cyclone_id_string,
                SYNOPTIC_TIME_TOLERANCE_SEC,
                time_conversion.unix_sec_to_string(
                    this_time_unix_sec, TIME_FORMAT_FOR_LOG_MESSAGES
                )
            )

            warnings.warn(warning_string)
            continue

        if len(these_indices) > 1:
            warning_string = (
                'POTENTIAL ERROR: Found {0:d} instances of cyclone {1:s} '
                'within {2:d} seconds of {3:s} in A-deck data:\n{4:s}'
            ).format(
                len(these_indices),
                this_cyclone_id_string,
                SYNOPTIC_TIME_TOLERANCE_SEC,
                time_conversion.unix_sec_to_string(
                    this_time_unix_sec, TIME_FORMAT_FOR_LOG_MESSAGES
                ),
                str(a_deck_table_xarray.isel(
                    indexers={a_deck_io.STORM_OBJECT_DIM: these_indices}
                ))
            )

            warnings.warn(warning_string)

        adt = a_deck_table_xarray
        j = these_indices[0]

        if make_storm_types_binary:
            if adt[a_deck_io.UNNORM_TROPICAL_FLAG_KEY].values[j] == 1:
                prediction_storm_type_strings[i] = TROPICAL_TYPE_STRING
            else:
                prediction_storm_type_strings[i] = NON_TROPICAL_TYPE_STRING

            continue

        if adt[a_deck_io.UNNORM_TROPICAL_FLAG_KEY].values[j] == 1:
            prediction_storm_type_strings[i] = TROPICAL_TYPE_STRING
        elif adt[a_deck_io.UNNORM_SUBTROPICAL_FLAG_KEY].values[j] == 1:
            prediction_storm_type_strings[i] = SUBTROPICAL_TYPE_STRING
        elif adt[a_deck_io.UNNORM_EXTRATROPICAL_FLAG_KEY].values[j] == 1:
            prediction_storm_type_strings[i] = EXTRATROPICAL_TYPE_STRING
        elif adt[a_deck_io.UNNORM_DISTURBANCE_FLAG_KEY].values[j] == 1:
            prediction_storm_type_strings[i] = DISTURBANCE_TYPE_STRING
        else:
            prediction_storm_type_strings[i] = MISC_TYPE_STRING

    print('Have sought storm type for all {0:d} examples!'.format(num_examples))
    print(SEPARATOR_STRING)

    unique_storm_type_strings = numpy.unique(prediction_storm_type_strings)

    for this_storm_type_string in unique_storm_type_strings:
        this_output_dir_name = '{0:s}/storm-type={1:s}'.format(
            top_output_prediction_dir_name, this_storm_type_string
        )
        file_system_utils.mkdir_recursive_if_necessary(
            directory_name=this_output_dir_name
        )

        these_indices = numpy.where(
            prediction_storm_type_strings == this_storm_type_string
        )[0]

        split_by_basin._write_scalar_predictions_1basin(
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
        a_deck_file_name=getattr(INPUT_ARG_OBJECT, A_DECK_FILE_ARG_NAME),
        make_storm_types_binary=bool(
            getattr(INPUT_ARG_OBJECT, MAKE_STORM_TYPES_BINARY_ARG_NAME)
        ),
        top_output_prediction_dir_name=getattr(
            INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME
        )
    )
