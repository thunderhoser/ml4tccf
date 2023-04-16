"""Creates PIT (prob integ transform) histogram for each target variable."""

import os
import sys
import glob
import argparse

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import prediction_io
import scalar_prediction_utils
import pit_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILE_PATTERN_ARG_NAME = 'input_prediction_file_pattern'
NUM_BINS_ARG_NAME = 'num_bins'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_PATTERN_HELP_STRING = (
    'Glob pattern for input files.  Each will be read by '
    '`prediction_io.read_file`, and predictions in all these files will be '
    'evaluated together.'
)
NUM_BINS_HELP_STRING = 'Number of bins in each histogram.'
OUTPUT_FILE_HELP_STRING = (
    'Path to output (NetCDF) file.  Results will be written here by '
    '`pit_utils.write_results`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_PATTERN_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_PATTERN_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BINS_ARG_NAME, type=int, required=True,
    help=NUM_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(prediction_file_pattern, num_bins, output_file_name):
    """Creates PIT (prob integ transform) histogram for each target variable.

    This is effectively the main method.

    :param prediction_file_pattern: See documentation at top of file.
    :param num_bins: Same.
    :param output_file_name: Same.
    """

    prediction_file_names = glob.glob(prediction_file_pattern)
    if len(prediction_file_names) == 0:
        error_string = (
            'Could not find any prediction file with the following pattern: '
            '"{0:s}"'
        ).format(prediction_file_pattern)

        raise ValueError(error_string)

    prediction_file_names.sort()

    print('Reading data from: "{0:s}"...'.format(prediction_file_names[0]))
    first_prediction_table_xarray = prediction_io.read_file(
        prediction_file_names[0]
    )

    are_predictions_gridded = (
        scalar_prediction_utils.PREDICTED_ROW_OFFSET_KEY
        not in first_prediction_table_xarray
    )
    if are_predictions_gridded:
        raise ValueError(
            'This script does not yet work for gridded predictions.'
        )

    result_table_xarray = pit_utils.get_results_all_vars(
        prediction_file_names=prediction_file_names, num_bins=num_bins
    )
    print(SEPARATOR_STRING)

    t = result_table_xarray
    target_names = t.coords[pit_utils.TARGET_FIELD_DIM].values.tolist()

    for j in range(len(target_names)):
        print((
            'Variable = {0:s} ... PITD = {1:f} ... low-PIT freq bias = {2:f} '
            '... medium-PIT freq bias = {3:f} ... high-PIT freq bias = {4:f}'
        ).format(
            target_names[j],
            t[pit_utils.PIT_DEVIATION_KEY].values[j],
            t[pit_utils.LOW_BIN_BIAS_KEY].values[j],
            t[pit_utils.MIDDLE_BIN_BIAS_KEY].values[j],
            t[pit_utils.HIGH_BIN_BIAS_KEY].values[j]
        ))

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    pit_utils.write_results(
        result_table_xarray=result_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_pattern=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_PATTERN_ARG_NAME
        ),
        num_bins=getattr(INPUT_ARG_OBJECT, NUM_BINS_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
