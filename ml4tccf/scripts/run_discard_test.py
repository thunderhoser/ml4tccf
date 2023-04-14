"""Runs discard test to determine quality of uncertainty estimates."""

import glob
import argparse
import numpy
from ml4tccf.io import prediction_io
from ml4tccf.utils import scalar_prediction_utils
from ml4tccf.utils import discard_test_utils as dt_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILE_PATTERN_ARG_NAME = 'input_prediction_file_pattern'
DISCARD_FRACTIONS_ARG_NAME = 'discard_fractions'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_PATTERN_HELP_STRING = (
    'Glob pattern for input files.  Each will be read by '
    '`prediction_io.read_file`, and predictions in all these files will be '
    'evaluated together.'
)
DISCARD_FRACTIONS_HELP_STRING = (
    'List of discard fractions, ranging from (0, 1).  This script will '
    'automatically use 0 as the lowest discard fraction.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Results will be written here by '
    '`discard_test_utils.write_results`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_PATTERN_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_PATTERN_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DISCARD_FRACTIONS_ARG_NAME, type=float, nargs='+', required=True,
    help=DISCARD_FRACTIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(prediction_file_pattern, discard_fractions, output_file_name):
    """Runs discard test to determine quality of uncertainty estimates.

    This is effectively the main method.

    :param prediction_file_pattern: See documentation at top of file.
    :param discard_fractions: Same.
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

    result_table_xarray = dt_utils.run_discard_test(
        prediction_file_names=prediction_file_names,
        discard_fractions=discard_fractions
    )
    print(SEPARATOR_STRING)

    t = result_table_xarray
    target_names = t.coords[dt_utils.TARGET_FIELD_DIM].values.tolist()

    for j in range(len(target_names)):
        print((
            'Variable = {0:s} ... mono fraction = {1:f} ... '
            'mean MAE improvement = {2:f}'
        ).format(
            target_names[j],
            t[dt_utils.MONO_FRACTION_KEY].values[j],
            t[dt_utils.MEAN_MAE_IMPROVEMENT_KEY].values[j]
        ))

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    dt_utils.write_results(
        result_table_xarray=result_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_pattern=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_PATTERN_ARG_NAME
        ),
        discard_fractions=numpy.array(
            getattr(INPUT_ARG_OBJECT, DISCARD_FRACTIONS_ARG_NAME), dtype=float
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
