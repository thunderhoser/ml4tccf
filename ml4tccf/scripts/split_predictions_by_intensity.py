"""Splits predictions by TC intensity."""

import glob
import argparse
import warnings
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4tccf.io import prediction_io
from ml4tccf.io import extended_best_track_io as xbt_io
from ml4tccf.utils import prediction_utils
from ml4tccf.utils import extended_best_track_utils as xbt_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HOURS_TO_SECONDS = 3600
TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H%M'

MAX_WIND_KT = 250.
MAX_PRESSURE_MB = 1030.

METRES_PER_SECOND_TO_KT = 3.6 / 1.852
PASCALS_TO_MB = 0.01

INPUT_PREDICTION_FILES_ARG_NAME = 'input_prediction_file_pattern'
XBT_FILE_ARG_NAME = 'input_extended_best_track_file_name'
MAX_WIND_CUTOFFS_ARG_NAME = 'max_wind_cutoffs_kt'
MIN_PRESSURE_CUTOFFS_ARG_NAME = 'min_pressure_cutoffs_mb'
OUTPUT_DIR_ARG_NAME = 'output_prediction_dir_name'

INPUT_PREDICTION_FILES_HELP_STRING = (
    'Glob pattern for input files.  Each will be read by '
    '`prediction_io.read_file`.'
)
XBT_FILE_HELP_STRING = (
    'Path to file with extended best-track data (contains intensity measures).'
    '  Will be read by `extended_best_track_io.read_file`.'
)
MAX_WIND_CUTOFFS_HELP_STRING = (
    'Category cutoffs for max sustained wind (one measure of intensity).  '
    'Please leave 0 and infinity out of this list, as they will be added '
    'automatically.'
)
MIN_PRESSURE_CUTOFFS_HELP_STRING = (
    'Category cutoffs for min surface pressure (one measure of intensity).  '
    'Please leave 0 and infinity out of this list, as they will be added '
    'automatically.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Within this directory, one '
    'subdirectory will be created for every wind-based intensity category, and '
    'another for every pressure-based intensity category.  Then subset '
    'predictions will be written to these subdirectories by '
    '`prediction_io.write_file`, to exact locations determined by '
    '`prediction_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_PREDICTION_FILES_ARG_NAME, type=str, required=True,
    help=INPUT_PREDICTION_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + XBT_FILE_ARG_NAME, type=str, required=True, help=XBT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_WIND_CUTOFFS_ARG_NAME, type=float, nargs='+', required=False,
    default=[50, 85], help=MAX_WIND_CUTOFFS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_PRESSURE_CUTOFFS_ARG_NAME, type=float, nargs='+', required=False,
    default=[], help=MIN_PRESSURE_CUTOFFS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _write_predictions_one_category(prediction_table_1cat_xarray,
                                    output_dir_name_1cat):
    """Writes predictions for one intensity category.

    :param prediction_table_1cat_xarray: xarray table with predictions for the
        one intensity category.
    :param output_dir_name_1cat: Name of output directory for the one intensity
        category.
    """

    all_cyclone_id_strings = numpy.array([
        s.decode('utf-8') for s in
        prediction_table_1cat_xarray[prediction_utils.CYCLONE_ID_KEY].values
    ])

    unique_cyclone_id_strings = numpy.unique(all_cyclone_id_strings)

    for cyclone_id_string in unique_cyclone_id_strings:
        these_indices = numpy.where(
            all_cyclone_id_strings == cyclone_id_string
        )[0]
        prediction_table_xarray_1cyclone = prediction_table_1cat_xarray.isel(
            indexers={prediction_utils.EXAMPLE_DIM_KEY: these_indices}
        )

        output_file_name = prediction_io.find_file(
            directory_name=output_dir_name_1cat,
            cyclone_id_string=cyclone_id_string,
            raise_error_if_missing=False
        )

        pt1cyc = prediction_table_xarray_1cyclone

        target_matrix = numpy.transpose(numpy.vstack((
            pt1cyc[prediction_utils.ACTUAL_ROW_OFFSET_KEY].values,
            pt1cyc[prediction_utils.ACTUAL_COLUMN_OFFSET_KEY].values,
            pt1cyc[prediction_utils.GRID_SPACING_KEY].values,
            pt1cyc[prediction_utils.ACTUAL_CENTER_LATITUDE_KEY].values
        )))

        prediction_matrix = numpy.stack((
            pt1cyc[prediction_utils.PREDICTED_ROW_OFFSET_KEY].values,
            pt1cyc[prediction_utils.PREDICTED_COLUMN_OFFSET_KEY].values
        ), axis=-2)

        print('Writing data to: "{0:s}"...'.format(output_file_name))
        prediction_io.write_file(
            netcdf_file_name=output_file_name,
            target_matrix=target_matrix,
            prediction_matrix=prediction_matrix,
            cyclone_id_string=cyclone_id_string,
            target_times_unix_sec=
            pt1cyc[prediction_utils.TARGET_TIME_KEY].values,
            model_file_name=pt1cyc.attrs[prediction_utils.MODEL_FILE_KEY]
        )


def _run(input_prediction_file_pattern, xbt_file_name,
         max_wind_cutoffs_kt, min_pressure_cutoffs_mb,
         top_output_prediction_dir_name):
    """Splits predictions by TC intensity.

    This is effectively the main method.

    :param input_prediction_file_pattern: See documentation at top of file.
    :param xbt_file_name: Same.
    :param max_wind_cutoffs_kt: Same.
    :param min_pressure_cutoffs_mb: Same.
    :param top_output_prediction_dir_name: Same.
    :raises: ValueError: if no prediction files could be found.
    :raises: ValueError: if an example in the prediction files cannot be found
        in extended best-track data.
    """

    # Check input args.
    if len(max_wind_cutoffs_kt) > 0:
        max_wind_cutoffs_kt = number_rounding.round_to_nearest(
            max_wind_cutoffs_kt, 0.1
        )
        max_wind_cutoffs_kt = numpy.sort(max_wind_cutoffs_kt)

        error_checking.assert_is_greater_numpy_array(max_wind_cutoffs_kt, 0.)
        error_checking.assert_is_leq_numpy_array(
            max_wind_cutoffs_kt, MAX_WIND_KT
        )
        error_checking.assert_is_greater_numpy_array(
            numpy.diff(max_wind_cutoffs_kt), 0.
        )

        max_wind_cutoffs_kt = numpy.concatenate((
            numpy.array([0.]),
            max_wind_cutoffs_kt,
            numpy.array([numpy.inf])
        ))

    if len(min_pressure_cutoffs_mb) > 0:
        min_pressure_cutoffs_mb = number_rounding.round_to_nearest(
            min_pressure_cutoffs_mb, 0.1
        )
        min_pressure_cutoffs_mb = numpy.sort(min_pressure_cutoffs_mb)

        error_checking.assert_is_greater_numpy_array(
            min_pressure_cutoffs_mb, 0.
        )
        error_checking.assert_is_leq_numpy_array(
            min_pressure_cutoffs_mb, MAX_PRESSURE_MB
        )
        error_checking.assert_is_greater_numpy_array(
            numpy.diff(min_pressure_cutoffs_mb), 0.
        )

        min_pressure_cutoffs_mb = numpy.concatenate((
            numpy.array([0.]),
            min_pressure_cutoffs_mb,
            numpy.array([numpy.inf])
        ))

    error_checking.assert_is_greater(
        len(max_wind_cutoffs_kt) + len(min_pressure_cutoffs_mb), 0
    )

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

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(
            input_prediction_file_names[i]
        ))
        prediction_tables_xarray[i] = prediction_io.read_file(
            input_prediction_file_names[i]
        )
        prediction_tables_xarray[i] = prediction_utils.get_ensemble_mean(
            prediction_tables_xarray[i]
        )

    prediction_table_xarray = prediction_utils.concat_over_examples(
        prediction_tables_xarray
    )
    pt = prediction_table_xarray

    print('Reading data from: "{0:s}"...'.format(xbt_file_name))
    xbt_table_xarray = xbt_io.read_file(xbt_file_name)
    xbt_times_unix_sec = (
        HOURS_TO_SECONDS * xbt_table_xarray[xbt_utils.VALID_TIME_KEY]
    )

    try:
        xbt_cyclone_id_strings = numpy.array([
            s.decode('utf-8')
            for s in xbt_table_xarray[xbt_utils.STORM_ID_KEY].values
        ])
    except AttributeError:
        xbt_cyclone_id_strings = xbt_table_xarray[xbt_utils.STORM_ID_KEY].values

    # Find intensity corresponding to each prediction.
    num_examples = len(pt[prediction_utils.TARGET_TIME_KEY].values)
    prediction_max_winds_m_s01 = numpy.full(num_examples, numpy.nan)
    prediction_min_pressures_pa = numpy.full(num_examples, numpy.nan)

    print(SEPARATOR_STRING)

    for i in range(num_examples):
        if numpy.mod(i, 100) == 0:
            print((
                'Have found extended best-track data for {0:d} of {1:d} '
                'examples...'
            ).format(
                i, num_examples
            ))

        this_cyclone_id_string = (
            pt[prediction_utils.CYCLONE_ID_KEY].values[i].decode('utf-8')
        )
        this_time_unix_sec = pt[prediction_utils.TARGET_TIME_KEY].values[i]

        these_indices = numpy.where(numpy.logical_and(
            xbt_cyclone_id_strings == this_cyclone_id_string,
            xbt_times_unix_sec == this_time_unix_sec
        ))[0]

        if len(these_indices) == 0:
            error_string = (
                'Cannot find cyclone {0:s} at {1:s} in extended best-track '
                'data.'
            ).format(
                this_cyclone_id_string,
                time_conversion.unix_sec_to_string(
                    this_time_unix_sec, TIME_FORMAT_FOR_LOG_MESSAGES
                )
            )

            raise ValueError(error_string)

        if len(these_indices) > 1:
            warning_string = (
                'POTENTIAL ERROR: Found {0:d} instances of cyclone {1:s} at '
                '{2:s} in extended best-track data:\n{3:s}'
            ).format(
                len(these_indices),
                this_cyclone_id_string,
                time_conversion.unix_sec_to_string(
                    this_time_unix_sec, TIME_FORMAT_FOR_LOG_MESSAGES
                ),
                str(xbt_table_xarray.isel(
                    indexers={xbt_utils.STORM_OBJECT_DIM: these_indices}
                ))
            )

            warnings.warn(warning_string)

        prediction_max_winds_m_s01[i] = xbt_table_xarray[
            xbt_utils.MAX_SUSTAINED_WIND_KEY
        ].values[these_indices[0]]

        prediction_min_pressures_pa[i] = xbt_table_xarray[
            xbt_utils.MIN_PRESSURE_KEY
        ].values[these_indices[0]]

    print('Have found extended best-track data for all {0:d} examples!'.format(
        num_examples
    ))
    print(SEPARATOR_STRING)

    prediction_max_winds_kt = (
        METRES_PER_SECOND_TO_KT * prediction_max_winds_m_s01
    )
    prediction_min_pressures_mb = PASCALS_TO_MB * prediction_min_pressures_pa

    for i in range(len(max_wind_cutoffs_kt) - 1):
        these_indices = numpy.where(numpy.logical_and(
            prediction_max_winds_kt >= max_wind_cutoffs_kt[i],
            prediction_max_winds_kt <= max_wind_cutoffs_kt[i + 1]
        ))[0]

        this_output_dir_name = '{0:s}/max_wind_kt={1:.1f}-{2:.1f}'.format(
            top_output_prediction_dir_name,
            max_wind_cutoffs_kt[i], max_wind_cutoffs_kt[i + 1]
        )
        file_system_utils.mkdir_recursive_if_necessary(
            directory_name=this_output_dir_name
        )

        if len(these_indices) == 0:
            continue

        _write_predictions_one_category(
            prediction_table_1cat_xarray=prediction_table_xarray.isel(
                indexers={prediction_utils.EXAMPLE_DIM_KEY: these_indices}
            ),
            output_dir_name_1cat=this_output_dir_name
        )

    for i in range(len(min_pressure_cutoffs_mb) - 1):
        these_indices = numpy.where(numpy.logical_and(
            prediction_min_pressures_mb >= min_pressure_cutoffs_mb[i],
            prediction_min_pressures_mb <= min_pressure_cutoffs_mb[i + 1]
        ))[0]

        this_output_dir_name = '{0:s}/min_pressure_mb={1:.1f}-{2:.1f}'.format(
            top_output_prediction_dir_name,
            min_pressure_cutoffs_mb[i], min_pressure_cutoffs_mb[i + 1]
        )
        file_system_utils.mkdir_recursive_if_necessary(
            directory_name=this_output_dir_name
        )

        if len(these_indices) == 0:
            continue

        _write_predictions_one_category(
            prediction_table_1cat_xarray=prediction_table_xarray.isel(
                indexers={prediction_utils.EXAMPLE_DIM_KEY: these_indices}
            ),
            output_dir_name_1cat=this_output_dir_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_file_pattern=getattr(
            INPUT_ARG_OBJECT, INPUT_PREDICTION_FILES_ARG_NAME
        ),
        xbt_file_name=getattr(INPUT_ARG_OBJECT, XBT_FILE_ARG_NAME),
        max_wind_cutoffs_kt=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_WIND_CUTOFFS_ARG_NAME),
            dtype=float
        ),
        min_pressure_cutoffs_mb=numpy.array(
            getattr(INPUT_ARG_OBJECT, MIN_PRESSURE_CUTOFFS_ARG_NAME),
            dtype=float
        ),
        top_output_prediction_dir_name=getattr(
            INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME
        )
    )
