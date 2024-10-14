"""Converts prediction file to Zhixing's CSV format."""

import argparse
import numpy
from scipy.interpolate import interp1d
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import file_system_utils
from ml4tccf.io import prediction_io
from ml4tccf.io import satellite_io
from ml4tccf.utils import satellite_utils
from ml4tccf.utils import scalar_prediction_utils as prediction_utils

TIME_FORMAT_IN_OUTPUT_FILE_NAMES = '%Y%m%d_%H%M'
TIME_FORMAT_IN_OUTPUT_FILES = '%Y%m%d, %H%M'

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
SATELLITE_DIR_ARG_NAME = 'input_satellite_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_prediction_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file (will be read by `prediction_io.read_file`).'
)
SATELLITE_DIR_HELP_STRING = (
    'Path to directory with satellite data.  Files therein will be found by '
    '`satellite_io.find_file` and read by `satellite_io.read_file`.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  This script will write one CSV file per time '
    'step to the given directory.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SATELLITE_DIR_ARG_NAME, type=str, required=True,
    help=SATELLITE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _convert_predictions_one_sample(
        prediction_table_xarray, cyclone_id_string, target_time_unix_sec,
        satellite_dir_name, output_prediction_dir_name):
    """Converts predictions to new file format for one TC sample.

    :param prediction_table_xarray: xarray table in format returned by
        `prediction_io.read_file`.
    :param cyclone_id_string: Cyclone ID.
    :param target_time_unix_sec: Target time.
    :param satellite_dir_name: See documentation at top of this script.
    :param output_prediction_dir_name: Same.
    """

    satellite_file_name = satellite_io.find_file(
        directory_name=satellite_dir_name,
        cyclone_id_string=cyclone_id_string,
        valid_date_string=time_conversion.unix_sec_to_string(
            target_time_unix_sec, satellite_io.DATE_FORMAT
        ),
        raise_error_if_missing=True
    )

    print('Reading data from: "{0:s}"...'.format(satellite_file_name))
    satellite_table_xarray = satellite_io.read_file(satellite_file_name)
    stx = satellite_table_xarray

    time_idx = numpy.where(
        stx.coords[satellite_utils.TIME_DIM].values == target_time_unix_sec
    )[0][0]
    grid_latitudes_deg_n = (
        stx[satellite_utils.LATITUDE_LOW_RES_KEY].values[time_idx, :]
    )
    grid_longitudes_deg_e = (
        stx[satellite_utils.LONGITUDE_LOW_RES_KEY].values[time_idx, :]
    )
    grid_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        grid_longitudes_deg_e
    )

    # Does not allow for domain crossing International Date Line!
    assert not numpy.any(numpy.diff(grid_longitudes_deg_e) > 0.1)
    assert numpy.all(numpy.diff(grid_latitudes_deg_n) > 0.)
    assert numpy.all(numpy.diff(grid_longitudes_deg_e) > 0.)

    num_grid_rows = len(grid_latitudes_deg_n)
    num_grid_columns = len(grid_longitudes_deg_e)

    all_row_indices = numpy.linspace(
        0, num_grid_rows - 1, num=num_grid_rows, dtype=float
    )
    all_column_indices = numpy.linspace(
        0, num_grid_columns - 1, num=num_grid_columns, dtype=float
    )

    latitude_interp_object = interp1d(
        x=all_row_indices, y=grid_latitudes_deg_n,
        kind='linear', assume_sorted=True, bounds_error=True
    )
    longitude_interp_object = interp1d(
        x=all_column_indices, y=grid_longitudes_deg_e,
        kind='linear', assume_sorted=True, bounds_error=True
    )

    ptx = prediction_table_xarray
    sample_idxs = numpy.where(numpy.logical_and(
        ptx[prediction_utils.CYCLONE_ID_KEY].values == cyclone_id_string,
        ptx[prediction_utils.TARGET_TIME_KEY].values == target_time_unix_sec
    ))[0]

    assert len(sample_idxs) > 0

    print((
        'Found {0:d} TC samples with cyclone ID "{1:s}" and target time {2:s}!'
    ).format(
        len(sample_idxs),
        cyclone_id_string,
        time_conversion.unix_sec_to_string(
            target_time_unix_sec, '%Y-%m-%d-%H%M'
        )
    ))

    new_ptx = ptx.isel({prediction_utils.EXAMPLE_DIM_KEY: sample_idxs})
    print(new_ptx)

    predicted_row_index = numpy.mean(
        ptx[prediction_utils.PREDICTED_ROW_OFFSET_KEY].values[sample_idxs, :]
    )
    predicted_column_index = numpy.mean(
        ptx[prediction_utils.PREDICTED_COLUMN_OFFSET_KEY].values[sample_idxs, :]
    )
    predicted_latitude_deg_n = latitude_interp_object(predicted_row_index)
    predicted_longitude_deg_e = longitude_interp_object(predicted_column_index)

    fake_cyclone_id_string = '{0:s}{1:s}'.format(
        cyclone_id_string[4:].lower(), cyclone_id_string[:4]
    )

    output_file_name = '{0:s}/{1:s}/{1:s}_{2:s}.txt'.format(
        output_prediction_dir_name,
        fake_cyclone_id_string,
        time_conversion.unix_sec_to_string(
            target_time_unix_sec, TIME_FORMAT_IN_OUTPUT_FILE_NAMES
        )
    )

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    print('Writing prediction to file: "{0:s}"...'.format(output_file_name))

    with open(output_file_name, "w") as output_file_handle:
        output_file_handle.write('{0:s}, {1:s}, {2:.2f}, {3:.2f}'.format(
            fake_cyclone_id_string,
            time_conversion.unix_sec_to_string(
                target_time_unix_sec, TIME_FORMAT_IN_OUTPUT_FILES
            ),
            predicted_latitude_deg_n,
            lng_conversion.convert_lng_positive_in_west(
                predicted_longitude_deg_e
            )
        ))


def _run(input_prediction_file_name, satellite_dir_name,
         output_prediction_dir_name):
    """Converts prediction file to Zhixing's CSV format.

    This is effectively the main method.

    :param input_prediction_file_name: See documentation at top of this script.
    :param satellite_dir_name: Same.
    :param output_prediction_dir_name: Same.
    """

    print('Reading data from: "{0:s}"...'.format(input_prediction_file_name))
    prediction_table_xarray = prediction_io.read_file(
        input_prediction_file_name
    )
    prediction_table_xarray = prediction_utils.get_ensemble_mean(
        prediction_table_xarray
    )

    ptx = prediction_table_xarray
    cyclone_id_time_matrix = numpy.transpose(numpy.vstack([
        ptx[prediction_utils.CYCLONE_ID_KEY].values,
        ptx[prediction_utils.TARGET_TIME_KEY].values
    ]))
    unique_id_time_matrix = numpy.unique(cyclone_id_time_matrix, axis=0)

    for i in range(unique_id_time_matrix.shape[0]):
        _convert_predictions_one_sample(
            prediction_table_xarray=prediction_table_xarray,
            cyclone_id_string=unique_id_time_matrix[i, 0].decode('utf-8'),
            target_time_unix_sec=int(unique_id_time_matrix[i, 1]),
            satellite_dir_name=satellite_dir_name,
            output_prediction_dir_name=output_prediction_dir_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME
        ),
        satellite_dir_name=getattr(INPUT_ARG_OBJECT, SATELLITE_DIR_ARG_NAME),
        output_prediction_dir_name=getattr(
            INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME
        )
    )
