"""Splits predictions by lat-long coordinates."""

import glob
import argparse
import warnings
import numpy
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from ml4tccf.io import prediction_io
from ml4tccf.io import extended_best_track_io as ebtrk_io
from ml4tccf.utils import misc_utils
from ml4tccf.utils import scalar_prediction_utils
from ml4tccf.utils import extended_best_track_utils as ebtrk_utils
from ml4tccf.machine_learning import neural_net_training_cira_ir as nn_training
from ml4tccf.scripts import split_predictions_by_intensity as split_by_intensity

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H%M'

MIN_LATITUDE_DEG_N = -60.
MAX_LATITUDE_DEG_N = 60.
MIN_LONGITUDE_DEG_E = 0.
MAX_LONGITUDE_DEG_E = 360.

HOURS_TO_SECONDS = 3600

SYNOPTIC_TIME_TOLERANCE_SEC = nn_training.SYNOPTIC_TIME_TOLERANCE_SEC

INPUT_PREDICTION_FILES_ARG_NAME = 'input_prediction_file_pattern'
EBTRK_FILE_ARG_NAME = 'input_extended_best_track_file_name'
LATITUDE_SPACING_ARG_NAME = 'latitude_spacing_deg'
LONGITUDE_SPACING_ARG_NAME = 'longitude_spacing_deg'
OUTPUT_DIR_ARG_NAME = 'output_prediction_dir_name'

INPUT_PREDICTION_FILES_HELP_STRING = (
    'Glob pattern for input files.  Each will be read by '
    '`prediction_io.read_file`.'
)
EBTRK_FILE_HELP_STRING = (
    'Path to file with extended best-track data (contains TC-center locations).'
    '  Will be read by `extended_best_track_io.read_file`.'
)
LATITUDE_SPACING_HELP_SPACING = 'Meridional grid spacing (degrees).'
LONGITUDE_SPACING_HELP_SPACING = 'Zonal grid spacing (degrees).'
OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Within this directory, one category '
    'will be created for every 2-D bin (based on both lat and long).  Then '
    'subset predictions will be written to these subdirectories by '
    '`scalar_prediction_io.write_file` or `gridded_prediction_io.write_file`, '
    'to exact locations determined by `prediction_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_PREDICTION_FILES_ARG_NAME, type=str, required=True,
    help=INPUT_PREDICTION_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EBTRK_FILE_ARG_NAME, type=str, required=True,
    help=EBTRK_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LATITUDE_SPACING_ARG_NAME, type=float, required=True,
    help=LATITUDE_SPACING_HELP_SPACING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LONGITUDE_SPACING_ARG_NAME, type=float, required=True,
    help=LONGITUDE_SPACING_HELP_SPACING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_prediction_file_pattern, ebtrk_file_name,
         latitude_spacing_deg, longitude_spacing_deg,
         top_output_prediction_dir_name):
    """Splits predictions by lat-long coordinates.

    This is effectively the main method.

    :param input_prediction_file_pattern: See documentation at top of file.
    :param ebtrk_file_name: Same.
    :param latitude_spacing_deg: Same.
    :param longitude_spacing_deg: Same.
    :param top_output_prediction_dir_name: Same.
    :raises: ValueError: if no prediction files could be found.
    """

    latitude_spacing_deg = number_rounding.round_to_nearest(
        latitude_spacing_deg, 0.1
    )
    longitude_spacing_deg = number_rounding.round_to_nearest(
        longitude_spacing_deg, 0.1
    )

    grid_latitudes_deg_n, grid_longitudes_deg_e = misc_utils.create_latlng_grid(
        min_latitude_deg_n=MIN_LATITUDE_DEG_N,
        max_latitude_deg_n=MAX_LATITUDE_DEG_N,
        latitude_spacing_deg=latitude_spacing_deg,
        min_longitude_deg_e=MIN_LONGITUDE_DEG_E,
        max_longitude_deg_e=MAX_LONGITUDE_DEG_E - longitude_spacing_deg,
        longitude_spacing_deg=longitude_spacing_deg
    )

    grid_longitudes_deg_e += longitude_spacing_deg / 2
    num_grid_rows = len(grid_latitudes_deg_n)
    num_grid_columns = len(grid_longitudes_deg_e)

    grid_edge_latitudes_deg, grid_edge_longitudes_deg = (
        grids.get_latlng_grid_cell_edges(
            min_latitude_deg=grid_latitudes_deg_n[0],
            min_longitude_deg=grid_longitudes_deg_e[0],
            lat_spacing_deg=numpy.diff(grid_latitudes_deg_n[:2])[0],
            lng_spacing_deg=numpy.diff(grid_longitudes_deg_e[:2])[0],
            num_rows=num_grid_rows, num_columns=num_grid_columns
        )
    )

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

    print('Reading data from: "{0:s}"...'.format(ebtrk_file_name))
    ebtrk_table_xarray = ebtrk_io.read_file(ebtrk_file_name)
    ebtrk_times_unix_sec = (
        HOURS_TO_SECONDS * ebtrk_table_xarray[ebtrk_utils.VALID_TIME_KEY]
    )

    try:
        ebtrk_cyclone_id_strings = numpy.array([
            s.decode('utf-8')
            for s in ebtrk_table_xarray[ebtrk_utils.STORM_ID_KEY].values
        ])
    except AttributeError:
        ebtrk_cyclone_id_strings = (
            ebtrk_table_xarray[ebtrk_utils.STORM_ID_KEY].values
        )

    num_examples = len(pt[scalar_prediction_utils.TARGET_TIME_KEY].values)
    prediction_latitudes_deg_n = numpy.full(num_examples, numpy.nan)
    prediction_longitudes_deg_e = numpy.full(num_examples, numpy.nan)

    print(SEPARATOR_STRING)

    for i in range(num_examples):
        if numpy.mod(i, 100) == 0:
            print((
                'Have found lat-long coords for {0:d} of {1:d} examples...'
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
            ebtrk_cyclone_id_strings == this_cyclone_id_string,
            numpy.absolute(ebtrk_times_unix_sec - this_time_unix_sec) <=
            SYNOPTIC_TIME_TOLERANCE_SEC
        ))[0]

        if len(these_indices) == 0:
            warning_string = (
                'POTENTIAL ERROR: Cannot find cyclone {0:s} within {1:d} '
                'seconds of {2:s} in extended best-track data.'
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
                'within {2:d} seconds of {3:s} in extended best-track data:'
                '\n{4:s}'
            ).format(
                len(these_indices),
                this_cyclone_id_string,
                SYNOPTIC_TIME_TOLERANCE_SEC,
                time_conversion.unix_sec_to_string(
                    this_time_unix_sec, TIME_FORMAT_FOR_LOG_MESSAGES
                ),
                str(ebtrk_table_xarray.isel(
                    indexers={ebtrk_utils.STORM_OBJECT_DIM: these_indices}
                ))
            )

            warnings.warn(warning_string)

        prediction_latitudes_deg_n[i] = ebtrk_table_xarray[
            ebtrk_utils.CENTER_LATITUDE_KEY
        ].values[these_indices[0]]

        prediction_longitudes_deg_e[i] = ebtrk_table_xarray[
            ebtrk_utils.CENTER_LONGITUDE_KEY
        ].values[these_indices[0]]

    print('Have found lat-long coords for all {0:d} examples!'.format(
        num_examples
    ))
    print(SEPARATOR_STRING)

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            these_indices = grids.find_events_in_grid_cell(
                event_x_coords_metres=prediction_longitudes_deg_e,
                event_y_coords_metres=prediction_latitudes_deg_n,
                grid_edge_x_coords_metres=grid_edge_longitudes_deg,
                grid_edge_y_coords_metres=grid_edge_latitudes_deg,
                row_index=i, column_index=j, verbose=False
            )

            this_output_dir_name = (
                '{0:s}/latitude-deg-n={1:+05.1f}_{2:+05.1f}_'
                'longitude-deg-e={3:+06.1f}_{4:+06.1f}'
            ).format(
                top_output_prediction_dir_name,
                grid_edge_latitudes_deg[i],
                grid_edge_latitudes_deg[i + 1],
                grid_edge_longitudes_deg[j],
                grid_edge_longitudes_deg[j + 1]
            )

            file_system_utils.mkdir_recursive_if_necessary(
                directory_name=this_output_dir_name
            )

            if len(these_indices) == 0:
                continue

            split_by_intensity._write_scalar_predictions_1category(
                prediction_table_1cat_xarray=prediction_table_xarray.isel(
                    indexers=
                    {scalar_prediction_utils.EXAMPLE_DIM_KEY: these_indices}
                ),
                output_dir_name_1cat=this_output_dir_name
            )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_file_pattern=getattr(
            INPUT_ARG_OBJECT, INPUT_PREDICTION_FILES_ARG_NAME
        ),
        ebtrk_file_name=getattr(INPUT_ARG_OBJECT, EBTRK_FILE_ARG_NAME),
        latitude_spacing_deg=getattr(
            INPUT_ARG_OBJECT, LATITUDE_SPACING_ARG_NAME
        ),
        longitude_spacing_deg=getattr(
            INPUT_ARG_OBJECT, LONGITUDE_SPACING_ARG_NAME
        ),
        top_output_prediction_dir_name=getattr(
            INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME
        )
    )
