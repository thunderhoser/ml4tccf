"""Finds interesting case studies for 2024 WAF paper.

WAF = Weather and Forecasting
"""

import os
import sys
import glob
import warnings
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import longitude_conversion as lng_conversion
import error_checking
import a_deck_io
import prediction_io
import extended_best_track_io as ebtrk_io
import misc_utils
import extended_best_track_utils as ebtrk_utils
import scalar_prediction_utils as prediction_utils
import neural_net_training_cira_ir as nn_training

TOLERANCE = 1e-6
SYNOPTIC_TIME_TOLERANCE_SEC = nn_training.SYNOPTIC_TIME_TOLERANCE_SEC

METRES_TO_KM = 0.001
HOURS_TO_SECONDS = 3600
METRES_PER_SECOND_TO_KT = 3.6 / 1.852

TIME_FORMAT_FOR_LOG = '%Y-%m-%d-%H%M'

TROPICAL_TYPE_STRING = 'tropical'
SUBTROPICAL_TYPE_STRING = 'subtropical'
EXTRATROPICAL_TYPE_STRING = 'extratropical'
DISTURBANCE_TYPE_STRING = 'disturbance'
MISC_TYPE_STRING = 'miscellaneous'
VALID_STORM_TYPE_STRINGS = [
    TROPICAL_TYPE_STRING, SUBTROPICAL_TYPE_STRING, EXTRATROPICAL_TYPE_STRING,
    DISTURBANCE_TYPE_STRING, MISC_TYPE_STRING
]

INPUT_FILE_PATTERN_ARG_NAME = 'input_prediction_file_pattern'
LATITUDE_LIMITS_ARG_NAME = 'latitude_limits_deg_n'
LONGITUDE_LIMITS_ARG_NAME = 'longitude_limits_deg_e'
NADIR_X_LIMITS_ARG_NAME = 'nadir_relative_x_limits_km'
NADIR_Y_LIMITS_ARG_NAME = 'nadir_relative_y_limits_km'
INTENSITY_LIMITS_ARG_NAME = 'intensity_limits_kt'
STORM_TYPES_ARG_NAME = 'storm_type_strings'
A_DECK_FILE_ARG_NAME = 'input_a_deck_file_name'
EBTRK_FILE_ARG_NAME = 'input_ebtrk_file_name'
EUCLIDEAN_ERROR_LIMITS_ARG_NAME = 'euclidean_error_limits_km'
X_ERROR_LIMITS_ARG_NAME = 'x_error_limits_km'
Y_ERROR_LIMITS_ARG_NAME = 'y_error_limits_km'

INPUT_FILE_PATTERN_HELP_STRING = (
    'Glob pattern for input files, containing predictions and targets.  Each '
    'will be read by `prediction_io.read_file`, and TC samples in all these '
    'files will be searched together.'
)
LATITUDE_LIMITS_HELP_STRING = (
    'List with [min, max] latitude in deg north.  Will look for TC samples in '
    'this latitude range.  If you do not care about latitude range, leave this '
    'argument alone.'
)
LONGITUDE_LIMITS_HELP_STRING = (
    'List with [min, max] longitude in deg east.  Will look for TC samples in '
    'this longitude range.  If you do not care about longitude range, leave '
    'this argument alone.'
)
NADIR_X_LIMITS_HELP_STRING = (
    'List with [min, max] nadir-relative x-coordinate.  Will look for TC '
    'samples in this range.  If you do not care about nadir-relative x, leave '
    'this argument alone.'
)
NADIR_Y_LIMITS_HELP_STRING = (
    'List with [min, max] nadir-relative y-coordinate.  Will look for TC '
    'samples in this range.  If you do not care about nadir-relative y, leave '
    'this argument alone.'
)
INTENSITY_LIMITS_HELP_STRING = (
    'List with [min, max] TC intesity.  Will look for TC samples in this '
    'intensity range.  If you do not care about intensity range, leave this '
    'argument alone.'
)
STORM_TYPES_HELP_STRING = (
    'List of desired storm types.  If you do not care about storm type, leave '
    'this argument alone.  Each storm type must be in the following list:'
    '\n{0:s}'
).format(str(VALID_STORM_TYPE_STRINGS))

A_DECK_FILE_HELP_STRING = (
    'Path to A-deck file.  Will be read by `a_deck_io.read_file`.'
)
EBTRK_FILE_HELP_STRING = (
    ' Path to extended best-track file.  Will be read by '
    '`extended_best_track_io.read_file`.'
)
EUCLIDEAN_ERROR_LIMITS_HELP_STRING = (
    'List with [min, max] Euclidean errors.  Will look for TC samples with '
    'this range of Euclidean errors.  If you do not care about Euclidean '
    'error, leave this argument alone.'
)
X_ERROR_LIMITS_HELP_STRING = (
    'List with [min, max] x-coord errors.  Will look for TC samples with this '
    'range of x-errors.  If you do not care about x-error, leave this argument '
    'alone.'
)
Y_ERROR_LIMITS_HELP_STRING = (
    'List with [min, max] y-coord errors.  Will look for TC samples with this '
    'range of y-errors.  If you do not care about y-error, leave this argument '
    'alone.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_PATTERN_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_PATTERN_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LATITUDE_LIMITS_ARG_NAME, type=float, nargs=2, required=False,
    default=[1, -1], help=LATITUDE_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LONGITUDE_LIMITS_ARG_NAME, type=float, nargs=2, required=False,
    default=[0, 0], help=LONGITUDE_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NADIR_X_LIMITS_ARG_NAME, type=float, nargs=2, required=False,
    default=[1, -1], help=NADIR_X_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NADIR_Y_LIMITS_ARG_NAME, type=float, nargs=2, required=False,
    default=[1, -1], help=NADIR_Y_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INTENSITY_LIMITS_ARG_NAME, type=float, nargs=2, required=False,
    default=[1, -1], help=INTENSITY_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + STORM_TYPES_ARG_NAME, type=str, nargs='+', required=False,
    default=[''], help=STORM_TYPES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + A_DECK_FILE_ARG_NAME, type=str, required=True,
    help=A_DECK_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EBTRK_FILE_ARG_NAME, type=str, required=True,
    help=EBTRK_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EUCLIDEAN_ERROR_LIMITS_ARG_NAME, type=float, nargs=2, required=False,
    default=[1, -1], help=EUCLIDEAN_ERROR_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + X_ERROR_LIMITS_ARG_NAME, type=float, nargs=2, required=False,
    default=[1, -1], help=X_ERROR_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + Y_ERROR_LIMITS_ARG_NAME, type=float, nargs=2, required=False,
    default=[1, -1], help=Y_ERROR_LIMITS_HELP_STRING
)


def _check_storm_type(storm_type_string):
    """Error-checks storm type.

    :param storm_type_string: Storm type.
    :raises: ValueError: if `storm_type_string not in VALID_STORM_TYPE_STRINGS`.
    """

    if storm_type_string in VALID_STORM_TYPE_STRINGS:
        return

    error_string = (
        'Storm type ({0:s}) is not in list of accepted storm types:\n{1:s}'
    ).format(
        storm_type_string, str(VALID_STORM_TYPE_STRINGS)
    )

    raise ValueError(error_string)


def _create_pred_and_target_matrices(prediction_table_xarray):
    """Creates simple prediction and target matrices from xarray table.

    E = number of examples

    :param prediction_table_xarray: xarray table in format returned by
        `prediction_io.read_file`.
    :return: prediction_matrix_km: E-by-2 numpy array with predicted y-offsets
        (first column) and x-offsets (second column) in km.
    :return: target_matrix_km: E-by-2 numpy array with actual y-offsets (first
        column) and x-offsets (second column) in km.
    """

    ptx = prediction_table_xarray

    grid_spacings_km = ptx[prediction_utils.GRID_SPACING_KEY].values
    prediction_matrix_km = numpy.transpose(numpy.vstack([
        grid_spacings_km *
        ptx[prediction_utils.PREDICTED_ROW_OFFSET_KEY].values[:, 0],
        grid_spacings_km *
        ptx[prediction_utils.PREDICTED_COLUMN_OFFSET_KEY].values[:, 0]
    ]))
    target_matrix_km = numpy.transpose(numpy.vstack([
        grid_spacings_km * ptx[prediction_utils.ACTUAL_ROW_OFFSET_KEY].values,
        grid_spacings_km * ptx[prediction_utils.ACTUAL_COLUMN_OFFSET_KEY].values
    ]))

    return prediction_matrix_km, target_matrix_km


def _get_intensities(prediction_table_xarray, ebtrk_file_name):
    """Returns TC intensities.

    E = number of examples

    :param prediction_table_xarray: xarray table in format returned by
        `prediction_io.read_file`.
    :param ebtrk_file_name: See documentation at top of this script.
    :return: prediction_intensities_kt: length-E numpy array of intensities.
    """

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

    ptx = prediction_table_xarray
    num_examples = len(ptx[prediction_utils.CYCLONE_ID_KEY].values)
    prediction_intensities_m_s01 = numpy.full(num_examples, numpy.nan)

    for i in range(num_examples):
        this_time_unix_sec = ptx[prediction_utils.TARGET_TIME_KEY].values[i]
        this_cyclone_id_string = ptx[prediction_utils.CYCLONE_ID_KEY].values[i]
        try:
            this_cyclone_id_string = this_cyclone_id_string.decode('utf-8')
        except:
            pass

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
                    this_time_unix_sec, TIME_FORMAT_FOR_LOG
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
                    this_time_unix_sec, TIME_FORMAT_FOR_LOG
                ),
                str(ebtrk_table_xarray.isel(
                    indexers={ebtrk_utils.STORM_OBJECT_DIM: these_indices}
                ))
            )

            warnings.warn(warning_string)

        prediction_intensities_m_s01[i] = ebtrk_table_xarray[
            ebtrk_utils.MAX_SUSTAINED_WIND_KEY
        ].values[these_indices[0]]

    return METRES_PER_SECOND_TO_KT * prediction_intensities_m_s01


def _get_storm_types(prediction_table_xarray, a_deck_file_name):
    """Returns storm types.

    E = number of examples

    :param prediction_table_xarray: xarray table in format returned by
        `prediction_io.read_file`.
    :param a_deck_file_name: See documentation at top of this script.
    :return: prediction_storm_type_strings: length-E numpy array of storm types.
    """

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

    ptx = prediction_table_xarray
    num_examples = len(ptx[prediction_utils.CYCLONE_ID_KEY].values)
    prediction_storm_type_strings = numpy.full(num_examples, '', dtype=object)

    for i in range(num_examples):
        this_time_unix_sec = ptx[prediction_utils.TARGET_TIME_KEY].values[i]
        this_cyclone_id_string = ptx[prediction_utils.CYCLONE_ID_KEY].values[i]
        try:
            this_cyclone_id_string = this_cyclone_id_string.decode('utf-8')
        except:
            pass

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
                    this_time_unix_sec, TIME_FORMAT_FOR_LOG
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
                    this_time_unix_sec, TIME_FORMAT_FOR_LOG
                ),
                str(a_deck_table_xarray.isel(
                    indexers={a_deck_io.STORM_OBJECT_DIM: these_indices}
                ))
            )

            warnings.warn(warning_string)

        adt = a_deck_table_xarray
        j = these_indices[0]

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

    return prediction_storm_type_strings


def _run(prediction_file_pattern, latitude_limits_deg_n, longitude_limits_deg_e,
         nadir_relative_x_limits_km, nadir_relative_y_limits_km,
         intensity_limits_kt, desired_storm_type_strings,
         a_deck_file_name, ebtrk_file_name,
         euclidean_error_limits_km, x_error_limits_km, y_error_limits_km):
    """Finds interesting case studies for 2024 WAF paper.

    This is effectively the main method.

    :param prediction_file_pattern: See documentation at top of this script.
    :param latitude_limits_deg_n: Same.
    :param longitude_limits_deg_e: Same.
    :param nadir_relative_x_limits_km: Same.
    :param nadir_relative_y_limits_km: Same.
    :param intensity_limits_kt: Same.
    :param desired_storm_type_strings: Same.
    :param a_deck_file_name: Same.
    :param ebtrk_file_name: Same.
    :param euclidean_error_limits_km: Same.
    :param x_error_limits_km: Same.
    :param y_error_limits_km: Same.
    """

    # Check input args.
    prediction_file_names = glob.glob(prediction_file_pattern)
    if len(prediction_file_names) == 0:
        error_string = (
            'Could not find any prediction file with the following pattern: '
            '"{0:s}"'
        ).format(prediction_file_pattern)

        raise ValueError(error_string)

    prediction_file_names.sort()

    if numpy.diff(latitude_limits_deg_n)[0] <= 0:
        latitude_limits_deg_n = None
    if latitude_limits_deg_n is not None:
        error_checking.assert_is_valid_lat_numpy_array(latitude_limits_deg_n)

    is_lng_positive_in_west = True

    if numpy.absolute(numpy.diff(longitude_limits_deg_e)[0]) <= TOLERANCE:
        longitude_limits_deg_e = None
    if longitude_limits_deg_e is not None:
        longitude_limits_deg_e = lng_conversion.convert_lng_positive_in_west(
            longitude_limits_deg_e, allow_nan=False
        )

        if numpy.diff(longitude_limits_deg_e)[0] < 0:
            longitude_limits_deg_e = (
                lng_conversion.convert_lng_negative_in_west(
                    longitude_limits_deg_e, allow_nan=False
                )
            )
            is_lng_positive_in_west = False

    if numpy.diff(nadir_relative_x_limits_km)[0] <= 0:
        nadir_relative_x_limits_km = None
    if nadir_relative_x_limits_km is not None:
        error_checking.assert_is_numpy_array_without_nan(
            nadir_relative_x_limits_km
        )

    if numpy.diff(nadir_relative_y_limits_km)[0] <= 0:
        nadir_relative_y_limits_km = None
    if nadir_relative_y_limits_km is not None:
        error_checking.assert_is_numpy_array_without_nan(
            nadir_relative_y_limits_km
        )

    if numpy.diff(intensity_limits_kt)[0] <= 0:
        intensity_limits_kt = None
    if intensity_limits_kt is not None:
        error_checking.assert_is_geq_numpy_array(intensity_limits_kt, 0.)

    if (
            len(desired_storm_type_strings) == 1
            and desired_storm_type_strings[0] == ''
    ):
        desired_storm_type_strings = None

    if desired_storm_type_strings is not None:
        desired_storm_type_strings = list(set(desired_storm_type_strings))
        for st in desired_storm_type_strings:
            _check_storm_type(st)

    if numpy.diff(euclidean_error_limits_km)[0] <= 0:
        euclidean_error_limits_km = None
    if euclidean_error_limits_km is not None:
        error_checking.assert_is_geq_numpy_array(euclidean_error_limits_km, 0.)

    if numpy.diff(x_error_limits_km)[0] <= 0:
        x_error_limits_km = None
    if x_error_limits_km is not None:
        error_checking.assert_is_numpy_array_without_nan(x_error_limits_km)

    if numpy.diff(y_error_limits_km)[0] <= 0:
        y_error_limits_km = None
    if y_error_limits_km is not None:
        error_checking.assert_is_numpy_array_without_nan(y_error_limits_km)

    print('Reading data from: "{0:s}"...'.format(prediction_file_names[0]))
    first_prediction_table_xarray = prediction_io.read_file(
        prediction_file_names[0]
    )

    are_predictions_gridded = (
        prediction_utils.PREDICTED_ROW_OFFSET_KEY
        not in first_prediction_table_xarray
    )
    if are_predictions_gridded:
        raise ValueError(
            'This script does not yet work for gridded predictions.'
        )

    # Do actual stuff.
    num_files = len(prediction_file_names)
    prediction_tables_xarray = [None] * num_files

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(prediction_file_names[i]))
        prediction_tables_xarray[i] = prediction_io.read_file(
            prediction_file_names[i]
        )

    prediction_table_xarray = prediction_utils.concat_over_examples(
        prediction_tables_xarray
    )
    prediction_table_xarray = prediction_utils.get_ensemble_mean(
        prediction_table_xarray
    )
    ptx = prediction_table_xarray

    if latitude_limits_deg_n is not None:
        print('Reading data from: "{0:s}"...'.format(ebtrk_file_name))
        ebtrk_table_xarray = ebtrk_io.read_file(ebtrk_file_name)
        prediction_latitudes_deg_n = misc_utils.match_predictions_to_tc_centers(
            prediction_table_xarray=ptx,
            ebtrk_table_xarray=ebtrk_table_xarray,
            return_xy=False
        )[1]

        orig_num_examples = len(ptx[prediction_utils.CYCLONE_ID_KEY].values)

        keep_indices = numpy.where(numpy.logical_and(
            prediction_latitudes_deg_n >= latitude_limits_deg_n[0],
            prediction_latitudes_deg_n <= latitude_limits_deg_n[1]
        ))[0]
        ptx = ptx.isel({prediction_utils.EXAMPLE_DIM_KEY: keep_indices})

        num_examples = len(ptx[prediction_utils.CYCLONE_ID_KEY].values)

        print((
            'Found {0:d} of {1:d} examples in the latitude range '
            '[{2:.2f}, {3:.2f}] deg N!'
        ).format(
            num_examples,
            orig_num_examples,
            latitude_limits_deg_n[0],
            latitude_limits_deg_n[1]
        ))

    if longitude_limits_deg_e is not None:
        print('Reading data from: "{0:s}"...'.format(ebtrk_file_name))
        ebtrk_table_xarray = ebtrk_io.read_file(ebtrk_file_name)
        prediction_longitudes_deg_e = misc_utils.match_predictions_to_tc_centers(
            prediction_table_xarray=ptx,
            ebtrk_table_xarray=ebtrk_table_xarray,
            return_xy=False
        )[0]

        if is_lng_positive_in_west:
            prediction_longitudes_deg_e = (
                lng_conversion.convert_lng_positive_in_west(
                    prediction_longitudes_deg_e
                )
            )
        else:
            prediction_longitudes_deg_e = (
                lng_conversion.convert_lng_negative_in_west(
                    prediction_longitudes_deg_e
                )
            )

        orig_num_examples = len(ptx[prediction_utils.CYCLONE_ID_KEY].values)

        keep_indices = numpy.where(numpy.logical_and(
            prediction_longitudes_deg_e >= longitude_limits_deg_e[0],
            prediction_longitudes_deg_e <= longitude_limits_deg_e[1]
        ))[0]
        ptx = ptx.isel({prediction_utils.EXAMPLE_DIM_KEY: keep_indices})

        num_examples = len(ptx[prediction_utils.CYCLONE_ID_KEY].values)

        print((
            'Found {0:d} of {1:d} examples in the longitude range '
            '[{2:.2f}, {3:.2f}] deg E!'
        ).format(
            num_examples,
            orig_num_examples,
            longitude_limits_deg_e[0],
            longitude_limits_deg_e[1]
        ))

    if nadir_relative_x_limits_km is not None:
        print('Reading data from: "{0:s}"...'.format(ebtrk_file_name))
        ebtrk_table_xarray = ebtrk_io.read_file(ebtrk_file_name)
        prediction_x_coords_km = METRES_TO_KM * (
            misc_utils.match_predictions_to_tc_centers(
                prediction_table_xarray=ptx,
                ebtrk_table_xarray=ebtrk_table_xarray,
                return_xy=True
            )[0]
        )

        orig_num_examples = len(ptx[prediction_utils.CYCLONE_ID_KEY].values)

        keep_indices = numpy.where(numpy.logical_and(
            prediction_x_coords_km >= nadir_relative_x_limits_km[0],
            prediction_x_coords_km <= nadir_relative_x_limits_km[1]
        ))[0]
        ptx = ptx.isel({prediction_utils.EXAMPLE_DIM_KEY: keep_indices})

        num_examples = len(ptx[prediction_utils.CYCLONE_ID_KEY].values)

        print((
            'Found {0:d} of {1:d} examples with nadir-relative x-coord in '
            '[{2:.1f}, {3:.1f}] km!'
        ).format(
            num_examples,
            orig_num_examples,
            nadir_relative_x_limits_km[0],
            nadir_relative_x_limits_km[1]
        ))

    if nadir_relative_y_limits_km is not None:
        print('Reading data from: "{0:s}"...'.format(ebtrk_file_name))
        ebtrk_table_xarray = ebtrk_io.read_file(ebtrk_file_name)
        prediction_y_coords_km = METRES_TO_KM * (
            misc_utils.match_predictions_to_tc_centers(
                prediction_table_xarray=ptx,
                ebtrk_table_xarray=ebtrk_table_xarray,
                return_xy=True
            )[1]
        )

        orig_num_examples = len(ptx[prediction_utils.CYCLONE_ID_KEY].values)

        keep_indices = numpy.where(numpy.logical_and(
            prediction_y_coords_km >= nadir_relative_y_limits_km[0],
            prediction_y_coords_km <= nadir_relative_y_limits_km[1]
        ))[0]
        ptx = ptx.isel({prediction_utils.EXAMPLE_DIM_KEY: keep_indices})

        num_examples = len(ptx[prediction_utils.CYCLONE_ID_KEY].values)

        print((
            'Found {0:d} of {1:d} examples with nadir-relative y-coord in '
            '[{2:.1f}, {3:.1f}] km!'
        ).format(
            num_examples,
            orig_num_examples,
            nadir_relative_y_limits_km[0],
            nadir_relative_y_limits_km[1]
        ))

    if intensity_limits_kt is not None:
        prediction_intensities_kt = _get_intensities(
            prediction_table_xarray=ptx, ebtrk_file_name=ebtrk_file_name
        )

        orig_num_examples = len(ptx[prediction_utils.CYCLONE_ID_KEY].values)

        keep_indices = numpy.where(numpy.logical_and(
            prediction_intensities_kt >= intensity_limits_kt[0],
            prediction_intensities_kt <= intensity_limits_kt[1]
        ))[0]
        ptx = ptx.isel({prediction_utils.EXAMPLE_DIM_KEY: keep_indices})

        num_examples = len(ptx[prediction_utils.CYCLONE_ID_KEY].values)

        print((
            'Found {0:d} of {1:d} examples with TC intensity in '
            '[{2:.1f}, {3:.1f}] kt!'
        ).format(
            num_examples,
            orig_num_examples,
            intensity_limits_kt[0],
            intensity_limits_kt[1]
        ))

    if desired_storm_type_strings is not None:
        prediction_storm_type_strings = _get_storm_types(
            prediction_table_xarray=ptx, a_deck_file_name=a_deck_file_name
        )

        orig_num_examples = len(ptx[prediction_utils.CYCLONE_ID_KEY].values)

        keep_indices = numpy.where(numpy.isin(
            element=numpy.array(prediction_storm_type_strings),
            test_elements=numpy.array(desired_storm_type_strings)
        ))[0]
        ptx = ptx.isel({prediction_utils.EXAMPLE_DIM_KEY: keep_indices})

        num_examples = len(ptx[prediction_utils.CYCLONE_ID_KEY].values)

        print((
            'Found {0:d} of {1:d} examples with storm type in the following '
            'list:\n{2:s}'
        ).format(
            num_examples,
            orig_num_examples,
            str(desired_storm_type_strings)
        ))

    if euclidean_error_limits_km is not None:
        prediction_matrix_km, target_matrix_km = (
            _create_pred_and_target_matrices(ptx)
        )
        all_euclidean_errors_km = numpy.sqrt(
            (prediction_matrix_km[:, 0] - target_matrix_km[:, 0]) ** 2 +
            (prediction_matrix_km[:, 1] - target_matrix_km[:, 1]) ** 2
        )
        orig_num_examples = len(all_euclidean_errors_km)

        keep_indices = numpy.where(numpy.logical_and(
            all_euclidean_errors_km >= euclidean_error_limits_km[0],
            all_euclidean_errors_km <= euclidean_error_limits_km[1]
        ))[0]
        ptx = ptx.isel({prediction_utils.EXAMPLE_DIM_KEY: keep_indices})

        num_examples = len(
            ptx[prediction_utils.ACTUAL_CENTER_LATITUDE_KEY].values
        )

        print((
            'Found {0:d} of {1:d} examples in the Euclidean-error range '
            '[{2:.2f}, {3:.2f}] km!'
        ).format(
            num_examples,
            orig_num_examples,
            euclidean_error_limits_km[0],
            euclidean_error_limits_km[1]
        ))

    if x_error_limits_km is not None:
        prediction_matrix_km, target_matrix_km = (
            _create_pred_and_target_matrices(ptx)
        )
        all_x_errors_km = prediction_matrix_km[:, 1] - target_matrix_km[:, 1]
        orig_num_examples = len(all_x_errors_km)

        keep_indices = numpy.where(numpy.logical_and(
            all_x_errors_km >= x_error_limits_km[0],
            all_x_errors_km <= x_error_limits_km[1]
        ))[0]
        ptx = ptx.isel({prediction_utils.EXAMPLE_DIM_KEY: keep_indices})

        num_examples = len(
            ptx[prediction_utils.ACTUAL_CENTER_LATITUDE_KEY].values
        )

        print((
            'Found {0:d} of {1:d} examples in the x-error range '
            '[{2:.2f}, {3:.2f}] km!'
        ).format(
            num_examples,
            orig_num_examples,
            x_error_limits_km[0],
            x_error_limits_km[1]
        ))

    if y_error_limits_km is not None:
        prediction_matrix_km, target_matrix_km = (
            _create_pred_and_target_matrices(ptx)
        )
        all_y_errors_km = prediction_matrix_km[:, 0] - target_matrix_km[:, 0]
        orig_num_examples = len(all_y_errors_km)

        keep_indices = numpy.where(numpy.logical_and(
            all_y_errors_km >= y_error_limits_km[0],
            all_y_errors_km <= y_error_limits_km[1]
        ))[0]
        ptx = ptx.isel({prediction_utils.EXAMPLE_DIM_KEY: keep_indices})

        num_examples = len(
            ptx[prediction_utils.ACTUAL_CENTER_LATITUDE_KEY].values
        )

        print((
            'Found {0:d} of {1:d} examples in the y-error range '
            '[{2:.2f}, {3:.2f}] km!'
        ).format(
            num_examples,
            orig_num_examples,
            y_error_limits_km[0],
            y_error_limits_km[1]
        ))

    sort_indices = numpy.argsort(ptx[prediction_utils.TARGET_TIME_KEY].values)
    ptx = ptx.isel({prediction_utils.EXAMPLE_DIM_KEY: sort_indices})

    prediction_matrix_km, target_matrix_km = (
        _create_pred_and_target_matrices(ptx)
    )
    y_errors_km = prediction_matrix_km[:, 0] - target_matrix_km[:, 0]
    x_errors_km = prediction_matrix_km[:, 1] - target_matrix_km[:, 1]
    euclidean_errors_km = numpy.sqrt(
        (prediction_matrix_km[:, 0] - target_matrix_km[:, 0]) ** 2 +
        (prediction_matrix_km[:, 1] - target_matrix_km[:, 1]) ** 2
    )
    cyclone_id_strings = ptx[prediction_utils.CYCLONE_ID_KEY].values
    try:
        cyclone_id_strings = numpy.array([
            c.decode('utf-8') for c in cyclone_id_strings
        ])
    except:
        pass

    valid_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT_FOR_LOG)
        for t in ptx[prediction_utils.TARGET_TIME_KEY].values
    ]

    print('Reading data from: "{0:s}"...'.format(ebtrk_file_name))
    ebtrk_table_xarray = ebtrk_io.read_file(ebtrk_file_name)
    longitudes_deg_e, latitudes_deg_n = (
        misc_utils.match_predictions_to_tc_centers(
            prediction_table_xarray=ptx,
            ebtrk_table_xarray=ebtrk_table_xarray,
            return_xy=False
        )
    )
    x_coords_km, y_coords_km = misc_utils.match_predictions_to_tc_centers(
        prediction_table_xarray=ptx,
        ebtrk_table_xarray=ebtrk_table_xarray,
        return_xy=True
    )
    intensities_kt = _get_intensities(
        prediction_table_xarray=ptx, ebtrk_file_name=ebtrk_file_name
    )
    storm_type_strings = _get_storm_types(
        prediction_table_xarray=ptx, a_deck_file_name=a_deck_file_name
    )

    x_coords_km *= METRES_TO_KM
    y_coords_km *= METRES_TO_KM

    for i in range(len(cyclone_id_strings)):
        print((
            '{0:d}th example remaining ... '
            '{1:s} at {2:s} ({3:.1f} deg N, {4:.1f} deg E, {5:.1f} kt) ... '
            'type = {6:s} ... nadir-relative x/y = {7:.0f}, {8:.0f} km ... '
            'Euclidean/x/y errors = {9:.2f}, {10:.2f}, {11:.2f} km'
        ).format(
            i + 1,
            cyclone_id_strings[i],
            valid_time_strings[i],
            latitudes_deg_n[i],
            longitudes_deg_e[i],
            intensities_kt[i],
            storm_type_strings[i],
            x_coords_km[i],
            y_coords_km[i],
            euclidean_errors_km[i],
            x_errors_km[i],
            y_errors_km[i]
        ))


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_pattern=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_PATTERN_ARG_NAME
        ),
        latitude_limits_deg_n=numpy.array(
            getattr(INPUT_ARG_OBJECT, LATITUDE_LIMITS_ARG_NAME), dtype=float
        ),
        longitude_limits_deg_e=numpy.array(
            getattr(INPUT_ARG_OBJECT, LONGITUDE_LIMITS_ARG_NAME), dtype=float
        ),
        nadir_relative_x_limits_km=numpy.array(
            getattr(INPUT_ARG_OBJECT, NADIR_X_LIMITS_ARG_NAME), dtype=float
        ),
        nadir_relative_y_limits_km=numpy.array(
            getattr(INPUT_ARG_OBJECT, NADIR_Y_LIMITS_ARG_NAME), dtype=float
        ),
        intensity_limits_kt=numpy.array(
            getattr(INPUT_ARG_OBJECT, INTENSITY_LIMITS_ARG_NAME), dtype=float
        ),
        desired_storm_type_strings=getattr(
            INPUT_ARG_OBJECT, STORM_TYPES_ARG_NAME
        ),
        a_deck_file_name=getattr(INPUT_ARG_OBJECT, A_DECK_FILE_ARG_NAME),
        ebtrk_file_name=getattr(INPUT_ARG_OBJECT, EBTRK_FILE_ARG_NAME),
        euclidean_error_limits_km=numpy.array(
            getattr(INPUT_ARG_OBJECT, EUCLIDEAN_ERROR_LIMITS_ARG_NAME),
            dtype=float
        ),
        x_error_limits_km=numpy.array(
            getattr(INPUT_ARG_OBJECT, X_ERROR_LIMITS_ARG_NAME), dtype=float
        ),
        y_error_limits_km=numpy.array(
            getattr(INPUT_ARG_OBJECT, Y_ERROR_LIMITS_ARG_NAME), dtype=float
        )
    )
