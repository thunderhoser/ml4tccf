"""Converts raw HURDAT2 data to fake A-decks."""

import argparse
import numpy
import xarray
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking
from ml4tccf.io import a_deck_io
from ml4tccf.io import raw_a_deck_io
from ml4tccf.utils import misc_utils
from ml4tccf.scripts import convert_ebtrk_to_fake_a_decks as fake_a_decks

HOURS_TO_SECONDS = 3600
KT_TO_METRES_PER_SECOND = 1.852 / 3.6
MB_TO_PASCALS = 100.
NAUTICAL_MILES_TO_METRES = 1852.

MAX_NUM_WIND_THRESHOLDS = raw_a_deck_io.MAX_NUM_WIND_THRESHOLDS
MAX_NUM_WAVE_HEIGHT_THRESHOLDS = raw_a_deck_io.MAX_NUM_WAVE_HEIGHT_THRESHOLDS

INPUT_FILE_ARG_NAME = 'input_raw_hurdat2_file_name'
START_YEAR_ARG_NAME = 'start_year'
END_YEAR_ARG_NAME = 'end_year'
OUTPUT_FILE_ARG_NAME = 'output_a_deck_file'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Should be downloaded from: '
    'https://www.nhc.noaa.gov/data/#hurdat'
)
START_YEAR_HELP_STRING = (
    'Start year.  Will do the conversion only for TCs in the continuous period '
    '`{0:s}`...`{1:s}`.'
).format(START_YEAR_ARG_NAME, END_YEAR_ARG_NAME)

END_YEAR_HELP_STRING = 'Same as {0:s} but end year.'.format(START_YEAR_ARG_NAME)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by `a_deck_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + START_YEAR_ARG_NAME, type=int, required=True,
    help=START_YEAR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + END_YEAR_ARG_NAME, type=int, required=True,
    help=END_YEAR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _convert_storm_types(ebtrk_storm_type_strings):
    """Converts storm types from EBTRK format to A-deck format.

    E = number of TC objects

    :param ebtrk_storm_type_strings: length-E list of storm types.
    :return: a_deck_storm_type_strings: length-E list of storm types in new
        format.
    """

    renaming_dict = {
        'DB': a_deck_io.TROPICAL_DISTURBANCE_TYPE_STRING,
        'LO': a_deck_io.LOW_TYPE_STRING,
        'HU': a_deck_io.TROPICAL_HURRICANE_TYPE_STRING,
        'TD': a_deck_io.TROPICAL_DEPRESSION_TYPE_STRING,
        'TS': a_deck_io.TROPICAL_STORM_TYPE_STRING,
        'EX': a_deck_io.EXTRATROPICAL_TYPE_STRING,
        'SD': a_deck_io.SUBTROPICAL_DEPRESSION_TYPE_STRING,
        'SS': a_deck_io.SUBTROPICAL_STORM_TYPE_STRING,
        'WV': a_deck_io.WAVE_TYPE_STRING
    }

    return [renaming_dict[s] for s in ebtrk_storm_type_strings]


def _read_hurdat2_file(hurdat2_file_name, start_year, end_year):
    """Reads data from HURDAT2 file.

    :param hurdat2_file_name: See documentation at top of file.
    :param start_year: Same.
    :param end_year: Same.
    :return: a_deck_table_xarray: xarray table.  Metadata and variable names in
        the table should make it self-explanatory.
    """

    error_checking.assert_is_geq(end_year, start_year)
    desired_years = numpy.linspace(
        start_year, end_year, num=end_year - start_year + 1, dtype=int
    )

    cyclone_id_strings = []
    valid_times_unix_sec = []
    storm_type_strings = []
    center_latitudes_deg_n = []
    center_longitudes_deg_e = []
    max_sustained_winds_kt = []
    min_pressures_mb = []

    wind_radii_ne_by_threshold_naut_miles = [] * MAX_NUM_WIND_THRESHOLDS
    wind_radii_se_by_threshold_naut_miles = [] * MAX_NUM_WIND_THRESHOLDS
    wind_radii_sw_by_threshold_naut_miles = [] * MAX_NUM_WIND_THRESHOLDS
    wind_radii_nw_by_threshold_naut_miles = [] * MAX_NUM_WIND_THRESHOLDS
    max_wind_radii_naut_miles = []

    hurdat2_file_handle = open(hurdat2_file_name, 'r')
    current_cyclone_id_string = None

    for this_line in hurdat2_file_handle.readlines():
        these_words = this_line.split(',')
        these_words = [w.strip() for w in these_words]

        if len(these_words) < 5:
            current_cyclone_id_string = '{0:s}{1:s}'.format(
                these_words[0][-4:], these_words[0][:4]
            )
            misc_utils.parse_cyclone_id(current_cyclone_id_string)
            continue

        current_year = misc_utils.parse_cyclone_id(current_cyclone_id_string)[0]
        if current_year not in desired_years:
            continue

        cyclone_id_strings.append(current_cyclone_id_string)

        this_time_string = '{0:s}-{1:s}'.format(these_words[0], these_words[1])
        this_time_unix_sec = time_conversion.string_to_unix_sec(
            this_time_string, '%Y%m%d-%H%M'
        )
        valid_times_unix_sec.append(this_time_unix_sec)

        # TODO(thunderhoser): Have to figure out what these look like.
        storm_type_strings.append(these_words[3])

        this_hemisphere_string = these_words[4][-1]
        assert this_hemisphere_string in ['N', 'S']
        this_multiplier = 1. if this_hemisphere_string == 'N' else -1.
        center_latitudes_deg_n.append(
            this_multiplier * float(these_words[4][:-1])
        )

        this_hemisphere_string = these_words[5][-1]
        assert this_hemisphere_string in ['E', 'W']
        this_multiplier = 1. if this_hemisphere_string == 'E' else -1.
        center_longitudes_deg_e.append(
            this_multiplier * float(these_words[5][:-1])
        )

        max_sustained_winds_kt.append(int(these_words[6]))
        min_pressures_mb.append(int(these_words[7]))

        these_wind_radii_naut_miles = numpy.array([
            int(these_words[8]),
            int(these_words[12]),
            int(these_words[16])
        ])
        wind_radii_ne_by_threshold_naut_miles.append(
            these_wind_radii_naut_miles
        )

        these_wind_radii_naut_miles = numpy.array([
            int(these_words[9]),
            int(these_words[13]),
            int(these_words[17])
        ])
        wind_radii_se_by_threshold_naut_miles.append(
            these_wind_radii_naut_miles
        )

        these_wind_radii_naut_miles = numpy.array([
            int(these_words[10]),
            int(these_words[14]),
            int(these_words[18])
        ])
        wind_radii_sw_by_threshold_naut_miles.append(
            these_wind_radii_naut_miles
        )

        these_wind_radii_naut_miles = numpy.array([
            int(these_words[11]),
            int(these_words[15]),
            int(these_words[19])
        ])
        wind_radii_nw_by_threshold_naut_miles.append(
            these_wind_radii_naut_miles
        )

        max_wind_radii_naut_miles.append(int(these_words[20]))

    num_storm_objects = len(cyclone_id_strings)

    wind_threshold_matrix_m_s01 = numpy.full(
        (num_storm_objects, MAX_NUM_WIND_THRESHOLDS), numpy.nan
    )
    wind_threshold_matrix_m_s01[:, 0] = 34 * KT_TO_METRES_PER_SECOND
    wind_threshold_matrix_m_s01[:, 1] = 50 * KT_TO_METRES_PER_SECOND
    wind_threshold_matrix_m_s01[:, 2] = 64 * KT_TO_METRES_PER_SECOND

    center_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        numpy.array(center_longitudes_deg_e),
        allow_nan=False
    )
    max_sustained_winds_m_s01 = (
        KT_TO_METRES_PER_SECOND * numpy.array(max_sustained_winds_kt)
    )
    min_pressures_pa = MB_TO_PASCALS * numpy.array(min_pressures_mb)
    wind_radius_matrix_ne_quadrant_metres = (
        NAUTICAL_MILES_TO_METRES *
        numpy.stack(wind_radii_ne_by_threshold_naut_miles, axis=0)
    )
    wind_radius_matrix_se_quadrant_metres = (
        NAUTICAL_MILES_TO_METRES *
        numpy.stack(wind_radii_se_by_threshold_naut_miles, axis=0)
    )
    wind_radius_matrix_sw_quadrant_metres = (
        NAUTICAL_MILES_TO_METRES *
        numpy.stack(wind_radii_sw_by_threshold_naut_miles, axis=0)
    )
    wind_radius_matrix_nw_quadrant_metres = (
        NAUTICAL_MILES_TO_METRES *
        numpy.stack(wind_radii_nw_by_threshold_naut_miles, axis=0)
    )
    max_wind_radii_metres = (
        NAUTICAL_MILES_TO_METRES * numpy.array(max_wind_radii_naut_miles)
    )

    storm_object_indices = numpy.linspace(
        0, num_storm_objects - 1, num=num_storm_objects, dtype=int
    )
    wind_threshold_indices = numpy.linspace(
        0, MAX_NUM_WIND_THRESHOLDS - 1, num=MAX_NUM_WIND_THRESHOLDS, dtype=int
    )
    wave_height_threshold_indices = numpy.linspace(
        0, MAX_NUM_WAVE_HEIGHT_THRESHOLDS - 1,
        num=MAX_NUM_WAVE_HEIGHT_THRESHOLDS, dtype=int
    )

    metadata_dict = {
        a_deck_io.STORM_OBJECT_DIM: storm_object_indices,
        a_deck_io.WIND_THRESHOLD_DIM: wind_threshold_indices,
        a_deck_io.WAVE_HEIGHT_THRESHOLD_DIM: wave_height_threshold_indices
    }

    storm_type_strings = _convert_storm_types(storm_type_strings)

    these_dim = (a_deck_io.STORM_OBJECT_DIM,)
    main_data_dict = {
        a_deck_io.CYCLONE_ID_KEY: (these_dim, cyclone_id_strings),
        a_deck_io.VALID_TIME_KEY: (these_dim, valid_times_unix_sec),
        a_deck_io.TECHNIQUE_KEY: (these_dim, ['CARQ'] * num_storm_objects),
        a_deck_io.SYSTEM_DEPTH_KEY: (these_dim, ['D'] * num_storm_objects),
        a_deck_io.STORM_TYPE_KEY: (these_dim, storm_type_strings),
        a_deck_io.LATITUDE_KEY: (these_dim, center_latitudes_deg_n),
        a_deck_io.LONGITUDE_KEY: (these_dim, center_longitudes_deg_e),
        a_deck_io.INTENSITY_KEY: (these_dim, max_sustained_winds_m_s01),
        a_deck_io.SEA_LEVEL_PRESSURE_KEY: (these_dim, min_pressures_pa),
        a_deck_io.LAST_ISOBAR_PRESSURE_KEY: (
            these_dim, numpy.full(num_storm_objects, 1e5)
        ),
        a_deck_io.LAST_ISOBAR_RADIUS_KEY: (
            these_dim, numpy.full(num_storm_objects, 1e5)
        ),
        a_deck_io.MAX_WIND_RADIUS_KEY: (these_dim, max_wind_radii_metres),
        a_deck_io.GUST_SPEED_KEY: (
            these_dim, numpy.full(num_storm_objects, 0.)
        ),
        a_deck_io.EYE_DIAMETER_KEY: (
            these_dim, numpy.full(num_storm_objects, 0.)
        ),
        a_deck_io.MAX_SEA_HEIGHT_KEY: (
            these_dim, numpy.full(num_storm_objects, 0.)
        )
    }

    these_dim = (a_deck_io.STORM_OBJECT_DIM, a_deck_io.WIND_THRESHOLD_DIM)
    main_data_dict.update({
        a_deck_io.WIND_THRESHOLD_KEY: (
            these_dim, wind_threshold_matrix_m_s01
        ),
        a_deck_io.WIND_RADIUS_CIRCULAR_KEY: (
            these_dim,
            numpy.full(wind_radius_matrix_ne_quadrant_metres.shape, numpy.nan)
        ),
        a_deck_io.WIND_RADIUS_NE_QUADRANT_KEY: (
            these_dim, wind_radius_matrix_ne_quadrant_metres
        ),
        a_deck_io.WIND_RADIUS_SE_QUADRANT_KEY: (
            these_dim, wind_radius_matrix_se_quadrant_metres
        ),
        a_deck_io.WIND_RADIUS_SW_QUADRANT_KEY: (
            these_dim, wind_radius_matrix_sw_quadrant_metres
        ),
        a_deck_io.WIND_RADIUS_NW_QUADRANT_KEY: (
            these_dim, wind_radius_matrix_nw_quadrant_metres
        )
    })

    dummy_wave_height_threshold_matrix_metres = numpy.full(
        (num_storm_objects, MAX_NUM_WAVE_HEIGHT_THRESHOLDS), numpy.nan
    )
    dummy_wave_height_radius_matrix_metres = numpy.full(
        (num_storm_objects, MAX_NUM_WAVE_HEIGHT_THRESHOLDS), numpy.nan
    )

    these_dim = (
        a_deck_io.STORM_OBJECT_DIM, a_deck_io.WAVE_HEIGHT_THRESHOLD_DIM
    )
    main_data_dict.update({
        a_deck_io.WAVE_HEIGHT_THRESHOLD_KEY: (
            these_dim, dummy_wave_height_threshold_matrix_metres
        ),
        a_deck_io.WAVE_HEIGHT_RADIUS_CIRCULAR_KEY: (
            these_dim, dummy_wave_height_radius_matrix_metres + 0.
        ),
        a_deck_io.WAVE_HEIGHT_RADIUS_NE_QUADRANT_KEY: (
            these_dim, dummy_wave_height_radius_matrix_metres + 0.
        ),
        a_deck_io.WAVE_HEIGHT_RADIUS_SE_QUADRANT_KEY: (
            these_dim, dummy_wave_height_radius_matrix_metres + 0.
        ),
        a_deck_io.WAVE_HEIGHT_RADIUS_SW_QUADRANT_KEY: (
            these_dim, dummy_wave_height_radius_matrix_metres + 0.
        ),
        a_deck_io.WAVE_HEIGHT_RADIUS_NW_QUADRANT_KEY: (
            these_dim, dummy_wave_height_radius_matrix_metres + 0.
        )
    })

    a_deck_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )
    a_deck_table_xarray = fake_a_decks._compute_storm_motions(
        a_deck_table_xarray
    )
    a_deck_table_xarray = raw_a_deck_io._create_extrap_forecasts(
        a_deck_table_xarray
    )

    return a_deck_table_xarray


def _run(hurdat2_file_name, start_year, end_year, output_file_name):
    """Converts raw HURDAT2 data to fake A-decks.

    This is effectively the same method.

    :param hurdat2_file_name: See documentation at top of file.
    :param start_year: Same.
    :param end_year: Same.
    :param output_file_name: Same.
    """

    a_deck_table_xarray = _read_hurdat2_file(
        hurdat2_file_name=hurdat2_file_name,
        start_year=start_year, end_year=end_year
    )

    print('Writing fake A-decks to: "{0:s}"...'.format(output_file_name))
    a_deck_io.write_file(
        a_deck_table_xarray=a_deck_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        hurdat2_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        start_year=getattr(INPUT_ARG_OBJECT, START_YEAR_ARG_NAME),
        end_year=getattr(INPUT_ARG_OBJECT, END_YEAR_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
