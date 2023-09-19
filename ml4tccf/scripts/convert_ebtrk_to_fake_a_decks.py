"""Converts extended best-track (EBTRK) data to fake A-decks."""

import argparse
import numpy
import xarray
from pyproj import Geod
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from ml4tccf.io import a_deck_io
from ml4tccf.io import raw_a_deck_io
from ml4tccf.io import extended_best_track_io as ebtrk_io
from ml4tccf.utils import extended_best_track_utils as ebtrk_utils
from ml4tccf.utils import misc_utils

HOURS_TO_SECONDS = 3600
KT_TO_METRES_PER_SECOND = 1.852 / 3.6

MAX_NUM_WIND_THRESHOLDS = raw_a_deck_io.MAX_NUM_WIND_THRESHOLDS
MAX_NUM_WAVE_HEIGHT_THRESHOLDS = raw_a_deck_io.MAX_NUM_WAVE_HEIGHT_THRESHOLDS

INPUT_FILE_ARG_NAME = 'input_ebtrk_file_name'
BASINS_ARG_NAME = 'basin_id_strings'
START_YEAR_ARG_NAME = 'start_year'
END_YEAR_ARG_NAME = 'end_year'
OUTPUT_FILE_ARG_NAME = 'output_a_deck_file'

INPUT_FILE_HELP_STRING = (
    'Path to input (extended best-track) file.  Will be read by '
    '`extended_best_track_io.read_file`.'
)
BASINS_HELP_STRING = (
    '1-D list of ocean basins.  Will do the conversion only for TCs in these '
    'basins.  If you want all ocean basins, leave this alone.'
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
    '--' + BASINS_ARG_NAME, type=str, nargs='+', required=False, default=[''],
    help=BASINS_HELP_STRING
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
        'disturbance': 'tropical_disturbance',
        'extratropical': 'extratropical_system',
        'remnant_low': 'low',
        'subtropical': 'subtropical_storm',
        'tropical': 'tropical_hurricane',
        'unknown': 'low',
        'wave': 'wave'
    }

    return [renaming_dict[s] for s in ebtrk_storm_type_strings]


def _compute_storm_motions(a_deck_table_xarray):
    """Computes motion (speed and heading) for each TC object.

    :param a_deck_table_xarray: xarray table.
    :return: a_deck_table_xarray: Same but with extra variables.
    """

    adt = a_deck_table_xarray
    storm_latitudes_deg_n = adt[a_deck_io.LATITUDE_KEY].values
    storm_longitudes_deg_e = adt[a_deck_io.LONGITUDE_KEY].values

    num_cyclone_objects = len(storm_latitudes_deg_n)
    storm_speeds_m_s01 = numpy.full(num_cyclone_objects, numpy.nan)
    storm_headings_deg = numpy.full(num_cyclone_objects, numpy.nan)

    geodesic_object = Geod(ellps='WGS84')

    for i in range(num_cyclone_objects):
        desired_indices = numpy.where(numpy.logical_and(
            adt[a_deck_io.VALID_TIME_KEY].values <
            adt[a_deck_io.VALID_TIME_KEY].values[i],
            adt[a_deck_io.CYCLONE_ID_KEY].values ==
            adt[a_deck_io.CYCLONE_ID_KEY].values[i]
        ))[0]

        if len(desired_indices) == 0:
            continue

        best_subindex = numpy.argmax(
            adt[a_deck_io.VALID_TIME_KEY].values[desired_indices]
        )
        desired_idx = desired_indices[best_subindex]

        storm_headings_deg[i], _, storm_speeds_m_s01[i] = geodesic_object.inv(
            storm_longitudes_deg_e[desired_idx],
            storm_latitudes_deg_n[desired_idx],
            storm_longitudes_deg_e[i],
            storm_latitudes_deg_n[i]
        )

        print('Storm {0:s} ending {1:s} moves from [{2:.1f} deg N, {3:.1f} deg E] to [{4:.1f} deg N, {5:.1f} deg E] ... heading = {6:.2f} deg'.format(
            adt[a_deck_io.CYCLONE_ID_KEY].values[i],
            time_conversion.unix_sec_to_string(adt[a_deck_io.VALID_TIME_KEY].values[i], '%Y-%m-%d-%H'),
            storm_latitudes_deg_n[desired_idx], storm_longitudes_deg_e[desired_idx],
            storm_latitudes_deg_n[i], storm_longitudes_deg_e[i],
            storm_headings_deg[i]
        ))

        time_diff_sec = (
            adt[a_deck_io.VALID_TIME_KEY].values[i] -
            adt[a_deck_io.VALID_TIME_KEY].values[desired_idx]
        )
        storm_speeds_m_s01[i] = storm_speeds_m_s01[i] / time_diff_sec

    for this_percentile_level in [1, 5, 25, 50, 75, 95, 99, 99.5, 99.9]:
        print('{0:.1f}th percentile of storm speeds = {1:.4f} m/s'.format(
            this_percentile_level,
            numpy.nanpercentile(storm_speeds_m_s01, this_percentile_level)
        ))

    storm_headings_deg[storm_headings_deg < 0] += 360.

    for min_cutoff_deg, max_cutoff_deg in zip(
            [0, 90, 180, 270], [90, 180, 270, numpy.inf]
    ):
        print((
            'Number of TC objects with motion heading in range '
            '{0:.0f}...{1:.0f} deg = {2:d}'
        ).format(
            min_cutoff_deg,
            max_cutoff_deg,
            numpy.sum(numpy.logical_and(
                storm_headings_deg >= min_cutoff_deg,
                storm_headings_deg < max_cutoff_deg
            ))
        ))

    return a_deck_table_xarray.assign({
        a_deck_io.MOTION_SPEED_KEY: (
            (a_deck_io.STORM_OBJECT_DIM,), storm_speeds_m_s01
        ),
        a_deck_io.MOTION_HEADING_KEY: (
            (a_deck_io.STORM_OBJECT_DIM,), storm_headings_deg
        )
    })


def _run(ebtrk_file_name, basin_id_strings, start_year, end_year,
         output_file_name):
    """Converts extended best-track (EBTRK) data to fake A-decks.

    This is effectively the main method.

    :param ebtrk_file_name: See documentation at top of file.
    :param basin_id_strings: Same.
    :param start_year: Same.
    :param end_year: Same.
    :param output_file_name: Same.
    """

    # Check input args.
    if len(basin_id_strings) == 1 and basin_id_strings[0] == '':
        basin_id_strings = None
    else:
        for this_basin_id_string in basin_id_strings:
            misc_utils.check_basin_id(this_basin_id_string)

    error_checking.assert_is_geq(end_year, start_year)
    desired_years = numpy.linspace(
        start_year, end_year, num=end_year - start_year + 1, dtype=int
    )

    # Do actual stuff.
    print('Reading data from: "{0:s}"...'.format(ebtrk_file_name))
    ebtrk_table_xarray = ebtrk_io.read_file(ebtrk_file_name)

    if basin_id_strings is not None:
        cyclone_basin_id_strings = [
            misc_utils.parse_cyclone_id(c)[1] for c in
            ebtrk_table_xarray[ebtrk_utils.STORM_ID_KEY].values
        ]

        good_indices = numpy.where(numpy.isin(
            element=numpy.array(cyclone_basin_id_strings),
            test_elements=numpy.array(basin_id_strings)
        ))[0]

        print((
            '{0:d} of {1:d} cyclones are in one of the desired basins (below):'
            '\n{2:s}'
        ).format(
            len(good_indices),
            len(cyclone_basin_id_strings),
            str(basin_id_strings)
        ))

        ebtrk_table_xarray = ebtrk_table_xarray.isel(
            {ebtrk_utils.STORM_OBJECT_DIM: good_indices}
        )
        del cyclone_basin_id_strings

    cyclone_years = numpy.array([
        misc_utils.parse_cyclone_id(c)[0] for c in
        ebtrk_table_xarray[ebtrk_utils.STORM_ID_KEY].values
    ], dtype=int)

    good_indices = numpy.where(numpy.isin(
        element=cyclone_years, test_elements=desired_years
    ))[0]

    print((
        '{0:d} of {1:d} cyclones are in one of the desired years (below):'
        '\n{2:s}'
    ).format(
        len(good_indices),
        len(cyclone_years),
        str(desired_years)
    ))

    ebtrk_table_xarray = ebtrk_table_xarray.isel(
        {ebtrk_utils.STORM_OBJECT_DIM: good_indices}
    )
    del cyclone_years

    cyclone_id_strings = ebtrk_table_xarray[ebtrk_utils.STORM_ID_KEY].values
    valid_times_unix_sec = (
        HOURS_TO_SECONDS * ebtrk_table_xarray[ebtrk_utils.VALID_TIME_KEY].values
    )

    num_storm_objects = len(cyclone_id_strings)
    dummy_technique_strings = ['CARQ'] * num_storm_objects
    dummy_system_depth_strings = ['D'] * num_storm_objects
    storm_type_strings = _convert_storm_types(
        ebtrk_table_xarray[ebtrk_utils.STORM_TYPE_KEY].values
    )
    storm_latitudes_deg_n = (
        ebtrk_table_xarray[ebtrk_utils.CENTER_LATITUDE_KEY].values
    )
    storm_longitudes_deg_e = (
        ebtrk_table_xarray[ebtrk_utils.CENTER_LONGITUDE_KEY].values
    )
    storm_intensities_m_s01 = (
        ebtrk_table_xarray[ebtrk_utils.MAX_SUSTAINED_WIND_KEY].values
    )
    storm_min_pressures_pa = (
        ebtrk_table_xarray[ebtrk_utils.MIN_PRESSURE_KEY].values
    )
    dummy_last_closed_isobar_pressures_pa = numpy.full(num_storm_objects, 1e5)
    dummy_last_closed_isobar_radii_metres = numpy.full(num_storm_objects, 1e5)
    dummy_max_wind_ratii_metres = numpy.full(num_storm_objects, 5e4)
    dummy_gust_speeds_m_s01 = numpy.full(num_storm_objects, 0.)
    dummy_eye_diameters_metres = numpy.full(num_storm_objects, 0.)
    dummy_max_sea_heights_metres = numpy.full(num_storm_objects, 0.)

    dummy_wind_threshold_matrix_m_s01 = numpy.full(
        (num_storm_objects, MAX_NUM_WIND_THRESHOLDS), numpy.nan
    )
    dummy_wind_threshold_matrix_m_s01[:, 0] = 34 * KT_TO_METRES_PER_SECOND
    dummy_wind_threshold_matrix_m_s01[:, 1] = 50 * KT_TO_METRES_PER_SECOND
    dummy_wind_threshold_matrix_m_s01[:, 2] = 64 * KT_TO_METRES_PER_SECOND
    dummy_wind_radius_matrix_metres = numpy.full(
        (num_storm_objects, MAX_NUM_WIND_THRESHOLDS), numpy.nan
    )

    dummy_wave_height_threshold_matrix_metres = numpy.full(
        (num_storm_objects, MAX_NUM_WAVE_HEIGHT_THRESHOLDS), numpy.nan
    )
    dummy_wave_height_radius_matrix_metres = numpy.full(
        (num_storm_objects, MAX_NUM_WAVE_HEIGHT_THRESHOLDS), numpy.nan
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

    these_dim = (a_deck_io.STORM_OBJECT_DIM,)
    main_data_dict = {
        a_deck_io.CYCLONE_ID_KEY: (these_dim, cyclone_id_strings),
        a_deck_io.VALID_TIME_KEY: (these_dim, valid_times_unix_sec),
        a_deck_io.TECHNIQUE_KEY: (these_dim, dummy_technique_strings),
        a_deck_io.SYSTEM_DEPTH_KEY: (these_dim, dummy_system_depth_strings),
        a_deck_io.STORM_TYPE_KEY: (these_dim, storm_type_strings),
        a_deck_io.LATITUDE_KEY: (these_dim, storm_latitudes_deg_n),
        a_deck_io.LONGITUDE_KEY: (these_dim, storm_longitudes_deg_e),
        a_deck_io.INTENSITY_KEY: (these_dim, storm_intensities_m_s01),
        a_deck_io.SEA_LEVEL_PRESSURE_KEY: (these_dim, storm_min_pressures_pa),
        a_deck_io.LAST_ISOBAR_PRESSURE_KEY: (
            these_dim, dummy_last_closed_isobar_pressures_pa
        ),
        a_deck_io.LAST_ISOBAR_RADIUS_KEY: (
            these_dim, dummy_last_closed_isobar_radii_metres
        ),
        a_deck_io.MAX_WIND_RADIUS_KEY: (these_dim, dummy_max_wind_ratii_metres),
        a_deck_io.GUST_SPEED_KEY: (these_dim, dummy_gust_speeds_m_s01),
        a_deck_io.EYE_DIAMETER_KEY: (these_dim, dummy_eye_diameters_metres),
        a_deck_io.MAX_SEA_HEIGHT_KEY: (these_dim, dummy_max_sea_heights_metres)
    }

    these_dim = (a_deck_io.STORM_OBJECT_DIM, a_deck_io.WIND_THRESHOLD_DIM)
    main_data_dict.update({
        a_deck_io.WIND_THRESHOLD_KEY: (
            these_dim, dummy_wind_threshold_matrix_m_s01
        ),
        a_deck_io.WIND_RADIUS_CIRCULAR_KEY: (
            these_dim, dummy_wind_radius_matrix_metres + 0.
        ),
        a_deck_io.WIND_RADIUS_NE_QUADRANT_KEY: (
            these_dim, dummy_wind_radius_matrix_metres + 0.
        ),
        a_deck_io.WIND_RADIUS_SE_QUADRANT_KEY: (
            these_dim, dummy_wind_radius_matrix_metres + 0.
        ),
        a_deck_io.WIND_RADIUS_SW_QUADRANT_KEY: (
            these_dim, dummy_wind_radius_matrix_metres + 0.
        ),
        a_deck_io.WIND_RADIUS_NW_QUADRANT_KEY: (
            these_dim, dummy_wind_radius_matrix_metres + 0.
        )
    })

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
    a_deck_table_xarray = _compute_storm_motions(a_deck_table_xarray)
    a_deck_table_xarray = raw_a_deck_io._create_extrap_forecasts(
        a_deck_table_xarray
    )

    print('Writing fake A-decks to: "{0:s}"...'.format(output_file_name))
    a_deck_io.write_file(
        a_deck_table_xarray=a_deck_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        ebtrk_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        basin_id_strings=getattr(INPUT_ARG_OBJECT, BASINS_ARG_NAME),
        start_year=getattr(INPUT_ARG_OBJECT, START_YEAR_ARG_NAME),
        end_year=getattr(INPUT_ARG_OBJECT, END_YEAR_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
