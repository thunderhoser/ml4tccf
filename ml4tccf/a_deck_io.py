"""IO methods for processed A-deck files from the ATCF.

ATCF = Automated Tropical-cyclone-forecasting System
"""

import os
import sys
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils

CYCLONE_ID_REGEX = '[0-9][0-9][0-9][0-9][A-Z][A-Z][0-9][0-9]'

STORM_OBJECT_DIM = 'storm_object_index'
WIND_THRESHOLD_DIM = 'wind_threshold_index'
WAVE_HEIGHT_THRESHOLD_DIM = 'wave_height_threshold_index'

CYCLONE_ID_KEY = 'cyclone_id_string'
VALID_TIME_KEY = 'valid_time_unix_sec'
TECHNIQUE_KEY = 'technique_string'
LATITUDE_KEY = 'latitude_deg_n'
LONGITUDE_KEY = 'longitude_deg_e'
EXTRAP_LATITUDE_KEY = 'extrapolated_6h_latitude_deg_n'
EXTRAP_LONGITUDE_KEY = 'extrapolated_6h_longitude_deg_e'
UNNORM_EXTRAP_LATITUDE_KEY = 'unnormalized_extrapolated_6h_latitude_deg_n'
UNNORM_EXTRAP_LONGITUDE_KEY = 'unnormalized_extrapolated_6h_longitude_deg_e'
INTENSITY_KEY = 'intensity_m_s01'
SEA_LEVEL_PRESSURE_KEY = 'sea_level_pressure_pa'
STORM_TYPE_KEY = 'storm_type_string'
LAST_ISOBAR_PRESSURE_KEY = 'last_closed_isobar_pressure_pa'
LAST_ISOBAR_RADIUS_KEY = 'last_closed_isobar_radius_metres'
MAX_WIND_RADIUS_KEY = 'max_wind_radius_metres'
GUST_SPEED_KEY = 'gust_speed_m_s01'
EYE_DIAMETER_KEY = 'eye_diameter_metres'
MAX_SEA_HEIGHT_KEY = 'max_sea_height_metres'
MOTION_HEADING_KEY = 'motion_heading_deg'
MOTION_SPEED_KEY = 'motion_speed_m_s01'
SYSTEM_DEPTH_KEY = 'system_depth_string'
WIND_THRESHOLD_KEY = 'wind_threshold_m_s01'
WIND_RADIUS_CIRCULAR_KEY = 'wind_radius_circular_metres'
WIND_RADIUS_NE_QUADRANT_KEY = 'wind_radius_ne_quadrant_metres'
WIND_RADIUS_NW_QUADRANT_KEY = 'wind_radius_nw_quadrant_metres'
WIND_RADIUS_SW_QUADRANT_KEY = 'wind_radius_sw_quadrant_metres'
WIND_RADIUS_SE_QUADRANT_KEY = 'wind_radius_se_quadrant_metres'
WAVE_HEIGHT_THRESHOLD_KEY = 'wave_height_threshold_metres'
WAVE_HEIGHT_RADIUS_CIRCULAR_KEY = 'wave_height_radius_circular_metres'
WAVE_HEIGHT_RADIUS_NE_QUADRANT_KEY = 'wave_height_radius_ne_quadrant_metres'
WAVE_HEIGHT_RADIUS_NW_QUADRANT_KEY = 'wave_height_radius_nw_quadrant_metres'
WAVE_HEIGHT_RADIUS_SW_QUADRANT_KEY = 'wave_height_radius_sw_quadrant_metres'
WAVE_HEIGHT_RADIUS_SE_QUADRANT_KEY = 'wave_height_radius_se_quadrant_metres'

ABSOLUTE_LATITUDE_KEY = 'absolute_latitude_deg_n'
LONGITUDE_COSINE_KEY = 'longitude_cosine'
LONGITUDE_SINE_KEY = 'longitude_sine'
EASTWARD_MOTION_KEY = 'eastward_motion_m_s01'
NORTHWARD_MOTION_KEY = 'northward_motion_m_s01'

TRAINING_YEARS_FOR_NORM_KEY = 'training_years_for_normalization'

TROPICAL_DISTURBANCE_TYPE_STRING = 'tropical_disturbance'
TROPICAL_DEPRESSION_TYPE_STRING = 'tropical_depression'
TROPICAL_STORM_TYPE_STRING = 'tropical_storm'
TROPICAL_TYPHOON_TYPE_STRING = 'tropical_typhoon'
TROPICAL_SUPER_TYPHOON_TYPE_STRING = 'tropical_super_typhoon'
TROPICAL_CYCLONE_TYPE_STRING = 'tropical_cyclone'
TROPICAL_HURRICANE_TYPE_STRING = 'tropical_hurricane'
SUBTROPICAL_DEPRESSION_TYPE_STRING = 'subtropical_depression'
SUBTROPICAL_STORM_TYPE_STRING = 'subtropical_storm'
EXTRATROPICAL_TYPE_STRING = 'extratropical_system'
POSTTROPICAL_TYPE_STRING = 'posttropical_system'
INLAND_TYPE_STRING = 'inland'
DISSIPATING_TYPE_STRING = 'dissipating'
LOW_TYPE_STRING = 'low'
WAVE_TYPE_STRING = 'wave'
EXTRAPOLATED_TYPE_STRING = 'extrapolated'
MONSOON_DEPRESSION_TYPE_STRING = 'monsoon_depression'
UNKNOWN_TYPE_STRING = 'unknown'


def read_file(netcdf_file_name):
    """Reads A-deck data from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: a_deck_table_xarray: xarray table.  Documentation in the xarray
        table should make values self-explanatory.
    """

    return xarray.open_dataset(netcdf_file_name)


def write_file(a_deck_table_xarray, netcdf_file_name):
    """Writes A-deck data to NetCDF file.

    :param a_deck_table_xarray: xarray table in format returned by `read_file`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    a_deck_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def concat_tables_over_storm_object(a_deck_tables_xarray):
    """Concatenates tables with A-deck data over the storm-object dimension.

    :param a_deck_tables_xarray: 1-D list of xarray tables in format returned by
        `read_file`.
    :return: a_deck_table_xarray: One xarray table, in format returned by
        `read_file`, created by concatenating inputs.
    """

    num_storm_objects_found = 0

    for i in range(len(a_deck_tables_xarray)):
        assert numpy.array_equal(
            a_deck_tables_xarray[0].coords[WIND_THRESHOLD_DIM].values,
            a_deck_tables_xarray[i].coords[WIND_THRESHOLD_DIM].values
        )
        assert numpy.array_equal(
            a_deck_tables_xarray[0].coords[WAVE_HEIGHT_THRESHOLD_DIM].values,
            a_deck_tables_xarray[i].coords[WAVE_HEIGHT_THRESHOLD_DIM].values
        )

        this_num_storm_objects = len(
            a_deck_tables_xarray[i].coords[STORM_OBJECT_DIM].values
        )
        these_indices = numpy.linspace(
            num_storm_objects_found,
            num_storm_objects_found + this_num_storm_objects - 1,
            num=this_num_storm_objects, dtype=int
        )
        num_storm_objects_found += this_num_storm_objects

        a_deck_tables_xarray[i] = a_deck_tables_xarray[i].assign_coords({
            STORM_OBJECT_DIM: these_indices
        })

    return xarray.concat(objs=a_deck_tables_xarray, dim=STORM_OBJECT_DIM)