"""Helper methods for XBT (extended best track) data."""

STORM_ID_KEY = 'storm_id_string'
STORM_NAME_KEY = 'storm_name'
VALID_TIME_KEY = 'valid_time_unix_hours'
CENTER_LATITUDE_KEY = 'center_latitude_deg_n'
CENTER_LONGITUDE_KEY = 'center_longitude_deg_e'
MAX_SUSTAINED_WIND_KEY = 'max_sustained_wind_m_s01'
MIN_PRESSURE_KEY = 'min_pressure_pa'
MAX_WIND_RADIUS_KEY = 'max_wind_radius_metres'
EYE_DIAMETER_KEY = 'eye_diameter_metres'
OUTERMOST_PRESSURE_KEY = 'pressure_of_outer_closed_isobar_pa'
OUTERMOST_RADIUS_KEY = 'radius_of_outer_closed_isobar_metres'
RADII_OF_34KT_WIND_KEY = 'radius_of_34kt_wind_metres'
RADII_OF_50KT_WIND_KEY = 'radius_of_50kt_wind_metres'
RADII_OF_64KT_WIND_KEY = 'radius_of_64kt_wind_metres'
STORM_TYPE_KEY = 'storm_type_string'
DISTANCE_TO_LAND_KEY = 'distance_to_land_metres'
MAX_WIND_RADIUS_SOURCE_KEY = 'max_wind_radius_source_string'
EYE_DIAMETER_SOURCE_KEY = 'eye_diameter_source_string'
OUTERMOST_PRESSURE_SOURCE_KEY = 'outermost_pressure_source_string'
OUTERMOST_RADIUS_SOURCE_KEY = 'outermost_radius_source_string'

STORM_OBJECT_DIM = 'storm_object'
QUADRANT_DIM = 'quadrant'
QUADRANT_STRINGS = ['NE', 'SE', 'SW', 'NW']

TROPICAL_TYPE_STRING = 'tropical'
WAVE_TYPE_STRING = 'wave'
DISTURBANCE_TYPE_STRING = 'disturbance'
SUBTROPICAL_TYPE_STRING = 'subtropical'
EXTRATROPICAL_TYPE_STRING = 'extratropical'
REMNANT_LOW_TYPE_STRING = 'remnant_low'
UNKNOWN_TYPE_STRING = 'unknown'

B_DECK_SOURCE_STRING = 'b_deck'
A_DECK_SOURCE_STRING = 'a_deck_carq_line'
OFFICIAL_0H_FCST_SOURCE_STRING = 'official_forecast_0hour_lead'
OFFICIAL_3H_FCST_SOURCE_STRING = 'official_forecast_3hour_lead'
PREVIOUS_XBT_VERSION_SOURCE_STRING = 'previous_version_of_xbt_file'
MISSING_SOURCE_STRING = 'missing'