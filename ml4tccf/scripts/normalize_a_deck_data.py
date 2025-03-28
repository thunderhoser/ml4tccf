"""Normalizes A-deck data."""

import argparse
import numpy
import xarray
from gewittergefahr.gg_utils import time_conversion
from ml4tccf.io import a_deck_io
from ml4tccf.utils import normalization
from ml4tccf.utils import misc_utils

DEGREES_TO_RADIANS = numpy.pi / 180.

FIELD_NAMES_TO_KEEP = [
    a_deck_io.CYCLONE_ID_KEY, a_deck_io.VALID_TIME_KEY, a_deck_io.STORM_TYPE_KEY
]

FIELD_NAMES_TO_NORMALIZE = [
    a_deck_io.LATITUDE_KEY, a_deck_io.LONGITUDE_KEY,
    a_deck_io.INTENSITY_KEY, a_deck_io.SEA_LEVEL_PRESSURE_KEY,
    a_deck_io.MOTION_SPEED_KEY, a_deck_io.MOTION_HEADING_KEY
]

INPUT_FILE_ARG_NAME = 'input_new_a_deck_file_name'
REFERENCE_FILE_ARG_NAME = 'input_reference_a_deck_file_name'
REFERENCE_YEARS_ARG_NAME = 'reference_years'
REFERENCE_CYCLONES_ARG_NAME = 'reference_cyclone_id_strings'
OUTPUT_FILE_ARG_NAME = 'output_new_a_deck_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to file with A-decks to be normalized.  Will be read by '
    '`a_deck_io.read_file`.'
)
REFERENCE_FILE_HELP_STRING = (
    'Path to file with reference A-decks (in physical units, i.e., not '
    'normalized).  Normalization params will be taken from these reference '
    'samples.'
)
REFERENCE_YEARS_HELP_STRING = (
    'List of reference years, used to compute normalization params.  If you '
    'would rather specify cyclone IDs, leave this argument alone.'
)
REFERENCE_CYCLONES_HELP_STRING = (
    'List of reference cyclones (format "yyyyBBnn"), used to compute '
    'normalization params.  If you would rather specify years, leave this '
    'argument alone.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to file for normalized A-decks, which will be written here by '
    '`a_deck_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + REFERENCE_FILE_ARG_NAME, type=str, required=True,
    help=REFERENCE_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + REFERENCE_YEARS_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=REFERENCE_YEARS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + REFERENCE_CYCLONES_ARG_NAME, type=str, nargs='+', required=False,
    default=[''], help=REFERENCE_CYCLONES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_new_a_deck_file_name, reference_a_deck_file_name,
         reference_years, reference_cyclone_id_strings,
         output_new_a_deck_file_name):
    """Normalizes A-deck data.

    This is effectively the main method.

    :param input_new_a_deck_file_name: See documentation at top of this script.
    :param reference_a_deck_file_name: Same.
    :param reference_years: Same.
    :param reference_cyclone_id_strings: Same.
    :param output_new_a_deck_file_name: Same.
    """

    # Check input args.
    if len(reference_years) == 1 and reference_years[0] < 1900:
        reference_years = None

    if (
            len(reference_cyclone_id_strings) == 1 and
            reference_cyclone_id_strings[0] == ''
    ):
        reference_cyclone_id_strings = None

    assert not (
        reference_years is None and reference_cyclone_id_strings is None
    )

    # Do actual stuff.
    print('Reading reference A-decks from: "{0:s}"...'.format(
        reference_a_deck_file_name
    ))
    reference_table_xarray = a_deck_io.read_file(reference_a_deck_file_name)

    if reference_cyclone_id_strings is None:
        all_years = numpy.array([
            int(time_conversion.unix_sec_to_string(t, '%Y'))
            for t in reference_table_xarray[a_deck_io.VALID_TIME_KEY].values
        ], dtype=int)

        good_flags = numpy.isin(
            element=all_years, test_elements=reference_years
        )
    else:
        good_flags = numpy.isin(
            element=reference_table_xarray[a_deck_io.CYCLONE_ID_KEY].values,
            test_elements=numpy.array(reference_cyclone_id_strings)
        )

    good_indices = numpy.where(good_flags)[0]
    reference_table_xarray = reference_table_xarray.isel(
        {a_deck_io.STORM_OBJECT_DIM: good_indices}
    )

    print((
        'Reading new A-decks (those to be normalized) from: "{0:s}"...'
    ).format(
        input_new_a_deck_file_name
    ))
    new_table_xarray = a_deck_io.read_file(input_new_a_deck_file_name)

    ref_table = reference_table_xarray
    new_table = new_table_xarray

    norm_absolute_latitudes = normalization._normalize_one_variable(
        actual_values_new=
        numpy.absolute(new_table[a_deck_io.LATITUDE_KEY].values),
        actual_values_training=
        numpy.absolute(ref_table[a_deck_io.LATITUDE_KEY].values)
    )

    real_ref_indices = numpy.where(numpy.invert(numpy.isnan(
        ref_table[a_deck_io.EXTRAP_LATITUDE_KEY].values
    )))[0]

    norm_abs_extrap_latitudes = normalization._normalize_one_variable(
        actual_values_new=numpy.absolute(
            new_table[a_deck_io.EXTRAP_LATITUDE_KEY].values + 0.
        ),
        actual_values_training=numpy.absolute(
            ref_table[a_deck_io.EXTRAP_LATITUDE_KEY].values[real_ref_indices]
        )
    )

    norm_longitude_sines = normalization._normalize_one_variable(
        actual_values_new=numpy.sin(
            DEGREES_TO_RADIANS * new_table[a_deck_io.LONGITUDE_KEY].values
        ),
        actual_values_training=numpy.sin(
            DEGREES_TO_RADIANS * ref_table[a_deck_io.LONGITUDE_KEY].values
        )
    )

    norm_longitude_cosines = normalization._normalize_one_variable(
        actual_values_new=numpy.cos(
            DEGREES_TO_RADIANS * new_table[a_deck_io.LONGITUDE_KEY].values
        ),
        actual_values_training=numpy.cos(
            DEGREES_TO_RADIANS * ref_table[a_deck_io.LONGITUDE_KEY].values
        )
    )

    norm_intensities = normalization._normalize_one_variable(
        actual_values_new=new_table[a_deck_io.INTENSITY_KEY].values + 0.,
        actual_values_training=ref_table[a_deck_io.INTENSITY_KEY].values
    )

    real_ref_indices = numpy.where(numpy.invert(numpy.isnan(
        ref_table[a_deck_io.SEA_LEVEL_PRESSURE_KEY].values
    )))[0]

    norm_central_pressures = normalization._normalize_one_variable(
        actual_values_new=new_table[a_deck_io.SEA_LEVEL_PRESSURE_KEY].values,
        actual_values_training=
        ref_table[a_deck_io.SEA_LEVEL_PRESSURE_KEY].values[real_ref_indices]
    )

    ref_motion_headings_standard_deg = numpy.full(
        len(ref_table[a_deck_io.MOTION_HEADING_KEY].values), numpy.nan
    )
    real_ref_indices = numpy.where(numpy.invert(numpy.isnan(
        ref_table[a_deck_io.MOTION_HEADING_KEY].values
    )))[0]
    ref_motion_headings_standard_deg[real_ref_indices] = (
        misc_utils.geodetic_to_standard_angles(
            ref_table[a_deck_io.MOTION_HEADING_KEY].values[real_ref_indices]
        )
    )

    ref_eastward_motions_m_s01 = (
        ref_table[a_deck_io.MOTION_SPEED_KEY].values *
        numpy.cos(DEGREES_TO_RADIANS * ref_motion_headings_standard_deg)
    )
    ref_northward_motions_m_s01 = (
        ref_table[a_deck_io.MOTION_SPEED_KEY].values *
        numpy.sin(DEGREES_TO_RADIANS * ref_motion_headings_standard_deg)
    )

    new_motion_headings_standard_deg = numpy.full(
        len(new_table[a_deck_io.MOTION_HEADING_KEY].values), numpy.nan
    )
    real_new_indices = numpy.where(numpy.invert(numpy.isnan(
        new_table[a_deck_io.MOTION_HEADING_KEY].values
    )))[0]
    new_motion_headings_standard_deg[real_new_indices] = (
        misc_utils.geodetic_to_standard_angles(
            new_table[a_deck_io.MOTION_HEADING_KEY].values[real_new_indices]
        )
    )

    new_eastward_motions_m_s01 = (
        new_table[a_deck_io.MOTION_SPEED_KEY].values *
        numpy.cos(DEGREES_TO_RADIANS * new_motion_headings_standard_deg)
    )
    new_northward_motions_m_s01 = (
        new_table[a_deck_io.MOTION_SPEED_KEY].values *
        numpy.sin(DEGREES_TO_RADIANS * new_motion_headings_standard_deg)
    )

    real_ref_indices = numpy.where(numpy.invert(numpy.isnan(
        ref_eastward_motions_m_s01
    )))[0]
    norm_eastward_motions_m_s01 = normalization._normalize_one_variable(
        actual_values_new=new_eastward_motions_m_s01,
        actual_values_training=ref_eastward_motions_m_s01[real_ref_indices]
    )

    real_ref_indices = numpy.where(numpy.invert(numpy.isnan(
        ref_northward_motions_m_s01
    )))[0]
    norm_northward_motions_m_s01 = normalization._normalize_one_variable(
        actual_values_new=new_northward_motions_m_s01,
        actual_values_training=ref_northward_motions_m_s01[real_ref_indices]
    )

    # TODO(thunderhoser): The code dealing with wind radii is a bit HACKY.
    # Not fully vetted yet.  Also, I assume that the three wind thresholds are
    # 34, 50, and 64 kt.
    ref_wind_radius_matrix_metres = numpy.stack([
        ref_table[a_deck_io.WIND_RADIUS_NE_QUADRANT_KEY].values,
        ref_table[a_deck_io.WIND_RADIUS_NW_QUADRANT_KEY].values,
        ref_table[a_deck_io.WIND_RADIUS_SW_QUADRANT_KEY].values,
        ref_table[a_deck_io.WIND_RADIUS_SE_QUADRANT_KEY].values,
    ], axis=-1)

    ref_wind_radius_matrix_metres = numpy.mean(
        ref_wind_radius_matrix_metres, axis=-1
    )

    new_wind_radius_matrix_metres = numpy.stack([
        new_table[a_deck_io.WIND_RADIUS_NE_QUADRANT_KEY].values,
        new_table[a_deck_io.WIND_RADIUS_NW_QUADRANT_KEY].values,
        new_table[a_deck_io.WIND_RADIUS_SW_QUADRANT_KEY].values,
        new_table[a_deck_io.WIND_RADIUS_SE_QUADRANT_KEY].values,
    ], axis=-1)

    new_wind_radius_matrix_metres = numpy.mean(
        new_wind_radius_matrix_metres, axis=-1
    )
    new_34kt_wind_radii_metres = new_wind_radius_matrix_metres[:, 0] + 0.
    new_50kt_wind_radii_metres = new_wind_radius_matrix_metres[:, 1] + 0.
    new_64kt_wind_radii_metres = new_wind_radius_matrix_metres[:, 2] + 0.

    real_ref_indices = numpy.where(numpy.invert(numpy.isnan(
        ref_wind_radius_matrix_metres[:, 0]
    )))[0]
    norm_34kt_wind_radii_metres = normalization._normalize_one_variable(
        actual_values_new=new_wind_radius_matrix_metres[:, 0],
        actual_values_training=
        ref_wind_radius_matrix_metres[real_ref_indices, 0]
    )

    real_ref_indices = numpy.where(numpy.invert(numpy.isnan(
        ref_wind_radius_matrix_metres[:, 1]
    )))[0]
    norm_50kt_wind_radii_metres = normalization._normalize_one_variable(
        actual_values_new=new_wind_radius_matrix_metres[:, 1],
        actual_values_training=
        ref_wind_radius_matrix_metres[real_ref_indices, 1]
    )

    real_ref_indices = numpy.where(numpy.invert(numpy.isnan(
        ref_wind_radius_matrix_metres[:, 2]
    )))[0]
    norm_64kt_wind_radii_metres = normalization._normalize_one_variable(
        actual_values_new=new_wind_radius_matrix_metres[:, 2],
        actual_values_training=
        ref_wind_radius_matrix_metres[real_ref_indices, 2]
    )

    real_ref_indices = numpy.where(numpy.invert(numpy.isnan(
        ref_table[a_deck_io.MAX_WIND_RADIUS_KEY].values
    )))[0]
    new_max_wind_radii_metres = new_table[a_deck_io.MAX_WIND_RADIUS_KEY].values
    norm_max_wind_radii_metres = normalization._normalize_one_variable(
        actual_values_new=new_max_wind_radii_metres + 0.,
        actual_values_training=
        ref_table[a_deck_io.MAX_WIND_RADIUS_KEY].values[real_ref_indices]
    )

    real_ref_indices = numpy.where(numpy.invert(numpy.isnan(
        ref_table[a_deck_io.SEA_LEVEL_PRESSURE_KEY].values
    )))[0]

    norm_central_pressures = normalization._normalize_one_variable(
        actual_values_new=new_table[a_deck_io.SEA_LEVEL_PRESSURE_KEY].values,
        actual_values_training=
        ref_table[a_deck_io.SEA_LEVEL_PRESSURE_KEY].values[real_ref_indices]
    )

    num_storm_objects = len(norm_northward_motions_m_s01)
    storm_object_indices = numpy.linspace(
        0, num_storm_objects - 1, num=num_storm_objects, dtype=int
    )
    metadata_dict = {
        a_deck_io.STORM_OBJECT_DIM: storm_object_indices
    }

    these_dim = (a_deck_io.STORM_OBJECT_DIM,)
    main_data_dict = {
        a_deck_io.CYCLONE_ID_KEY: (
            these_dim, new_table[a_deck_io.CYCLONE_ID_KEY].values
        ),
        a_deck_io.VALID_TIME_KEY: (
            these_dim, new_table[a_deck_io.VALID_TIME_KEY].values
        ),
        a_deck_io.STORM_TYPE_KEY: (
            these_dim, new_table[a_deck_io.STORM_TYPE_KEY].values
        ),
        a_deck_io.UNNORM_EXTRAP_LATITUDE_KEY: (
            these_dim, new_table[a_deck_io.EXTRAP_LATITUDE_KEY].values
        ),
        a_deck_io.UNNORM_EXTRAP_LONGITUDE_KEY: (
            these_dim, new_table[a_deck_io.EXTRAP_LONGITUDE_KEY].values
        ),
        a_deck_io.ABSOLUTE_LATITUDE_KEY: (
            these_dim, norm_absolute_latitudes
        ),
        a_deck_io.ABSOLUTE_EXTRAP_LATITUDE_KEY: (
            these_dim, norm_abs_extrap_latitudes
        ),
        a_deck_io.LONGITUDE_COSINE_KEY: (
            these_dim, norm_longitude_cosines
        ),
        a_deck_io.LONGITUDE_SINE_KEY: (
            these_dim, norm_longitude_sines
        ),
        a_deck_io.INTENSITY_KEY: (
            these_dim, norm_intensities
        ),
        a_deck_io.UNNORM_INTENSITY_KEY: (
            these_dim, new_table[a_deck_io.INTENSITY_KEY].values
        ),
        a_deck_io.SEA_LEVEL_PRESSURE_KEY: (
            these_dim, norm_central_pressures
        ),
        a_deck_io.EASTWARD_MOTION_KEY: (
            these_dim, norm_eastward_motions_m_s01
        ),
        a_deck_io.NORTHWARD_MOTION_KEY: (
            these_dim, norm_northward_motions_m_s01
        ),
        a_deck_io.WIND_RADIUS_34KT_KEY: (
            these_dim, norm_34kt_wind_radii_metres
        ),
        a_deck_io.WIND_RADIUS_50KT_KEY: (
            these_dim, norm_50kt_wind_radii_metres
        ),
        a_deck_io.WIND_RADIUS_64KT_KEY: (
            these_dim, norm_64kt_wind_radii_metres
        ),
        a_deck_io.MAX_WIND_RADIUS_KEY: (
            these_dim, norm_max_wind_radii_metres
        ),
        a_deck_io.UNNORM_WIND_RADIUS_34KT_KEY: (
            these_dim, new_34kt_wind_radii_metres
        ),
        a_deck_io.UNNORM_WIND_RADIUS_50KT_KEY: (
            these_dim, new_50kt_wind_radii_metres
        ),
        a_deck_io.UNNORM_WIND_RADIUS_64KT_KEY: (
            these_dim, new_64kt_wind_radii_metres
        ),
        a_deck_io.UNNORM_MAX_WIND_RADIUS_KEY: (
            these_dim, new_max_wind_radii_metres
        )
    }

    attribute_dict = {
        a_deck_io.TRAINING_YEARS_FOR_NORM_KEY:
            [] if reference_years is None else reference_years,
        a_deck_io.TRAINING_CYCLONES_FOR_NORM_KEY:
            '' if reference_cyclone_id_strings is None
            else ' '.join(reference_cyclone_id_strings)
    }

    new_normalized_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict, attrs=attribute_dict
    )
    new_normalized_table_xarray = a_deck_io.storm_types_to_1hot_encoding(
        new_normalized_table_xarray
    )

    print('Writing normalized A-decks to: "{0:s}"...'.format(
        output_new_a_deck_file_name
    ))
    a_deck_io.write_file(
        netcdf_file_name=output_new_a_deck_file_name,
        a_deck_table_xarray=new_normalized_table_xarray
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_new_a_deck_file_name=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME
        ),
        reference_a_deck_file_name=getattr(
            INPUT_ARG_OBJECT, REFERENCE_FILE_ARG_NAME
        ),
        reference_years=numpy.array(
            getattr(INPUT_ARG_OBJECT, REFERENCE_YEARS_ARG_NAME), dtype=int
        ),
        reference_cyclone_id_strings=getattr(
            INPUT_ARG_OBJECT, REFERENCE_CYCLONES_ARG_NAME
        ),
        output_new_a_deck_file_name=getattr(
            INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME
        )
    )
