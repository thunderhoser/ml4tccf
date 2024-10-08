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

INPUT_FILE_ARG_NAME = 'input_a_deck_file_name'
TRAINING_YEARS_ARG_NAME = 'training_years'
TRAINING_CYCLONES_ARG_NAME = 'training_cyclone_id_strings'
OUTPUT_FILE_ARG_NAME = 'output_a_deck_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing unnormalized data.  Will be read by '
    '`a_deck_io.read_file`.'
)
TRAINING_YEARS_HELP_STRING = (
    'List of training years.  Normalization parameters will be computed only '
    'from these years.  If you would rather specify cyclone IDs, leave this '
    'argument alone.'
)
TRAINING_CYCLONES_HELP_STRING = (
    'List of training cyclones (IDs in format "yyyyBBnn").  Normalization '
    'parameters will be computed only from these cyclones.  If you would '
    'rather specify years, leave this argument alone.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Normalized data will be written here by '
    '`a_deck_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TRAINING_YEARS_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=TRAINING_YEARS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TRAINING_CYCLONES_ARG_NAME, type=str, nargs='+', required=False,
    default=[''], help=TRAINING_CYCLONES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_file_name, training_years, training_cyclone_id_strings,
         output_file_name):
    """Normalizes A-deck data.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param training_years: Same.
    :param training_cyclone_id_strings: Same.
    :param output_file_name: Same.
    """

    if len(training_years) == 1 and training_years[0] < 1900:
        training_years = None
    if (
            len(training_cyclone_id_strings) == 1 and
            training_cyclone_id_strings[0] == ''
    ):
        training_cyclone_id_strings = None

    assert not (training_years is None and training_cyclone_id_strings is None)

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    a_deck_table_xarray = a_deck_io.read_file(input_file_name)

    if training_cyclone_id_strings is None:
        a_deck_years = numpy.array([
            int(time_conversion.unix_sec_to_string(t, '%Y'))
            for t in a_deck_table_xarray[a_deck_io.VALID_TIME_KEY].values
        ], dtype=int)

        training_row_flags = numpy.isin(
            element=a_deck_years, test_elements=training_years
        )
        training_row_indices = numpy.where(training_row_flags)[0]
    else:
        training_row_flags = numpy.isin(
            element=a_deck_table_xarray[a_deck_io.CYCLONE_ID_KEY].values,
            test_elements=numpy.array(training_cyclone_id_strings)
        )
        training_row_indices = numpy.where(training_row_flags)[0]

    adt = a_deck_table_xarray

    norm_absolute_latitudes = normalization._normalize_one_variable(
        actual_values_new=numpy.absolute(adt[a_deck_io.LATITUDE_KEY].values),
        actual_values_training=numpy.absolute(
            adt[a_deck_io.LATITUDE_KEY].values[training_row_indices]
        )
    )

    real_subindices = numpy.where(numpy.invert(numpy.isnan(
        adt[a_deck_io.EXTRAP_LATITUDE_KEY].values[training_row_indices]
    )))[0]

    norm_abs_extrap_latitudes = normalization._normalize_one_variable(
        actual_values_new=numpy.absolute(
            adt[a_deck_io.EXTRAP_LATITUDE_KEY].values
        ),
        actual_values_training=numpy.absolute(
            adt[a_deck_io.EXTRAP_LATITUDE_KEY].values[
                training_row_indices[real_subindices]
            ]
        )
    )

    orig_longitude_sines = numpy.sin(
        DEGREES_TO_RADIANS * adt[a_deck_io.LONGITUDE_KEY].values
    )
    norm_longitude_sines = normalization._normalize_one_variable(
        actual_values_new=orig_longitude_sines,
        actual_values_training=orig_longitude_sines[training_row_indices]
    )

    orig_longitude_cosines = numpy.cos(
        DEGREES_TO_RADIANS * adt[a_deck_io.LONGITUDE_KEY].values
    )
    norm_longitude_cosines = normalization._normalize_one_variable(
        actual_values_new=orig_longitude_cosines,
        actual_values_training=orig_longitude_cosines[training_row_indices]
    )

    norm_intensities = normalization._normalize_one_variable(
        actual_values_new=adt[a_deck_io.INTENSITY_KEY].values + 0.,
        actual_values_training=
        adt[a_deck_io.INTENSITY_KEY].values[training_row_indices]
    )

    real_subindices = numpy.where(numpy.invert(numpy.isnan(
        adt[a_deck_io.SEA_LEVEL_PRESSURE_KEY].values[training_row_indices]
    )))[0]

    norm_central_pressures = normalization._normalize_one_variable(
        actual_values_new=adt[a_deck_io.SEA_LEVEL_PRESSURE_KEY].values,
        actual_values_training=adt[a_deck_io.SEA_LEVEL_PRESSURE_KEY].values[
            training_row_indices[real_subindices]
        ]
    )

    orig_motion_headings_standard_deg = numpy.full(
        len(adt[a_deck_io.MOTION_HEADING_KEY].values), numpy.nan
    )
    real_indices = numpy.where(numpy.invert(numpy.isnan(
        orig_motion_headings_standard_deg
    )))[0]
    orig_motion_headings_standard_deg[real_indices] = (
        misc_utils.geodetic_to_standard_angles(
            adt[a_deck_io.MOTION_HEADING_KEY].values[real_indices]
        )
    )

    orig_eastward_motions_m_s01 = (
        adt[a_deck_io.MOTION_SPEED_KEY].values *
        numpy.cos(DEGREES_TO_RADIANS * orig_motion_headings_standard_deg)
    )
    orig_northward_motions_m_s01 = (
        adt[a_deck_io.MOTION_SPEED_KEY].values *
        numpy.sin(DEGREES_TO_RADIANS * orig_motion_headings_standard_deg)
    )

    real_subindices = numpy.where(numpy.invert(numpy.isnan(
        orig_eastward_motions_m_s01[training_row_indices]
    )))[0]

    norm_eastward_motions_m_s01 = normalization._normalize_one_variable(
        actual_values_new=orig_eastward_motions_m_s01,
        actual_values_training=orig_eastward_motions_m_s01[
            training_row_indices[real_subindices]
        ]
    )

    real_subindices = numpy.where(numpy.invert(numpy.isnan(
        orig_northward_motions_m_s01[training_row_indices]
    )))[0]

    norm_northward_motions_m_s01 = normalization._normalize_one_variable(
        actual_values_new=orig_northward_motions_m_s01,
        actual_values_training=
        orig_northward_motions_m_s01[training_row_indices[real_subindices]]
    )

    num_storm_objects = len(norm_northward_motions_m_s01)
    storm_object_indices = numpy.linspace(
        0, num_storm_objects - 1, num=num_storm_objects, dtype=int
    )
    metadata_dict = {
        a_deck_io.STORM_OBJECT_DIM: storm_object_indices
    }

    # Process actual data.
    these_dim = (a_deck_io.STORM_OBJECT_DIM,)
    main_data_dict = {
        a_deck_io.CYCLONE_ID_KEY: (
            these_dim, adt[a_deck_io.CYCLONE_ID_KEY].values
        ),
        a_deck_io.VALID_TIME_KEY: (
            these_dim, adt[a_deck_io.VALID_TIME_KEY].values
        ),
        a_deck_io.STORM_TYPE_KEY: (
            these_dim, adt[a_deck_io.STORM_TYPE_KEY].values
        ),
        a_deck_io.UNNORM_EXTRAP_LATITUDE_KEY: (
            these_dim, adt[a_deck_io.EXTRAP_LATITUDE_KEY].values
        ),
        a_deck_io.UNNORM_EXTRAP_LONGITUDE_KEY: (
            these_dim, adt[a_deck_io.EXTRAP_LONGITUDE_KEY].values
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
            these_dim, adt[a_deck_io.INTENSITY_KEY].values
        ),
        a_deck_io.SEA_LEVEL_PRESSURE_KEY: (
            these_dim, norm_central_pressures
        ),
        a_deck_io.EASTWARD_MOTION_KEY: (
            these_dim, norm_eastward_motions_m_s01
        ),
        a_deck_io.NORTHWARD_MOTION_KEY: (
            these_dim, norm_northward_motions_m_s01
        )
    }

    attribute_dict = {
        a_deck_io.TRAINING_YEARS_FOR_NORM_KEY:
            [] if training_years is None else training_years,
        a_deck_io.TRAINING_CYCLONES_FOR_NORM_KEY:
            '' if training_cyclone_id_strings is None
            else ' '.join([c for c in training_cyclone_id_strings])
    }

    a_deck_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict, attrs=attribute_dict
    )
    a_deck_table_xarray = a_deck_io.storm_types_to_1hot_encoding(
        a_deck_table_xarray
    )

    print('Writing normalized data to: "{0:s}"...'.format(output_file_name))
    a_deck_io.write_file(
        netcdf_file_name=output_file_name,
        a_deck_table_xarray=a_deck_table_xarray
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        training_years=numpy.array(
            getattr(INPUT_ARG_OBJECT, TRAINING_YEARS_ARG_NAME), dtype=int
        ),
        training_cyclone_id_strings=getattr(
            INPUT_ARG_OBJECT, TRAINING_CYCLONES_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
