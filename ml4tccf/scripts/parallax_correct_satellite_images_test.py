"""Unit tests for parallax_correct_satellite_images.py."""

import unittest
import numpy
import xarray
from ml4tccf.utils import satellite_utils
from ml4tccf.scripts import \
    parallax_correct_satellite_images as parallax_correct

TOLERANCE = 1e-6

FIRST_LATITUDES_DEG_N = numpy.array([
    39.5, 39.6, 39.7, 39.8, 39.9, 40.0,
    40.1, 40.2, 40.3, 40.4, 40.5, 40.6
])
FIRST_LONGITUDES_DEG_E = numpy.array([
    -105.7, -105.6, -105.5, -105.4, -105.3,
    -105.2, -105.1, -105.0, -104.9, -104.8
])
DUMMY_HIGH_RES_LONGITUDES_DEG_E = numpy.array([0.0, 0.1])

THIS_DATA_DICT = {
    satellite_utils.LATITUDE_LOW_RES_KEY: (
        (satellite_utils.TIME_DIM, satellite_utils.LOW_RES_ROW_DIM),
        numpy.expand_dims(FIRST_LATITUDES_DEG_N, axis=0)
    ),
    satellite_utils.LONGITUDE_LOW_RES_KEY: (
        (satellite_utils.TIME_DIM, satellite_utils.LOW_RES_COLUMN_DIM),
        numpy.expand_dims(FIRST_LONGITUDES_DEG_E, axis=0)
    ),
    satellite_utils.LONGITUDE_HIGH_RES_KEY: (
        (satellite_utils.TIME_DIM, satellite_utils.HIGH_RES_COLUMN_DIM),
        numpy.expand_dims(DUMMY_HIGH_RES_LONGITUDES_DEG_E, axis=0)
    )
}

FIRST_SATELLITE_TABLE_COORDS_ONLY = xarray.Dataset(data_vars=THIS_DATA_DICT)
FIRST_CENTER_LATITUDE_DEG_N = 40.05
FIRST_CENTER_LONGITUDE_DEG_E = -105.25

SECOND_LATITUDES_DEG_N = FIRST_LATITUDES_DEG_N + 0.
SECOND_LONGITUDES_DEG_E = numpy.array([
    254.3, 254.4, 254.5, 254.6, 254.7,
    254.8, 254.9, 255.0, 255.1, 255.2
])

THIS_DATA_DICT = {
    satellite_utils.LATITUDE_LOW_RES_KEY: (
        (satellite_utils.TIME_DIM, satellite_utils.LOW_RES_ROW_DIM),
        numpy.expand_dims(SECOND_LATITUDES_DEG_N, axis=0)
    ),
    satellite_utils.LONGITUDE_LOW_RES_KEY: (
        (satellite_utils.TIME_DIM, satellite_utils.LOW_RES_COLUMN_DIM),
        numpy.expand_dims(SECOND_LONGITUDES_DEG_E, axis=0)
    ),
    satellite_utils.LONGITUDE_HIGH_RES_KEY: (
        (satellite_utils.TIME_DIM, satellite_utils.HIGH_RES_COLUMN_DIM),
        numpy.expand_dims(DUMMY_HIGH_RES_LONGITUDES_DEG_E, axis=0)
    )
}

SECOND_SATELLITE_TABLE_COORDS_ONLY = xarray.Dataset(data_vars=THIS_DATA_DICT)
SECOND_CENTER_LATITUDE_DEG_N = 40.05
SECOND_CENTER_LONGITUDE_DEG_E = 254.75

THIRD_LATITUDES_DEG_N = FIRST_LATITUDES_DEG_N + 0.
THIRD_LONGITUDES_DEG_E = FIRST_LONGITUDES_DEG_E[::-1]

THIS_DATA_DICT = {
    satellite_utils.LATITUDE_LOW_RES_KEY: (
        (satellite_utils.TIME_DIM, satellite_utils.LOW_RES_ROW_DIM),
        numpy.expand_dims(THIRD_LATITUDES_DEG_N, axis=0)
    ),
    satellite_utils.LONGITUDE_LOW_RES_KEY: (
        (satellite_utils.TIME_DIM, satellite_utils.LOW_RES_COLUMN_DIM),
        numpy.expand_dims(THIRD_LONGITUDES_DEG_E, axis=0)
    ),
    satellite_utils.LONGITUDE_HIGH_RES_KEY: (
        (satellite_utils.TIME_DIM, satellite_utils.HIGH_RES_COLUMN_DIM),
        numpy.expand_dims(DUMMY_HIGH_RES_LONGITUDES_DEG_E, axis=0)
    )
}

THIRD_SATELLITE_TABLE_COORDS_ONLY = xarray.Dataset(data_vars=THIS_DATA_DICT)

FOURTH_LATITUDES_DEG_N = numpy.array([
    39.5, 39.6, 39.7, 39.8, 39.9, 39.91,
    40.1, 40.2, 40.3, 40.4, 40.5, 40.6
])
FOURTH_LONGITUDES_DEG_E = numpy.array([
    -105.7, -105.6, -105.5, -105.4, -105.22,
    -105.2, -105.1, -105.0, -104.9, -104.8
])

THIS_DATA_DICT = {
    satellite_utils.LATITUDE_LOW_RES_KEY: (
        (satellite_utils.TIME_DIM, satellite_utils.LOW_RES_ROW_DIM),
        numpy.expand_dims(FOURTH_LATITUDES_DEG_N, axis=0)
    ),
    satellite_utils.LONGITUDE_LOW_RES_KEY: (
        (satellite_utils.TIME_DIM, satellite_utils.LOW_RES_COLUMN_DIM),
        numpy.expand_dims(FOURTH_LONGITUDES_DEG_E, axis=0)
    ),
    satellite_utils.LONGITUDE_HIGH_RES_KEY: (
        (satellite_utils.TIME_DIM, satellite_utils.HIGH_RES_COLUMN_DIM),
        numpy.expand_dims(DUMMY_HIGH_RES_LONGITUDES_DEG_E, axis=0)
    )
}

FOURTH_SATELLITE_TABLE_COORDS_ONLY = xarray.Dataset(data_vars=THIS_DATA_DICT)
FOURTH_CENTER_LATITUDE_DEG_N = 40.005
FOURTH_CENTER_LONGITUDE_DEG_E = -105.21

THESE_LATITUDES_DEG_N = numpy.array([
    39.5, 39.6, 39.7, 39.8, 39.9, 40.0,
    40.1, 40.2, 40.3, 40.4, 40.5, 40.6
])
THESE_LONGITUDES_DEG_E = numpy.array([
    -105.7, -105.6, -105.5, -105.4, -105.3,
    -105.2, -105.1, -105.0, -104.9, -104.8
])

THIS_BDRF_MATRIX = numpy.array([
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
    [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0],
    [3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0],
    [4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0],
    [5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0],
    [6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0],
    [7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0],
    [8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0],
    [9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0],
    [10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0],
    [11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.0]
])
THIS_BDRF_MATRIX = numpy.expand_dims(THIS_BDRF_MATRIX, axis=-1)

BDRF_DIMENSIONS = (
    satellite_utils.TIME_DIM,
    satellite_utils.HIGH_RES_ROW_DIM,
    satellite_utils.HIGH_RES_COLUMN_DIM,
    satellite_utils.HIGH_RES_WAVELENGTH_DIM
)

THIS_DATA_DICT = {
    satellite_utils.LATITUDE_LOW_RES_KEY: (
        (satellite_utils.TIME_DIM, satellite_utils.LOW_RES_ROW_DIM),
        numpy.expand_dims(THESE_LATITUDES_DEG_N, axis=0)
    ),
    satellite_utils.LONGITUDE_LOW_RES_KEY: (
        (satellite_utils.TIME_DIM, satellite_utils.LOW_RES_COLUMN_DIM),
        numpy.expand_dims(THESE_LONGITUDES_DEG_E, axis=0)
    ),
    satellite_utils.LATITUDE_HIGH_RES_KEY: (
        (satellite_utils.TIME_DIM, satellite_utils.HIGH_RES_ROW_DIM),
        numpy.expand_dims(THESE_LATITUDES_DEG_N, axis=0)
    ),
    satellite_utils.LONGITUDE_HIGH_RES_KEY: (
        (satellite_utils.TIME_DIM, satellite_utils.HIGH_RES_COLUMN_DIM),
        numpy.expand_dims(THESE_LONGITUDES_DEG_E, axis=0)
    ),
    satellite_utils.CYCLONE_ID_KEY: (
        (satellite_utils.TIME_DIM,),
        ['2020AL01']
    ),
    satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY: (
        BDRF_DIMENSIONS,
        numpy.expand_dims(THIS_BDRF_MATRIX, axis=0)
    ),
}

THIS_COORD_DICT = {
    satellite_utils.HIGH_RES_WAVELENGTH_DIM: numpy.array([0.64 * 1e-6])
}

SATELLITE_TABLE_HIGH_RES = xarray.Dataset(
    data_vars=THIS_DATA_DICT, coords=THIS_COORD_DICT
)

LATITUDE_CORRECTION_DEG = -0.1
LONGITUDE_CORRECTION_DEG = 0.2
CORRECTED_BDRF_MATRIX = numpy.array([
    [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
    [1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8],
    [2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8],
    [3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8],
    [4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8],
    [5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8],
    [6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8],
    [7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8],
    [8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8],
    [9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8],
    [10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8],
    [11.9, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8]
])
CORRECTED_BDRF_MATRIX = numpy.expand_dims(CORRECTED_BDRF_MATRIX, axis=-1)


class ParallaxCorrectSatelliteImagesTests(unittest.TestCase):
    """Each method is a unit test for parallax_correct_satellite_images.py."""

    def test_find_grid_center_one_time_first(self):
        """Ensures correct output from _find_grid_center_one_time.

        In this case, using first set of inputs.
        """

        this_center_latitude_deg_n, this_center_longitude_deg_e = (
            parallax_correct._find_grid_center_one_time(
                satellite_table_xarray=FIRST_SATELLITE_TABLE_COORDS_ONLY,
                time_index=0
            )
        )

        self.assertTrue(numpy.isclose(
            this_center_latitude_deg_n, FIRST_CENTER_LATITUDE_DEG_N,
            atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            this_center_longitude_deg_e, FIRST_CENTER_LONGITUDE_DEG_E,
            atol=TOLERANCE
        ))

    def test_find_grid_center_one_time_second(self):
        """Ensures correct output from _find_grid_center_one_time.

        In this case, using second set of inputs.
        """

        this_center_latitude_deg_n, this_center_longitude_deg_e = (
            parallax_correct._find_grid_center_one_time(
                satellite_table_xarray=SECOND_SATELLITE_TABLE_COORDS_ONLY,
                time_index=0
            )
        )

        self.assertTrue(numpy.isclose(
            this_center_latitude_deg_n, SECOND_CENTER_LATITUDE_DEG_N,
            atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            this_center_longitude_deg_e, SECOND_CENTER_LONGITUDE_DEG_E,
            atol=TOLERANCE
        ))

    def test_find_grid_center_one_time_third(self):
        """Ensures correct output from _find_grid_center_one_time.

        In this case, using third set of inputs.
        """

        with self.assertRaises(AssertionError):
            parallax_correct._find_grid_center_one_time(
                satellite_table_xarray=THIRD_SATELLITE_TABLE_COORDS_ONLY,
                time_index=0
            )

    def test_find_grid_center_one_time_fourth(self):
        """Ensures correct output from _find_grid_center_one_time.

        In this case, using fourth set of inputs.
        """

        this_center_latitude_deg_n, this_center_longitude_deg_e = (
            parallax_correct._find_grid_center_one_time(
                satellite_table_xarray=FOURTH_SATELLITE_TABLE_COORDS_ONLY,
                time_index=0
            )
        )

        self.assertTrue(numpy.isclose(
            this_center_latitude_deg_n, FOURTH_CENTER_LATITUDE_DEG_N,
            atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            this_center_longitude_deg_e, FOURTH_CENTER_LONGITUDE_DEG_E,
            atol=TOLERANCE
        ))

    def test_parallax_correct_high_res_one_time(self):
        """Ensures correct output from _parallax_correct_high_res_one_time."""

        this_bdrf_matrix = parallax_correct._parallax_correct_high_res_one_time(
            satellite_table_xarray=SATELLITE_TABLE_HIGH_RES,
            time_index=0,
            latitude_shift_testing_only_deg=LATITUDE_CORRECTION_DEG,
            longitude_shift_testing_only_deg=LONGITUDE_CORRECTION_DEG
        )

        self.assertTrue(numpy.allclose(
            this_bdrf_matrix, CORRECTED_BDRF_MATRIX, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
