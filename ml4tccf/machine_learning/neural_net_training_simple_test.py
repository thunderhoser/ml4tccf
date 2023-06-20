"""Unit tests for neural_net_training_simple.py."""

import unittest
import numpy
from ml4tccf.io import a_deck_io
from ml4tccf.machine_learning import neural_net_training_simple as nn_training

TOLERANCE = 1e-6

# The following constants are used to test _extrap_based_forecasts_to_rowcol.
EXTRAP_LATITUDES_DEG_N = numpy.array([
    40.038, 39.990, 39.980, 40.110, 40.120, 40.110,
    40.111, 39.990, 39.989, 40.050
])
EXTRAP_LONGITUDES_DEG_E = numpy.array([
    -105.25578, -105.31000, -105.31500, -105.32000, -105.32000, -105.15000,
    -105.15000, -105.15000, -105.14700, -105.23000
])

GRID_LATITUDES_DEG_N = numpy.array([40.00, 40.02, 40.04, 40.06, 40.08, 40.10])
GRID_LONGITUDES_DEG_E = numpy.array([
    -105.30, -105.28, -105.26, -105.24, -105.22, -105.20, -105.18, -105.16
])

GRID_LATITUDE_MATRIX_DEG_N = numpy.expand_dims(GRID_LATITUDES_DEG_N, axis=0)
GRID_LATITUDE_MATRIX_DEG_N = numpy.expand_dims(
    GRID_LATITUDE_MATRIX_DEG_N, axis=-1
)
GRID_LATITUDE_MATRIX_DEG_N = numpy.repeat(
    GRID_LATITUDE_MATRIX_DEG_N, axis=0, repeats=len(EXTRAP_LATITUDES_DEG_N)
)

GRID_LONGITUDE_MATRIX_DEG_E = numpy.expand_dims(GRID_LONGITUDES_DEG_E, axis=0)
GRID_LONGITUDE_MATRIX_DEG_E = numpy.expand_dims(
    GRID_LONGITUDE_MATRIX_DEG_E, axis=-1
)
GRID_LONGITUDE_MATRIX_DEG_E = numpy.repeat(
    GRID_LONGITUDE_MATRIX_DEG_E, axis=0, repeats=len(EXTRAP_LATITUDES_DEG_N)
)

SATELLITE_DATA_DICT = {
    nn_training.LOW_RES_LATITUDES_KEY: GRID_LATITUDE_MATRIX_DEG_N,
    nn_training.LOW_RES_LONGITUDES_KEY: GRID_LONGITUDE_MATRIX_DEG_E
}

SCALAR_PREDICTOR_MATRIX_LATLNG = numpy.transpose(numpy.vstack((
    EXTRAP_LATITUDES_DEG_N, EXTRAP_LONGITUDES_DEG_E
)))
SCALAR_A_DECK_FIELD_NAMES = [
    a_deck_io.UNNORM_EXTRAP_LATITUDE_KEY, a_deck_io.UNNORM_EXTRAP_LONGITUDE_KEY
]

EXTRAP_ROW_OFFSETS_PX = numpy.array([
    -0.60, -3.00, -3.50, 3.00, 3.50, 3.00, 3.05, -3.00, -3.05, 0.00
])
EXTRAP_COLUMN_OFFSETS_PX = numpy.array([
    -1.289, -4.000, -4.250, -4.500, -4.500, 4.000, 4.000, 4.000, 4.150, 0.000
])
SCALAR_PREDICTOR_MATRIX_ROWCOL = numpy.transpose(numpy.vstack((
    EXTRAP_ROW_OFFSETS_PX, EXTRAP_COLUMN_OFFSETS_PX
)))


class NeuralNetTrainingSimpleTests(unittest.TestCase):
    """Each method is a unit test for neural_net_training_simple.py."""

    def test_extrap_based_forecasts_to_rowcol(self):
        """Ensures correct output from _extrap_based_forecasts_to_rowcol."""

        this_predictor_matrix = nn_training._extrap_based_forecasts_to_rowcol(
            scalar_predictor_matrix=SCALAR_PREDICTOR_MATRIX_LATLNG,
            scalar_a_deck_field_names=SCALAR_A_DECK_FIELD_NAMES,
            satellite_data_dict=SATELLITE_DATA_DICT
        )

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, SCALAR_PREDICTOR_MATRIX_ROWCOL,
            atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
