"""Unit tests for custom_metrics_scalar.py."""

import unittest
import numpy
from keras import backend as K
from ml4tccf.machine_learning import custom_metrics_scalar
from ml4tccf.machine_learning import custom_losses_scalar_test

TOLERANCE = 1e-6

TARGET_MATRIX = custom_losses_scalar_test.TARGET_MATRIX
PREDICTION_MATRIX = custom_losses_scalar_test.PREDICTION_MATRIX
TARGET_TENSOR = custom_losses_scalar_test.TARGET_TENSOR
PREDICTION_TENSOR = custom_losses_scalar_test.PREDICTION_TENSOR

MEAN_PREDICTION = -20. / 48

PREDICTIVE_RANGE_MATRIX = numpy.array([
    [1, 1],
    [1, 3],
    [2, 1],
    [2, 1],
    [1, 0],
    [1, 1],
    [1, 1],
    [1, 2]
], dtype=float)

MEAN_PREDICTIVE_RANGE = numpy.mean(PREDICTIVE_RANGE_MATRIX)
MEAN_TARGET = 1. / 16
MEAN_GRID_SPACING_KM = 2.0375

MEAN_DISTANCE_ERRORS_KM = numpy.sqrt(
    custom_losses_scalar_test.MEAN_ROW_DISTANCES_KM ** 2 +
    custom_losses_scalar_test.MEAN_COLUMN_DISTANCES_KM ** 2
)
MEAN_OVERALL_DISTANCE_KILOMETRES = numpy.mean(MEAN_DISTANCE_ERRORS_KM)

PREDICTION_DIFF_MATRIX_KM = numpy.sqrt(
    custom_losses_scalar_test.ROW_PREDICTION_DIFF_MATRIX_KM ** 2 +
    custom_losses_scalar_test.COLUMN_PREDICTION_DIFF_MATRIX_KM ** 2
)
MEAN_PREDICTION_DIFFS_KM = numpy.mean(
    PREDICTION_DIFF_MATRIX_KM, axis=(-2, -1)
)
CRPS_KILOMETRES = numpy.mean(
    MEAN_DISTANCE_ERRORS_KM - 0.5 * MEAN_PREDICTION_DIFFS_KM
)


class CustomMetricsScalarTests(unittest.TestCase):
    """Each method is a unit test for custom_metrics_scalar.py."""

    def test_mean_prediction(self):
        """Ensures correct output from mean_prediction."""

        this_mean_prediction = custom_metrics_scalar.mean_prediction(
            target_tensor=TARGET_TENSOR, prediction_tensor=PREDICTION_TENSOR
        )
        self.assertTrue(numpy.isclose(
            K.eval(this_mean_prediction), MEAN_PREDICTION, atol=TOLERANCE
        ))

    def test_mean_predictive_range(self):
        """Ensures correct output from mean_predictive_range."""

        this_predictive_range = custom_metrics_scalar.mean_predictive_range(
            target_tensor=TARGET_TENSOR, prediction_tensor=PREDICTION_TENSOR
        )
        self.assertTrue(numpy.isclose(
            K.eval(this_predictive_range), MEAN_PREDICTIVE_RANGE, atol=TOLERANCE
        ))

    def test_mean_target(self):
        """Ensures correct output from mean_target."""

        this_mean_target = custom_metrics_scalar.mean_target(
            target_tensor=TARGET_TENSOR, prediction_tensor=PREDICTION_TENSOR
        )
        self.assertTrue(numpy.isclose(
            K.eval(this_mean_target), MEAN_TARGET, atol=TOLERANCE
        ))

    def test_mean_grid_spacing_kilometres(self):
        """Ensures correct output from mean_grid_spacing_kilometres."""

        this_mean_spacing_km = custom_metrics_scalar.mean_grid_spacing_kilometres(
            target_tensor=TARGET_TENSOR, prediction_tensor=PREDICTION_TENSOR
        )
        self.assertTrue(numpy.isclose(
            K.eval(this_mean_spacing_km), MEAN_GRID_SPACING_KM, atol=TOLERANCE
        ))

    def test_mean_distance_kilometres(self):
        """Ensures correct output from mean_distance_kilometres."""

        this_mean_distance_km = custom_metrics_scalar.mean_distance_kilometres(
            target_tensor=TARGET_TENSOR, prediction_tensor=PREDICTION_TENSOR
        )
        self.assertTrue(numpy.isclose(
            K.eval(this_mean_distance_km), MEAN_OVERALL_DISTANCE_KILOMETRES,
            atol=TOLERANCE
        ))

    def test_crps_kilometres(self):
        """Ensures correct output from crps_kilometres."""

        this_crps_kilometres = custom_metrics_scalar.crps_kilometres(
            target_tensor=TARGET_TENSOR, prediction_tensor=PREDICTION_TENSOR
        )
        self.assertTrue(numpy.isclose(
            K.eval(this_crps_kilometres), CRPS_KILOMETRES, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
