"""Unit tests for custom_losses.py."""

import unittest
import numpy
import tensorflow
from keras import backend as K
from ml4tccf.machine_learning import custom_losses

TOLERANCE = 1e-6

# The following constants are used to test every method.
TARGET_MATRIX = numpy.array([
    [1, -1, 2.00, 0],
    [2, -2, 2.01, 0],
    [3, -3, 2.02, 30],
    [4, -4, 2.03, 30],
    [5, -5, 2.04, 45],
    [6, -6, 2.05, 45],
    [7, -7, 2.07, 60],
    [-8, 9, 2.08, 60]
])

FIRST_PREDICTION_MATRIX = numpy.array([
    [3, -3],
    [2, -8],
    [-1, -5],
    [6, -7],
    [9, -5],
    [5, -7],
    [7, -5],
    [-11, 13]
], dtype=float)

SECOND_PREDICTION_MATRIX = numpy.array([
    [2, -2],
    [3, -9],
    [0, -4],
    [5, -6],
    [9, -5],
    [4, -8],
    [6, -4],
    [-10, 14]
], dtype=float)

THIRD_PREDICTION_MATRIX = numpy.array([
    [3, -2],
    [2, -6],
    [-2, -5],
    [4, -7],
    [10, -5],
    [4, -7],
    [7, -5],
    [-11, 12]
], dtype=float)

PREDICTION_MATRIX = numpy.stack((
    FIRST_PREDICTION_MATRIX, SECOND_PREDICTION_MATRIX, THIRD_PREDICTION_MATRIX
), axis=-1)

MEAN_PREDICTION_MATRIX = (1. / 3) * numpy.array([
    [8, -7],
    [7, -23],
    [-3, -14],
    [15, -20],
    [28, -15],
    [13, -22],
    [20, -14],
    [-32, 39],
], dtype=float)

MEAN_ROW_DISTANCES_KM = (1. / 3) * numpy.array([
    10.00, 2.01, 24.24, 6.09, 26.52, 10.25, 2.07, 16.64
])
MEAN_COLUMN_DISTANCES_KM = (1. / 3) * numpy.array([
    8.00, 34.17, 10.10, 16.24, 0.00, 8.20, 14.49, 24.96
])
MEAN_ROW_DISTANCES_DISCRETIZED_KM = (1. / 3) * numpy.array([
    0, 0, 24.24, 0, 26.52, 0, 0, 0
])
MEAN_COLUMN_DISTANCES_DISCRETIZED_KM = (1. / 3) * numpy.array([
    0, 34.17, 0, 16.24, 0, 0, 14.49, 24.96
])

MEAN_OVERALL_SQUARED_DISTANCE_KILOMETRES2 = numpy.mean(
    MEAN_ROW_DISTANCES_KM ** 2 + MEAN_COLUMN_DISTANCES_KM ** 2
)
MEAN_OVERALL_SQ_DIST_DISCRETIZED_KILOMETRES2 = numpy.mean(
    MEAN_ROW_DISTANCES_DISCRETIZED_KM ** 2 +
    MEAN_COLUMN_DISTANCES_DISCRETIZED_KM ** 2
)

TARGET_TENSOR = tensorflow.constant(TARGET_MATRIX, dtype=tensorflow.float64)
PREDICTION_TENSOR = tensorflow.constant(
    PREDICTION_MATRIX, dtype=tensorflow.float64
)

# The following constants are used to test mean_squared_distance_kilometres2 and
# discretized_mean_sq_dist_kilometres2.
MEAN_SQUARED_DISTANCES_KILOMETRES2 = (
    MEAN_ROW_DISTANCES_KM ** 2 + MEAN_COLUMN_DISTANCES_KM ** 2
)
MEAN_SQ_DISTANCES_DISCRETIZED_KM2 = (
    MEAN_ROW_DISTANCES_DISCRETIZED_KM ** 2 +
    MEAN_COLUMN_DISTANCES_DISCRETIZED_KM ** 2
)

# The following constants are used to test weird_crps_kilometres2,
# discretized_weird_crps_kilometres2, coord_avg_crps_kilometres, and
# discretized_coord_avg_crps_kilometres.
FIRST_DIFF_MATRIX_KM = 2.00 * numpy.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=float)

SECOND_DIFF_MATRIX_KM = 2.01 * numpy.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=float)

THIRD_DIFF_MATRIX_KM = 2.02 * numpy.array([
    [0, 1, 1],
    [1, 0, 2],
    [1, 2, 0]
], dtype=float)

FOURTH_DIFF_MATRIX_KM = 2.03 * numpy.array([
    [0, 1, 2],
    [1, 0, 1],
    [2, 1, 0]
], dtype=float)

FIFTH_DIFF_MATRIX_KM = 2.04 * numpy.array([
    [0, 0, 1],
    [0, 0, 1],
    [1, 1, 0]
], dtype=float)

SIXTH_DIFF_MATRIX_KM = 2.05 * numpy.array([
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 0]
], dtype=float)

SEVENTH_DIFF_MATRIX_KM = 2.07 * numpy.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=float)

EIGHTH_DIFF_MATRIX_KM = 2.08 * numpy.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=float)

ROW_PREDICTION_DIFF_MATRIX_KM = numpy.stack((
    FIRST_DIFF_MATRIX_KM, SECOND_DIFF_MATRIX_KM, THIRD_DIFF_MATRIX_KM,
    FOURTH_DIFF_MATRIX_KM, FIFTH_DIFF_MATRIX_KM, SIXTH_DIFF_MATRIX_KM,
    SEVENTH_DIFF_MATRIX_KM, EIGHTH_DIFF_MATRIX_KM
), axis=0)

FIRST_DIFF_MATRIX_KM = 2.00 * numpy.array([
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 0]
], dtype=float)

SECOND_DIFF_MATRIX_KM = 2.01 * numpy.array([
    [0, 1, 2],
    [1, 0, 3],
    [2, 3, 0]
], dtype=float)

THIRD_DIFF_MATRIX_KM = 2.02 * numpy.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=float)

FOURTH_DIFF_MATRIX_KM = 2.03 * numpy.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=float)

FIFTH_DIFF_MATRIX_KM = 2.04 * numpy.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
], dtype=float)

SIXTH_DIFF_MATRIX_KM = 2.05 * numpy.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=float)

SEVENTH_DIFF_MATRIX_KM = 2.07 * numpy.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=float)

EIGHTH_DIFF_MATRIX_KM = 2.08 * numpy.array([
    [0, 1, 1],
    [1, 0, 2],
    [1, 2, 0]
], dtype=float)

COLUMN_PREDICTION_DIFF_MATRIX_KM = numpy.stack((
    FIRST_DIFF_MATRIX_KM, SECOND_DIFF_MATRIX_KM, THIRD_DIFF_MATRIX_KM,
    FOURTH_DIFF_MATRIX_KM, FIFTH_DIFF_MATRIX_KM, SIXTH_DIFF_MATRIX_KM,
    SEVENTH_DIFF_MATRIX_KM, EIGHTH_DIFF_MATRIX_KM
), axis=0)

MEAN_ROW_PREDICTION_DIFFS_KM = numpy.mean(
    ROW_PREDICTION_DIFF_MATRIX_KM, axis=(-2, -1)
)
MEAN_COLUMN_PREDICTION_DIFFS_KM = numpy.mean(
    COLUMN_PREDICTION_DIFF_MATRIX_KM, axis=(-2, -1)
)

# The following constants are used to test coord_avg_crps_kilometres and
# discretized_coord_avg_crps_kilometres.
ROW_CRPS_KILOMETRES = numpy.mean(
    MEAN_ROW_DISTANCES_KM - 0.5 * MEAN_ROW_PREDICTION_DIFFS_KM
)
COLUMN_CRPS_KILOMETRES = numpy.mean(
    MEAN_COLUMN_DISTANCES_KM - 0.5 * MEAN_COLUMN_PREDICTION_DIFFS_KM
)
COORD_AVERAGED_CRPS_KM = (ROW_CRPS_KILOMETRES + COLUMN_CRPS_KILOMETRES) / 2

DISCRETIZED_ROW_CRPS_KILOMETRES = numpy.mean(
    MEAN_ROW_DISTANCES_DISCRETIZED_KM - 0.5 * MEAN_ROW_PREDICTION_DIFFS_KM
)
DISCRETIZED_COLUMN_CRPS_KILOMETRES = numpy.mean(
    MEAN_COLUMN_DISTANCES_DISCRETIZED_KM - 0.5 * MEAN_COLUMN_PREDICTION_DIFFS_KM
)
DISCRETIZED_COORD_AVERAGED_CRPS_KM = (
    (DISCRETIZED_ROW_CRPS_KILOMETRES + DISCRETIZED_COLUMN_CRPS_KILOMETRES) / 2
)

# The following constants are used to test weird_crps_kilometres2 and
# discretized_weird_crps_kilometres2.
MEAN_DISTANCE_ERRORS_KM2 = (
    MEAN_ROW_DISTANCES_KM ** 2 + MEAN_COLUMN_DISTANCES_KM ** 2
)
PREDICTION_DIFF_MATRIX_KM2 = (
    ROW_PREDICTION_DIFF_MATRIX_KM ** 2 + COLUMN_PREDICTION_DIFF_MATRIX_KM ** 2
)
MEAN_PREDICTION_DIFFS_KM2 = numpy.mean(
    PREDICTION_DIFF_MATRIX_KM2, axis=(-2, -1)
)
WEIRD_CRPS_KILOMETRES2 = numpy.mean(
    MEAN_DISTANCE_ERRORS_KM2 - 0.5 * MEAN_PREDICTION_DIFFS_KM2
)

DISCRETIZED_MEAN_DISTANCE_ERRORS_KM2 = (
    MEAN_ROW_DISTANCES_DISCRETIZED_KM ** 2 +
    MEAN_COLUMN_DISTANCES_DISCRETIZED_KM ** 2
)
DISCRETIZED_WEIRD_CRPS_KILOMETRES2 = numpy.mean(
    DISCRETIZED_MEAN_DISTANCE_ERRORS_KM2 - 0.5 * MEAN_PREDICTION_DIFFS_KM2
)


class CustomLossesTests(unittest.TestCase):
    """Each method is a unit test for custom_losses.py."""

    def test_mean_squared_distance_kilometres2(self):
        """Ensures correct output from mean_squared_distance_kilometres2."""

        this_mean_squared_dist_km2 = (
            custom_losses.mean_squared_distance_kilometres2(
                target_tensor=TARGET_TENSOR, prediction_tensor=PREDICTION_TENSOR
            )
        )

        self.assertTrue(numpy.isclose(
            K.eval(this_mean_squared_dist_km2),
            MEAN_OVERALL_SQUARED_DISTANCE_KILOMETRES2,
            atol=TOLERANCE
        ))

    def test_discretized_mean_sq_dist_kilometres2(self):
        """Ensures correct output from discretized_mean_sq_dist_kilometres2."""

        this_mean_squared_dist_km2 = (
            custom_losses.discretized_mean_sq_dist_kilometres2(
                target_tensor=TARGET_TENSOR, prediction_tensor=PREDICTION_TENSOR
            )
        )

        # print(K.eval(this_mean_squared_dist_km2))

        self.assertTrue(numpy.isclose(
            K.eval(this_mean_squared_dist_km2),
            MEAN_OVERALL_SQ_DIST_DISCRETIZED_KILOMETRES2,
            atol=TOLERANCE
        ))

    def test_coord_avg_crps_kilometres(self):
        """Ensures correct output from coord_avg_crps_kilometres."""

        this_crps_kilometres = custom_losses.coord_avg_crps_kilometres(
            target_tensor=TARGET_TENSOR, prediction_tensor=PREDICTION_TENSOR
        )
        self.assertTrue(numpy.isclose(
            K.eval(this_crps_kilometres), COORD_AVERAGED_CRPS_KM, atol=TOLERANCE
        ))

    def test_discretized_coord_avg_crps_kilometres(self):
        """Ensures correct output from discretized_coord_avg_crps_kilometres."""

        this_crps_kilometres = (
            custom_losses.discretized_coord_avg_crps_kilometres(
                target_tensor=TARGET_TENSOR, prediction_tensor=PREDICTION_TENSOR
            )
        )
        self.assertTrue(numpy.isclose(
            K.eval(this_crps_kilometres), DISCRETIZED_COORD_AVERAGED_CRPS_KM,
            atol=TOLERANCE
        ))

    def test_weird_crps_kilometres2(self):
        """Ensures correct output from weird_crps_kilometres2."""

        this_weird_crps_km2 = custom_losses.weird_crps_kilometres2(
            target_tensor=TARGET_TENSOR, prediction_tensor=PREDICTION_TENSOR
        )
        self.assertTrue(numpy.isclose(
            K.eval(this_weird_crps_km2), WEIRD_CRPS_KILOMETRES2, atol=TOLERANCE
        ))

    def test_discretized_weird_crps_kilometres2(self):
        """Ensures correct output from discretized_weird_crps_kilometres2."""

        this_weird_crps_km2 = custom_losses.discretized_weird_crps_kilometres2(
            target_tensor=TARGET_TENSOR, prediction_tensor=PREDICTION_TENSOR
        )
        self.assertTrue(numpy.isclose(
            K.eval(this_weird_crps_km2), DISCRETIZED_WEIRD_CRPS_KILOMETRES2,
            atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
