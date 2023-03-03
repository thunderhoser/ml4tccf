"""Unit tests for scalar_evaluation.py."""

import unittest
import numpy
from ml4tccf.utils import scalar_evaluation

TOLERANCE = 1e-6

# The following constants are used to test _get_angular_diffs.
TARGET_ANGLES_DEG = numpy.array([
    183, 223, 14, 222, 177, 317, 291, 91, 309, 202, 33, 83, 92, 49, 69,
    209, 62, 262, 280, 225
], dtype=float)

PREDICTED_ANGLES_DEG = numpy.array([
    177, 258, 34, 273, 324, 207, 186, 140, 338, 46, 213, 74, 351, 353, 299,
    86, 274, 183, 329, 34
], dtype=float)

ANGULAR_DIFFS_DEG = numpy.array([
    -6, 35, 20, 51, 147, -110, -105, 49, 29, -156, 180, -9, -101, -56, -130,
    -123, -148, -79, 49, 169
], dtype=float)

# The following constants are used to test _get_offset_angles.
X_OFFSETS = numpy.array([0, 1, 0, -1, 0, 1], dtype=float)
Y_OFFSETS = numpy.array([0, 0, 1, 0, -1, 1], dtype=float)
OFFSET_ANGLES_DEG = numpy.array([numpy.nan, 0, 90, 180, 270, 45])

# The following constants are used to test _get_mean_distance and
# _get_mean_squared_distance.
TARGET_OFFSET_MATRIX = numpy.array([
    [0, 0],
    [1, -1],
    [2, -2],
    [-3, 6]
], dtype=float)

PREDICTED_OFFSET_MATRIX = numpy.array([
    [0, 2],
    [2, 2],
    [3, -1],
    [-5, 5]
], dtype=float)

MEAN_SQUARED_DISTANCE = numpy.mean([4, 10, 2, 5], dtype=float)
MEAN_DISTANCE = numpy.mean(
    [2, numpy.sqrt(10), numpy.sqrt(2), numpy.sqrt(5)], dtype=float
)


class ScalarEvaluationTests(unittest.TestCase):
    """Each method is a unit test for scalar_evaluation.py."""

    def test_get_angular_diffs(self):
        """Ensures correct output from _get_angular_diffs."""

        these_angular_diffs_deg = scalar_evaluation._get_angular_diffs(
            target_angles_deg=TARGET_ANGLES_DEG,
            predicted_angles_deg=PREDICTED_ANGLES_DEG
        )

        self.assertTrue(numpy.allclose(
            these_angular_diffs_deg, ANGULAR_DIFFS_DEG, atol=TOLERANCE
        ))

    def test_get_offset_angles(self):
        """Ensures correct output from _get_offset_angles."""

        these_offset_angles_deg = scalar_evaluation._get_offset_angles(
            x_offsets=X_OFFSETS, y_offsets=Y_OFFSETS
        )

        self.assertTrue(numpy.allclose(
            these_offset_angles_deg, OFFSET_ANGLES_DEG, atol=TOLERANCE,
            equal_nan=True
        ))

    def test_get_mean_distance(self):
        """Ensures correct output from _get_mean_distance."""

        this_mean_distance = scalar_evaluation._get_mean_distance(
            target_offset_matrix=TARGET_OFFSET_MATRIX,
            predicted_offset_matrix=PREDICTED_OFFSET_MATRIX
        )
        self.assertTrue(numpy.isclose(this_mean_distance, MEAN_DISTANCE))

    def test_get_mean_squared_distance(self):
        """Ensures correct output from _get_mean_squared_distance."""

        this_mean_squared_distance = (
            scalar_evaluation._get_mean_squared_distance(
                target_offset_matrix=TARGET_OFFSET_MATRIX,
                predicted_offset_matrix=PREDICTED_OFFSET_MATRIX
            )
        )
        self.assertTrue(numpy.isclose(
            this_mean_squared_distance, MEAN_SQUARED_DISTANCE
        ))


if __name__ == '__main__':
    unittest.main()
