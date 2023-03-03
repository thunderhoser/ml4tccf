"""Unit tests for misc_utils.py."""

import unittest
import numpy
from ml4tccf.utils import misc_utils

TOLERANCE = 1e-6

# The following constants are used to test target_matrix_to_centroid and
# prediction_matrix_to_centroid.
FIRST_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=float)

FIRST_MATRIX = FIRST_MATRIX / numpy.sum(FIRST_MATRIX)
FIRST_ROW_OFFSET_FOR_TARGETS = -3
FIRST_COLUMN_OFFSET_FOR_TARGETS = -4
FIRST_ROW_OFFSET_FOR_PREDICTIONS = -3.
FIRST_COLUMN_OFFSET_FOR_PREDICTIONS = -4.

SECOND_MATRIX = numpy.array([
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=float)

SECOND_MATRIX = SECOND_MATRIX / numpy.sum(SECOND_MATRIX)
SECOND_ROW_OFFSET_FOR_TARGETS = -4
SECOND_COLUMN_OFFSET_FOR_TARGETS = -5
SECOND_ROW_OFFSET_FOR_PREDICTIONS = -4.
SECOND_COLUMN_OFFSET_FOR_PREDICTIONS = -5.

THIRD_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=float)

THIRD_MATRIX = THIRD_MATRIX / numpy.sum(THIRD_MATRIX)
THIRD_ROW_OFFSET_FOR_TARGETS = 0
THIRD_COLUMN_OFFSET_FOR_TARGETS = 0
THIRD_ROW_OFFSET_FOR_PREDICTIONS = 0.
THIRD_COLUMN_OFFSET_FOR_PREDICTIONS = 0.

FOURTH_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=float)

# FOURTH_ROW_OFFSET_FOR_TARGETS = None
# FOURTH_COLUMN_OFFSET_FOR_TARGETS = None
# FOURTH_ROW_OFFSET_FOR_PREDICTIONS = None
# FOURTH_COLUMN_OFFSET_FOR_PREDICTIONS = None

FIFTH_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=float)

FIFTH_MATRIX = FIFTH_MATRIX / numpy.sum(FIFTH_MATRIX)
# FIFTH_ROW_OFFSET_FOR_TARGETS = None
# FIFTH_COLUMN_OFFSET_FOR_TARGETS = None
FIFTH_ROW_OFFSET_FOR_PREDICTIONS = -0.3
FIFTH_COLUMN_OFFSET_FOR_PREDICTIONS = -0.1

SIXTH_MATRIX = numpy.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
], dtype=float)

SIXTH_MATRIX = SIXTH_MATRIX / numpy.sum(SIXTH_MATRIX)
# SIXTH_ROW_OFFSET_FOR_TARGETS = None
# SIXTH_COLUMN_OFFSET_FOR_TARGETS = None
SIXTH_ROW_OFFSET_FOR_PREDICTIONS = 0.
SIXTH_COLUMN_OFFSET_FOR_PREDICTIONS = 0.


class MiscUtilsTests(unittest.TestCase):
    """Each method is a unit test for misc_utils.py."""

    def test_target_matrix_to_centroid_first(self):
        """Ensures correctness of target_matrix_to_centroid w/ first input."""

        this_row_offset, this_column_offset = (
            misc_utils.target_matrix_to_centroid(FIRST_MATRIX)
        )
        self.assertTrue(this_row_offset == FIRST_ROW_OFFSET_FOR_TARGETS)
        self.assertTrue(this_column_offset == FIRST_COLUMN_OFFSET_FOR_TARGETS)

    def test_target_matrix_to_centroid_second(self):
        """Ensures correctness of target_matrix_to_centroid w/ second input."""

        this_row_offset, this_column_offset = (
            misc_utils.target_matrix_to_centroid(SECOND_MATRIX)
        )
        self.assertTrue(this_row_offset == SECOND_ROW_OFFSET_FOR_TARGETS)
        self.assertTrue(this_column_offset == SECOND_COLUMN_OFFSET_FOR_TARGETS)

    def test_target_matrix_to_centroid_third(self):
        """Ensures correctness of target_matrix_to_centroid w/ third input."""

        this_row_offset, this_column_offset = (
            misc_utils.target_matrix_to_centroid(THIRD_MATRIX)
        )
        self.assertTrue(this_row_offset == THIRD_ROW_OFFSET_FOR_TARGETS)
        self.assertTrue(this_column_offset == THIRD_COLUMN_OFFSET_FOR_TARGETS)

    def test_target_matrix_to_centroid_fourth(self):
        """Ensures correctness of target_matrix_to_centroid w/ fourth input."""

        with self.assertRaises(AssertionError):
            misc_utils.target_matrix_to_centroid(FOURTH_MATRIX)

    def test_target_matrix_to_centroid_fifth(self):
        """Ensures correctness of target_matrix_to_centroid w/ fifth input."""

        with self.assertRaises(AssertionError):
            misc_utils.target_matrix_to_centroid(FIFTH_MATRIX)

    def test_target_matrix_to_centroid_sixth(self):
        """Ensures correctness of target_matrix_to_centroid w/ sixth input."""

        with self.assertRaises(AssertionError):
            misc_utils.target_matrix_to_centroid(SIXTH_MATRIX)

    def test_prediction_matrix_to_centroid_first(self):
        """Ensures crrctnss of prediction_matrix_to_centroid w/ first input."""

        this_row_offset, this_column_offset = (
            misc_utils.prediction_matrix_to_centroid(FIRST_MATRIX)
        )

        self.assertTrue(numpy.isclose(
            this_row_offset, FIRST_ROW_OFFSET_FOR_PREDICTIONS
        ))
        self.assertTrue(numpy.isclose(
            this_column_offset, FIRST_COLUMN_OFFSET_FOR_PREDICTIONS
        ))

    def test_prediction_matrix_to_centroid_second(self):
        """Ensures crrctnss of prediction_matrix_to_centroid w/ second input."""

        this_row_offset, this_column_offset = (
            misc_utils.prediction_matrix_to_centroid(SECOND_MATRIX)
        )

        self.assertTrue(numpy.isclose(
            this_row_offset, SECOND_ROW_OFFSET_FOR_PREDICTIONS
        ))
        self.assertTrue(numpy.isclose(
            this_column_offset, SECOND_COLUMN_OFFSET_FOR_PREDICTIONS
        ))

    def test_prediction_matrix_to_centroid_third(self):
        """Ensures crrctnss of prediction_matrix_to_centroid w/ third input."""

        this_row_offset, this_column_offset = (
            misc_utils.prediction_matrix_to_centroid(THIRD_MATRIX)
        )

        self.assertTrue(numpy.isclose(
            this_row_offset, THIRD_ROW_OFFSET_FOR_PREDICTIONS
        ))
        self.assertTrue(numpy.isclose(
            this_column_offset, THIRD_COLUMN_OFFSET_FOR_PREDICTIONS
        ))

    def test_prediction_matrix_to_centroid_fourth(self):
        """Ensures crrctnss of prediction_matrix_to_centroid w/ fourth input."""

        with self.assertRaises(AssertionError):
            misc_utils.prediction_matrix_to_centroid(FOURTH_MATRIX)

    def test_prediction_matrix_to_centroid_fifth(self):
        """Ensures crrctnss of prediction_matrix_to_centroid w/ fifth input."""

        this_row_offset, this_column_offset = (
            misc_utils.prediction_matrix_to_centroid(FIFTH_MATRIX)
        )

        self.assertTrue(numpy.isclose(
            this_row_offset, FIFTH_ROW_OFFSET_FOR_PREDICTIONS
        ))
        self.assertTrue(numpy.isclose(
            this_column_offset, FIFTH_COLUMN_OFFSET_FOR_PREDICTIONS
        ))

    def test_prediction_matrix_to_centroid_sixth(self):
        """Ensures crrctnss of prediction_matrix_to_centroid w/ sixth input."""

        this_row_offset, this_column_offset = (
            misc_utils.prediction_matrix_to_centroid(SIXTH_MATRIX)
        )

        self.assertTrue(numpy.isclose(
            this_row_offset, SIXTH_ROW_OFFSET_FOR_PREDICTIONS
        ))
        self.assertTrue(numpy.isclose(
            this_column_offset, SIXTH_COLUMN_OFFSET_FOR_PREDICTIONS
        ))


if __name__ == '__main__':
    unittest.main()
