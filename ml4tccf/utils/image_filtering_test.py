"""Unit tests for image_filtering.py."""

import unittest
import numpy
from ml4tccf.utils import image_filtering

TOLERANCE = 1e-6

# The following constants are used to test
# _get_structure_matrix_for_target_discretizn and undo_target_discretization.
GRID_SPACING_KM = 2.
CENTER_LATITUDES_FOR_STRUCT_DEG_N = numpy.array(
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float
)

FIRST_STRUCTURE_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0]
], dtype=bool)

SECOND_STRUCTURE_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
], dtype=bool)

THIRD_STRUCTURE_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
], dtype=bool)

STRUCTURE_MATRICES = [
    FIRST_STRUCTURE_MATRIX, FIRST_STRUCTURE_MATRIX, FIRST_STRUCTURE_MATRIX,
    SECOND_STRUCTURE_MATRIX, SECOND_STRUCTURE_MATRIX, SECOND_STRUCTURE_MATRIX,
    THIRD_STRUCTURE_MATRIX, THIRD_STRUCTURE_MATRIX, THIRD_STRUCTURE_MATRIX,
    THIRD_STRUCTURE_MATRIX
]

INTEGER_TARGET_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=int)

FIRST_DEDISCRETIZED_MATRIX = numpy.array([
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=float)

SECOND_DEDISCRETIZED_MATRIX = numpy.array([
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=float)

THIRD_DEDISCRETIZED_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=float)

# FIRST_DEDISCRETIZED_MATRIX = (
#     FIRST_DEDISCRETIZED_MATRIX / numpy.sum(FIRST_DEDISCRETIZED_MATRIX)
# )
# SECOND_DEDISCRETIZED_MATRIX = (
#     SECOND_DEDISCRETIZED_MATRIX / numpy.sum(SECOND_DEDISCRETIZED_MATRIX)
# )
# THIRD_DEDISCRETIZED_MATRIX = (
#     THIRD_DEDISCRETIZED_MATRIX / numpy.sum(THIRD_DEDISCRETIZED_MATRIX)
# )

DEDISCRETIZED_TARGET_MATRICES = [
    FIRST_DEDISCRETIZED_MATRIX, FIRST_DEDISCRETIZED_MATRIX,
    FIRST_DEDISCRETIZED_MATRIX,
    SECOND_DEDISCRETIZED_MATRIX, SECOND_DEDISCRETIZED_MATRIX,
    SECOND_DEDISCRETIZED_MATRIX,
    THIRD_DEDISCRETIZED_MATRIX, THIRD_DEDISCRETIZED_MATRIX,
    THIRD_DEDISCRETIZED_MATRIX, THIRD_DEDISCRETIZED_MATRIX
]

# The following constants are used to test create_mean_conv_filter.
FIRST_HALF_NUM_ROWS = 0
FIRST_HALF_NUM_COLUMNS = 1
FIRST_NUM_CHANNELS = 3

FIRST_WEIGHT_MATRIX = numpy.full((1, 3), 1. / 3)
FIRST_WEIGHT_MATRIX = numpy.expand_dims(FIRST_WEIGHT_MATRIX, axis=-1)
FIRST_WEIGHT_MATRIX = numpy.expand_dims(FIRST_WEIGHT_MATRIX, axis=-1)
FIRST_WEIGHT_MATRIX = numpy.repeat(FIRST_WEIGHT_MATRIX, repeats=3, axis=-2)
FIRST_WEIGHT_MATRIX = numpy.repeat(FIRST_WEIGHT_MATRIX, repeats=3, axis=-1)

SECOND_HALF_NUM_ROWS = 2
SECOND_HALF_NUM_COLUMNS = 2
SECOND_NUM_CHANNELS = 1

SECOND_WEIGHT_MATRIX = numpy.full((5, 5), 1. / 25)
SECOND_WEIGHT_MATRIX = numpy.expand_dims(SECOND_WEIGHT_MATRIX, axis=-1)
SECOND_WEIGHT_MATRIX = numpy.expand_dims(SECOND_WEIGHT_MATRIX, axis=-1)


class ImageFilteringTests(unittest.TestCase):
    """Each method is a unit test for image_filtering.py."""

    def test_get_structure_matrix_for_target_discretizn(self):
        """Ensures crrctness of _get_structure_matrix_for_target_discretizn."""

        for i in range(len(CENTER_LATITUDES_FOR_STRUCT_DEG_N)):
            this_structure_matrix = (
                image_filtering._get_structure_matrix_for_target_discretizn(
                    grid_spacing_km=GRID_SPACING_KM,
                    cyclone_center_latitude_deg_n=
                    CENTER_LATITUDES_FOR_STRUCT_DEG_N[i]
                )
            )

            self.assertTrue(numpy.array_equal(
                this_structure_matrix, STRUCTURE_MATRICES[i]
            ))

    def test_undo_target_discretization(self):
        """Ensures correct output from undo_target_discretization."""

        for i in range(len(CENTER_LATITUDES_FOR_STRUCT_DEG_N)):
            this_target_matrix = image_filtering.undo_target_discretization(
                integer_target_matrix=INTEGER_TARGET_MATRIX,
                grid_spacing_km=GRID_SPACING_KM,
                cyclone_center_latitude_deg_n=
                CENTER_LATITUDES_FOR_STRUCT_DEG_N[i]
            )

            self.assertTrue(numpy.allclose(
                this_target_matrix, DEDISCRETIZED_TARGET_MATRICES[i],
                atol=TOLERANCE
            ))

    def test_create_mean_conv_filter_first(self):
        """Ensures correct output from _create_mean_conv_filter.

        In this case, using first set of inputs.
        """

        this_weight_matrix = image_filtering.create_mean_conv_filter(
            half_num_rows=FIRST_HALF_NUM_ROWS,
            half_num_columns=FIRST_HALF_NUM_COLUMNS,
            num_channels=FIRST_NUM_CHANNELS
        )

        self.assertTrue(numpy.allclose(
            this_weight_matrix, FIRST_WEIGHT_MATRIX, atol=TOLERANCE
        ))

    def test_create_mean_conv_filter_second(self):
        """Ensures correct output from _create_mean_conv_filter.

        In this case, using second set of inputs.
        """

        this_weight_matrix = image_filtering.create_mean_conv_filter(
            half_num_rows=SECOND_HALF_NUM_ROWS,
            half_num_columns=SECOND_HALF_NUM_COLUMNS,
            num_channels=SECOND_NUM_CHANNELS
        )

        self.assertTrue(numpy.allclose(
            this_weight_matrix, SECOND_WEIGHT_MATRIX, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
