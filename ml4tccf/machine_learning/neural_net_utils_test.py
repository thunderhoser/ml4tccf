"""Unit tests for neural_net_utils.py."""

import unittest
import numpy
from ml4tccf.machine_learning import neural_net_utils as nn_utils

TOLERANCE = 1e-6

# The following constants are used to test make_targets_for_semantic_seg.
ROW_TRANSLATIONS_PX = numpy.array([-3, -2, -1, 1, 2, 3], dtype=int)
COLUMN_TRANSLATIONS_PX = numpy.array([-3, -2, -1, 1, 2, 3], dtype=int)
GRID_SPACINGS_KM = numpy.array([2, 2, 2, 2, 2, 2], dtype=float)
CYCLONE_CENTER_LATITUDES_DEG_N = numpy.array([0, 0, 0, 0, 0, 0], dtype=float)
GAUSSIAN_SMOOTHER_STDEV_KM = 1e-6
NUM_GRID_ROWS = 10
NUM_GRID_COLUMNS = 12

FIRST_TARGET_MATRIX = numpy.array([
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=float)

SECOND_TARGET_MATRIX = numpy.array([
    [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=float)

THIRD_TARGET_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=float)

FOURTH_TARGET_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=float)

FIFTH_TARGET_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
], dtype=float)

SIXTH_TARGET_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
], dtype=float)

# FIRST_TARGET_MATRIX = FIRST_TARGET_MATRIX / numpy.sum(FIRST_TARGET_MATRIX)
# SECOND_TARGET_MATRIX = SECOND_TARGET_MATRIX / numpy.sum(SECOND_TARGET_MATRIX)
# THIRD_TARGET_MATRIX = THIRD_TARGET_MATRIX / numpy.sum(THIRD_TARGET_MATRIX)
# FOURTH_TARGET_MATRIX = FOURTH_TARGET_MATRIX / numpy.sum(FOURTH_TARGET_MATRIX)
# FIFTH_TARGET_MATRIX = FIFTH_TARGET_MATRIX / numpy.sum(FIFTH_TARGET_MATRIX)
# SIXTH_TARGET_MATRIX = SIXTH_TARGET_MATRIX / numpy.sum(SIXTH_TARGET_MATRIX)

TARGET_MATRIX = numpy.stack((
    FIRST_TARGET_MATRIX, SECOND_TARGET_MATRIX, THIRD_TARGET_MATRIX,
    FOURTH_TARGET_MATRIX, FIFTH_TARGET_MATRIX, SIXTH_TARGET_MATRIX
), axis=0)

TARGET_MATRIX = numpy.expand_dims(TARGET_MATRIX, axis=-1)

# The following constants are used to test combine_lag_times_and_wavelengths and
# separate_lag_times_and_wavelengths.
FIRST_SATELLITE_DATA_MATRIX_5D = numpy.random.normal(
    loc=0., scale=1., size=(3, 50, 50, 4, 4)
)
SECOND_SATELLITE_DATA_MATRIX_5D = numpy.random.normal(
    loc=0., scale=1., size=(3, 50, 50, 4, 1)
)
THIRD_SATELLITE_DATA_MATRIX_5D = numpy.random.normal(
    loc=0., scale=1., size=(3, 50, 50, 1, 4)
)


class NeuralNetUtilsTests(unittest.TestCase):
    """Each method is a unit test for neural_net_utils.py."""

    def test_make_targets_for_semantic_seg(self):
        """Ensures correct output from make_targets_for_semantic_seg."""

        this_target_matrix = nn_utils.make_targets_for_semantic_seg(
            row_translations_px=ROW_TRANSLATIONS_PX,
            column_translations_px=COLUMN_TRANSLATIONS_PX,
            grid_spacings_km=GRID_SPACINGS_KM,
            cyclone_center_latitudes_deg_n=CYCLONE_CENTER_LATITUDES_DEG_N,
            gaussian_smoother_stdev_km=GAUSSIAN_SMOOTHER_STDEV_KM,
            num_grid_rows=NUM_GRID_ROWS, num_grid_columns=NUM_GRID_COLUMNS
        )

        self.assertTrue(numpy.allclose(
            this_target_matrix, TARGET_MATRIX, atol=TOLERANCE
        ))

    def test_combine_lag_times_and_wavelengths_first(self):
        """Ensures correct output from combine_lag_times_and_wavelengths.

        In this case, using first set of inputs.  This method also tests
        separate_lag_times_and_wavelengths.
        """

        orig_data_matrix_4d = nn_utils.combine_lag_times_and_wavelengths(
            FIRST_SATELLITE_DATA_MATRIX_5D
        )
        new_data_matrix_5d = nn_utils.separate_lag_times_and_wavelengths(
            satellite_data_matrix=orig_data_matrix_4d,
            num_lag_times=FIRST_SATELLITE_DATA_MATRIX_5D.shape[-2]
        )
        new_data_matrix_4d = nn_utils.combine_lag_times_and_wavelengths(
            new_data_matrix_5d
        )

        self.assertTrue(numpy.allclose(
            new_data_matrix_5d, FIRST_SATELLITE_DATA_MATRIX_5D, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            new_data_matrix_4d, orig_data_matrix_4d, atol=TOLERANCE
        ))

    def test_combine_lag_times_and_wavelengths_second(self):
        """Ensures correct output from combine_lag_times_and_wavelengths.

        In this case, using second set of inputs.  This method also tests
        separate_lag_times_and_wavelengths.
        """

        orig_data_matrix_4d = nn_utils.combine_lag_times_and_wavelengths(
            SECOND_SATELLITE_DATA_MATRIX_5D
        )
        new_data_matrix_5d = nn_utils.separate_lag_times_and_wavelengths(
            satellite_data_matrix=orig_data_matrix_4d,
            num_lag_times=SECOND_SATELLITE_DATA_MATRIX_5D.shape[-2]
        )
        new_data_matrix_4d = nn_utils.combine_lag_times_and_wavelengths(
            new_data_matrix_5d
        )

        self.assertTrue(numpy.allclose(
            new_data_matrix_5d, SECOND_SATELLITE_DATA_MATRIX_5D, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            new_data_matrix_4d, orig_data_matrix_4d, atol=TOLERANCE
        ))

    def test_combine_lag_times_and_wavelengths_third(self):
        """Ensures correct output from combine_lag_times_and_wavelengths.

        In this case, using third set of inputs.  This method also tests
        separate_lag_times_and_wavelengths.
        """

        orig_data_matrix_4d = nn_utils.combine_lag_times_and_wavelengths(
            THIRD_SATELLITE_DATA_MATRIX_5D
        )
        new_data_matrix_5d = nn_utils.separate_lag_times_and_wavelengths(
            satellite_data_matrix=orig_data_matrix_4d,
            num_lag_times=THIRD_SATELLITE_DATA_MATRIX_5D.shape[-2]
        )
        new_data_matrix_4d = nn_utils.combine_lag_times_and_wavelengths(
            new_data_matrix_5d
        )

        self.assertTrue(numpy.allclose(
            new_data_matrix_5d, THIRD_SATELLITE_DATA_MATRIX_5D, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            new_data_matrix_4d, orig_data_matrix_4d, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
