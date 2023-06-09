"""Unit tests for data_augmentation.py."""

import unittest
import numpy
from ml4tccf.machine_learning import data_augmentation

TOLERANCE = 1e-6
MINUTES_TO_SECONDS = 60
DAYS_TO_SECONDS = 86400

# The following constants are used to test _translate_images.
THIS_FIRST_MATRIX = numpy.array([
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11]
], dtype=float)

THIS_SECOND_MATRIX = numpy.array([
    [2, 5, 8, 11],
    [3, 6, 9, 12],
    [4, 7, 10, 13]
], dtype=float)

DATA_MATRIX_3D_NO_TRANSLATION = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0
)
DATA_MATRIX_4D_NO_TRANSLATION = numpy.stack(
    (DATA_MATRIX_3D_NO_TRANSLATION,) * 6, axis=-1
)
DATA_MATRIX_5D_NO_TRANSLATION = numpy.stack(
    (DATA_MATRIX_4D_NO_TRANSLATION,) * 5, axis=-2
)

FIRST_X_OFFSET_PIXELS = 2
FIRST_Y_OFFSET_PIXELS = 1
FIRST_SENTINEL_VALUE = -999.

THIS_FIRST_MATRIX = numpy.array([
    [-999, -999, -999, -999],
    [-999, -999, 0, 1],
    [-999, -999, 4, 5]
], dtype=float)

THIS_SECOND_MATRIX = numpy.array([
    [-999, -999, -999, -999],
    [-999, -999, 2, 5],
    [-999, -999, 3, 6]
], dtype=float)

DATA_MATRIX_3D_FIRST_TRANS = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0
)
DATA_MATRIX_4D_FIRST_TRANS = numpy.stack(
    (DATA_MATRIX_3D_FIRST_TRANS,) * 6, axis=-1
)
DATA_MATRIX_5D_FIRST_TRANS = numpy.stack(
    (DATA_MATRIX_4D_FIRST_TRANS,) * 5, axis=-2
)

SECOND_X_OFFSET_PIXELS = -1
SECOND_Y_OFFSET_PIXELS = -1
SECOND_SENTINEL_VALUE = 5000.

THIS_FIRST_MATRIX = numpy.array([
    [5, 6, 7, 5000],
    [9, 10, 11, 5000],
    [5000, 5000, 5000, 5000]
], dtype=float)

THIS_SECOND_MATRIX = numpy.array([
    [6, 9, 12, 5000],
    [7, 10, 13, 5000],
    [5000, 5000, 5000, 5000]
], dtype=float)

DATA_MATRIX_3D_SECOND_TRANS = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0
)
DATA_MATRIX_4D_SECOND_TRANS = numpy.stack(
    (DATA_MATRIX_3D_SECOND_TRANS,) * 6, axis=-1
)
DATA_MATRIX_5D_SECOND_TRANS = numpy.stack(
    (DATA_MATRIX_4D_SECOND_TRANS,) * 5, axis=-2
)


class DataAugmentationTests(unittest.TestCase):
    """Each method is a unit test for data_augmentation.py."""

    def test_translate_images_first_3d(self):
        """Ensures correct output from _translate_images.

        With first set of options for 3-D data.
        """

        this_image_matrix = data_augmentation._translate_images(
            image_matrix=DATA_MATRIX_3D_NO_TRANSLATION,
            row_translation_px=FIRST_Y_OFFSET_PIXELS,
            column_translation_px=FIRST_X_OFFSET_PIXELS,
            padding_value=FIRST_SENTINEL_VALUE
        )

        self.assertTrue(numpy.allclose(
            this_image_matrix, DATA_MATRIX_3D_FIRST_TRANS, atol=TOLERANCE
        ))

    def test_translate_images_first_4d(self):
        """Ensures correct output from _translate_images.

        With first set of options for 4-D data.
        """

        this_image_matrix = data_augmentation._translate_images(
            image_matrix=DATA_MATRIX_4D_NO_TRANSLATION,
            row_translation_px=FIRST_Y_OFFSET_PIXELS,
            column_translation_px=FIRST_X_OFFSET_PIXELS,
            padding_value=FIRST_SENTINEL_VALUE
        )

        self.assertTrue(numpy.allclose(
            this_image_matrix, DATA_MATRIX_4D_FIRST_TRANS, atol=TOLERANCE
        ))

    def test_translate_images_first_5d(self):
        """Ensures correct output from _translate_images.

        With first set of options for 5-D data.
        """

        this_image_matrix = data_augmentation._translate_images(
            image_matrix=DATA_MATRIX_5D_NO_TRANSLATION,
            row_translation_px=FIRST_Y_OFFSET_PIXELS,
            column_translation_px=FIRST_X_OFFSET_PIXELS,
            padding_value=FIRST_SENTINEL_VALUE
        )

        self.assertTrue(numpy.allclose(
            this_image_matrix, DATA_MATRIX_5D_FIRST_TRANS, atol=TOLERANCE
        ))

    def test_translate_images_second_3d(self):
        """Ensures correct output from _translate_images.

        With second set of options for 3-D data.
        """

        this_image_matrix = data_augmentation._translate_images(
            image_matrix=DATA_MATRIX_3D_NO_TRANSLATION,
            row_translation_px=SECOND_Y_OFFSET_PIXELS,
            column_translation_px=SECOND_X_OFFSET_PIXELS,
            padding_value=SECOND_SENTINEL_VALUE
        )

        self.assertTrue(numpy.allclose(
            this_image_matrix, DATA_MATRIX_3D_SECOND_TRANS, atol=TOLERANCE
        ))

    def test_translate_images_second_4d(self):
        """Ensures correct output from _translate_images.

        With second set of options for 4-D data.
        """

        this_image_matrix = data_augmentation._translate_images(
            image_matrix=DATA_MATRIX_4D_NO_TRANSLATION,
            row_translation_px=SECOND_Y_OFFSET_PIXELS,
            column_translation_px=SECOND_X_OFFSET_PIXELS,
            padding_value=SECOND_SENTINEL_VALUE
        )

        self.assertTrue(numpy.allclose(
            this_image_matrix, DATA_MATRIX_4D_SECOND_TRANS, atol=TOLERANCE
        ))

    def test_translate_images_second_5d(self):
        """Ensures correct output from _translate_images.

        With second set of options for 5-D data.
        """

        this_image_matrix = data_augmentation._translate_images(
            image_matrix=DATA_MATRIX_5D_NO_TRANSLATION,
            row_translation_px=SECOND_Y_OFFSET_PIXELS,
            column_translation_px=SECOND_X_OFFSET_PIXELS,
            padding_value=SECOND_SENTINEL_VALUE
        )

        self.assertTrue(numpy.allclose(
            this_image_matrix, DATA_MATRIX_5D_SECOND_TRANS, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
