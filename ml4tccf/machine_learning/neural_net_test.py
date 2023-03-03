"""Unit tests for neural_net.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import time_conversion
from ml4tccf.io import satellite_io
from ml4tccf.machine_learning import neural_net

TOLERANCE = 1e-6
MINUTES_TO_SECONDS = 60

# The following constants are used to test _date_in_time_period.
ZEROTH_DATE_STRING = '2023-02-03'
ZEROTH_START_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2023-02-03-030000', '%Y-%m-%d-%H%M%S'
)
ZEROTH_END_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2023-02-03-030100', '%Y-%m-%d-%H%M%S'
)
ZEROTH_DATE_IN_TIME_PERIOD_FLAG = True

FIRST_DATE_STRING = '2023-02-03'
FIRST_START_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2023-02-02-235959', '%Y-%m-%d-%H%M%S'
)
FIRST_END_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2023-02-03-000000', '%Y-%m-%d-%H%M%S'
)
FIRST_DATE_IN_TIME_PERIOD_FLAG = True

SECOND_DATE_STRING = '2023-02-03'
SECOND_START_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2023-02-02-235959', '%Y-%m-%d-%H%M%S'
)
SECOND_END_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2023-02-02-235959', '%Y-%m-%d-%H%M%S'
)
SECOND_DATE_IN_TIME_PERIOD_FLAG = False

THIRD_DATE_STRING = '2023-02-03'
THIRD_START_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2023-02-03-235959', '%Y-%m-%d-%H%M%S'
)
THIRD_END_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2023-02-04-000000', '%Y-%m-%d-%H%M%S'
)
THIRD_DATE_IN_TIME_PERIOD_FLAG = True

FOURTH_DATE_STRING = '2023-02-03'
FOURTH_START_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2023-02-04-000000', '%Y-%m-%d-%H%M%S'
)
FOURTH_END_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2023-02-04-000000', '%Y-%m-%d-%H%M%S'
)
FOURTH_DATE_IN_TIME_PERIOD_FLAG = False

FIFTH_DATE_STRING = '2023-02-03'
FIFTH_START_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2023-02-02-000000', '%Y-%m-%d-%H%M%S'
)
FIFTH_END_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2023-02-06-000000', '%Y-%m-%d-%H%M%S'
)
FIFTH_DATE_IN_TIME_PERIOD_FLAG = True

SIXTH_DATE_STRING = '2023-02-03'
SIXTH_START_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2023-02-04-000000', '%Y-%m-%d-%H%M%S'
)
SIXTH_END_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2023-02-06-000000', '%Y-%m-%d-%H%M%S'
)
SIXTH_DATE_IN_TIME_PERIOD_FLAG = False

# The following constants are used to test _decide_files_to_read_one_cyclone.
# _decide_files_to_read_one_cyclone(
#         satellite_file_names, target_times_unix_sec,
#         lag_times_minutes, lag_time_tolerance_sec, max_interp_gap_sec)

VALID_DATE_STRINGS = [
    '2005-08-30', '2005-08-26', '2005-08-28', '2005-08-27', '2006-01-01'
]
SATELLITE_FILE_NAMES = [
    satellite_io.find_file(
        directory_name='foo', cyclone_id_string='2005AL12', valid_date_string=d,
        raise_error_if_missing=False
    ) for d in VALID_DATE_STRINGS
]

TARGET_TIME_STRINGS = [
    '2005-08-26-00', '2005-08-26-06', '2005-08-28-00', '2005-08-28-12',
    '2005-08-31-00'
]
TARGET_TIMES_UNIX_SEC = [
    time_conversion.string_to_unix_sec(s, '%Y-%m-%d-%H')
    for s in TARGET_TIME_STRINGS
]

FIRST_LAG_TIMES_MINUTES = numpy.array([120, 0, 60], dtype=int)
FIRST_LAG_TIMES_SEC = MINUTES_TO_SECONDS * FIRST_LAG_TIMES_MINUTES
FIRST_LAG_TIME_TOLERANCE_SEC = 1800
FIRST_MAX_INTERP_GAP_SEC = 1800

THESE_START_TIMES_UNIX_SEC = numpy.array([
    TARGET_TIMES_UNIX_SEC[0] - numpy.max(FIRST_LAG_TIMES_SEC) -
    FIRST_MAX_INTERP_GAP_SEC,
    TARGET_TIMES_UNIX_SEC[1] - numpy.max(FIRST_LAG_TIMES_SEC) -
    FIRST_MAX_INTERP_GAP_SEC
], dtype=int)

THESE_END_TIMES_UNIX_SEC = numpy.array([
    TARGET_TIMES_UNIX_SEC[0] - numpy.min(FIRST_LAG_TIMES_SEC),
    TARGET_TIMES_UNIX_SEC[1] - numpy.min(FIRST_LAG_TIMES_SEC)
], dtype=int)

THESE_END_TIMES_UNIX_SEC[
    THESE_END_TIMES_UNIX_SEC != TARGET_TIMES_UNIX_SEC[:2]
] += FIRST_MAX_INTERP_GAP_SEC

FIRST_DESIRED_FILE_TO_TIMES_DICT = {
    SATELLITE_FILE_NAMES[1]:
        [THESE_START_TIMES_UNIX_SEC, THESE_END_TIMES_UNIX_SEC]
}

THESE_START_TIMES_UNIX_SEC = numpy.array([
    TARGET_TIMES_UNIX_SEC[2] - numpy.max(FIRST_LAG_TIMES_SEC) -
    FIRST_MAX_INTERP_GAP_SEC
], dtype=int)

THESE_END_TIMES_UNIX_SEC = numpy.array([
    TARGET_TIMES_UNIX_SEC[2] - numpy.min(FIRST_LAG_TIMES_SEC)
], dtype=int)

THESE_END_TIMES_UNIX_SEC[
    THESE_END_TIMES_UNIX_SEC != TARGET_TIMES_UNIX_SEC[2]
] += FIRST_MAX_INTERP_GAP_SEC

FIRST_DESIRED_FILE_TO_TIMES_DICT.update({
    SATELLITE_FILE_NAMES[3]:
        [THESE_START_TIMES_UNIX_SEC, THESE_END_TIMES_UNIX_SEC]
})

THESE_START_TIMES_UNIX_SEC = numpy.array([
    TARGET_TIMES_UNIX_SEC[2] - numpy.max(FIRST_LAG_TIMES_SEC) -
    FIRST_MAX_INTERP_GAP_SEC,
    TARGET_TIMES_UNIX_SEC[3] - numpy.max(FIRST_LAG_TIMES_SEC) -
    FIRST_MAX_INTERP_GAP_SEC
], dtype=int)

THESE_END_TIMES_UNIX_SEC = numpy.array([
    TARGET_TIMES_UNIX_SEC[2] - numpy.min(FIRST_LAG_TIMES_SEC),
    TARGET_TIMES_UNIX_SEC[3] - numpy.min(FIRST_LAG_TIMES_SEC)
], dtype=int)

THESE_END_TIMES_UNIX_SEC[
    THESE_END_TIMES_UNIX_SEC != TARGET_TIMES_UNIX_SEC[2:4]
] += FIRST_MAX_INTERP_GAP_SEC

FIRST_DESIRED_FILE_TO_TIMES_DICT.update({
    SATELLITE_FILE_NAMES[2]:
        [THESE_START_TIMES_UNIX_SEC, THESE_END_TIMES_UNIX_SEC]
})

THESE_START_TIMES_UNIX_SEC = numpy.array([
    TARGET_TIMES_UNIX_SEC[4] - numpy.max(FIRST_LAG_TIMES_SEC) -
    FIRST_MAX_INTERP_GAP_SEC
], dtype=int)

THESE_END_TIMES_UNIX_SEC = numpy.array([
    TARGET_TIMES_UNIX_SEC[4] - numpy.min(FIRST_LAG_TIMES_SEC)
], dtype=int)

THESE_END_TIMES_UNIX_SEC[
    THESE_END_TIMES_UNIX_SEC != TARGET_TIMES_UNIX_SEC[4]
] += FIRST_MAX_INTERP_GAP_SEC

FIRST_DESIRED_FILE_TO_TIMES_DICT.update({
    SATELLITE_FILE_NAMES[0]:
        [THESE_START_TIMES_UNIX_SEC, THESE_END_TIMES_UNIX_SEC]
})

SECOND_LAG_TIMES_MINUTES = numpy.array([0, 1], dtype=int)
SECOND_LAG_TIMES_SEC = MINUTES_TO_SECONDS * SECOND_LAG_TIMES_MINUTES
SECOND_LAG_TIME_TOLERANCE_SEC = 0
SECOND_MAX_INTERP_GAP_SEC = 90000

THESE_START_TIMES_UNIX_SEC = numpy.array([
    TARGET_TIMES_UNIX_SEC[0] - numpy.max(SECOND_LAG_TIMES_SEC) -
    SECOND_MAX_INTERP_GAP_SEC,
    TARGET_TIMES_UNIX_SEC[1] - numpy.max(SECOND_LAG_TIMES_SEC) -
    SECOND_MAX_INTERP_GAP_SEC,
    TARGET_TIMES_UNIX_SEC[2] - numpy.max(SECOND_LAG_TIMES_SEC) -
    SECOND_MAX_INTERP_GAP_SEC
], dtype=int)

THESE_END_TIMES_UNIX_SEC = numpy.array([
    TARGET_TIMES_UNIX_SEC[0] - numpy.min(SECOND_LAG_TIMES_SEC),
    TARGET_TIMES_UNIX_SEC[1] - numpy.min(SECOND_LAG_TIMES_SEC),
    TARGET_TIMES_UNIX_SEC[2] - numpy.min(SECOND_LAG_TIMES_SEC)
], dtype=int)

SECOND_DESIRED_FILE_TO_TIMES_DICT = {
    SATELLITE_FILE_NAMES[1]:
        [THESE_START_TIMES_UNIX_SEC, THESE_END_TIMES_UNIX_SEC]
}

THESE_START_TIMES_UNIX_SEC = numpy.array([
    TARGET_TIMES_UNIX_SEC[2] - numpy.max(SECOND_LAG_TIMES_SEC) -
    SECOND_MAX_INTERP_GAP_SEC,
    TARGET_TIMES_UNIX_SEC[3] - numpy.max(SECOND_LAG_TIMES_SEC) -
    SECOND_MAX_INTERP_GAP_SEC
], dtype=int)

THESE_END_TIMES_UNIX_SEC = numpy.array([
    TARGET_TIMES_UNIX_SEC[2] - numpy.min(SECOND_LAG_TIMES_SEC),
    TARGET_TIMES_UNIX_SEC[3] - numpy.min(SECOND_LAG_TIMES_SEC)
], dtype=int)

SECOND_DESIRED_FILE_TO_TIMES_DICT.update({
    SATELLITE_FILE_NAMES[3]:
        [THESE_START_TIMES_UNIX_SEC, THESE_END_TIMES_UNIX_SEC]
})

SECOND_DESIRED_FILE_TO_TIMES_DICT.update({
    SATELLITE_FILE_NAMES[2]:
        [THESE_START_TIMES_UNIX_SEC, THESE_END_TIMES_UNIX_SEC]
})

THESE_START_TIMES_UNIX_SEC = numpy.array([
    TARGET_TIMES_UNIX_SEC[4] - numpy.max(SECOND_LAG_TIMES_SEC) -
    SECOND_MAX_INTERP_GAP_SEC
], dtype=int)

THESE_END_TIMES_UNIX_SEC = numpy.array([
    TARGET_TIMES_UNIX_SEC[4] - numpy.min(SECOND_LAG_TIMES_SEC)
], dtype=int)

SECOND_DESIRED_FILE_TO_TIMES_DICT.update({
    SATELLITE_FILE_NAMES[0]:
        [THESE_START_TIMES_UNIX_SEC, THESE_END_TIMES_UNIX_SEC]
})

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

# The following constants are used to test _make_targets_for_semantic_seg.
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

FIRST_TARGET_MATRIX = FIRST_TARGET_MATRIX / numpy.sum(FIRST_TARGET_MATRIX)
SECOND_TARGET_MATRIX = SECOND_TARGET_MATRIX / numpy.sum(SECOND_TARGET_MATRIX)
THIRD_TARGET_MATRIX = THIRD_TARGET_MATRIX / numpy.sum(THIRD_TARGET_MATRIX)
FOURTH_TARGET_MATRIX = FOURTH_TARGET_MATRIX / numpy.sum(FOURTH_TARGET_MATRIX)
FIFTH_TARGET_MATRIX = FIFTH_TARGET_MATRIX / numpy.sum(FIFTH_TARGET_MATRIX)
SIXTH_TARGET_MATRIX = SIXTH_TARGET_MATRIX / numpy.sum(SIXTH_TARGET_MATRIX)

TARGET_MATRIX = numpy.stack((
    FIRST_TARGET_MATRIX, SECOND_TARGET_MATRIX, THIRD_TARGET_MATRIX,
    FOURTH_TARGET_MATRIX, FIFTH_TARGET_MATRIX, SIXTH_TARGET_MATRIX
), axis=0)


def _compare_file_to_times_dicts(first_dict, second_dict):
    """Determines whether or not two dictionaries are equal.

    These are dictionaries in the format returned by
    `neural_net._decide_files_to_read_one_cyclone`.

    :param first_dict: First dictionary.
    :param second_dict: Second dictionary.
    :return: are_dicts_equal: Boolean flag.
    """

    first_keys = list(first_dict.keys())
    second_keys = list(second_dict.keys())
    if set(first_keys) != set(second_keys):
        return False

    for this_key in first_keys:
        first_array_list = first_dict[this_key]
        second_array_list = second_dict[this_key]
        if len(first_array_list) != len(second_array_list):
            return False

        for i in range(len(first_array_list)):
            if not numpy.array_equal(
                    first_array_list[i], second_array_list[i]
            ):
                return False

    return True


class NeuralNetTests(unittest.TestCase):
    """Each method is a unit test for neural_net.py."""

    def test_date_in_time_period_zeroth(self):
        """Ensures correct output from _date_in_time_period.

        With zeroth set of options.
        """

        this_flag = neural_net._date_in_time_period(
            date_string=ZEROTH_DATE_STRING,
            start_time_unix_sec=ZEROTH_START_TIME_UNIX_SEC,
            end_time_unix_sec=ZEROTH_END_TIME_UNIX_SEC
        )
        self.assertEqual(this_flag, ZEROTH_DATE_IN_TIME_PERIOD_FLAG)

    def test_date_in_time_period_first(self):
        """Ensures correct output from _date_in_time_period.

        With first set of options.
        """

        this_flag = neural_net._date_in_time_period(
            date_string=FIRST_DATE_STRING,
            start_time_unix_sec=FIRST_START_TIME_UNIX_SEC,
            end_time_unix_sec=FIRST_END_TIME_UNIX_SEC
        )
        self.assertEqual(this_flag, FIRST_DATE_IN_TIME_PERIOD_FLAG)

    def test_date_in_time_period_second(self):
        """Ensures correct output from _date_in_time_period.

        With second set of options.
        """

        this_flag = neural_net._date_in_time_period(
            date_string=SECOND_DATE_STRING,
            start_time_unix_sec=SECOND_START_TIME_UNIX_SEC,
            end_time_unix_sec=SECOND_END_TIME_UNIX_SEC
        )
        self.assertEqual(this_flag, SECOND_DATE_IN_TIME_PERIOD_FLAG)

    def test_date_in_time_period_third(self):
        """Ensures correct output from _date_in_time_period.

        With third set of options.
        """

        this_flag = neural_net._date_in_time_period(
            date_string=THIRD_DATE_STRING,
            start_time_unix_sec=THIRD_START_TIME_UNIX_SEC,
            end_time_unix_sec=THIRD_END_TIME_UNIX_SEC
        )
        self.assertEqual(this_flag, THIRD_DATE_IN_TIME_PERIOD_FLAG)

    def test_date_in_time_period_fourth(self):
        """Ensures correct output from _date_in_time_period.

        With fourth set of options.
        """

        this_flag = neural_net._date_in_time_period(
            date_string=FOURTH_DATE_STRING,
            start_time_unix_sec=FOURTH_START_TIME_UNIX_SEC,
            end_time_unix_sec=FOURTH_END_TIME_UNIX_SEC
        )
        self.assertEqual(this_flag, FOURTH_DATE_IN_TIME_PERIOD_FLAG)

    def test_date_in_time_period_fifth(self):
        """Ensures correct output from _date_in_time_period.

        With fifth set of options.
        """

        this_flag = neural_net._date_in_time_period(
            date_string=FIFTH_DATE_STRING,
            start_time_unix_sec=FIFTH_START_TIME_UNIX_SEC,
            end_time_unix_sec=FIFTH_END_TIME_UNIX_SEC
        )
        self.assertEqual(this_flag, FIFTH_DATE_IN_TIME_PERIOD_FLAG)

    def test_date_in_time_period_sixth(self):
        """Ensures correct output from _date_in_time_period.

        With sixth set of options.
        """

        this_flag = neural_net._date_in_time_period(
            date_string=SIXTH_DATE_STRING,
            start_time_unix_sec=SIXTH_START_TIME_UNIX_SEC,
            end_time_unix_sec=SIXTH_END_TIME_UNIX_SEC
        )
        self.assertEqual(this_flag, SIXTH_DATE_IN_TIME_PERIOD_FLAG)

    def test_decide_files_to_read_first(self):
        """Ensures correct output from _decide_files_to_read_one_cyclone.

        With first set of options.
        """

        this_dict = neural_net._decide_files_to_read_one_cyclone(
            satellite_file_names=SATELLITE_FILE_NAMES,
            target_times_unix_sec=TARGET_TIMES_UNIX_SEC,
            lag_times_minutes=FIRST_LAG_TIMES_MINUTES,
            lag_time_tolerance_sec=FIRST_LAG_TIME_TOLERANCE_SEC,
            max_interp_gap_sec=FIRST_MAX_INTERP_GAP_SEC
        )

        self.assertTrue(
            _compare_file_to_times_dicts(
                FIRST_DESIRED_FILE_TO_TIMES_DICT, this_dict
            )
        )

    def test_decide_files_to_read_second(self):
        """Ensures correct output from _decide_files_to_read_one_cyclone.

        With second set of options.
        """

        this_dict = neural_net._decide_files_to_read_one_cyclone(
            satellite_file_names=SATELLITE_FILE_NAMES,
            target_times_unix_sec=TARGET_TIMES_UNIX_SEC,
            lag_times_minutes=SECOND_LAG_TIMES_MINUTES,
            lag_time_tolerance_sec=SECOND_LAG_TIME_TOLERANCE_SEC,
            max_interp_gap_sec=SECOND_MAX_INTERP_GAP_SEC
        )

        self.assertTrue(
            _compare_file_to_times_dicts(
                SECOND_DESIRED_FILE_TO_TIMES_DICT, this_dict
            )
        )

    def test_translate_images_first_3d(self):
        """Ensures correct output from _translate_images.

        With first set of options for 3-D data.
        """

        this_image_matrix = neural_net._translate_images(
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

        this_image_matrix = neural_net._translate_images(
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

        this_image_matrix = neural_net._translate_images(
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

        this_image_matrix = neural_net._translate_images(
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

        this_image_matrix = neural_net._translate_images(
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

        this_image_matrix = neural_net._translate_images(
            image_matrix=DATA_MATRIX_5D_NO_TRANSLATION,
            row_translation_px=SECOND_Y_OFFSET_PIXELS,
            column_translation_px=SECOND_X_OFFSET_PIXELS,
            padding_value=SECOND_SENTINEL_VALUE
        )

        self.assertTrue(numpy.allclose(
            this_image_matrix, DATA_MATRIX_5D_SECOND_TRANS, atol=TOLERANCE
        ))

    def test_make_targets_for_semantic_seg(self):
        """Ensures correct output from _make_targets_for_semantic_seg."""

        this_target_matrix = neural_net._make_targets_for_semantic_seg(
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


if __name__ == '__main__':
    unittest.main()
