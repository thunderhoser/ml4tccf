"""Unit tests for neural_net_training_fancy.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import number_rounding
from ml4tccf.io import satellite_io
from ml4tccf.machine_learning import \
    neural_net_training_fancy as nn_training_fancy

TOLERANCE = 1e-6
MINUTES_TO_SECONDS = 60
DAYS_TO_SECONDS = 86400

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

# The following constants are used to test decide_files_to_read_one_cyclone.
# decide_files_to_read_one_cyclone(
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
TARGET_TIMES_UNIX_SEC = numpy.array([
    time_conversion.string_to_unix_sec(s, '%Y-%m-%d-%H')
    for s in TARGET_TIME_STRINGS
], dtype=int)

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

# The following constants are used to test get_synoptic_target_times.
ALL_TIME_STRINGS = [
    '2019-10-30-180000', '2020-05-19-191326', '1978-10-18-175144',
    '1995-12-31-000000', '2022-12-12-115648', '2014-08-18-171725',
    '1974-12-27-160938', '1973-12-01-033442', '1976-09-05-111004',
    '2026-07-09-060000', '2004-12-11-060404', '2030-03-02-031855',
    '1983-09-27-033019', '2005-06-10-201341', '2021-05-31-041708',
    '1997-03-12-120000', '2029-01-01-023524', '1986-03-02-130239',
    '2023-04-29-055818', '2012-09-26-112921'
]

ALL_TIMES_UNIX_SEC = numpy.array([
    time_conversion.string_to_unix_sec(t, '%Y-%m-%d-%H%M%S')
    for t in ALL_TIME_STRINGS
], dtype=int)

SYNOPTIC_TIME_STRINGS = [
    '2019-10-30-180000', '1995-12-31-000000', '2026-07-09-060000',
    '1997-03-12-120000'
]

SYNOPTIC_TIMES_UNIX_SEC = numpy.array([
    time_conversion.string_to_unix_sec(t, '%Y-%m-%d-%H%M%S')
    for t in SYNOPTIC_TIME_STRINGS
], dtype=int)

# The following constants are used to test choose_random_target_times.
RANDOM_TIME_STRINGS = [
    '2019-10-30-180000', '2020-05-19-191326', '1978-10-18-175144',
    '2019-10-30-000000', '2020-05-19-115648', '1978-10-18-171725',
    '2019-10-30-160938', '2020-05-19-033442', '1978-10-18-111004',
    '2019-10-30-060000', '2020-05-19-060404', '1978-10-18-031855',
    '2019-10-30-033019', '2020-05-19-201341', '1978-10-18-041708',
    '2019-10-30-120000', '2020-05-19-023524', '1978-10-18-130239',
    '2019-10-30-055818', '2020-05-19-112921'
]

RANDOM_TIMES_UNIX_SEC = numpy.array([
    time_conversion.string_to_unix_sec(t, '%Y-%m-%d-%H%M%S')
    for t in RANDOM_TIME_STRINGS
], dtype=int)

NUM_TIMES_TO_CHOOSE = 6


def _compare_file_to_times_dicts(first_dict, second_dict):
    """Determines whether or not two dictionaries are equal.

    These are dictionaries in the format returned by
    `neural_net.decide_files_to_read_one_cyclone`.

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


class NeuralNetTrainingFancyTests(unittest.TestCase):
    """Each method is a unit test for neural_net_training_fancy.py."""

    def test_date_in_time_period_zeroth(self):
        """Ensures correct output from _date_in_time_period.

        With zeroth set of options.
        """

        this_flag = nn_training_fancy._date_in_time_period(
            date_string=ZEROTH_DATE_STRING,
            start_time_unix_sec=ZEROTH_START_TIME_UNIX_SEC,
            end_time_unix_sec=ZEROTH_END_TIME_UNIX_SEC
        )
        self.assertEqual(this_flag, ZEROTH_DATE_IN_TIME_PERIOD_FLAG)

    def test_date_in_time_period_first(self):
        """Ensures correct output from _date_in_time_period.

        With first set of options.
        """

        this_flag = nn_training_fancy._date_in_time_period(
            date_string=FIRST_DATE_STRING,
            start_time_unix_sec=FIRST_START_TIME_UNIX_SEC,
            end_time_unix_sec=FIRST_END_TIME_UNIX_SEC
        )
        self.assertEqual(this_flag, FIRST_DATE_IN_TIME_PERIOD_FLAG)

    def test_date_in_time_period_second(self):
        """Ensures correct output from _date_in_time_period.

        With second set of options.
        """

        this_flag = nn_training_fancy._date_in_time_period(
            date_string=SECOND_DATE_STRING,
            start_time_unix_sec=SECOND_START_TIME_UNIX_SEC,
            end_time_unix_sec=SECOND_END_TIME_UNIX_SEC
        )
        self.assertEqual(this_flag, SECOND_DATE_IN_TIME_PERIOD_FLAG)

    def test_date_in_time_period_third(self):
        """Ensures correct output from _date_in_time_period.

        With third set of options.
        """

        this_flag = nn_training_fancy._date_in_time_period(
            date_string=THIRD_DATE_STRING,
            start_time_unix_sec=THIRD_START_TIME_UNIX_SEC,
            end_time_unix_sec=THIRD_END_TIME_UNIX_SEC
        )
        self.assertEqual(this_flag, THIRD_DATE_IN_TIME_PERIOD_FLAG)

    def test_date_in_time_period_fourth(self):
        """Ensures correct output from _date_in_time_period.

        With fourth set of options.
        """

        this_flag = nn_training_fancy._date_in_time_period(
            date_string=FOURTH_DATE_STRING,
            start_time_unix_sec=FOURTH_START_TIME_UNIX_SEC,
            end_time_unix_sec=FOURTH_END_TIME_UNIX_SEC
        )
        self.assertEqual(this_flag, FOURTH_DATE_IN_TIME_PERIOD_FLAG)

    def test_date_in_time_period_fifth(self):
        """Ensures correct output from _date_in_time_period.

        With fifth set of options.
        """

        this_flag = nn_training_fancy._date_in_time_period(
            date_string=FIFTH_DATE_STRING,
            start_time_unix_sec=FIFTH_START_TIME_UNIX_SEC,
            end_time_unix_sec=FIFTH_END_TIME_UNIX_SEC
        )
        self.assertEqual(this_flag, FIFTH_DATE_IN_TIME_PERIOD_FLAG)

    def test_date_in_time_period_sixth(self):
        """Ensures correct output from _date_in_time_period.

        With sixth set of options.
        """

        this_flag = nn_training_fancy._date_in_time_period(
            date_string=SIXTH_DATE_STRING,
            start_time_unix_sec=SIXTH_START_TIME_UNIX_SEC,
            end_time_unix_sec=SIXTH_END_TIME_UNIX_SEC
        )
        self.assertEqual(this_flag, SIXTH_DATE_IN_TIME_PERIOD_FLAG)

    def test_decide_files_to_read_first(self):
        """Ensures correct output from decide_files_to_read_one_cyclone.

        With first set of options.
        """

        this_dict = nn_training_fancy.decide_files_to_read_one_cyclone(
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
        """Ensures correct output from decide_files_to_read_one_cyclone.

        With second set of options.
        """

        this_dict = nn_training_fancy.decide_files_to_read_one_cyclone(
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

    def test_get_synoptic_target_times(self):
        """Ensures correct output from get_synoptic_target_times."""

        these_times_unix_sec = nn_training_fancy.get_synoptic_target_times(
            all_target_times_unix_sec=ALL_TIMES_UNIX_SEC + 0
        )
        self.assertTrue(numpy.array_equal(
            these_times_unix_sec, SYNOPTIC_TIMES_UNIX_SEC
        ))

    def test_choose_random_target_times(self):
        """Ensures correct output from choose_random_target_times."""

        chosen_times_unix_sec = nn_training_fancy.choose_random_target_times(
            all_target_times_unix_sec=RANDOM_TIMES_UNIX_SEC,
            num_times_desired=NUM_TIMES_TO_CHOOSE
        )[0]
        chosen_dates_unix_sec = number_rounding.floor_to_nearest(
            chosen_times_unix_sec, DAYS_TO_SECONDS
        )

        self.assertTrue(len(numpy.unique(chosen_dates_unix_sec)) == 1)


if __name__ == '__main__':
    unittest.main()
