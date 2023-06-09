"""Unit tests for neural_net_training_cira_ir.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import number_rounding
from ml4tccf.machine_learning import \
    neural_net_training_cira_ir as nn_training_cira_ir

TOLERANCE = 1e-6
DAYS_TO_SECONDS = 86400

# The following constants are used to test _get_synoptic_target_times.
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
    '1978-10-18-175144', '1995-12-31-000000', '1997-03-12-120000',
    '2004-12-11-060404', '2019-10-30-180000', '2022-12-12-115648',
    '2023-04-29-055818', '2026-07-09-060000'
]

SYNOPTIC_TIMES_UNIX_SEC = numpy.array([
    time_conversion.string_to_unix_sec(t, '%Y-%m-%d-%H%M%S')
    for t in SYNOPTIC_TIME_STRINGS
], dtype=int)

# The following constants are used to test _choose_random_target_times.
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


class NeuralNetTrainingCiraIrTests(unittest.TestCase):
    """Each method is a unit test for neural_net_training_cira_ir.py."""

    def test_get_synoptic_target_times(self):
        """Ensures correct output from _get_synoptic_target_times."""

        these_times_unix_sec = nn_training_cira_ir._get_synoptic_target_times(
            all_target_times_unix_sec=ALL_TIMES_UNIX_SEC + 0
        )
        self.assertTrue(numpy.array_equal(
            these_times_unix_sec, SYNOPTIC_TIMES_UNIX_SEC
        ))

    def test_choose_random_target_times(self):
        """Ensures correct output from _choose_random_target_times."""

        for _ in range(10):
            chosen_times_unix_sec = (
                nn_training_cira_ir._choose_random_target_times(
                    all_target_times_unix_sec=RANDOM_TIMES_UNIX_SEC,
                    num_times_desired=NUM_TIMES_TO_CHOOSE
                )[0]
            )
            chosen_dates_unix_sec = number_rounding.floor_to_nearest(
                chosen_times_unix_sec, DAYS_TO_SECONDS
            )

            if len(numpy.unique(chosen_dates_unix_sec)) > 1:
                return

        self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()
