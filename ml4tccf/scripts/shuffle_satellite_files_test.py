"""Unit tests for shuffle_satellite_files.py."""

import unittest
import numpy
from ml4tccf.scripts import shuffle_satellite_files

HOURS_TO_SECONDS = 3600

CHUNK_NUMS_PER_INPUT_FILE = numpy.array([
    1, 4, 7, 10,
    1, 4, 7, 10,
    1, 4, 7, 10
], dtype=int)

OUTPUT_TIME_INTERVALS_MIN = numpy.array([
    30, 30, 30, 30,
    120, 120, 120, 120,
    210, 210, 210, 210,
], dtype=int)

START_TIMES_HOURS_INTO_DAY_ARRAYLIST = [
    numpy.array([0]),
    numpy.array([0, 6, 12, 18]),
    numpy.array([0, 3.5, 7, 10.5, 14, 17.5, 21]),
    numpy.array([0, 2.5, 5, 7.5, 10, 12, 14.5, 17, 19.5, 22]),
    numpy.array([0]),
    numpy.array([0, 6, 12, 18]),
    numpy.array([0, 4, 8, 12, 14, 18, 22]),
    numpy.array([0, 4, 6, 8, 10, 12, 16, 18, 20, 22]),
    numpy.array([0]),
    numpy.array([0, 7, 14, 21]),
    numpy.array([0, 3.5, 7, 10.5, 14, 17.5, 21]),
    numpy.array([0, 3.5, 7, 10.5, 14, 17.5, 21])
]

END_TIMES_HOURS_INTO_DAY_ARRAYLIST = [
    numpy.array([24]),
    numpy.array([6, 12, 18, 24]),
    numpy.array([3.5, 7, 10.5, 14, 17.5, 21, 24]),
    numpy.array([2.5, 5, 7.5, 10, 12, 14.5, 17, 19.5, 22, 24]),
    numpy.array([24]),
    numpy.array([6, 12, 18, 24]),
    numpy.array([4, 8, 12, 14, 18, 22, 24]),
    numpy.array([4, 6, 8, 10, 12, 16, 18, 20, 22, 24]),
    numpy.array([24]),
    numpy.array([7, 14, 21, 24]),
    numpy.array([3.5, 7, 10.5, 14, 17.5, 21, 24]),
    numpy.array([3.5, 7, 10.5, 14, 17.5, 21, 24])
]

START_TIMES_SEC_INTO_DAY_ARRAYLIST = [
    numpy.round(HOURS_TO_SECONDS * t).astype(int)
    for t in START_TIMES_HOURS_INTO_DAY_ARRAYLIST
]

END_TIMES_SEC_INTO_DAY_ARRAYLIST = [
    numpy.round(HOURS_TO_SECONDS * t).astype(int)
    for t in END_TIMES_HOURS_INTO_DAY_ARRAYLIST
]


class ShuffleSatelliteFilesTests(unittest.TestCase):
    """Each method is a unit test for shuffle_satellite_files.py."""

    def test_get_time_range_by_chunk(self):
        """Ensures correct output from _get_time_range_by_chunk."""

        for i in range(len(CHUNK_NUMS_PER_INPUT_FILE)):
            these_start_times, these_end_times = (
                shuffle_satellite_files._get_time_range_by_chunk(
                    num_chunks_per_input_file=CHUNK_NUMS_PER_INPUT_FILE[i],
                    output_time_interval_minutes=OUTPUT_TIME_INTERVALS_MIN[i]
                )
            )

            self.assertTrue(numpy.array_equal(
                these_start_times, START_TIMES_SEC_INTO_DAY_ARRAYLIST[i]
            ))
            self.assertTrue(numpy.array_equal(
                these_end_times, END_TIMES_SEC_INTO_DAY_ARRAYLIST[i]
            ))


if __name__ == '__main__':
    unittest.main()
