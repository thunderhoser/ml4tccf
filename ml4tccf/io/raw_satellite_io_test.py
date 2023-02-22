"""Unit tests for raw_satellite_io.py."""

import unittest
from gewittergefahr.gg_utils import time_conversion
from ml4tccf.io import raw_satellite_io

DIRECTORY_NAME = 'foo'
CYCLONE_ID_STRING = '2021AL01'
VALID_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2021-05-20-0030', '%Y-%m-%d-%H%M'
)
LOOK_FOR_HIGH_RES = False

SATELLITE_FILE_NAME_UNZIPPED = (
    'foo/al012021/2021-05-20/2021-05-20T00-30-00/'
    '2000m_2021-05-20T00-30-00_al012021.nc'
)
SATELLITE_FILE_NAME_ZIPPED = (
    'foo/al012021/2021-05-20/2021-05-20T00-30-00/'
    '2000m_2021-05-20T00-30-00_al012021.nc.gz'
)

SATELLITE_FILE_PATTERN_UNZIPPED = (
    'foo/al012021/{0:s}/{1:s}/2000m_{1:s}_al012021.nc'
).format(raw_satellite_io.DATE_REGEX, raw_satellite_io.TIME_REGEX)

SATELLITE_FILE_PATTERN_ZIPPED = (
    'foo/al012021/{0:s}/{1:s}/2000m_{1:s}_al012021.nc.gz'
).format(raw_satellite_io.DATE_REGEX, raw_satellite_io.TIME_REGEX)


class RawSatelliteIoTests(unittest.TestCase):
    """Each method is a unit test for raw_satellite_io.py."""

    def test_find_file_zipped_allow(self):
        """Ensures correct output from find_file.

        In this case, looking for zipped file but will allow unzipped file.
        """

        this_file_name = raw_satellite_io.find_file(
            directory_name=DIRECTORY_NAME,
            cyclone_id_string=CYCLONE_ID_STRING,
            valid_time_unix_sec=VALID_TIME_UNIX_SEC,
            look_for_high_res=LOOK_FOR_HIGH_RES,
            prefer_zipped_format=True, allow_other_format=True,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == SATELLITE_FILE_NAME_UNZIPPED)

    def test_find_file_zipped_no_allow(self):
        """Ensures correct output from find_file.

        In this case, looking for zipped file and will *not* allow unzipped
        file.
        """

        this_file_name = raw_satellite_io.find_file(
            directory_name=DIRECTORY_NAME,
            cyclone_id_string=CYCLONE_ID_STRING,
            valid_time_unix_sec=VALID_TIME_UNIX_SEC,
            look_for_high_res=LOOK_FOR_HIGH_RES,
            prefer_zipped_format=True, allow_other_format=False,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == SATELLITE_FILE_NAME_ZIPPED)

    def test_find_file_unzipped_allow(self):
        """Ensures correct output from find_file.

        In this case, looking for unzipped file but will allow zipped file.
        """

        this_file_name = raw_satellite_io.find_file(
            directory_name=DIRECTORY_NAME,
            cyclone_id_string=CYCLONE_ID_STRING,
            valid_time_unix_sec=VALID_TIME_UNIX_SEC,
            look_for_high_res=LOOK_FOR_HIGH_RES,
            prefer_zipped_format=False, allow_other_format=True,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == SATELLITE_FILE_NAME_ZIPPED)

    def test_find_file_unzipped_no_allow(self):
        """Ensures correct output from find_file.

        In this case, looking for unzipped file and will *not* allow zipped
        file.
        """

        this_file_name = raw_satellite_io.find_file(
            directory_name=DIRECTORY_NAME,
            cyclone_id_string=CYCLONE_ID_STRING,
            valid_time_unix_sec=VALID_TIME_UNIX_SEC,
            look_for_high_res=LOOK_FOR_HIGH_RES,
            prefer_zipped_format=False, allow_other_format=False,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == SATELLITE_FILE_NAME_UNZIPPED)

    def test_file_name_to_time_zipped(self):
        """Ensures correct output from file_name_to_time_zipped.

        In this case, using name of zipped file.
        """

        self.assertTrue(
            raw_satellite_io.file_name_to_time(SATELLITE_FILE_NAME_ZIPPED) ==
            VALID_TIME_UNIX_SEC
        )

    def test_file_name_to_time_unzipped(self):
        """Ensures correct output from file_name_to_time_zipped.

        In this case, using name of unzipped file.
        """

        self.assertTrue(
            raw_satellite_io.file_name_to_time(SATELLITE_FILE_NAME_UNZIPPED) ==
            VALID_TIME_UNIX_SEC
        )

    def test_find_files_one_tc_zipped_allow(self):
        """Ensures correct output from find_files_one_tc.

        In this case, looking for zipped files but will allow unzipped files.
        """

        these_file_names = raw_satellite_io.find_files_one_tc(
            directory_name=DIRECTORY_NAME,
            cyclone_id_string=CYCLONE_ID_STRING,
            look_for_high_res=LOOK_FOR_HIGH_RES,
            prefer_zipped_format=True, allow_other_format=True,
            raise_error_if_all_missing=False, test_mode=True
        )

        self.assertTrue(these_file_names == [SATELLITE_FILE_PATTERN_ZIPPED])

    def test_find_files_one_tc_zipped_no_allow(self):
        """Ensures correct output from find_files_one_tc.

        In this case, looking for zipped files and will *not* allow unzipped
        files.
        """

        these_file_names = raw_satellite_io.find_files_one_tc(
            directory_name=DIRECTORY_NAME,
            cyclone_id_string=CYCLONE_ID_STRING,
            look_for_high_res=LOOK_FOR_HIGH_RES,
            prefer_zipped_format=True, allow_other_format=False,
            raise_error_if_all_missing=False, test_mode=True
        )

        self.assertTrue(these_file_names == [SATELLITE_FILE_PATTERN_ZIPPED])

    def test_find_files_one_tc_unzipped_allow(self):
        """Ensures correct output from find_files_one_tc.

        In this case, looking for unzipped files but will allow zipped files.
        """

        these_file_names = raw_satellite_io.find_files_one_tc(
            directory_name=DIRECTORY_NAME,
            cyclone_id_string=CYCLONE_ID_STRING,
            look_for_high_res=LOOK_FOR_HIGH_RES,
            prefer_zipped_format=False, allow_other_format=True,
            raise_error_if_all_missing=False, test_mode=True
        )

        self.assertTrue(these_file_names == [SATELLITE_FILE_PATTERN_UNZIPPED])

    def test_find_files_one_tc_unzipped_no_allow(self):
        """Ensures correct output from find_files_one_tc.

        In this case, looking for unzipped files and will *not* allow zipped
        files.
        """

        these_file_names = raw_satellite_io.find_files_one_tc(
            directory_name=DIRECTORY_NAME,
            cyclone_id_string=CYCLONE_ID_STRING,
            look_for_high_res=LOOK_FOR_HIGH_RES,
            prefer_zipped_format=False, allow_other_format=False,
            raise_error_if_all_missing=False, test_mode=True
        )

        self.assertTrue(these_file_names == [SATELLITE_FILE_PATTERN_UNZIPPED])


if __name__ == '__main__':
    unittest.main()
