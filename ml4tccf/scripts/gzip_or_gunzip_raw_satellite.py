"""Uses gzip to compress or decompress raw satellite data."""

import os
import argparse
from ml4tccf.io import raw_satellite_io
from ml4tccf.utils import misc_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

RAW_SATELLITE_DIR_ARG_NAME = 'raw_satellite_dir_name'
CYCLONE_ID_ARG_NAME = 'cyclone_id_string'
COMPRESS_FLAG_ARG_NAME = 'compress_flag'

RAW_SATELLITE_DIR_HELP_STRING = (
    'Name of working directory.  Input (either compressed or decompressed) '
    'files will be found therein by `raw_satellite_io.find_file`, and output '
    '(either decompressed or compressed, respectively) files created by this '
    'script will be written to the same directory, to exact locations also '
    'determined by `raw_satellte_io.find_file`.'
)
CYCLONE_ID_HELP_STRING = (
    'Cyclone ID in format "yyyyBBnn".  Will compress or decompress files for '
    'this cyclone.'
)
COMPRESS_FLAG_HELP_STRING = (
    'Boolean flag.  If 1 (0), will compress (decompress) files.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + RAW_SATELLITE_DIR_ARG_NAME, type=str, required=True,
    help=RAW_SATELLITE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CYCLONE_ID_ARG_NAME, type=str, required=True,
    help=CYCLONE_ID_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + COMPRESS_FLAG_ARG_NAME, type=int, required=True,
    help=COMPRESS_FLAG_HELP_STRING
)


def _run(raw_satellite_dir_name, cyclone_id_string, compress_flag):
    """Uses gzip to compress or decompress raw satellite data.

    This is effectively the main method.

    :param raw_satellite_dir_name: See documentation at top of file.
    :param cyclone_id_string: Same.
    :param compress_flag: Same.
    :raises: ValueError: if no input files are found.
    """

    input_file_names = raw_satellite_io.find_files_one_tc(
        directory_name=raw_satellite_dir_name,
        cyclone_id_string=cyclone_id_string,
        look_for_high_res=False,
        prefer_zipped_format=not compress_flag,
        allow_other_format=False,
        raise_error_if_all_missing=False
    )

    input_file_names += raw_satellite_io.find_files_one_tc(
        directory_name=raw_satellite_dir_name,
        cyclone_id_string=cyclone_id_string,
        look_for_high_res=True,
        prefer_zipped_format=not compress_flag,
        allow_other_format=False,
        raise_error_if_all_missing=len(input_file_names) == 0
    )

    for this_file_name in input_file_names:
        if compress_flag:
            print('Compressing file with gzip: "{0:s}"...'.format(
                this_file_name
            ))
            misc_utils.gzip_file(this_file_name)
        else:
            print('Decompressing file with gunzip: "{0:s}"...'.format(
                this_file_name
            ))
            misc_utils.gunzip_file(this_file_name)

        os.remove(this_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        raw_satellite_dir_name=getattr(
            INPUT_ARG_OBJECT, RAW_SATELLITE_DIR_ARG_NAME
        ),
        cyclone_id_string=getattr(INPUT_ARG_OBJECT, CYCLONE_ID_ARG_NAME),
        compress_flag=bool(getattr(INPUT_ARG_OBJECT, COMPRESS_FLAG_ARG_NAME))
    )
