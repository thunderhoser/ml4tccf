"""Parallax-corrects satellite images."""

import argparse
import numpy
import xarray
from scipy.interpolate import RegularGridInterpolator
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking
from ml4tccf.io import satellite_io
from ml4tccf.utils import misc_utils
from ml4tccf.utils import satellite_utils
from ml4tccf.outside_code import parallax

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

METRES_TO_MICRONS = 1e6
DUMMY_CLOUD_TOP_HEIGHT_FOR_VISIBLE_M_AGL = 2000.

INPUT_DIR_ARG_NAME = 'input_satellite_dir_name'
CYCLONE_IDS_ARG_NAME = 'cyclone_id_strings'
NEIGH_HALF_WIDTH_ARG_NAME = 'neigh_half_width_low_res_px'
OUTPUT_DIR_ARG_NAME = 'output_satellite_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing uncorrected images.  Files therein '
    'will be found by `satellite_io.find_file` and read by '
    '`satellite_io.read_file`.'
)
CYCLONE_IDS_HELP_STRING = (
    'List of cyclone IDs.  This script will parallax-correct images for all '
    'given cyclones.'
)
NEIGH_HALF_WIDTH_HELP_STRING = (
    'Neighbourhood half-width (in low-resolution pixels, or IR pixels) for '
    'determining cloud-top height.  For example, if neigh half-width is 50 px, '
    'this script will average brightness temperatures over the middle '
    '100-by-100-px window of each image, convert this average BT to cloud-top '
    'height, and use this cloud-top height in the parallax correction.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Parallax-corrected images will be written here '
    'by `satellite_io.write_file`, to exact locations determined by '
    '`satellite_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CYCLONE_IDS_ARG_NAME, type=str, nargs='+', required=True,
    help=CYCLONE_IDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NEIGH_HALF_WIDTH_ARG_NAME, type=int, required=True,
    help=NEIGH_HALF_WIDTH_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _cyclone_id_to_satellite_metadata(cyclone_id_string):
    """Returns satellite metadata for the given cyclone.

    :param cyclone_id_string: Cyclone ID.
    :return: satellite_longitude_deg_e: Longitude (deg east) of satellite
        subpoint.
    :return: satellite_altitude_m_agl: Satellite altitude (metres above ground
        level).
    """

    basin_id_string = misc_utils.parse_cyclone_id(cyclone_id_string)[1]

    if basin_id_string == misc_utils.NORTH_ATLANTIC_ID_STRING:
        return -75.2, 35786650.

    if basin_id_string == misc_utils.NORTHWEST_PACIFIC_ID_STRING:
        return 140.7, 35793000.

    if basin_id_string == misc_utils.NORTHEAST_PACIFIC_ID_STRING:
        return -137.2, 35794900.


def _find_grid_center_one_time(satellite_table_xarray, time_index):
    """Finds grid center at one time step.

    :param satellite_table_xarray: xarray table returned by
        `satellite_io.read_file`, i.e., for a single file.
    :param time_index: Will find grid center for the [i]th time step, where
        i == `time_index`.
    :return: center_latitude_deg_n: Latitude of grid center (deg north).
    :return: center_longitude_deg_e: Longitude of grid center (deg east).
    """

    stx = satellite_table_xarray
    i = time_index

    assert numpy.all(
        numpy.diff(stx[satellite_utils.LONGITUDE_LOW_RES_KEY].values[i, :]) > 0
    )
    assert numpy.all(
        numpy.diff(stx[satellite_utils.LONGITUDE_HIGH_RES_KEY].values[i, :]) > 0
    )

    num_rows_low_res = (
        stx[satellite_utils.LATITUDE_LOW_RES_KEY].values.shape[1]
    )
    center_row_indices_low_res = numpy.array([
        0.5 * num_rows_low_res - 1,
        0.5 * num_rows_low_res
    ])
    center_row_indices_low_res = numpy.round(
        center_row_indices_low_res
    ).astype(int)

    num_columns_low_res = (
        stx[satellite_utils.LONGITUDE_LOW_RES_KEY].values.shape[1]
    )
    center_column_indices_low_res = numpy.array([
        0.5 * num_columns_low_res - 1,
        0.5 * num_columns_low_res
    ])
    center_column_indices_low_res = numpy.round(
        center_column_indices_low_res
    ).astype(int)

    center_latitude_deg_n = numpy.mean(
        stx[satellite_utils.LATITUDE_LOW_RES_KEY].values[
            i, center_row_indices_low_res
        ]
    )

    center_longitude_deg_e = numpy.mean(
        stx[satellite_utils.LONGITUDE_LOW_RES_KEY].values[
            i, center_column_indices_low_res
        ]
    )

    return center_latitude_deg_n, center_longitude_deg_e


def _parallax_correct_high_res_one_time(
        satellite_table_xarray, time_index,
        latitude_shift_testing_only_deg=None,
        longitude_shift_testing_only_deg=None):
    """Parallax-corrects high-resolution (visible) data for one time step.

    M = number of rows in high-resolution grid
    N = number of columns in high-resolution grid
    C = number of high-resolution channels

    :param satellite_table_xarray: xarray table returned by
        `satellite_io.read_file`, i.e., for a single file.
    :param time_index: Will parallax-correct data for the [i]th time step, where
        i == `time_index`.
    :param latitude_shift_testing_only_deg: Leave this alone.
    :param longitude_shift_testing_only_deg: Leave this alone.
    :return: bidirectional_reflectance_matrix: M-by-N-by-C numpy array of
        parallax-corrected data for the given time step.
    """

    test_mode = not (
        latitude_shift_testing_only_deg is None
        or longitude_shift_testing_only_deg is None
    )

    if test_mode:
        error_checking.assert_is_real_number(latitude_shift_testing_only_deg)
        error_checking.assert_is_real_number(longitude_shift_testing_only_deg)

    center_latitude_deg_n, center_longitude_deg_e = _find_grid_center_one_time(
        satellite_table_xarray=satellite_table_xarray,
        time_index=time_index
    )

    stx = satellite_table_xarray
    i = time_index

    cyclone_id_string = stx[satellite_utils.CYCLONE_ID_KEY].values[0]
    satellite_longitude_deg_e, satellite_altitude_m_agl = (
        _cyclone_id_to_satellite_metadata(cyclone_id_string)
    )

    if test_mode:
        corrected_center_latitude_deg_n = (
            center_latitude_deg_n + latitude_shift_testing_only_deg
        )
        corrected_center_longitude_deg_e = (
            center_longitude_deg_e + longitude_shift_testing_only_deg
        )
    else:
        (
            corrected_center_longitude_deg_e, corrected_center_latitude_deg_n
        ) = parallax.get_parallax_corrected_lonlats(
            sat_lon=satellite_longitude_deg_e, sat_lat=0.,
            sat_alt=satellite_altitude_m_agl,
            lon=center_longitude_deg_e, lat=center_latitude_deg_n,
            height=DUMMY_CLOUD_TOP_HEIGHT_FOR_VISIBLE_M_AGL
        )

    center_longitude_deg_e = lng_conversion.convert_lng_positive_in_west(
        center_longitude_deg_e
    )
    corrected_center_longitude_deg_e = (
        lng_conversion.convert_lng_positive_in_west(
            corrected_center_longitude_deg_e
        )
    )

    if numpy.absolute(
            corrected_center_longitude_deg_e - center_longitude_deg_e
    ) > 1:
        center_longitude_deg_e = lng_conversion.convert_lng_negative_in_west(
            center_longitude_deg_e
        )
        corrected_center_longitude_deg_e = (
            lng_conversion.convert_lng_negative_in_west(
                corrected_center_longitude_deg_e
            )
        )

    assert numpy.absolute(
        corrected_center_longitude_deg_e - center_longitude_deg_e
    ) < 1

    print((
        'Parallax correction for {0:.3f}-micron data at {1:d}th time = '
        '{2:.4f} deg north, {3:.4f} deg east'
    ).format(
        METRES_TO_MICRONS *
        stx.coords[satellite_utils.HIGH_RES_WAVELENGTH_DIM].values[0],
        i + 1,
        corrected_center_latitude_deg_n - center_latitude_deg_n,
        corrected_center_longitude_deg_e - center_longitude_deg_e
    ))

    corrected_latitudes_deg_n = (
        stx[satellite_utils.LATITUDE_HIGH_RES_KEY].values[i, :] +
        (corrected_center_latitude_deg_n - center_latitude_deg_n)
    )
    corrected_longitudes_deg_e = (
        stx[satellite_utils.LONGITUDE_HIGH_RES_KEY].values[i, :] +
        (corrected_center_longitude_deg_e - center_longitude_deg_e)
    )

    # error_checking.assert_is_valid_lat_numpy_array(
    #     corrected_latitudes_deg_n, allow_nan=False
    # )

    bidirectional_reflectance_matrix = (
        stx[satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY].values[i, ...]
    )
    num_channels = bidirectional_reflectance_matrix.shape[-1]

    for k in range(num_channels):
        if numpy.all(numpy.isnan(bidirectional_reflectance_matrix[..., k])):
            continue

        assert not numpy.any(
            numpy.isnan(bidirectional_reflectance_matrix[..., k])
        )

        orig_point_tuple = (
            corrected_latitudes_deg_n,
            corrected_longitudes_deg_e
        )

        interp_object = RegularGridInterpolator(
            points=orig_point_tuple,
            values=bidirectional_reflectance_matrix[..., k],
            method='linear', bounds_error=False, fill_value=None
        )

        (
            query_longitude_matrix_deg_e, query_latitude_matrix_deg_n
        ) = numpy.meshgrid(
            stx[satellite_utils.LONGITUDE_HIGH_RES_KEY].values[i, :],
            stx[satellite_utils.LATITUDE_HIGH_RES_KEY].values[i, :]
        )

        query_point_matrix = numpy.stack(
            (query_latitude_matrix_deg_n, query_longitude_matrix_deg_e),
            axis=-1
        )

        new_matrix = interp_object(query_point_matrix)
        abs_diff_matrix = numpy.absolute(
            new_matrix - bidirectional_reflectance_matrix[..., k]
        )

        print((
            'Min/mean/max absolute BDRF difference between original and '
            'corrected = {0:.4f}/{1:.4f}/{2:.4f}'
        ).format(
            numpy.min(abs_diff_matrix),
            numpy.mean(abs_diff_matrix),
            numpy.max(abs_diff_matrix)
        ))

        bidirectional_reflectance_matrix[..., k] = new_matrix + 0.

    return bidirectional_reflectance_matrix


def _parallax_correct_low_res_one_time(satellite_table_xarray, time_index,
                                       neigh_half_width_px):
    """Parallax-corrects low-resolution (infrared) data for one time step.

    m = number of rows in low-resolution grid
    n = number of columns in low-resolution grid
    c = number of low-resolution channels

    :param satellite_table_xarray: xarray table returned by
        `satellite_io.read_file`, i.e., for a single file.
    :param time_index: Will parallax-correct data for the [i]th time step, where
        i == `time_index`.
    :param neigh_half_width_px: See documentation at top of file.
    :return: brightness_temp_matrix_kelvins: m-by-n-by-c numpy array of
        parallax-corrected data for the given time step.
    """

    center_latitude_deg_n, center_longitude_deg_e = _find_grid_center_one_time(
        satellite_table_xarray=satellite_table_xarray,
        time_index=time_index
    )

    stx = satellite_table_xarray
    i = time_index

    cyclone_id_string = stx[satellite_utils.CYCLONE_ID_KEY].values[0]
    satellite_longitude_deg_e, satellite_altitude_m_agl = (
        _cyclone_id_to_satellite_metadata(cyclone_id_string)
    )

    brightness_temp_matrix_kelvins = (
        stx[satellite_utils.BRIGHTNESS_TEMPERATURE_KEY].values[i, ...]
    )
    num_grid_rows = brightness_temp_matrix_kelvins.shape[0]
    num_grid_columns = brightness_temp_matrix_kelvins.shape[1]
    num_channels = brightness_temp_matrix_kelvins.shape[2]

    first_row_index_in_middle = (
        int(numpy.round(0.5 * num_grid_rows)) - neigh_half_width_px
    )
    first_row_index_in_middle = max([first_row_index_in_middle, 0])

    last_row_index_in_middle = (
        int(numpy.round(0.5 * num_grid_rows)) + neigh_half_width_px
    )
    last_row_index_in_middle = min([last_row_index_in_middle, num_grid_rows])

    first_column_index_in_middle = (
        int(numpy.round(0.5 * num_grid_columns)) - neigh_half_width_px
    )
    first_column_index_in_middle = max([first_column_index_in_middle, 0])

    last_column_index_in_middle = (
        int(numpy.round(0.5 * num_grid_columns)) + neigh_half_width_px
    )
    last_column_index_in_middle = min([
        last_column_index_in_middle, num_grid_columns
    ])

    for k in range(num_channels):
        if numpy.all(numpy.isnan(brightness_temp_matrix_kelvins[..., k])):
            continue

        assert not numpy.any(
            numpy.isnan(brightness_temp_matrix_kelvins[..., k])
        )

        i_start = first_row_index_in_middle
        i_end = last_row_index_in_middle
        j_start = first_column_index_in_middle
        j_end = last_column_index_in_middle

        mean_middle_bt_kelvins = numpy.mean(
            brightness_temp_matrix_kelvins[i_start:i_end, j_start:j_end, k]
        )
        cloud_top_height_m_agl = misc_utils.brightness_temp_to_cloud_top_height(
            numpy.array([mean_middle_bt_kelvins])
        )[0]

        (
            corrected_center_longitude_deg_e, corrected_center_latitude_deg_n
        ) = parallax.get_parallax_corrected_lonlats(
            sat_lon=satellite_longitude_deg_e, sat_lat=0.,
            sat_alt=satellite_altitude_m_agl,
            lon=center_longitude_deg_e, lat=center_latitude_deg_n,
            height=cloud_top_height_m_agl
        )

        center_longitude_deg_e = lng_conversion.convert_lng_positive_in_west(
            center_longitude_deg_e
        )
        corrected_center_longitude_deg_e = (
            lng_conversion.convert_lng_positive_in_west(
                corrected_center_longitude_deg_e
            )
        )

        if numpy.absolute(
                corrected_center_longitude_deg_e - center_longitude_deg_e
        ) > 1:
            center_longitude_deg_e = (
                lng_conversion.convert_lng_negative_in_west(
                    center_longitude_deg_e
                )
            )

            corrected_center_longitude_deg_e = (
                lng_conversion.convert_lng_negative_in_west(
                    corrected_center_longitude_deg_e
                )
            )

        assert numpy.absolute(
            corrected_center_longitude_deg_e - center_longitude_deg_e
        ) < 1

        print((
            'Parallax correction for {0:.3f}-micron data at {1:d}th time = '
            '{2:.4f} deg north, {3:.4f} deg east'
        ).format(
            METRES_TO_MICRONS *
            stx.coords[satellite_utils.LOW_RES_WAVELENGTH_DIM].values[k],
            i + 1,
            corrected_center_latitude_deg_n - center_latitude_deg_n,
            corrected_center_longitude_deg_e - center_longitude_deg_e
        ))

        corrected_latitudes_deg_n = (
            stx[satellite_utils.LATITUDE_LOW_RES_KEY].values[i, :] +
            (corrected_center_latitude_deg_n - center_latitude_deg_n)
        )
        corrected_longitudes_deg_e = (
            stx[satellite_utils.LONGITUDE_LOW_RES_KEY].values[i, :] +
            (corrected_center_longitude_deg_e - center_longitude_deg_e)
        )
        orig_point_tuple = (
            corrected_latitudes_deg_n,
            corrected_longitudes_deg_e
        )

        interp_object = RegularGridInterpolator(
            points=orig_point_tuple,
            values=brightness_temp_matrix_kelvins[..., k],
            method='linear', bounds_error=False, fill_value=None
        )

        (
            query_longitude_matrix_deg_e, query_latitude_matrix_deg_n
        ) = numpy.meshgrid(
            stx[satellite_utils.LONGITUDE_LOW_RES_KEY].values[i, :],
            stx[satellite_utils.LATITUDE_LOW_RES_KEY].values[i, :]
        )

        query_point_matrix = numpy.stack(
            (query_latitude_matrix_deg_n, query_longitude_matrix_deg_e),
            axis=-1
        )

        new_matrix = interp_object(query_point_matrix)
        abs_diff_matrix = numpy.absolute(
            new_matrix - brightness_temp_matrix_kelvins[..., k]
        )

        print((
            'Min/mean/max absolute BT difference between original and '
            'corrected = {0:.4f}/{1:.4f}/{2:.4f} K'
        ).format(
            numpy.min(abs_diff_matrix),
            numpy.mean(abs_diff_matrix),
            numpy.max(abs_diff_matrix)
        ))

        brightness_temp_matrix_kelvins[..., k] = new_matrix + 0.

    return brightness_temp_matrix_kelvins


def _parallax_correct_one_table(satellite_table_xarray,
                                neigh_half_width_low_res_px):
    """Parallax-corrects one data table.

    :param satellite_table_xarray: xarray table returned by
        `satellite_io.read_file`, i.e., for a single file.
    :param neigh_half_width_low_res_px: See documentation at top of file.
    :return: satellite_table_xarray: Same as input but parallax-corrected.
    """

    stx = satellite_table_xarray

    bidirectional_reflectance_matrix = (
        stx[satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY].values
    )
    brightness_temp_matrix_kelvins = (
        stx[satellite_utils.BRIGHTNESS_TEMPERATURE_KEY].values
    )
    num_times = bidirectional_reflectance_matrix.shape[0]

    for i in range(num_times):
        bidirectional_reflectance_matrix[i, ...] = (
            _parallax_correct_high_res_one_time(
                satellite_table_xarray=satellite_table_xarray, time_index=i
            )
        )

        brightness_temp_matrix_kelvins[i, ...] = (
            _parallax_correct_low_res_one_time(
                satellite_table_xarray=satellite_table_xarray, time_index=i,
                neigh_half_width_px=neigh_half_width_low_res_px
            )
        )

    main_data_dict = {}

    for var_name in satellite_table_xarray.data_vars:
        if var_name == satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY:
            main_data_dict[var_name] = (
                satellite_table_xarray[var_name].dims,
                bidirectional_reflectance_matrix
            )
            continue

        if var_name == satellite_utils.BRIGHTNESS_TEMPERATURE_KEY:
            main_data_dict[var_name] = (
                satellite_table_xarray[var_name].dims,
                brightness_temp_matrix_kelvins
            )
            continue

        main_data_dict[var_name] = (
            satellite_table_xarray[var_name].dims,
            satellite_table_xarray[var_name].values
        )

    metadata_dict = {}
    for coord_name in satellite_table_xarray.coords:
        metadata_dict[coord_name] = (
            satellite_table_xarray.coords[coord_name].values
        )

    return xarray.Dataset(data_vars=main_data_dict, coords=metadata_dict)


def _run(input_dir_name, cyclone_id_strings, neigh_half_width_low_res_px,
         output_dir_name):
    """Parallax-corrects satellite images.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param cyclone_id_strings: Same.
    :param neigh_half_width_low_res_px: Same.
    :param output_dir_name: Same.
    """

    error_checking.assert_is_greater(neigh_half_width_low_res_px, 0)

    input_file_names = []

    for this_cyclone_id_string in cyclone_id_strings:
        input_file_names += satellite_io.find_files_one_cyclone(
            directory_name=input_dir_name,
            cyclone_id_string=this_cyclone_id_string,
            raise_error_if_all_missing=True
        )

    for this_input_file_name in input_file_names:
        print('Reading data from: "{0:s}"...'.format(this_input_file_name))
        satellite_table_xarray = satellite_io.read_file(this_input_file_name)

        satellite_table_xarray = _parallax_correct_one_table(
            satellite_table_xarray=satellite_table_xarray,
            neigh_half_width_low_res_px=neigh_half_width_low_res_px
        )

        this_output_file_name = satellite_io.find_file(
            directory_name=output_dir_name,
            cyclone_id_string=
            satellite_io.file_name_to_cyclone_id(this_input_file_name),
            valid_date_string=
            satellite_io.file_name_to_date(this_input_file_name),
            raise_error_if_missing=False
        )

        print('Writing data to: "{0:s}"...'.format(this_output_file_name))
        satellite_io.write_file(
            satellite_table_xarray=satellite_table_xarray,
            zarr_file_name=this_output_file_name
        )

        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        cyclone_id_strings=getattr(INPUT_ARG_OBJECT, CYCLONE_IDS_ARG_NAME),
        neigh_half_width_low_res_px=getattr(
            INPUT_ARG_OBJECT, NEIGH_HALF_WIDTH_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
