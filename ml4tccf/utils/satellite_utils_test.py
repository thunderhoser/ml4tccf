"""Unit tests for satellite_utils.py."""

import copy
import unittest
import numpy
import xarray
from ml4tccf.utils import satellite_utils

NAN = numpy.nan
TOLERANCE = 1e-6
MICRONS_TO_METRES = 1e-6

VALID_TIMES_UNIX_SEC = numpy.array([int(1e9), int(1e9) + 900], dtype=int)
LOW_RES_WAVELENGTHS_MICRONS = numpy.array([3.3, 6.7, 10.8])
LOW_RES_WAVELENGTHS_METRES = MICRONS_TO_METRES * LOW_RES_WAVELENGTHS_MICRONS
HIGH_RES_WAVELENGTHS_MICRONS = numpy.array([0.64])
HIGH_RES_WAVELENGTHS_METRES = MICRONS_TO_METRES * HIGH_RES_WAVELENGTHS_MICRONS

LOW_RES_METADATA_DICT = {
    satellite_utils.TIME_DIM: VALID_TIMES_UNIX_SEC,
    satellite_utils.LOW_RES_WAVELENGTH_DIM: LOW_RES_WAVELENGTHS_METRES
}
HIGH_RES_METADATA_DICT = {
    satellite_utils.TIME_DIM: VALID_TIMES_UNIX_SEC,
    satellite_utils.HIGH_RES_WAVELENGTH_DIM: HIGH_RES_WAVELENGTHS_METRES
}

LOW_RES_MATRIX_DIM_KEYS = (
    satellite_utils.TIME_DIM, satellite_utils.LOW_RES_ROW_DIM,
    satellite_utils.LOW_RES_COLUMN_DIM, satellite_utils.LOW_RES_WAVELENGTH_DIM
)
HIGH_RES_MATRIX_DIM_KEYS = (
    satellite_utils.TIME_DIM, satellite_utils.HIGH_RES_ROW_DIM,
    satellite_utils.HIGH_RES_COLUMN_DIM, satellite_utils.HIGH_RES_WAVELENGTH_DIM
)

# The following constants are used to test quality_control_low_res.
FIRST_MAX_BAD_PIXELS_PER_TIME_CHANNEL = 1
SECOND_MAX_BAD_PIXELS_PER_TIME_CHANNEL = 10

# The 0 and 500 are invalid, as are the NaN's.
FIRST_2D_MATRIX_KELVINS = numpy.array([
    [171, 172, 173, 174, 175, 176],
    [172, 173, NAN, 175, 176, 177],
    [173, 174, NAN, 0, 177, 178],
    [174, 175, NAN, 500, NAN, NAN],
    [175, 176, 177, 178, 179, 180],
    [175, 176, 177, 178, 179, 180]
], dtype=float)

SECOND_2D_MATRIX_KELVINS = numpy.array([
    [172, 174, 176, 178, 180, 182],
    [174, 176, 178, 180, 182, 184],
    [176, 178, 180, 182, 184, 186],
    [178, 180, 182, 184, 186, 188],
    [180, 182, 184, 186, 188, 190],
    [180, 182, 184, 186, 188, 190]
], dtype=float)

THIRD_2D_MATRIX_KELVINS = numpy.array([
    [173, 176, 179, 182, 185, 188],
    [176, 179, 182, 185, 188, 191],
    [179, 182, 185, 188, 191, 194],
    [182, 185, 188, 191, 194, 197],
    [180, 182, 184, 186, 188, 190],
    [180, 182, 184, 186, 188, 190]
], dtype=float)

THIS_3D_MATRIX_KELVINS = numpy.stack((
    FIRST_2D_MATRIX_KELVINS, SECOND_2D_MATRIX_KELVINS, THIRD_2D_MATRIX_KELVINS
), axis=-1)

THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS = numpy.stack(
    (THIS_3D_MATRIX_KELVINS + 5, THIS_3D_MATRIX_KELVINS + 10), axis=0
)

THIS_MAIN_DATA_DICT = {
    satellite_utils.BRIGHTNESS_TEMPERATURE_KEY: (
        LOW_RES_MATRIX_DIM_KEYS, THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS
    )
}
TABLE_BEFORE_LOW_RES_QC = xarray.Dataset(
    data_vars=THIS_MAIN_DATA_DICT, coords=LOW_RES_METADATA_DICT
)

THIS_REFLECTANCE_MATRIX = THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS[..., [0]] + 0.
THIS_REFLECTANCE_MATRIX[
    THIS_REFLECTANCE_MATRIX < satellite_utils.MIN_BRIGHTNESS_TEMP_KELVINS
] = numpy.nan
THIS_REFLECTANCE_MATRIX[
    THIS_REFLECTANCE_MATRIX > satellite_utils.MAX_BRIGHTNESS_TEMP_KELVINS
] = numpy.nan

THIS_MAIN_DATA_DICT = {
    satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY: (
        HIGH_RES_MATRIX_DIM_KEYS, THIS_REFLECTANCE_MATRIX
    )
}
TABLE_BEFORE_HIGH_RES_QC = xarray.Dataset(
    data_vars=THIS_MAIN_DATA_DICT, coords=HIGH_RES_METADATA_DICT
)

FIRST_2D_MATRIX_KELVINS[:] = NAN
THIS_3D_MATRIX_KELVINS = numpy.stack((
    FIRST_2D_MATRIX_KELVINS, SECOND_2D_MATRIX_KELVINS, THIRD_2D_MATRIX_KELVINS
), axis=-1)

THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS = numpy.stack(
    (THIS_3D_MATRIX_KELVINS + 5, THIS_3D_MATRIX_KELVINS + 10), axis=0
)

THIS_MAIN_DATA_DICT = {
    satellite_utils.BRIGHTNESS_TEMPERATURE_KEY: (
        LOW_RES_MATRIX_DIM_KEYS, THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS
    )
}
FIRST_TABLE_AFTER_LOW_RES_QC = xarray.Dataset(
    data_vars=THIS_MAIN_DATA_DICT, coords=LOW_RES_METADATA_DICT
)

THIS_REFLECTANCE_MATRIX = THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS[..., [0]] + 0.
THIS_REFLECTANCE_MATRIX[
    THIS_REFLECTANCE_MATRIX < satellite_utils.MIN_BRIGHTNESS_TEMP_KELVINS
] = numpy.nan
THIS_REFLECTANCE_MATRIX[
    THIS_REFLECTANCE_MATRIX > satellite_utils.MAX_BRIGHTNESS_TEMP_KELVINS
] = numpy.nan

THIS_MAIN_DATA_DICT = {
    satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY: (
        HIGH_RES_MATRIX_DIM_KEYS, THIS_REFLECTANCE_MATRIX
    )
}
FIRST_TABLE_AFTER_HIGH_RES_QC = xarray.Dataset(
    data_vars=THIS_MAIN_DATA_DICT, coords=HIGH_RES_METADATA_DICT
)

FIRST_2D_MATRIX_KELVINS = numpy.array([
    [171, 172, 173, 174, 175, 176],
    [172, 173, 173, 175, 176, 177],
    [173, 174, 174, 175, 177, 178],
    [174, 175, 175, 178, 177, 178],
    [175, 176, 177, 178, 179, 180],
    [175, 176, 177, 178, 179, 180]
], dtype=float)

THIS_3D_MATRIX_KELVINS = numpy.stack((
    FIRST_2D_MATRIX_KELVINS, SECOND_2D_MATRIX_KELVINS, THIRD_2D_MATRIX_KELVINS
), axis=-1)

THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS = numpy.stack(
    (THIS_3D_MATRIX_KELVINS + 5, THIS_3D_MATRIX_KELVINS + 10), axis=0
)

THIS_MAIN_DATA_DICT = {
    satellite_utils.BRIGHTNESS_TEMPERATURE_KEY: (
        LOW_RES_MATRIX_DIM_KEYS, THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS
    )
}
SECOND_TABLE_AFTER_LOW_RES_QC = xarray.Dataset(
    data_vars=THIS_MAIN_DATA_DICT, coords=LOW_RES_METADATA_DICT
)

THIS_REFLECTANCE_MATRIX = THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS[..., [0]] + 0.
THIS_REFLECTANCE_MATRIX[
    THIS_REFLECTANCE_MATRIX < satellite_utils.MIN_BRIGHTNESS_TEMP_KELVINS
] = numpy.nan
THIS_REFLECTANCE_MATRIX[
    THIS_REFLECTANCE_MATRIX > satellite_utils.MAX_BRIGHTNESS_TEMP_KELVINS
] = numpy.nan

THIS_MAIN_DATA_DICT = {
    satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY: (
        HIGH_RES_MATRIX_DIM_KEYS, THIS_REFLECTANCE_MATRIX
    )
}
SECOND_TABLE_AFTER_HIGH_RES_QC = xarray.Dataset(
    data_vars=THIS_MAIN_DATA_DICT, coords=HIGH_RES_METADATA_DICT
)

# The following constants are used to test subset_grid.
FIRST_NUM_ROWS_TO_KEEP = 2
FIRST_NUM_COLUMNS_TO_KEEP = 2
SECOND_NUM_ROWS_TO_KEEP = 2
SECOND_NUM_COLUMNS_TO_KEEP = 4
THIRD_NUM_ROWS_TO_KEEP = 2
THIRD_NUM_COLUMNS_TO_KEEP = 6

TABLE_BEFORE_SUBSET_GRID_LOW_RES = copy.deepcopy(TABLE_BEFORE_LOW_RES_QC)
TABLE_BEFORE_SUBSET_GRID_HIGH_RES = copy.deepcopy(TABLE_BEFORE_HIGH_RES_QC)

FIRST_2D_MATRIX_KELVINS = numpy.array([
    [NAN, 0],
    [NAN, 500]
], dtype=float)

SECOND_2D_MATRIX_KELVINS = numpy.array([
    [180, 182],
    [182, 184]
], dtype=float)

THIRD_2D_MATRIX_KELVINS = numpy.array([
    [185, 188],
    [188, 191]
], dtype=float)

THIS_3D_MATRIX_KELVINS = numpy.stack((
    FIRST_2D_MATRIX_KELVINS, SECOND_2D_MATRIX_KELVINS, THIRD_2D_MATRIX_KELVINS
), axis=-1)

THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS = numpy.stack(
    (THIS_3D_MATRIX_KELVINS + 5, THIS_3D_MATRIX_KELVINS + 10), axis=0
)

THIS_MAIN_DATA_DICT = {
    satellite_utils.BRIGHTNESS_TEMPERATURE_KEY: (
        LOW_RES_MATRIX_DIM_KEYS, THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS
    )
}
FIRST_TABLE_AFTER_SUBSET_GRID_LOW_RES = xarray.Dataset(
    data_vars=THIS_MAIN_DATA_DICT, coords=LOW_RES_METADATA_DICT
)

THIS_REFLECTANCE_MATRIX = THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS[..., [0]] + 0.
THIS_REFLECTANCE_MATRIX[
    THIS_REFLECTANCE_MATRIX < satellite_utils.MIN_BRIGHTNESS_TEMP_KELVINS
] = numpy.nan
THIS_REFLECTANCE_MATRIX[
    THIS_REFLECTANCE_MATRIX > satellite_utils.MAX_BRIGHTNESS_TEMP_KELVINS
] = numpy.nan

THIS_MAIN_DATA_DICT = {
    satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY: (
        HIGH_RES_MATRIX_DIM_KEYS, THIS_REFLECTANCE_MATRIX
    )
}
FIRST_TABLE_AFTER_SUBSET_GRID_HIGH_RES = xarray.Dataset(
    data_vars=THIS_MAIN_DATA_DICT, coords=HIGH_RES_METADATA_DICT
)

FIRST_2D_MATRIX_KELVINS = numpy.array([
    [174, NAN, 0, 177],
    [175, NAN, 500, NAN]
], dtype=float)

SECOND_2D_MATRIX_KELVINS = numpy.array([
    [178, 180, 182, 184],
    [180, 182, 184, 186]
], dtype=float)

THIRD_2D_MATRIX_KELVINS = numpy.array([
    [182, 185, 188, 191],
    [185, 188, 191, 194]
], dtype=float)

THIS_3D_MATRIX_KELVINS = numpy.stack((
    FIRST_2D_MATRIX_KELVINS, SECOND_2D_MATRIX_KELVINS, THIRD_2D_MATRIX_KELVINS
), axis=-1)

THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS = numpy.stack(
    (THIS_3D_MATRIX_KELVINS + 5, THIS_3D_MATRIX_KELVINS + 10), axis=0
)

THIS_MAIN_DATA_DICT = {
    satellite_utils.BRIGHTNESS_TEMPERATURE_KEY: (
        LOW_RES_MATRIX_DIM_KEYS, THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS
    )
}
SECOND_TABLE_AFTER_SUBSET_GRID_LOW_RES = xarray.Dataset(
    data_vars=THIS_MAIN_DATA_DICT, coords=LOW_RES_METADATA_DICT
)

THIS_REFLECTANCE_MATRIX = THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS[..., [0]] + 0.
THIS_REFLECTANCE_MATRIX[
    THIS_REFLECTANCE_MATRIX < satellite_utils.MIN_BRIGHTNESS_TEMP_KELVINS
] = numpy.nan
THIS_REFLECTANCE_MATRIX[
    THIS_REFLECTANCE_MATRIX > satellite_utils.MAX_BRIGHTNESS_TEMP_KELVINS
] = numpy.nan

THIS_MAIN_DATA_DICT = {
    satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY: (
        HIGH_RES_MATRIX_DIM_KEYS, THIS_REFLECTANCE_MATRIX
    )
}
SECOND_TABLE_AFTER_SUBSET_GRID_HIGH_RES = xarray.Dataset(
    data_vars=THIS_MAIN_DATA_DICT, coords=HIGH_RES_METADATA_DICT
)

# The following constants are used to test subset_wavelengths.
FIRST_LOW_RES_WAVELENGTHS_TO_KEEP_MICRONS = numpy.array([10.8, 3.3])
FIRST_HIGH_RES_WAVELENGTHS_TO_KEEP_MICRONS = numpy.array([])
SECOND_LOW_RES_WAVELENGTHS_TO_KEEP_MICRONS = numpy.array([6.66])
SECOND_HIGH_RES_WAVELENGTHS_TO_KEEP_MICRONS = numpy.array([9.9])

TABLE_BEFORE_SUBSET_WVL_LOW_RES = copy.deepcopy(
    TABLE_BEFORE_LOW_RES_QC
)
TABLE_BEFORE_SUBSET_WVL_HIGH_RES = copy.deepcopy(
    TABLE_BEFORE_HIGH_RES_QC
)

FIRST_2D_MATRIX_KELVINS = numpy.array([
    [171, 172, 173, 174, 175, 176],
    [172, 173, NAN, 175, 176, 177],
    [173, 174, NAN, 0, 177, 178],
    [174, 175, NAN, 500, NAN, NAN],
    [175, 176, 177, 178, 179, 180],
    [175, 176, 177, 178, 179, 180]
], dtype=float)

THIRD_2D_MATRIX_KELVINS = numpy.array([
    [173, 176, 179, 182, 185, 188],
    [176, 179, 182, 185, 188, 191],
    [179, 182, 185, 188, 191, 194],
    [182, 185, 188, 191, 194, 197],
    [180, 182, 184, 186, 188, 190],
    [180, 182, 184, 186, 188, 190]
], dtype=float)

THIS_3D_MATRIX_KELVINS = numpy.stack((
    THIRD_2D_MATRIX_KELVINS, FIRST_2D_MATRIX_KELVINS
), axis=-1)

THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS = numpy.stack(
    (THIS_3D_MATRIX_KELVINS + 5, THIS_3D_MATRIX_KELVINS + 10), axis=0
)

THIS_MAIN_DATA_DICT = {
    satellite_utils.BRIGHTNESS_TEMPERATURE_KEY: (
        LOW_RES_MATRIX_DIM_KEYS, THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS
    )
}
THIS_METADATA_DICT = {
    satellite_utils.TIME_DIM: VALID_TIMES_UNIX_SEC,
    satellite_utils.LOW_RES_WAVELENGTH_DIM:
        MICRONS_TO_METRES * FIRST_LOW_RES_WAVELENGTHS_TO_KEEP_MICRONS
}
FIRST_TABLE_AFTER_SUBSET_WVL_LOW_RES = xarray.Dataset(
    data_vars=THIS_MAIN_DATA_DICT, coords=THIS_METADATA_DICT
)

THIS_REFLECTANCE_MATRIX = numpy.full(
    THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS.shape[:-1] + (0,),
    0.
)
THIS_MAIN_DATA_DICT = {
    satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY: (
        HIGH_RES_MATRIX_DIM_KEYS, THIS_REFLECTANCE_MATRIX
    )
}
THIS_METADATA_DICT = {
    satellite_utils.TIME_DIM: VALID_TIMES_UNIX_SEC,
    satellite_utils.HIGH_RES_WAVELENGTH_DIM:
        MICRONS_TO_METRES * FIRST_HIGH_RES_WAVELENGTHS_TO_KEEP_MICRONS
}
FIRST_TABLE_AFTER_SUBSET_WVL_HIGH_RES = xarray.Dataset(
    data_vars=THIS_MAIN_DATA_DICT, coords=THIS_METADATA_DICT
)

# The following constants are used to test subset_to_multiple_time_windows.
FIRST_START_TIMES_UNIX_SEC = numpy.array(
    [0, int(1e9) + 450, int(1e10)], dtype=int
)
FIRST_END_TIMES_UNIX_SEC = numpy.array(
    [int(1e9) + 1, int(1e9) + 1000, int(1e11)], dtype=int
)

SECOND_START_TIMES_UNIX_SEC = numpy.array(
    [int(1e10), int(1e9) + 450], dtype=int
)
SECOND_END_TIMES_UNIX_SEC = numpy.array(
    [int(1e11), int(1e9) + 1000], dtype=int
)

THIRD_START_TIMES_UNIX_SEC = numpy.array(
    [int(1e10)], dtype=int
)
THIRD_END_TIMES_UNIX_SEC = numpy.array(
    [int(1e11)], dtype=int
)

FIRST_2D_MATRIX_KELVINS = numpy.array([
    [171, 172, 173, 174, 175, 176],
    [172, 173, NAN, 175, 176, 177],
    [173, 174, NAN, 0, 177, 178],
    [174, 175, NAN, 500, NAN, NAN],
    [175, 176, 177, 178, 179, 180],
    [175, 176, 177, 178, 179, 180]
], dtype=float)

SECOND_2D_MATRIX_KELVINS = numpy.array([
    [172, 174, 176, 178, 180, 182],
    [174, 176, 178, 180, 182, 184],
    [176, 178, 180, 182, 184, 186],
    [178, 180, 182, 184, 186, 188],
    [180, 182, 184, 186, 188, 190],
    [180, 182, 184, 186, 188, 190]
], dtype=float)

THIRD_2D_MATRIX_KELVINS = numpy.array([
    [173, 176, 179, 182, 185, 188],
    [176, 179, 182, 185, 188, 191],
    [179, 182, 185, 188, 191, 194],
    [182, 185, 188, 191, 194, 197],
    [180, 182, 184, 186, 188, 190],
    [180, 182, 184, 186, 188, 190]
], dtype=float)

THIS_3D_MATRIX_KELVINS = numpy.stack((
    FIRST_2D_MATRIX_KELVINS, SECOND_2D_MATRIX_KELVINS, THIRD_2D_MATRIX_KELVINS
), axis=-1)

THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS = numpy.stack(
    (THIS_3D_MATRIX_KELVINS + 5, THIS_3D_MATRIX_KELVINS + 10), axis=0
)

THIS_REFLECTANCE_MATRIX = THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS[..., [0]] + 0.
THIS_REFLECTANCE_MATRIX[
    THIS_REFLECTANCE_MATRIX < satellite_utils.MIN_BRIGHTNESS_TEMP_KELVINS
] = numpy.nan
THIS_REFLECTANCE_MATRIX[
    THIS_REFLECTANCE_MATRIX > satellite_utils.MAX_BRIGHTNESS_TEMP_KELVINS
] = numpy.nan

THIS_MAIN_DATA_DICT = {
    satellite_utils.BRIGHTNESS_TEMPERATURE_KEY: (
        LOW_RES_MATRIX_DIM_KEYS, THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS
    ),
    satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY: (
        HIGH_RES_MATRIX_DIM_KEYS, THIS_REFLECTANCE_MATRIX
    )
}
THIS_METADATA_DICT = copy.deepcopy(LOW_RES_METADATA_DICT)
THIS_METADATA_DICT.update(HIGH_RES_METADATA_DICT)

TABLE_BEFORE_SUBSET_WINDOWS = xarray.Dataset(
    data_vars=THIS_MAIN_DATA_DICT, coords=THIS_METADATA_DICT
)

FIRST_TABLE_AFTER_SUBSET_WINDOWS = copy.deepcopy(TABLE_BEFORE_SUBSET_WINDOWS)

THIS_MAIN_DATA_DICT = {
    satellite_utils.BRIGHTNESS_TEMPERATURE_KEY: (
        LOW_RES_MATRIX_DIM_KEYS, THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS[[1], ...]
    ),
    satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY: (
        HIGH_RES_MATRIX_DIM_KEYS, THIS_REFLECTANCE_MATRIX[[1], ...]
    )
}
THIS_METADATA_DICT = copy.deepcopy(LOW_RES_METADATA_DICT)
THIS_METADATA_DICT.update(HIGH_RES_METADATA_DICT)
THIS_METADATA_DICT.update({
    satellite_utils.TIME_DIM: VALID_TIMES_UNIX_SEC[[1]]
})
SECOND_TABLE_AFTER_SUBSET_WINDOWS = xarray.Dataset(
    data_vars=THIS_MAIN_DATA_DICT, coords=THIS_METADATA_DICT
)

THIS_MAIN_DATA_DICT = {
    satellite_utils.BRIGHTNESS_TEMPERATURE_KEY: (
        LOW_RES_MATRIX_DIM_KEYS, THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS[[], ...]
    ),
    satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY: (
        HIGH_RES_MATRIX_DIM_KEYS, THIS_REFLECTANCE_MATRIX[[], ...]
    )
}
THIS_METADATA_DICT = copy.deepcopy(LOW_RES_METADATA_DICT)
THIS_METADATA_DICT.update(HIGH_RES_METADATA_DICT)
THIS_METADATA_DICT.update({
    satellite_utils.TIME_DIM: VALID_TIMES_UNIX_SEC[[]]
})
THIRD_TABLE_AFTER_SUBSET_WINDOWS = xarray.Dataset(
    data_vars=THIS_MAIN_DATA_DICT, coords=THIS_METADATA_DICT
)

# The following constants are used to test _compute_interpolation_gap.
SOURCE_TIMES_UNIX_SEC = numpy.array(
    [10, 20, 30, 40, 50, 65, 75, 88, 99], dtype=int
)
FIRST_TARGET_TIME_UNIX_SEC = 8
SECOND_TARGET_TIME_UNIX_SEC = 150
THIRD_TARGET_TIME_UNIX_SEC = 75
FOURTH_TARGET_TIME_UNIX_SEC = 76

FIRST_INTERP_GAP_SEC = 4
SECOND_INTERP_GAP_SEC = 102
THIRD_INTERP_GAP_SEC = 0
FOURTH_INTERP_GAP_SEC = 13

# The following constants are used to test subset_times.
FIRST_DESIRED_TIMES_UNIX_SEC = numpy.array(
    [int(1e9) + 850, int(1e9) - 20], dtype=int
)
FIRST_TIME_TOLERANCES_SEC = numpy.array([50, 20], dtype=int)
FIRST_MAX_NUM_MISSING_TIMES = 0
FIRST_MAX_INTERP_GAPS_SEC = numpy.array([0, 0], dtype=int)

SECOND_DESIRED_TIMES_UNIX_SEC = numpy.array(
    [int(1e9) + 850, int(1e9) - 20], dtype=int
)
SECOND_TIME_TOLERANCES_SEC = numpy.array([49, 20], dtype=int)
SECOND_MAX_NUM_MISSING_TIMES = 1
SECOND_MAX_INTERP_GAPS_SEC = numpy.array([1000, 0], dtype=int)

THIRD_DESIRED_TIMES_UNIX_SEC = numpy.array(
    [int(1e9) + 850, int(1e9) - 20], dtype=int
)
THIRD_TIME_TOLERANCES_SEC = numpy.array([49, 19], dtype=int)
THIRD_MAX_NUM_MISSING_TIMES = 2
THIRD_MAX_INTERP_GAPS_SEC = numpy.array([1000, 100], dtype=int)

FOURTH_DESIRED_TIMES_UNIX_SEC = numpy.array(
    [int(1e9) + 850, int(1e9) - 20], dtype=int
)
FOURTH_TIME_TOLERANCES_SEC = numpy.array([49, 19], dtype=int)
FOURTH_MAX_NUM_MISSING_TIMES = 2
FOURTH_MAX_INTERP_GAPS_SEC = numpy.array([2, 100], dtype=int)

FIFTH_DESIRED_TIMES_UNIX_SEC = numpy.array(
    [int(1e9) + 850, int(1e9) - 20], dtype=int
)
FIFTH_TIME_TOLERANCES_SEC = numpy.array([49, 19], dtype=int)
FIFTH_MAX_NUM_MISSING_TIMES = 1
FIFTH_MAX_INTERP_GAPS_SEC = numpy.array([2, 100], dtype=int)

SIXTH_DESIRED_TIMES_UNIX_SEC = numpy.array(
    [int(1e9) + 850, int(1e9) - 20], dtype=int
)
SIXTH_TIME_TOLERANCES_SEC = numpy.array([49, 19], dtype=int)
SIXTH_MAX_NUM_MISSING_TIMES = 1
SIXTH_MAX_INTERP_GAPS_SEC = numpy.array([2, 2], dtype=int)

FIRST_2D_MATRIX_KELVINS = numpy.array([
    [171, 172, 173, 174, 175, 176],
    [172, 173, NAN, 175, 176, 177],
    [173, 174, NAN, NAN, 177, 178],
    [174, 175, NAN, NAN, NAN, NAN],
    [175, 176, 177, 178, 179, 180],
    [175, 176, 177, 178, 179, 180]
], dtype=float)

SECOND_2D_MATRIX_KELVINS = numpy.array([
    [172, 174, 176, 178, 180, 182],
    [174, 176, 178, 180, 182, 184],
    [176, 178, 180, 182, 184, 186],
    [178, 180, 182, 184, 186, 188],
    [180, 182, 184, 186, 188, 190],
    [180, 182, 184, 186, 188, 190]
], dtype=float)

THIRD_2D_MATRIX_KELVINS = numpy.array([
    [173, 176, 179, 182, 185, 188],
    [176, 179, 182, 185, 188, 191],
    [179, 182, 185, 188, 191, 194],
    [182, 185, 188, 191, 194, 197],
    [180, 182, 184, 186, 188, 190],
    [180, 182, 184, 186, 188, 190]
], dtype=float)

THIS_3D_MATRIX_KELVINS = numpy.stack((
    FIRST_2D_MATRIX_KELVINS, SECOND_2D_MATRIX_KELVINS, THIRD_2D_MATRIX_KELVINS
), axis=-1)

THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS = numpy.stack(
    (THIS_3D_MATRIX_KELVINS + 5, THIS_3D_MATRIX_KELVINS + 10), axis=0
)

THIS_MAIN_DATA_DICT = {
    satellite_utils.BRIGHTNESS_TEMPERATURE_KEY: (
        LOW_RES_MATRIX_DIM_KEYS, THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS
    ),
    satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY: (
        HIGH_RES_MATRIX_DIM_KEYS,
        THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS[..., [0]] + 0.
    )
}
THIS_METADATA_DICT = copy.deepcopy(LOW_RES_METADATA_DICT)
THIS_METADATA_DICT.update(HIGH_RES_METADATA_DICT)

TABLE_BEFORE_SUBSET_TIMES = xarray.Dataset(
    data_vars=THIS_MAIN_DATA_DICT, coords=THIS_METADATA_DICT
)

THIS_MAIN_DATA_DICT = {
    satellite_utils.BRIGHTNESS_TEMPERATURE_KEY: (
        LOW_RES_MATRIX_DIM_KEYS,
        THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS[[1, 0], ...]
    ),
    satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY: (
        HIGH_RES_MATRIX_DIM_KEYS, THIS_REFLECTANCE_MATRIX[[1, 0], ...]
    )
}
THIS_METADATA_DICT[satellite_utils.TIME_DIM] = FIRST_DESIRED_TIMES_UNIX_SEC
FIRST_TABLE_AFTER_SUBSET_TIMES = xarray.Dataset(
    data_vars=THIS_MAIN_DATA_DICT, coords=THIS_METADATA_DICT
)

THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS = numpy.stack(
    (THIS_3D_MATRIX_KELVINS + 5, THIS_3D_MATRIX_KELVINS + 5 + 5 * 850. / 900),
    axis=0
)
THIS_REFLECTANCE_MATRIX = THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS[..., [0]] + 0.
THIS_MAIN_DATA_DICT = {
    satellite_utils.BRIGHTNESS_TEMPERATURE_KEY: (
        LOW_RES_MATRIX_DIM_KEYS,
        THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS[[1, 0], ...]
    ),
    satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY: (
        HIGH_RES_MATRIX_DIM_KEYS, THIS_REFLECTANCE_MATRIX[[1, 0], ...]
    )
}
THIS_METADATA_DICT[satellite_utils.TIME_DIM] = SECOND_DESIRED_TIMES_UNIX_SEC
SECOND_TABLE_AFTER_SUBSET_TIMES = xarray.Dataset(
    data_vars=THIS_MAIN_DATA_DICT, coords=THIS_METADATA_DICT
)

THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS = numpy.stack((
    THIS_3D_MATRIX_KELVINS + 5 - 5 * 20. / 900,
    THIS_3D_MATRIX_KELVINS + 5 + 5 * 850. / 900
), axis=0)

THIS_REFLECTANCE_MATRIX = THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS[..., [0]] + 0.
THIS_MAIN_DATA_DICT = {
    satellite_utils.BRIGHTNESS_TEMPERATURE_KEY: (
        LOW_RES_MATRIX_DIM_KEYS,
        THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS[[1, 0], ...]
    ),
    satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY: (
        HIGH_RES_MATRIX_DIM_KEYS, THIS_REFLECTANCE_MATRIX[[1, 0], ...]
    )
}
THIS_METADATA_DICT[satellite_utils.TIME_DIM] = THIRD_DESIRED_TIMES_UNIX_SEC
THIRD_TABLE_AFTER_SUBSET_TIMES = xarray.Dataset(
    data_vars=THIS_MAIN_DATA_DICT, coords=THIS_METADATA_DICT
)

THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS[1, ...] = numpy.nan
THIS_REFLECTANCE_MATRIX = THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS[..., [0]] + 0.
THIS_MAIN_DATA_DICT = {
    satellite_utils.BRIGHTNESS_TEMPERATURE_KEY: (
        LOW_RES_MATRIX_DIM_KEYS,
        THIS_BRIGHTNESS_TEMP_MATRIX_KELVINS[[1, 0], ...]
    ),
    satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY: (
        HIGH_RES_MATRIX_DIM_KEYS, THIS_REFLECTANCE_MATRIX[[1, 0], ...]
    )
}
THIS_METADATA_DICT[satellite_utils.TIME_DIM] = FOURTH_DESIRED_TIMES_UNIX_SEC
FOURTH_TABLE_AFTER_SUBSET_TIMES = xarray.Dataset(
    data_vars=THIS_MAIN_DATA_DICT, coords=THIS_METADATA_DICT
)


def _compare_satellite_tables(first_table_xarray, second_table_xarray):
    """Compares two xarray tables with satellite data.

    :param first_table_xarray: First table.
    :param second_table_xarray: Second table.
    :return: are_tables_equal: Boolean flag.
    """

    integer_keys = [
        satellite_utils.HIGH_RES_ROW_DIM, satellite_utils.HIGH_RES_COLUMN_DIM,
        satellite_utils.LOW_RES_ROW_DIM, satellite_utils.LOW_RES_COLUMN_DIM,
        satellite_utils.TIME_DIM
    ]

    for this_key in integer_keys:
        if this_key not in first_table_xarray.coords:
            if this_key in second_table_xarray.coords:
                print(this_key)
                return False

            continue

        if not numpy.array_equal(
                first_table_xarray.coords[this_key].values,
                second_table_xarray.coords[this_key].values
        ):
            print(this_key)
            return False

    float_keys = [
        satellite_utils.HIGH_RES_WAVELENGTH_DIM,
        satellite_utils.LOW_RES_WAVELENGTH_DIM
    ]

    for this_key in float_keys:
        if this_key not in first_table_xarray.coords:
            if this_key in second_table_xarray.coords:
                print(this_key)
                return False

            continue

        if not numpy.allclose(
                first_table_xarray.coords[this_key].values,
                second_table_xarray.coords[this_key].values,
                atol=TOLERANCE, equal_nan=False
        ):
            print(this_key)
            return False

    string_keys = [
        satellite_utils.CYCLONE_ID_KEY, satellite_utils.PYPROJ_STRING_KEY
    ]

    for this_key in string_keys:
        if this_key not in first_table_xarray:
            if this_key in second_table_xarray:
                print(this_key)
                return False

            continue

        if not numpy.array_equal(
                first_table_xarray[this_key].values,
                second_table_xarray[this_key].values
        ):
            print(this_key)
            return False

    float_keys = [
        satellite_utils.X_COORD_HIGH_RES_KEY,
        satellite_utils.Y_COORD_HIGH_RES_KEY,
        satellite_utils.X_COORD_LOW_RES_KEY,
        satellite_utils.Y_COORD_LOW_RES_KEY,
        satellite_utils.LATITUDE_HIGH_RES_KEY,
        satellite_utils.LONGITUDE_HIGH_RES_KEY,
        satellite_utils.LATITUDE_LOW_RES_KEY,
        satellite_utils.LONGITUDE_LOW_RES_KEY,
        satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY,
        satellite_utils.BRIGHTNESS_TEMPERATURE_KEY
    ]

    for this_key in float_keys:
        if this_key not in first_table_xarray:
            if this_key in second_table_xarray:
                print(this_key + '1')
                return False

            continue

        if not numpy.allclose(
                first_table_xarray[this_key].values,
                second_table_xarray[this_key].values,
                atol=TOLERANCE, equal_nan=True
        ):
            print(this_key + '2')
            return False

    return True


class SatelliteUtilsTests(unittest.TestCase):
    """Each method is a unit test for satellite_utils.py."""

    def test_quality_control_low_res_first(self):
        """Ensures correct output from quality_control_low_res.

        With first set of options.
        """

        this_table_xarray = satellite_utils.quality_control_low_res(
            satellite_table_xarray=copy.deepcopy(TABLE_BEFORE_LOW_RES_QC),
            max_bad_pixels_per_time_channel=
            FIRST_MAX_BAD_PIXELS_PER_TIME_CHANNEL
        )

        self.assertTrue(
            _compare_satellite_tables(
                this_table_xarray, FIRST_TABLE_AFTER_LOW_RES_QC
            )
        )

    def test_quality_control_low_res_second(self):
        """Ensures correct output from quality_control_low_res.

        With second set of options.
        """

        this_table_xarray = satellite_utils.quality_control_low_res(
            satellite_table_xarray=copy.deepcopy(TABLE_BEFORE_LOW_RES_QC),
            max_bad_pixels_per_time_channel=
            SECOND_MAX_BAD_PIXELS_PER_TIME_CHANNEL
        )

        self.assertTrue(
            _compare_satellite_tables(
                this_table_xarray, SECOND_TABLE_AFTER_LOW_RES_QC
            )
        )

    def test_quality_control_high_res_first(self):
        """Ensures correct output from quality_control_high_res.

        With first set of options.
        """

        this_table_xarray = satellite_utils.quality_control_high_res(
            satellite_table_xarray=copy.deepcopy(TABLE_BEFORE_HIGH_RES_QC),
            max_bad_pixels_per_time_channel=
            FIRST_MAX_BAD_PIXELS_PER_TIME_CHANNEL
        )

        self.assertTrue(
            _compare_satellite_tables(
                this_table_xarray, FIRST_TABLE_AFTER_HIGH_RES_QC
            )
        )

    def test_quality_control_high_res_second(self):
        """Ensures correct output from quality_control_high_res.

        With second set of options.
        """

        this_table_xarray = satellite_utils.quality_control_high_res(
            satellite_table_xarray=copy.deepcopy(TABLE_BEFORE_HIGH_RES_QC),
            max_bad_pixels_per_time_channel=
            SECOND_MAX_BAD_PIXELS_PER_TIME_CHANNEL
        )

        self.assertTrue(
            _compare_satellite_tables(
                this_table_xarray, SECOND_TABLE_AFTER_HIGH_RES_QC
            )
        )

    def test_subset_grid_low_res_first(self):
        """Ensures correct output from subset_grid.

        For low-resolution data with first set of options.
        """

        this_table_xarray = satellite_utils.subset_grid(
            satellite_table_xarray=
            copy.deepcopy(TABLE_BEFORE_SUBSET_GRID_LOW_RES),
            num_rows_to_keep=FIRST_NUM_ROWS_TO_KEEP,
            num_columns_to_keep=FIRST_NUM_COLUMNS_TO_KEEP,
            for_high_res=False
        )

        self.assertTrue(
            _compare_satellite_tables(
                this_table_xarray, FIRST_TABLE_AFTER_SUBSET_GRID_LOW_RES
            )
        )

    def test_subset_grid_low_res_second(self):
        """Ensures correct output from subset_grid.

        For low-resolution data with second set of options.
        """

        this_table_xarray = satellite_utils.subset_grid(
            satellite_table_xarray=
            copy.deepcopy(TABLE_BEFORE_SUBSET_GRID_LOW_RES),
            num_rows_to_keep=SECOND_NUM_ROWS_TO_KEEP,
            num_columns_to_keep=SECOND_NUM_COLUMNS_TO_KEEP,
            for_high_res=False
        )

        self.assertTrue(
            _compare_satellite_tables(
                this_table_xarray, SECOND_TABLE_AFTER_SUBSET_GRID_LOW_RES
            )
        )

    def test_subset_grid_low_res_third(self):
        """Ensures correct output from subset_grid.

        For low-resolution data with third set of options.
        """

        with self.assertRaises(ValueError):
            satellite_utils.subset_grid(
                satellite_table_xarray=
                copy.deepcopy(TABLE_BEFORE_SUBSET_GRID_LOW_RES),
                num_rows_to_keep=THIRD_NUM_ROWS_TO_KEEP,
                num_columns_to_keep=THIRD_NUM_COLUMNS_TO_KEEP,
                for_high_res=False
            )

    def test_subset_grid_high_res_first(self):
        """Ensures correct output from subset_grid.

        For high-resolution data with first set of options.
        """

        this_table_xarray = satellite_utils.subset_grid(
            satellite_table_xarray=
            copy.deepcopy(TABLE_BEFORE_SUBSET_GRID_HIGH_RES),
            num_rows_to_keep=FIRST_NUM_ROWS_TO_KEEP,
            num_columns_to_keep=FIRST_NUM_COLUMNS_TO_KEEP,
            for_high_res=True
        )

        self.assertTrue(
            _compare_satellite_tables(
                this_table_xarray, FIRST_TABLE_AFTER_SUBSET_GRID_HIGH_RES
            )
        )

    def test_subset_grid_high_res_second(self):
        """Ensures correct output from subset_grid.

        For high-resolution data with second set of options.
        """

        this_table_xarray = satellite_utils.subset_grid(
            satellite_table_xarray=
            copy.deepcopy(TABLE_BEFORE_SUBSET_GRID_HIGH_RES),
            num_rows_to_keep=SECOND_NUM_ROWS_TO_KEEP,
            num_columns_to_keep=SECOND_NUM_COLUMNS_TO_KEEP,
            for_high_res=True
        )

        self.assertTrue(
            _compare_satellite_tables(
                this_table_xarray, SECOND_TABLE_AFTER_SUBSET_GRID_HIGH_RES
            )
        )

    def test_subset_grid_high_res_third(self):
        """Ensures correct output from subset_grid.

        For high-resolution data with third set of options.
        """

        with self.assertRaises(ValueError):
            satellite_utils.subset_grid(
                satellite_table_xarray=
                copy.deepcopy(TABLE_BEFORE_SUBSET_GRID_HIGH_RES),
                num_rows_to_keep=THIRD_NUM_ROWS_TO_KEEP,
                num_columns_to_keep=THIRD_NUM_COLUMNS_TO_KEEP,
                for_high_res=True
            )

    def test_subset_wavelengths_low_res_first(self):
        """Ensures correct output from subset_wavelengths.

        For low-resolution data with first set of options.
        """

        this_table_xarray = satellite_utils.subset_wavelengths(
            satellite_table_xarray=
            copy.deepcopy(TABLE_BEFORE_SUBSET_WVL_LOW_RES),
            wavelengths_to_keep_microns=
            FIRST_LOW_RES_WAVELENGTHS_TO_KEEP_MICRONS,
            for_high_res=False
        )

        self.assertTrue(
            _compare_satellite_tables(
                this_table_xarray, FIRST_TABLE_AFTER_SUBSET_WVL_LOW_RES
            )
        )

    def test_subset_wavelengths_low_res_second(self):
        """Ensures correct output from subset_wavelengths.

        For low-resolution data with second set of options.
        """

        with self.assertRaises(IndexError):
            satellite_utils.subset_wavelengths(
                satellite_table_xarray=
                copy.deepcopy(TABLE_BEFORE_SUBSET_WVL_LOW_RES),
                wavelengths_to_keep_microns=
                SECOND_LOW_RES_WAVELENGTHS_TO_KEEP_MICRONS,
                for_high_res=False
            )

    def test_subset_wavelengths_high_res_first(self):
        """Ensures correct output from subset_wavelengths.

        For high-resolution data with first set of options.
        """

        this_table_xarray = satellite_utils.subset_wavelengths(
            satellite_table_xarray=
            copy.deepcopy(TABLE_BEFORE_SUBSET_WVL_HIGH_RES),
            wavelengths_to_keep_microns=
            FIRST_HIGH_RES_WAVELENGTHS_TO_KEEP_MICRONS,
            for_high_res=True
        )

        self.assertTrue(
            _compare_satellite_tables(
                this_table_xarray, FIRST_TABLE_AFTER_SUBSET_WVL_HIGH_RES
            )
        )

    def test_subset_wavelengths_high_res_second(self):
        """Ensures correct output from subset_wavelengths.

        For high-resolution data with second set of options.
        """

        with self.assertRaises(IndexError):
            satellite_utils.subset_wavelengths(
                satellite_table_xarray=
                copy.deepcopy(TABLE_BEFORE_SUBSET_WVL_HIGH_RES),
                wavelengths_to_keep_microns=
                SECOND_HIGH_RES_WAVELENGTHS_TO_KEEP_MICRONS,
                for_high_res=True
            )

    def test_subset_to_multiple_time_windows_first(self):
        """Ensures correct output from subset_to_multiple_time_windows.

        For both resolutions with first set of options.
        """

        this_table_xarray = satellite_utils.subset_to_multiple_time_windows(
            satellite_table_xarray=copy.deepcopy(TABLE_BEFORE_SUBSET_WINDOWS),
            start_times_unix_sec=FIRST_START_TIMES_UNIX_SEC,
            end_times_unix_sec=FIRST_END_TIMES_UNIX_SEC
        )

        self.assertTrue(
            _compare_satellite_tables(
                this_table_xarray, FIRST_TABLE_AFTER_SUBSET_WINDOWS
            )
        )

    def test_subset_to_multiple_time_windows_second(self):
        """Ensures correct output from subset_to_multiple_time_windows.

        For both resolutions with second set of options.
        """

        this_table_xarray = satellite_utils.subset_to_multiple_time_windows(
            satellite_table_xarray=copy.deepcopy(TABLE_BEFORE_SUBSET_WINDOWS),
            start_times_unix_sec=SECOND_START_TIMES_UNIX_SEC,
            end_times_unix_sec=SECOND_END_TIMES_UNIX_SEC
        )

        self.assertTrue(
            _compare_satellite_tables(
                this_table_xarray, SECOND_TABLE_AFTER_SUBSET_WINDOWS
            )
        )

    def test_subset_to_multiple_time_windows_third(self):
        """Ensures correct output from subset_to_multiple_time_windows.

        For both resolutions with third set of options.
        """

        this_table_xarray = satellite_utils.subset_to_multiple_time_windows(
            satellite_table_xarray=copy.deepcopy(TABLE_BEFORE_SUBSET_WINDOWS),
            start_times_unix_sec=THIRD_START_TIMES_UNIX_SEC,
            end_times_unix_sec=THIRD_END_TIMES_UNIX_SEC
        )

        self.assertTrue(
            _compare_satellite_tables(
                this_table_xarray, THIRD_TABLE_AFTER_SUBSET_WINDOWS
            )
        )

    def test_compute_interpolation_gap_first(self):
        """Ensures correct output from _compute_interpolation_gap.

        With first set of options.
        """

        this_interp_gap_sec = satellite_utils._compute_interpolation_gap(
            source_times_unix_sec=SOURCE_TIMES_UNIX_SEC,
            target_time_unix_sec=FIRST_TARGET_TIME_UNIX_SEC
        )
        self.assertTrue(this_interp_gap_sec == FIRST_INTERP_GAP_SEC)

    def test_compute_interpolation_gap_second(self):
        """Ensures correct output from _compute_interpolation_gap.

        With second set of options.
        """

        this_interp_gap_sec = satellite_utils._compute_interpolation_gap(
            source_times_unix_sec=SOURCE_TIMES_UNIX_SEC,
            target_time_unix_sec=SECOND_TARGET_TIME_UNIX_SEC
        )
        self.assertTrue(this_interp_gap_sec == SECOND_INTERP_GAP_SEC)

    def test_compute_interpolation_gap_third(self):
        """Ensures correct output from _compute_interpolation_gap.

        With third set of options.
        """

        this_interp_gap_sec = satellite_utils._compute_interpolation_gap(
            source_times_unix_sec=SOURCE_TIMES_UNIX_SEC,
            target_time_unix_sec=THIRD_TARGET_TIME_UNIX_SEC
        )
        self.assertTrue(this_interp_gap_sec == THIRD_INTERP_GAP_SEC)

    def test_compute_interpolation_gap_fourth(self):
        """Ensures correct output from _compute_interpolation_gap.

        With fourth set of options.
        """

        this_interp_gap_sec = satellite_utils._compute_interpolation_gap(
            source_times_unix_sec=SOURCE_TIMES_UNIX_SEC,
            target_time_unix_sec=FOURTH_TARGET_TIME_UNIX_SEC
        )
        self.assertTrue(this_interp_gap_sec == FOURTH_INTERP_GAP_SEC)

    def test_subset_times_first(self):
        """Ensures correct output from subset_times.

        With first set of options.
        """

        this_table_xarray = satellite_utils.subset_times(
            satellite_table_xarray=copy.deepcopy(TABLE_BEFORE_SUBSET_TIMES),
            desired_times_unix_sec=FIRST_DESIRED_TIMES_UNIX_SEC,
            tolerances_sec=FIRST_TIME_TOLERANCES_SEC,
            max_num_missing_times=FIRST_MAX_NUM_MISSING_TIMES,
            max_interp_gaps_sec=FIRST_MAX_INTERP_GAPS_SEC
        )

        self.assertTrue(
            _compare_satellite_tables(
                this_table_xarray, FIRST_TABLE_AFTER_SUBSET_TIMES
            )
        )

    def test_subset_times_second(self):
        """Ensures correct output from subset_times.

        With second set of options.
        """

        this_table_xarray = satellite_utils.subset_times(
            satellite_table_xarray=copy.deepcopy(TABLE_BEFORE_SUBSET_TIMES),
            desired_times_unix_sec=SECOND_DESIRED_TIMES_UNIX_SEC,
            tolerances_sec=SECOND_TIME_TOLERANCES_SEC,
            max_num_missing_times=SECOND_MAX_NUM_MISSING_TIMES,
            max_interp_gaps_sec=SECOND_MAX_INTERP_GAPS_SEC
        )

        self.assertTrue(
            _compare_satellite_tables(
                this_table_xarray, SECOND_TABLE_AFTER_SUBSET_TIMES
            )
        )

    def test_subset_times_third(self):
        """Ensures correct output from subset_times.

        With third set of options.
        """

        this_table_xarray = satellite_utils.subset_times(
            satellite_table_xarray=copy.deepcopy(TABLE_BEFORE_SUBSET_TIMES),
            desired_times_unix_sec=THIRD_DESIRED_TIMES_UNIX_SEC,
            tolerances_sec=THIRD_TIME_TOLERANCES_SEC,
            max_num_missing_times=THIRD_MAX_NUM_MISSING_TIMES,
            max_interp_gaps_sec=THIRD_MAX_INTERP_GAPS_SEC
        )

        self.assertTrue(
            _compare_satellite_tables(
                this_table_xarray, THIRD_TABLE_AFTER_SUBSET_TIMES
            )
        )

    def test_subset_times_fourth(self):
        """Ensures correct output from subset_times.

        With fourth set of options.
        """

        this_table_xarray = satellite_utils.subset_times(
            satellite_table_xarray=copy.deepcopy(TABLE_BEFORE_SUBSET_TIMES),
            desired_times_unix_sec=FOURTH_DESIRED_TIMES_UNIX_SEC,
            tolerances_sec=FOURTH_TIME_TOLERANCES_SEC,
            max_num_missing_times=FOURTH_MAX_NUM_MISSING_TIMES,
            max_interp_gaps_sec=FOURTH_MAX_INTERP_GAPS_SEC
        )

        self.assertTrue(
            _compare_satellite_tables(
                this_table_xarray, FOURTH_TABLE_AFTER_SUBSET_TIMES
            )
        )

    def test_subset_times_fifth(self):
        """Ensures correct output from subset_times.

        With fifth set of options.
        """

        with self.assertRaises(ValueError):
            satellite_utils.subset_times(
                satellite_table_xarray=copy.deepcopy(TABLE_BEFORE_SUBSET_TIMES),
                desired_times_unix_sec=FIFTH_DESIRED_TIMES_UNIX_SEC,
                tolerances_sec=FIFTH_TIME_TOLERANCES_SEC,
                max_num_missing_times=FIFTH_MAX_NUM_MISSING_TIMES,
                max_interp_gaps_sec=FIFTH_MAX_INTERP_GAPS_SEC
            )

    def test_subset_times_sixth(self):
        """Ensures correct output from subset_times.

        With sixth set of options.
        """

        with self.assertRaises(ValueError):
            satellite_utils.subset_times(
                satellite_table_xarray=copy.deepcopy(TABLE_BEFORE_SUBSET_TIMES),
                desired_times_unix_sec=SIXTH_DESIRED_TIMES_UNIX_SEC,
                tolerances_sec=SIXTH_TIME_TOLERANCES_SEC,
                max_num_missing_times=SIXTH_MAX_NUM_MISSING_TIMES,
                max_interp_gaps_sec=SIXTH_MAX_INTERP_GAPS_SEC
            )


if __name__ == '__main__':
    unittest.main()
