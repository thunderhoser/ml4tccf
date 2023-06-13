"""Data augmentation."""

import numpy
import xarray
from gewittergefahr.gg_utils import error_checking
from ml4tccf.utils import satellite_utils


def _get_random_signs(array_length):
    """Returns array of random signs (1 for positive, -1 for negative).

    :param array_length: Number of random signs desired.
    :return: sign_array: numpy array of integers in {-1, 1}.
    """

    return 2 * numpy.random.randint(low=0, high=2, size=array_length) - 1


# def _translate_images_fast_attempt(
#         image_matrix, row_translation_px, column_translation_px, padding_value):
#     """Translates set of images in both the x- and y-directions.
#
#     :param image_matrix: numpy array, where the second (third) axis is the row
#         (column) dimension.
#     :param row_translation_px: Will translate each image by this many rows.
#     :param column_translation_px: Will translate each image by this many
#         columns.
#     :param padding_value: Padded pixels will be filled with this value.
#     :return: translated_image_matrix: Same as input but after translation.
#     """
#
#     transform_object = skimage.transform.AffineTransform(
#         translation=[column_translation_px, row_translation_px]
#     )
#
#     num_examples = image_matrix.shape[0]
#     num_channels = image_matrix.shape[3]
#     translated_image_matrix = image_matrix + 0.
#
#     for i in range(num_examples):
#         for j in range(num_channels):
#             translated_image_matrix[i, ..., j] = skimage.transform.warp(
#                 translated_image_matrix[i, ..., j], transform_object.inverse
#             )
#
#     return translated_image_matrix


def _translate_images(image_matrix, row_translation_px, column_translation_px,
                      padding_value):
    """Translates set of images in both the x- and y-directions.

    :param image_matrix: numpy array, where the second (third) axis is the row
        (column) dimension.
    :param row_translation_px: Will translate each image by this many rows.
    :param column_translation_px: Will translate each image by this many
        columns.
    :param padding_value: Padded pixels will be filled with this value.
    :return: translated_image_matrix: Same as input but after translation.
    """

    num_rows = image_matrix.shape[1]
    num_columns = image_matrix.shape[2]

    num_padded_columns_at_left = max([column_translation_px, 0])
    num_padded_columns_at_right = max([-column_translation_px, 0])
    num_padded_rows_at_top = max([row_translation_px, 0])
    num_padded_rows_at_bottom = max([-row_translation_px, 0])

    padding_arg = (
        (0, 0),
        (num_padded_rows_at_top, num_padded_rows_at_bottom),
        (num_padded_columns_at_left, num_padded_columns_at_right)
    )

    num_dimensions = len(image_matrix.shape)
    for _ in range(3, num_dimensions):
        padding_arg += ((0, 0),)

    translated_image_matrix = numpy.pad(
        image_matrix, pad_width=padding_arg, mode='constant',
        constant_values=padding_value
    )

    if column_translation_px >= 0:
        translated_image_matrix = (
            translated_image_matrix[:, :, :num_columns, ...]
        )
    else:
        translated_image_matrix = (
            translated_image_matrix[:, :, -num_columns:, ...]
        )

    if row_translation_px >= 0:
        translated_image_matrix = translated_image_matrix[:, :num_rows, ...]
    else:
        translated_image_matrix = translated_image_matrix[:, -num_rows:, ...]

    return translated_image_matrix


def get_translation_distances(
        mean_translation_px, stdev_translation_px, num_translations):
    """Samples translation distances from normal distribution.

    T = number of translations

    :param mean_translation_px: Mean translation distance (pixels).
    :param stdev_translation_px: Standard deviation of translation distance
        (pixels).
    :param num_translations: T in the above discussion.
    :return: row_translations_px: length-T numpy array of translation distances
        (pixels).
    :return: column_translations_px: Same but for columns.
    """

    error_checking.assert_is_greater(mean_translation_px, 0.)
    error_checking.assert_is_greater(stdev_translation_px, 0.)
    error_checking.assert_is_integer(num_translations)
    error_checking.assert_is_greater(num_translations, 0)

    euclidean_translations_low_res_px = numpy.random.normal(
        loc=mean_translation_px, scale=stdev_translation_px,
        size=num_translations
    )
    euclidean_translations_low_res_px = numpy.maximum(
        euclidean_translations_low_res_px, 0.
    )
    translation_directions_rad = numpy.random.uniform(
        low=0., high=2 * numpy.pi - 1e-6, size=num_translations
    )

    row_translations_low_res_px = (
        euclidean_translations_low_res_px *
        numpy.sin(translation_directions_rad)
    )
    column_translations_low_res_px = (
        euclidean_translations_low_res_px *
        numpy.cos(translation_directions_rad)
    )

    # these_flags = numpy.logical_and(
    #     row_translations_low_res_px < 0, row_translations_low_res_px >= -0.5
    # )
    # row_translations_low_res_px[these_flags] = -1.
    #
    # these_flags = numpy.logical_and(
    #     row_translations_low_res_px > 0, row_translations_low_res_px <= 0.5
    # )
    # row_translations_low_res_px[these_flags] = 1.
    #
    # these_flags = numpy.logical_and(
    #     column_translations_low_res_px < 0,
    #     column_translations_low_res_px >= -0.5
    # )
    # column_translations_low_res_px[these_flags] = -1.
    #
    # these_flags = numpy.logical_and(
    #     column_translations_low_res_px > 0,
    #     column_translations_low_res_px <= 0.5
    # )
    # column_translations_low_res_px[these_flags] = 1.

    row_translations_low_res_px = (
        numpy.round(row_translations_low_res_px).astype(int)
    )
    column_translations_low_res_px = (
        numpy.round(column_translations_low_res_px).astype(int)
    )

    error_checking.assert_is_geq_numpy_array(
        numpy.absolute(row_translations_low_res_px), 0
    )
    error_checking.assert_is_geq_numpy_array(
        numpy.absolute(column_translations_low_res_px), 0
    )

    return row_translations_low_res_px, column_translations_low_res_px


def augment_data(
        bidirectional_reflectance_matrix, brightness_temp_matrix_kelvins,
        num_translations_per_example, mean_translation_low_res_px,
        stdev_translation_low_res_px, sentinel_value):
    """Augments data via translation.

    E = number of examples
    T = number of translations per example
    L = number of lag times
    W = number of high-res wavelengths
    w = number of low-res wavelengths
    M = number of rows in high-res grid
    m = number of rows in low-res grid = M/4
    N = number of columns in high-res grid
    n = number of columns in low-res grid = M/4

    :param bidirectional_reflectance_matrix: E-by-M-by-N-by-L-by-W numpy array
        of reflectance values (unitless).  This may also be None.
    :param brightness_temp_matrix_kelvins: E-by-m-by-n-by-L-by-w numpy array
        of brightness temperatures.
    :param num_translations_per_example: T in the above discussion.
    :param mean_translation_low_res_px: Mean translation distance (in units of
        low-resolution pixels).
    :param stdev_translation_low_res_px: Standard deviation of translation
        distance (in units of low-resolution pixels).
    :param sentinel_value: Sentinel value (used for padded pixels around edge).
    :return: bidirectional_reflectance_matrix: ET-by-M-by-N-by-L-by-W numpy
        array of reflectance values (unitless).  This may also be None.
    :return: brightness_temp_matrix_kelvins: ET-by-m-by-n-by-L-by-w numpy array
        of brightness temperatures.
    :return: row_translations_low_res_px: length-(ET) numpy array of translation
        distances applied (in units of low-resolution pixels).
    :return: column_translations_low_res_px: Same but for columns.
    """

    # Check input args.
    error_checking.assert_is_numpy_array_without_nan(
        brightness_temp_matrix_kelvins
    )
    error_checking.assert_is_numpy_array(
        brightness_temp_matrix_kelvins, num_dimensions=4
    )
    error_checking.assert_is_integer(num_translations_per_example)
    error_checking.assert_is_geq(num_translations_per_example, 1)
    error_checking.assert_is_not_nan(sentinel_value)

    if bidirectional_reflectance_matrix is not None:
        error_checking.assert_is_numpy_array_without_nan(
            bidirectional_reflectance_matrix
        )
        error_checking.assert_is_numpy_array(
            bidirectional_reflectance_matrix, num_dimensions=4
        )

        num_examples = brightness_temp_matrix_kelvins.shape[0]
        num_rows_les_res = brightness_temp_matrix_kelvins.shape[1]
        num_columns_les_res = brightness_temp_matrix_kelvins.shape[2]
        expected_dim = numpy.array([
            num_examples, 4 * num_rows_les_res, 4 * num_columns_les_res,
            bidirectional_reflectance_matrix.shape[-1]
        ], dtype=int)

        error_checking.assert_is_numpy_array(
            bidirectional_reflectance_matrix, exact_dimensions=expected_dim
        )

    # Housekeeping.
    num_examples_orig = brightness_temp_matrix_kelvins.shape[0]
    num_examples_new = num_examples_orig * num_translations_per_example

    row_translations_low_res_px, column_translations_low_res_px = (
        get_translation_distances(
            mean_translation_px=mean_translation_low_res_px,
            stdev_translation_px=stdev_translation_low_res_px,
            num_translations=num_examples_new
        )
    )

    # Do actual stuff.
    new_bt_matrix_kelvins = numpy.full(
        (num_examples_new,) + brightness_temp_matrix_kelvins.shape[1:],
        numpy.nan
    )

    if bidirectional_reflectance_matrix is None:
        new_reflectance_matrix = None
    else:
        new_reflectance_matrix = numpy.full(
            (num_examples_new,) + bidirectional_reflectance_matrix.shape[1:],
            numpy.nan
        )

    for i in range(num_examples_orig):
        first_index = i * num_translations_per_example
        last_index = first_index + num_translations_per_example

        for j in range(first_index, last_index):
            new_bt_matrix_kelvins[j, ...] = _translate_images(
                image_matrix=brightness_temp_matrix_kelvins[[i], ...],
                row_translation_px=row_translations_low_res_px[j],
                column_translation_px=column_translations_low_res_px[j],
                padding_value=sentinel_value
            )[0, ...]

            if new_reflectance_matrix is None:
                continue

            new_reflectance_matrix[j, ...] = _translate_images(
                image_matrix=bidirectional_reflectance_matrix[[i], ...],
                row_translation_px=4 * row_translations_low_res_px[j],
                column_translation_px=4 * column_translations_low_res_px[j],
                padding_value=sentinel_value
            )[0, ...]

    return (
        new_reflectance_matrix, new_bt_matrix_kelvins,
        row_translations_low_res_px, column_translations_low_res_px
    )


def augment_data_specific_trans(
        bidirectional_reflectance_matrix, brightness_temp_matrix_kelvins,
        row_translations_low_res_px, column_translations_low_res_px,
        sentinel_value):
    """Augments data via specific (not random) translations.

    E = number of examples

    :param bidirectional_reflectance_matrix: See doc for `augment_data`.
    :param brightness_temp_matrix_kelvins: Same.
    :param row_translations_low_res_px: length-E numpy array of row
        translations.  The [i]th example will be shifted
        row_translations_low_res_px[i] rows up (towards the north).
    :param column_translations_low_res_px: length-E numpy array of column
        translations.  The [i]th example will be shifted
        column_translations_low_res_px[i] columns towards the right (east).
    :param sentinel_value: Sentinel value (used for padded pixels around edge).
    :return: bidirectional_reflectance_matrix: numpy array of translated images
        with same size as input.
    :return: brightness_temp_matrix_kelvins: numpy array of translated images
        with same size as input.
    """

    # Check input args.
    error_checking.assert_is_numpy_array_without_nan(
        brightness_temp_matrix_kelvins
    )
    error_checking.assert_is_numpy_array(
        brightness_temp_matrix_kelvins, num_dimensions=4
    )
    error_checking.assert_is_not_nan(sentinel_value)

    num_examples = brightness_temp_matrix_kelvins.shape[0]
    expected_dim = numpy.array([num_examples], dtype=int)

    error_checking.assert_is_integer_numpy_array(row_translations_low_res_px)
    error_checking.assert_is_numpy_array(
        row_translations_low_res_px, exact_dimensions=expected_dim
    )

    error_checking.assert_is_integer_numpy_array(column_translations_low_res_px)
    error_checking.assert_is_numpy_array(
        column_translations_low_res_px, exact_dimensions=expected_dim
    )

    if bidirectional_reflectance_matrix is not None:
        error_checking.assert_is_numpy_array_without_nan(
            bidirectional_reflectance_matrix
        )
        error_checking.assert_is_numpy_array(
            bidirectional_reflectance_matrix, num_dimensions=4
        )

        num_examples = brightness_temp_matrix_kelvins.shape[0]
        num_rows_les_res = brightness_temp_matrix_kelvins.shape[1]
        num_columns_les_res = brightness_temp_matrix_kelvins.shape[2]
        expected_dim = numpy.array([
            num_examples, 4 * num_rows_les_res, 4 * num_columns_les_res,
            bidirectional_reflectance_matrix.shape[-1]
        ], dtype=int)

        error_checking.assert_is_numpy_array(
            bidirectional_reflectance_matrix, exact_dimensions=expected_dim
        )

    # Do actual stuff.
    for i in range(num_examples):
        brightness_temp_matrix_kelvins[i, ...] = _translate_images(
            image_matrix=brightness_temp_matrix_kelvins[[i], ...],
            row_translation_px=row_translations_low_res_px[i],
            column_translation_px=column_translations_low_res_px[i],
            padding_value=sentinel_value
        )[0, ...]

        if bidirectional_reflectance_matrix is None:
            continue

        bidirectional_reflectance_matrix[i, ...] = _translate_images(
            image_matrix=bidirectional_reflectance_matrix[[i], ...],
            row_translation_px=4 * row_translations_low_res_px[i],
            column_translation_px=4 * column_translations_low_res_px[i],
            padding_value=sentinel_value
        )[0, ...]

    return bidirectional_reflectance_matrix, brightness_temp_matrix_kelvins


def subset_grid_after_data_aug(data_matrix, num_rows_to_keep,
                               num_columns_to_keep, for_high_res):
    """Subsets grid after data augmentation (more cropping than first time).

    E = number of examples
    M = number of rows in original grid
    N = number of columns in original grid
    m = number of rows in subset grid
    n = number of columns in subset grid
    W = number of wavelengths in grid

    :param data_matrix: E-by-M-by-N-by-W numpy array of data.
    :param num_rows_to_keep: m in the above discussion.
    :param num_columns_to_keep: n in the above discussion.
    :param for_high_res: Boolean flag.
    :return: data_matrix: E-by-m-by-n-by-W numpy array of data (subset of
        input).
    """

    # Check input args.
    error_checking.assert_is_numpy_array_without_nan(data_matrix)
    error_checking.assert_is_numpy_array(data_matrix, num_dimensions=4)

    error_checking.assert_is_integer(num_rows_to_keep)
    error_checking.assert_is_greater(num_rows_to_keep, 0)
    error_checking.assert_equals(
        numpy.mod(num_rows_to_keep, 2),
        0
    )

    error_checking.assert_is_integer(num_columns_to_keep)
    error_checking.assert_is_greater(num_columns_to_keep, 0)
    error_checking.assert_equals(
        numpy.mod(num_columns_to_keep, 2),
        0
    )

    error_checking.assert_is_boolean(for_high_res)

    # Do actual stuff.
    num_examples = data_matrix.shape[0]
    num_rows = data_matrix.shape[1]
    num_columns = data_matrix.shape[2]
    num_wavelengths = data_matrix.shape[3]

    row_dim = (
        satellite_utils.HIGH_RES_ROW_DIM if for_high_res
        else satellite_utils.LOW_RES_ROW_DIM
    )
    column_dim = (
        satellite_utils.HIGH_RES_COLUMN_DIM if for_high_res
        else satellite_utils.LOW_RES_COLUMN_DIM
    )
    wavelength_dim = (
        satellite_utils.HIGH_RES_WAVELENGTH_DIM if for_high_res
        else satellite_utils.LOW_RES_WAVELENGTH_DIM
    )
    main_data_key = (
        satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY if for_high_res
        else satellite_utils.BRIGHTNESS_TEMPERATURE_KEY
    )

    metadata_dict = {
        satellite_utils.TIME_DIM: numpy.linspace(
            0, num_examples - 1, num=num_examples, dtype=int
        ),
        row_dim: numpy.linspace(
            0, num_rows - 1, num=num_rows, dtype=int
        ),
        column_dim: numpy.linspace(
            0, num_columns - 1, num=num_columns, dtype=int
        ),
        wavelength_dim: numpy.linspace(
            1, num_wavelengths, num=num_wavelengths, dtype=float
        )
    }

    main_data_dict = {
        main_data_key: (
            (satellite_utils.TIME_DIM, row_dim, column_dim, wavelength_dim),
            data_matrix
        )
    }
    dummy_satellite_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )

    dummy_satellite_table_xarray = satellite_utils.subset_grid(
        satellite_table_xarray=dummy_satellite_table_xarray,
        num_rows_to_keep=num_rows_to_keep,
        num_columns_to_keep=num_columns_to_keep,
        for_high_res=for_high_res
    )
    return dummy_satellite_table_xarray[main_data_key].values
