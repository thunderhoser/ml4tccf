"""Methods for normalizing predictor variables."""

import os
import shutil
import numpy
import xarray
import scipy.stats
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4tccf.io import satellite_io
from ml4tccf.utils import satellite_utils

TOLERANCE = 1e-6
MIN_CUMULATIVE_DENSITY = 1e-6
# MAX_CUMULATIVE_DENSITY = 1. - 1e-6
MAX_CUMULATIVE_DENSITY = 0.9995  # To account for 16-bit floats.

NUM_TIMES_PER_FILE_FOR_PARAMS = 10

BIDIRECTIONAL_REFLECTANCE_KEY = satellite_utils.BIDIRECTIONAL_REFLECTANCE_KEY
BRIGHTNESS_TEMPERATURE_KEY = satellite_utils.BRIGHTNESS_TEMPERATURE_KEY

HIGH_RES_WAVELENGTH_DIM = satellite_utils.HIGH_RES_WAVELENGTH_DIM
LOW_RES_WAVELENGTH_DIM = satellite_utils.LOW_RES_WAVELENGTH_DIM
HIGH_RES_SAMPLE_DIM = 'high_res_sample'
LOW_RES_SAMPLE_DIM = 'low_res_sample'


def _actual_to_uniform_dist(actual_values_new, actual_values_training):
    """Converts values from actual to uniform distribution.

    :param actual_values_new: numpy array of actual (physical) values to
        convert.
    :param actual_values_training: numpy array of actual (physical) values in
        training data.
    :return: uniform_values_new: numpy array (same shape as `actual_values_new`)
        with rescaled values from 0...1.
    """

    assert numpy.all(numpy.isfinite(actual_values_training))

    actual_values_new_1d = numpy.ravel(actual_values_new)
    actual_values_new_1d[
        numpy.invert(numpy.isfinite(actual_values_new_1d))
    ] = numpy.nan

    real_indices = numpy.where(
        numpy.invert(numpy.isnan(actual_values_new_1d))
    )[0]

    if len(real_indices) == 0:
        return actual_values_new

    search_indices = numpy.searchsorted(
        numpy.sort(numpy.ravel(actual_values_training)),
        actual_values_new_1d[real_indices],
        side='left'
    ).astype(float)

    uniform_values_new_1d = actual_values_new_1d + 0.
    num_values = actual_values_training.size
    uniform_values_new_1d[real_indices] = search_indices / (num_values - 1)

    uniform_values_new_1d = numpy.minimum(uniform_values_new_1d, 1.)
    uniform_values_new_1d = numpy.maximum(uniform_values_new_1d, 0.)

    return numpy.reshape(uniform_values_new_1d, actual_values_new.shape)


def _uniform_to_actual_dist(uniform_values_new, actual_values_training):
    """Converts values from uniform to actual distribution.

    This method is the inverse of `_actual_to_uniform_dist`.

    :param uniform_values_new: See doc for `_actual_to_uniform_dist`.
    :param actual_values_training: Same.
    :return: actual_values_new: Same.
    """

    error_checking.assert_is_geq_numpy_array(
        uniform_values_new, 0., allow_nan=True
    )
    error_checking.assert_is_leq_numpy_array(
        uniform_values_new, 1., allow_nan=True
    )
    assert numpy.all(numpy.isfinite(actual_values_training))

    uniform_values_new_1d = numpy.ravel(uniform_values_new)
    real_indices = numpy.where(
        numpy.invert(numpy.isnan(uniform_values_new_1d))
    )[0]

    if len(real_indices) == 0:
        return uniform_values_new

    actual_values_new_1d = uniform_values_new_1d + 0.
    actual_values_new_1d[real_indices] = numpy.percentile(
        numpy.ravel(actual_values_training),
        100 * uniform_values_new_1d[real_indices],
        interpolation='linear'
    )

    return numpy.reshape(actual_values_new_1d, uniform_values_new.shape)


def _normalize_one_variable(actual_values_new, actual_values_training):
    """Normalizes one variable.

    :param actual_values_new: See doc for `_actual_to_uniform_dist`.
    :param actual_values_training: Same.
    :return: normalized_values_new: numpy array (same shape as
        `actual_values_new`) with normalized values (z-scores).
    """

    uniform_values_new = _actual_to_uniform_dist(
        actual_values_new=actual_values_new,
        actual_values_training=actual_values_training
    )

    uniform_values_new_1d = numpy.ravel(uniform_values_new)
    real_indices = numpy.where(
        numpy.invert(numpy.isnan(uniform_values_new_1d))
    )[0]

    uniform_values_new_1d[real_indices] = numpy.maximum(
        uniform_values_new_1d[real_indices], MIN_CUMULATIVE_DENSITY
    )
    uniform_values_new_1d[real_indices] = numpy.minimum(
        uniform_values_new_1d[real_indices], MAX_CUMULATIVE_DENSITY
    )
    uniform_values_new_1d[real_indices] = scipy.stats.norm.ppf(
        uniform_values_new_1d[real_indices], loc=0., scale=1.
    )

    return numpy.reshape(uniform_values_new_1d, uniform_values_new.shape)


def _denorm_one_variable(normalized_values_new, actual_values_training):
    """Denormalizes one variable.

    This method is the inverse of `_normalize_one_variable`.

    :param normalized_values_new: See doc for `_normalize_one_variable`.
    :param actual_values_training: Same.
    :return: actual_values_new: Same.
    """

    normalized_values_new_1d = numpy.ravel(normalized_values_new)
    real_indices = numpy.where(
        numpy.invert(numpy.isnan(normalized_values_new_1d))
    )[0]

    uniform_values_new_1d = normalized_values_new_1d + 0.
    uniform_values_new_1d[real_indices] = scipy.stats.norm.cdf(
        normalized_values_new_1d[real_indices], loc=0., scale=1.
    )
    uniform_values_new = numpy.reshape(
        uniform_values_new_1d, normalized_values_new.shape
    )

    return _uniform_to_actual_dist(
        uniform_values_new=uniform_values_new,
        actual_values_training=actual_values_training
    )


def get_normalization_params(
        satellite_file_names, num_values_per_low_res_channel,
        num_values_per_high_res_channel):
    """Computes normaliz'n params (set of reference values) for each predictor.

    :param satellite_file_names: 1-D list of paths to input files, each readable
        by `satellite_io.read_file`.
    :param num_values_per_low_res_channel: Number of reference values to keep
        for each low-resolution channel.
    :param num_values_per_high_res_channel: Number of reference values to keep
        for each high-resolution channel.
    :return: normalization_param_table_xarray: xarray table.  Metadata and
        variable names should make this table self-explanatory.
    """

    error_checking.assert_is_string_list(satellite_file_names)
    error_checking.assert_is_integer(num_values_per_low_res_channel)
    error_checking.assert_is_geq(num_values_per_low_res_channel, 10000)
    error_checking.assert_is_integer(num_values_per_high_res_channel)
    error_checking.assert_is_geq(num_values_per_high_res_channel, 50000)

    print('Reading data from: "{0:s}"...'.format(satellite_file_names[0]))
    first_table_xarray = satellite_io.read_file(satellite_file_names[0])

    low_res_indices = numpy.linspace(
        0, num_values_per_low_res_channel - 1,
        num=num_values_per_low_res_channel, dtype=int
    )
    high_res_indices = numpy.linspace(
        0, num_values_per_high_res_channel - 1,
        num=num_values_per_high_res_channel, dtype=int
    )

    metadata_dict = {
        HIGH_RES_WAVELENGTH_DIM:
            first_table_xarray.coords[HIGH_RES_WAVELENGTH_DIM].values,
        LOW_RES_WAVELENGTH_DIM:
            first_table_xarray.coords[LOW_RES_WAVELENGTH_DIM].values,
        HIGH_RES_SAMPLE_DIM: high_res_indices,
        LOW_RES_SAMPLE_DIM: low_res_indices
    }

    num_high_res_wavelengths = len(metadata_dict[HIGH_RES_WAVELENGTH_DIM])
    num_low_res_wavelengths = len(metadata_dict[LOW_RES_WAVELENGTH_DIM])

    main_data_dict = {
        BIDIRECTIONAL_REFLECTANCE_KEY: (
            (HIGH_RES_SAMPLE_DIM, HIGH_RES_WAVELENGTH_DIM),
            numpy.full(
                (num_values_per_high_res_channel, num_high_res_wavelengths),
                numpy.nan
            )
        ),
        BRIGHTNESS_TEMPERATURE_KEY: (
            (LOW_RES_SAMPLE_DIM, LOW_RES_WAVELENGTH_DIM),
            numpy.full(
                (num_values_per_low_res_channel, num_low_res_wavelengths),
                numpy.nan
            )
        )
    }

    normalization_param_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )
    npt = normalization_param_table_xarray

    num_files = len(satellite_file_names)
    num_values_per_hr_channel_per_file = int(numpy.ceil(
        float(num_values_per_high_res_channel) / num_files
    ))
    num_values_per_lr_channel_per_file = int(numpy.ceil(
        float(num_values_per_low_res_channel) / num_files
    ))

    satellite_file_names = 2 * satellite_file_names

    for i in range(len(satellite_file_names)):
        need_more_values = False

        for this_key in main_data_dict:
            need_more_values = (
                need_more_values or
                numpy.any(numpy.isnan(npt[this_key].values))
            )

        if not need_more_values:
            break

        print('\nReading data from: "{0:s}"...'.format(satellite_file_names[i]))
        satellite_table_xarray = satellite_io.read_file(satellite_file_names[i])

        for this_key in main_data_dict:
            predictor_matrix = satellite_table_xarray[this_key].values

            selected_time_indices = numpy.where(
                numpy.isfinite(predictor_matrix)
            )[0]
            selected_time_indices = numpy.unique(selected_time_indices)

            if len(selected_time_indices) > NUM_TIMES_PER_FILE_FOR_PARAMS:
                selected_time_indices = numpy.random.choice(
                    selected_time_indices, size=NUM_TIMES_PER_FILE_FOR_PARAMS,
                    replace=False
                )

            for j in range(len(selected_time_indices)):
                for k in range(predictor_matrix.shape[-1]):
                    nan_indices = numpy.where(
                        numpy.isnan(npt[this_key].values[:, k])
                    )[0]
                    if len(nan_indices) == 0:
                        continue

                    predictor_values = (
                        predictor_matrix[selected_time_indices[j], ..., k]
                    )
                    predictor_values = predictor_values[
                        numpy.isfinite(predictor_values)
                    ]
                    if len(predictor_values) == 0:
                        continue

                    multiplier = i + float(j + 1) / len(selected_time_indices)

                    if this_key == BIDIRECTIONAL_REFLECTANCE_KEY:
                        num_values_expected = int(numpy.round(
                            multiplier * num_values_per_hr_channel_per_file
                        ))
                        num_values_expected = min([
                            num_values_expected, num_values_per_high_res_channel
                        ])
                    else:
                        num_values_expected = int(numpy.round(
                            multiplier * num_values_per_lr_channel_per_file
                        ))
                        num_values_expected = min([
                            num_values_expected, num_values_per_low_res_channel
                        ])

                    first_index = nan_indices[0]
                    num_values_needed = num_values_expected - first_index

                    if num_values_needed < 1:
                        if this_key == BIDIRECTIONAL_REFLECTANCE_KEY:
                            num_values_needed = (
                                num_values_per_high_res_channel - first_index
                            )
                        else:
                            num_values_needed = (
                                num_values_per_low_res_channel - first_index
                            )

                    if len(predictor_values) > num_values_needed:
                        predictor_values = numpy.random.choice(
                            predictor_values, size=num_values_needed,
                            replace=False
                        )

                    print((
                        'Randomly selecting {0:d} {1:s} values from {2:d}th '
                        'time step and {3:d}th channel...'
                    ).format(
                        len(predictor_values),
                        this_key.upper(),
                        selected_time_indices[j] + 1,
                        k + 1
                    ))

                    last_index = first_index + len(predictor_values)

                    this_reference_value_matrix = npt[this_key].values
                    this_reference_value_matrix[first_index:last_index, k] = (
                        predictor_values
                    )
                    npt = npt.assign({
                        this_key: (
                            npt[this_key].dims, this_reference_value_matrix
                        )
                    })

    for this_key in main_data_dict:
        assert not numpy.any(numpy.isnan(npt[this_key].values))

    normalization_param_table_xarray = npt
    return normalization_param_table_xarray


def normalize_data(satellite_table_xarray, normalization_param_table_xarray):
    """Normalizes all predictor variables.

    :param satellite_table_xarray: xarray table in format returned by
        `satellite_io.read_file`.
    :param normalization_param_table_xarray: See doc for
        `get_normalization_params`.
    :return: satellite_table_xarray: Normalized version of input.
    """

    st = satellite_table_xarray
    npt = normalization_param_table_xarray

    high_res_wavelengths_orig_metres = st.coords[HIGH_RES_WAVELENGTH_DIM].values
    high_res_wavelengths_norm_metres = (
        npt.coords[HIGH_RES_WAVELENGTH_DIM].values
    )

    for j in range(len(high_res_wavelengths_orig_metres)):
        bidirectional_reflectance_matrix = (
            st[BIDIRECTIONAL_REFLECTANCE_KEY].values
        )

        k = numpy.where(
            numpy.absolute(
                high_res_wavelengths_norm_metres -
                high_res_wavelengths_orig_metres[j]
            )
            < TOLERANCE
        )[0][0]

        bidirectional_reflectance_matrix[..., j] = _normalize_one_variable(
            actual_values_new=bidirectional_reflectance_matrix[..., j],
            actual_values_training=
            npt[BIDIRECTIONAL_REFLECTANCE_KEY].values[:, k]
        )

        st = st.assign({
            BIDIRECTIONAL_REFLECTANCE_KEY: (
                st[BIDIRECTIONAL_REFLECTANCE_KEY].dims,
                bidirectional_reflectance_matrix
            )
        })

    low_res_wavelengths_orig_metres = st.coords[LOW_RES_WAVELENGTH_DIM].values
    low_res_wavelengths_norm_metres = npt.coords[LOW_RES_WAVELENGTH_DIM].values

    for j in range(len(low_res_wavelengths_orig_metres)):
        brightness_temp_matrix_kelvins = st[BRIGHTNESS_TEMPERATURE_KEY].values

        k = numpy.where(
            numpy.absolute(
                low_res_wavelengths_norm_metres -
                low_res_wavelengths_orig_metres[j]
            )
            < TOLERANCE
        )[0][0]

        brightness_temp_matrix_kelvins[..., j] = _normalize_one_variable(
            actual_values_new=brightness_temp_matrix_kelvins[..., j],
            actual_values_training=npt[BRIGHTNESS_TEMPERATURE_KEY].values[:, k]
        )

        st = st.assign({
            BRIGHTNESS_TEMPERATURE_KEY: (
                st[BRIGHTNESS_TEMPERATURE_KEY].dims,
                brightness_temp_matrix_kelvins
            )
        })

    satellite_table_xarray = st
    return satellite_table_xarray


def denormalize_data(satellite_table_xarray, normalization_param_table_xarray):
    """Denormalizes all predictor variables.

    :param satellite_table_xarray: xarray table in format returned by
        `satellite_io.read_file`.
    :param normalization_param_table_xarray: See doc for
        `get_normalization_params`.
    :return: satellite_table_xarray: Denormalized version of input.
    """

    st = satellite_table_xarray
    npt = normalization_param_table_xarray

    high_res_wavelengths_orig_metres = st.coords[HIGH_RES_WAVELENGTH_DIM].values
    high_res_wavelengths_norm_metres = (
        npt.coords[HIGH_RES_WAVELENGTH_DIM].values
    )

    for j in range(len(high_res_wavelengths_orig_metres)):
        bidirectional_reflectance_matrix = (
            st[BIDIRECTIONAL_REFLECTANCE_KEY].values
        )

        k = numpy.where(
            numpy.absolute(
                high_res_wavelengths_norm_metres -
                high_res_wavelengths_orig_metres[j]
            )
            < TOLERANCE
        )[0][0]

        bidirectional_reflectance_matrix[..., j] = _denorm_one_variable(
            normalized_values_new=bidirectional_reflectance_matrix[..., j],
            actual_values_training=
            npt[BIDIRECTIONAL_REFLECTANCE_KEY].values[:, k]
        )

        st = st.assign({
            BIDIRECTIONAL_REFLECTANCE_KEY: (
                st[BIDIRECTIONAL_REFLECTANCE_KEY].dims,
                bidirectional_reflectance_matrix
            )
        })

    low_res_wavelengths_orig_metres = st.coords[LOW_RES_WAVELENGTH_DIM].values
    low_res_wavelengths_norm_metres = npt.coords[LOW_RES_WAVELENGTH_DIM].values

    for j in range(len(low_res_wavelengths_orig_metres)):
        brightness_temp_matrix_kelvins = st[BRIGHTNESS_TEMPERATURE_KEY].values

        k = numpy.where(
            numpy.absolute(
                low_res_wavelengths_norm_metres -
                low_res_wavelengths_orig_metres[j]
            )
            < TOLERANCE
        )[0][0]

        brightness_temp_matrix_kelvins[..., j] = _denorm_one_variable(
            normalized_values_new=brightness_temp_matrix_kelvins[..., j],
            actual_values_training=npt[BRIGHTNESS_TEMPERATURE_KEY].values[:, k]
        )

        st = st.assign({
            BRIGHTNESS_TEMPERATURE_KEY: (
                st[BRIGHTNESS_TEMPERATURE_KEY].dims,
                brightness_temp_matrix_kelvins
            )
        })

    satellite_table_xarray = st
    return satellite_table_xarray


def write_file(normalization_param_table_xarray, zarr_file_name):
    """Writes normalization params to zarr file.

    :param normalization_param_table_xarray: xarray table in format returned by
        `read_file`.
    :param zarr_file_name: Path to output file.
    """

    error_checking.assert_is_string(zarr_file_name)
    if os.path.isdir(zarr_file_name):
        shutil.rmtree(zarr_file_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=zarr_file_name
    )

    normalization_param_table_xarray.to_zarr(store=zarr_file_name, mode='w')


def read_file(zarr_file_name):
    """Reads normalization params from zarr file.

    :param zarr_file_name: Path to input file.
    :return: normalization_param_table_xarray: xarray table.  Metadata and
        variable names should make the table self-explanatory.
    """

    return xarray.open_zarr(zarr_file_name)
