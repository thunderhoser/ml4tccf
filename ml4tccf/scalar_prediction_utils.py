"""Helper methods for scalar predictions (x- and y-coords)."""

import os
import sys
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import error_checking

MODEL_FILE_KEY = 'model_file_name'
ISOTONIC_MODEL_FILE_KEY = 'isotonic_model_file_name'

EXAMPLE_DIM_KEY = 'example'
ENSEMBLE_MEMBER_DIM_KEY = 'ensemble_member'
DUMMY_ENSEMBLE_MEMBER_DIM_KEY = 'dummy_ensemble_member'
CYCLONE_ID_CHAR_DIM_KEY = 'cyclone_id_char'

PREDICTED_ROW_OFFSET_KEY = 'predicted_row_offset'
PREDICTED_COLUMN_OFFSET_KEY = 'predicted_column_offset'
ACTUAL_ROW_OFFSET_KEY = 'actual_row_offset'
ACTUAL_COLUMN_OFFSET_KEY = 'actual_column_offset'
GRID_SPACING_KEY = 'grid_spacing_low_res_km'
ACTUAL_CENTER_LATITUDE_KEY = 'actual_cyclone_center_latitude_deg_n'
TARGET_TIME_KEY = 'target_time_unix_sec'
CYCLONE_ID_KEY = 'cyclone_id_string'


def get_ensemble_mean(prediction_table_xarray):
    """Computes ensemble mean for each set of predictions.

    :param prediction_table_xarray: xarray table in format returned by
        `prediction_io.read_file`.
    :return: prediction_table_xarray: Same but with only one prediction (the
        ensemble mean) per example and target variable.
    """

    t = prediction_table_xarray

    try:
        t = t.assign_coords({
            DUMMY_ENSEMBLE_MEMBER_DIM_KEY: numpy.array([0], dtype=int)
        })
    except:
        t = t.assign_coords(
            DUMMY_ENSEMBLE_MEMBER_DIM_KEY=numpy.array([0], dtype=int)
        )

    these_dim_keys = (EXAMPLE_DIM_KEY, DUMMY_ENSEMBLE_MEMBER_DIM_KEY)

    t = t.assign({
        PREDICTED_ROW_OFFSET_KEY: (
            these_dim_keys,
            numpy.mean(
                t[PREDICTED_ROW_OFFSET_KEY].values, axis=-1, keepdims=True
            )
        ),
        PREDICTED_COLUMN_OFFSET_KEY: (
            these_dim_keys,
            numpy.mean(
                t[PREDICTED_COLUMN_OFFSET_KEY].values, axis=-1, keepdims=True
            )
        )
    })

    t = t.rename({DUMMY_ENSEMBLE_MEMBER_DIM_KEY: ENSEMBLE_MEMBER_DIM_KEY})
    prediction_table_xarray = t
    return prediction_table_xarray


def concat_over_examples(prediction_tables_xarray):
    """Concatenates prediction data over many examples.

    All tables must contain predictions from the same model.

    :param prediction_tables_xarray: 1-D list of input tables, in format
        returned by `prediction_io.read_file`.
    :return: prediction_table_xarray: xarray table with all predictions.
    :raises: ValueError: if models are not identical.
    """

    model_file_names = [
        t.attrs[MODEL_FILE_KEY] for t in prediction_tables_xarray
    ]
    unique_model_file_names = list(set(model_file_names))

    if len(unique_model_file_names) > 1:
        error_string = (
            'Cannot concatenate predictions from different models.  In this '
            'case, predictions come from the following unique models:\n{0:s}'
        ).format(str(unique_model_file_names))

        raise ValueError(error_string)

    model_file_names = [
        t.attrs[ISOTONIC_MODEL_FILE_KEY] for t in prediction_tables_xarray
    ]
    unique_model_file_names = list(set(model_file_names))

    if len(unique_model_file_names) > 1:
        error_string = (
            'Cannot concatenate predictions bias-corrected with different '
            'isotonic-regression models.  In this case, predictions come from '
            'the following unique models:\n{0:s}'
        ).format(str(unique_model_file_names))

        raise ValueError(error_string)

    return xarray.concat(
        prediction_tables_xarray, dim=EXAMPLE_DIM_KEY, data_vars='all',
        coords='minimal', compat='identical', join='exact'
    )


def concat_over_ensemble_members(prediction_tables_xarray,
                                 use_only_common_examples=False):
    """Concatenates prediction data over many ensemble members.

    :param prediction_tables_xarray: 1-D list of input tables, in format
        returned by `prediction_io.read_file`.
    :param use_only_common_examples: Boolean flag.  If True, will allow
        different tables to contain different examples and will concatenate only
        for common examples.
    :return: prediction_table_xarray: xarray table with all predictions.
    :raises: ValueError: if models are not identical.
    """

    error_checking.assert_is_boolean(use_only_common_examples)

    if not use_only_common_examples:
        return xarray.concat(
            prediction_tables_xarray, dim=ENSEMBLE_MEMBER_DIM_KEY,
            data_vars='minimal', coords='minimal', compat='identical',
            join='exact'
        )

    enhanced_cyclone_id_strings = [
        '{0:s}_{1:s}_{2:d}_{3:d}'.format(
            c,
            time_conversion.unix_sec_to_string(t, '%Y-%m-%d-%H%M%S'),
            int(numpy.round(row)),
            int(numpy.round(col))
        ) for c, t, row, col in zip(
            prediction_tables_xarray[0][CYCLONE_ID_KEY].values,
            prediction_tables_xarray[0][TARGET_TIME_KEY].values,
            prediction_tables_xarray[0][ACTUAL_ROW_OFFSET_KEY].values,
            prediction_tables_xarray[0][ACTUAL_COLUMN_OFFSET_KEY].values
        )
    ]

    enhanced_cyclone_id_strings = set(enhanced_cyclone_id_strings)

    for this_table in prediction_tables_xarray:
        these_enhanced_cyclone_id_strings = [
            '{0:s}_{1:s}_{2:d}_{3:d}'.format(
                c,
                time_conversion.unix_sec_to_string(t, '%Y-%m-%d-%H%M%S'),
                int(numpy.round(row)),
                int(numpy.round(col))
            ) for c, t, row, col in zip(
                this_table[CYCLONE_ID_KEY].values,
                this_table[TARGET_TIME_KEY].values,
                this_table[ACTUAL_ROW_OFFSET_KEY].values,
                this_table[ACTUAL_COLUMN_OFFSET_KEY].values
            )
        ]

        these_enhanced_cyclone_id_strings = set(
            these_enhanced_cyclone_id_strings
        )

        enhanced_cyclone_id_strings = enhanced_cyclone_id_strings.intersection(
            these_enhanced_cyclone_id_strings
        )

    assert len(enhanced_cyclone_id_strings) > 0

    enhanced_cyclone_id_strings = list(enhanced_cyclone_id_strings)
    enhanced_cyclone_id_strings.sort()

    for i in range(len(prediction_tables_xarray)):
        these_enhanced_cyclone_id_strings = [
            '{0:s}_{1:s}_{2:d}_{3:d}'.format(
                c,
                time_conversion.unix_sec_to_string(t, '%Y-%m-%d-%H%M%S'),
                int(numpy.round(row)),
                int(numpy.round(col))
            ) for c, t, row, col in zip(
                prediction_tables_xarray[i][CYCLONE_ID_KEY].values,
                prediction_tables_xarray[i][TARGET_TIME_KEY].values,
                prediction_tables_xarray[i][ACTUAL_ROW_OFFSET_KEY].values,
                prediction_tables_xarray[i][ACTUAL_COLUMN_OFFSET_KEY].values
            )
        ]

        good_indices = numpy.array([
            these_enhanced_cyclone_id_strings.index(c)
            for c in enhanced_cyclone_id_strings
        ], dtype=int)

        prediction_tables_xarray[i] = prediction_tables_xarray[i].isel({
            EXAMPLE_DIM_KEY: good_indices
        })

    grid_spacing_matrix_km = numpy.concatenate([
        numpy.expand_dims(t[GRID_SPACING_KEY].values, axis=1)
        for t in prediction_tables_xarray
    ], axis=1)

    grid_spacings_km = numpy.mean(grid_spacing_matrix_km, axis=1)

    for i in range(len(prediction_tables_xarray)):
        prediction_tables_xarray[i] = prediction_tables_xarray[i].assign({
            GRID_SPACING_KEY: (
                prediction_tables_xarray[i][GRID_SPACING_KEY].dims,
                grid_spacings_km
            )
        })

    return xarray.concat(
        prediction_tables_xarray, dim=ENSEMBLE_MEMBER_DIM_KEY,
        data_vars='minimal', coords='minimal', compat='identical',
        join='exact'
    )
