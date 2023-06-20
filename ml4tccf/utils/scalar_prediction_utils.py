"""Helper methods for scalar predictions (x- and y-coords)."""

import numpy
import xarray

MODEL_FILE_KEY = 'model_file_name'

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
    t = t.assign_coords({
        DUMMY_ENSEMBLE_MEMBER_DIM_KEY: numpy.array([0], dtype=int)
    })

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

    return xarray.concat(
        prediction_tables_xarray, dim=EXAMPLE_DIM_KEY, data_vars='all',
        coords='minimal', compat='identical', join='exact'
    )


def concat_over_ensemble_members(prediction_tables_xarray):
    """Concatenates prediction data over many ensemble members.

    :param prediction_tables_xarray: 1-D list of input tables, in format
        returned by `prediction_io.read_file`.
    :return: prediction_table_xarray: xarray table with all predictions.
    :raises: ValueError: if models are not identical.
    """

    return xarray.concat(
        prediction_tables_xarray, dim=ENSEMBLE_MEMBER_DIM_KEY, data_vars='all',
        coords='minimal', compat='identical', join='exact'
    )
