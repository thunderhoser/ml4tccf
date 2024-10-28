"""Helper methods for TC-structure predictions."""

import numpy
import xarray

MODEL_FILE_KEY = 'model_file_name'

EXAMPLE_DIM = 'example'
ENSEMBLE_MEMBER_DIM = 'ensemble_member'
DUMMY_ENSEMBLE_MEMBER_DIM = 'dummy_ensemble_member'
TARGET_FIELD_DIM = 'target_field'
CYCLONE_ID_CHAR_DIM = 'cyclone_id_char'

PREDICTION_KEY = 'prediction'
TARGET_KEY = 'target'
BASELINE_PREDICTION_KEY = 'baseline_prediction'
TARGET_TIME_KEY = 'target_time_unix_sec'
CYCLONE_ID_KEY = 'cyclone_id_string'


def get_ensemble_mean(prediction_table_xarray):
    """Computes ensemble mean for each set of predictions.

    :param prediction_table_xarray: xarray table in format returned by
        `prediction_io.read_file`.
    :return: prediction_table_xarray: Same but with only one prediction (the
        ensemble mean) per example and target variable.
    """

    ptx = prediction_table_xarray
    ptx = ptx.assign_coords({
        DUMMY_ENSEMBLE_MEMBER_DIM: numpy.array([0], dtype=int)
    })

    these_dim_keys = (EXAMPLE_DIM, TARGET_FIELD_DIM, DUMMY_ENSEMBLE_MEMBER_DIM)
    ptx = ptx.assign({
        PREDICTION_KEY: (
            these_dim_keys,
            numpy.mean(ptx[PREDICTION_KEY].values, axis=-1, keepdims=True)
        )
    })

    ptx = ptx.rename({DUMMY_ENSEMBLE_MEMBER_DIM: ENSEMBLE_MEMBER_DIM})
    prediction_table_xarray = ptx
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
        ptx.attrs[MODEL_FILE_KEY] for ptx in prediction_tables_xarray
    ]
    unique_model_file_names = list(set(model_file_names))

    if len(unique_model_file_names) > 1:
        error_string = (
            'Cannot concatenate predictions from different models.  In this '
            'case, predictions come from the following unique models:\n{0:s}'
        ).format(str(unique_model_file_names))

        raise ValueError(error_string)

    try:
        return xarray.concat(
            prediction_tables_xarray, dim=EXAMPLE_DIM, data_vars='all',
            coords='minimal', compat='identical', join='exact'
        )
    except:
        return xarray.concat(
            prediction_tables_xarray, dim=EXAMPLE_DIM, data_vars='all',
            coords='minimal', compat='identical'
        )


def concat_over_ensemble_members(prediction_tables_xarray):
    """Concatenates prediction data over many ensemble members.

    :param prediction_tables_xarray: 1-D list of input tables, in format
        returned by `prediction_io.read_file`.
    :return: prediction_table_xarray: xarray table with all predictions.
    :raises: ValueError: if models are not identical.
    """

    try:
        return xarray.concat(
            prediction_tables_xarray, dim=ENSEMBLE_MEMBER_DIM,
            data_vars='minimal', coords='minimal', compat='identical',
            join='exact'
        )
    except:
        return xarray.concat(
            prediction_tables_xarray, dim=ENSEMBLE_MEMBER_DIM,
            data_vars='minimal', coords='minimal', compat='identical'
        )
