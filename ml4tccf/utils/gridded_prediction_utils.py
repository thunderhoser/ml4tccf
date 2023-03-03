"""Helper methods for scalar predictions (x- and y-coords)."""

import numpy
from ml4tccf.utils import scalar_prediction_utils

MODEL_FILE_KEY = 'model_file_name'

EXAMPLE_DIM = 'example'
ROW_DIM = 'row'
COLUMN_DIM = 'column'
ENSEMBLE_MEMBER_DIM = 'ensemble_member'
DUMMY_ENSEMBLE_MEMBER_DIM = 'dummy_ensemble_member'
CYCLONE_ID_CHAR_DIM = 'cyclone_id_char'

TARGET_MATRIX_KEY = 'target_matrix'
PREDICTION_MATRIX_KEY = 'prediction_matrix'
GRID_SPACING_KEY = 'grid_spacing_low_res_km'
ACTUAL_CENTER_LATITUDE_KEY = 'actual_cyclone_center_latitude_deg_n'
TARGET_TIME_KEY = 'target_time_unix_sec'
CYCLONE_ID_KEY = 'cyclone_id_string'


def get_ensemble_mean(prediction_table_xarray):
    """Computes ensemble mean for each set of predictions.

    :param prediction_table_xarray: xarray table in format returned by
        `prediction_io.read_file`.
    :return: prediction_table_xarray: Same but with only one prediction (the
        ensemble mean) per example and pixel.
    """

    t = prediction_table_xarray
    t = t.assign_coords({
        DUMMY_ENSEMBLE_MEMBER_DIM: numpy.array([0], dtype=int)
    })

    these_dim = (EXAMPLE_DIM, ROW_DIM, COLUMN_DIM, DUMMY_ENSEMBLE_MEMBER_DIM)

    t = t.assign({
        PREDICTION_MATRIX_KEY: (
            these_dim,
            numpy.mean(t[PREDICTION_MATRIX_KEY].values, axis=-1, keepdims=True)
        )
    })

    t = t.rename({DUMMY_ENSEMBLE_MEMBER_DIM: ENSEMBLE_MEMBER_DIM})
    prediction_table_xarray = t
    return prediction_table_xarray


def concat_over_examples(prediction_tables_xarray):
    """Concatenates prediction data over many examples.

    All tables must contain predictions from the same model.

    :param prediction_tables_xarray: 1-D list of input tables, in format
        returned by `prediction_io.read_file`.
    :return: prediction_table_xarray: xarray table with all predictions.
    """

    return scalar_prediction_utils.concat_over_examples(
        prediction_tables_xarray
    )
