"""Isotonic regression for TC-structure parameters."""

import os
import sys
import dill
import numpy
from sklearn.isotonic import IsotonicRegression

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import structure_prediction_utils as prediction_utils
import neural_net_utils as nn_utils
import neural_net_training_structure as nn_training


def train_models(prediction_table_xarray):
    """Trains one isotonic-regression model per target variable.

    F = number of target fields

    :param prediction_table_xarray: xarray table in format returned by
        `structure_prediction_io.read_file`.
    :return: target_field_to_model_object: Dictionary, where each key is the
        name of a target field and the corresponding value is a trained instance
        of `sklearn.isotonic.IsotonicRegression`.
    """

    ptx = prediction_table_xarray
    model_file_name = ptx.attrs[prediction_utils.MODEL_FILE_KEY]

    model_metafile_name = nn_utils.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = nn_utils.read_metafile(model_metafile_name)
    target_field_names = model_metadata_dict[nn_utils.TRAINING_OPTIONS_KEY][
        nn_training.TARGET_FIELDS_KEY
    ]

    num_target_fields = len(target_field_names)
    target_field_to_model_object = dict()

    for f in range(num_target_fields):
        this_model_object = IsotonicRegression(
            increasing=True, out_of_bounds='clip'
        )
        this_model_object.fit(
            X=numpy.mean(
                ptx[prediction_utils.PREDICTION_KEY].values[:, f, :],
                axis=-1
            ),
            y=ptx[prediction_utils.TARGET_KEY].values[:, f]
        )

        target_field_to_model_object[target_field_names[f]] = this_model_object

    return target_field_to_model_object


def apply_models(prediction_table_xarray, target_field_to_model_object):
    """Applies one isotonic-regression model per target variable.

    F = number of target fields

    :param prediction_table_xarray: xarray table in format returned by
        `structure_prediction_io.read_file`.
    :param target_field_to_model_object: See documentation for `train_models`.
    :return: prediction_table_xarray: Same as input but with bias-corrected
        predictions.
    """

    ptx = prediction_table_xarray
    model_file_name = ptx.attrs[prediction_utils.MODEL_FILE_KEY]

    model_metafile_name = nn_utils.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = nn_utils.read_metafile(model_metafile_name)
    target_field_names = model_metadata_dict[nn_utils.TRAINING_OPTIONS_KEY][
        nn_training.TARGET_FIELDS_KEY
    ]

    prediction_matrix = ptx[prediction_utils.PREDICTION_KEY].values
    num_target_fields = len(target_field_names)

    for f in range(num_target_fields):
        this_model_object = target_field_to_model_object[target_field_names[f]]

        orig_mean_predictions = numpy.mean(prediction_matrix[:, f, :], axis=-1)
        new_mean_predictions = this_model_object.predict(orig_mean_predictions)
        diff_matrix = numpy.expand_dims(
            new_mean_predictions - orig_mean_predictions, axis=-1
        )
        prediction_matrix[:, f, :] += diff_matrix

    try:
        r50_index = target_field_names.index(nn_training.R50_FIELD_NAME)
        r64_index = target_field_names.index(nn_training.R64_FIELD_NAME)
        prediction_matrix[:, r50_index, :] = numpy.maximum(
            prediction_matrix[:, r50_index, :],
            prediction_matrix[:, r64_index, :]
        )
    except:
        pass

    try:
        r34_index = target_field_names.index(nn_training.R34_FIELD_NAME)
        r50_index = target_field_names.index(nn_training.R50_FIELD_NAME)
        prediction_matrix[:, r34_index, :] = numpy.maximum(
            prediction_matrix[:, r34_index, :],
            prediction_matrix[:, r50_index, :]
        )
    except:
        pass

    return ptx.assign({
        prediction_utils.PREDICTION_KEY: (
            ptx[prediction_utils.PREDICTION_KEY].dims,
            prediction_matrix
        )
    })


def write_file(dill_file_name, target_field_to_model_object):
    """Writes set of isotonic-regression models to Dill file.

    :param dill_file_name: Path to output file.
    :param target_field_to_model_object: See doc for `train_models`.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=dill_file_name)

    dill_file_handle = open(dill_file_name, 'wb')
    dill.dump(target_field_to_model_object, dill_file_handle)
    dill_file_handle.close()


def read_file(dill_file_name):
    """Reads set of isotonic-regression models from Dill file.

    :param dill_file_name: Path to input file.
    :return: target_field_to_model_object: See doc for `train_models`.
    """

    error_checking.assert_file_exists(dill_file_name)

    dill_file_handle = open(dill_file_name, 'rb')
    target_field_to_model_object = dill.load(dill_file_handle)
    dill_file_handle.close()

    return target_field_to_model_object
