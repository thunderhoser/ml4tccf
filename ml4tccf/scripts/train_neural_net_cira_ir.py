"""Trains neural net with CIRA IR data."""

import os
import argparse
import numpy
from ml4tccf.machine_learning import neural_net_utils as nn_utils
from ml4tccf.machine_learning import neural_net_training_cira_ir as nn_training
from ml4tccf.scripts import training_args_cira_ir as training_args

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = training_args.add_input_args(parser_object=INPUT_ARG_PARSER)


def _run(template_file_name, output_dir_name, lag_times_minutes,
         num_examples_per_batch, max_examples_per_cyclone,
         num_rows_low_res, num_columns_low_res, data_aug_num_translations,
         data_aug_mean_translation_low_res_px,
         data_aug_stdev_translation_low_res_px, synoptic_times_only,
         a_deck_file_name, scalar_a_deck_field_names,
         remove_nontropical_systems,
         satellite_dir_name_for_training, training_years,
         satellite_dir_name_for_validation, validation_years,
         num_epochs,
         num_training_batches_per_epoch, num_validation_batches_per_epoch,
         plateau_patience_epochs, plateau_learning_rate_multiplier,
         early_stopping_patience_epochs):
    """Trains neural net.

    This is effectively the main method.

    :param template_file_name: See documentation at top of training_args.py.
    :param output_dir_name: Same.
    :param lag_times_minutes: Same.
    :param num_examples_per_batch: Same.
    :param max_examples_per_cyclone: Same.
    :param num_rows_low_res: Same.
    :param num_columns_low_res: Same.
    :param data_aug_num_translations: Same.
    :param data_aug_mean_translation_low_res_px: Same.
    :param data_aug_stdev_translation_low_res_px: Same.
    :param synoptic_times_only: Same.
    :param a_deck_file_name: Same.
    :param scalar_a_deck_field_names: Same.
    :param remove_nontropical_systems: Same.
    :param satellite_dir_name_for_training: Same.
    :param training_years: Same.
    :param satellite_dir_name_for_validation: Same.
    :param validation_years: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validation_batches_per_epoch: Same.
    :param plateau_patience_epochs: Same.
    :param plateau_learning_rate_multiplier: Same.
    :param early_stopping_patience_epochs: Same.
    """

    if num_rows_low_res <= 0:
        num_rows_low_res = None
    if num_columns_low_res <= 0:
        num_columns_low_res = None

    training_option_dict = {
        nn_utils.SATELLITE_DIRECTORY_KEY: satellite_dir_name_for_training,
        nn_utils.YEARS_KEY: training_years,
        nn_utils.LAG_TIMES_KEY: lag_times_minutes,
        nn_utils.HIGH_RES_WAVELENGTHS_KEY: numpy.array([]),
        nn_utils.LOW_RES_WAVELENGTHS_KEY: numpy.array([11.2]),
        nn_utils.BATCH_SIZE_KEY: num_examples_per_batch,
        nn_utils.MAX_EXAMPLES_PER_CYCLONE_KEY: max_examples_per_cyclone,
        nn_utils.NUM_GRID_ROWS_KEY: num_rows_low_res,
        nn_utils.NUM_GRID_COLUMNS_KEY: num_columns_low_res,
        nn_utils.DATA_AUG_NUM_TRANS_KEY: data_aug_num_translations,
        nn_utils.DATA_AUG_MEAN_TRANS_KEY:
            data_aug_mean_translation_low_res_px,
        nn_utils.DATA_AUG_STDEV_TRANS_KEY:
            data_aug_stdev_translation_low_res_px,
        nn_utils.LAG_TIME_TOLERANCE_KEY: 0,
        nn_utils.MAX_MISSING_LAG_TIMES_KEY: 0,
        nn_utils.MAX_INTERP_GAP_KEY: 0,
        nn_utils.SENTINEL_VALUE_KEY: -10.,
        nn_utils.TARGET_SMOOOTHER_STDEV_KEY: 1e-6,
        nn_utils.SYNOPTIC_TIMES_ONLY_KEY: synoptic_times_only,
        nn_utils.A_DECK_FILE_KEY: a_deck_file_name,
        nn_utils.SCALAR_A_DECK_FIELDS_KEY: scalar_a_deck_field_names,
        nn_utils.REMOVE_NONTROPICAL_KEY: remove_nontropical_systems
    }

    validation_option_dict = {
        nn_utils.SATELLITE_DIRECTORY_KEY: satellite_dir_name_for_validation,
        nn_utils.YEARS_KEY: validation_years,
        nn_utils.LAG_TIME_TOLERANCE_KEY: 0,
        nn_utils.MAX_MISSING_LAG_TIMES_KEY: 0,
        nn_utils.MAX_INTERP_GAP_KEY: 0
    }

    print('Reading model template from: "{0:s}"...'.format(template_file_name))
    model_object = nn_utils.read_model(keras_file_name=template_file_name)

    model_metafile_name = nn_utils.find_metafile(
        model_dir_name=os.path.split(template_file_name)[0],
        raise_error_if_missing=True
    )
    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = nn_utils.read_metafile(model_metafile_name)
    mmd = model_metadata_dict
    training_option_dict[nn_utils.SEMANTIC_SEG_FLAG_KEY] = (
        mmd[nn_utils.TRAINING_OPTIONS_KEY][nn_utils.SEMANTIC_SEG_FLAG_KEY]
    )

    nn_training.train_model(
        model_object=model_object, output_dir_name=output_dir_name,
        num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_option_dict=training_option_dict,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_option_dict=validation_option_dict,
        loss_function_string=mmd[nn_utils.LOSS_FUNCTION_KEY],
        optimizer_function_string=mmd[nn_utils.OPTIMIZER_FUNCTION_KEY],
        plateau_patience_epochs=plateau_patience_epochs,
        plateau_learning_rate_multiplier=plateau_learning_rate_multiplier,
        early_stopping_patience_epochs=early_stopping_patience_epochs,
        cnn_architecture_dict=mmd[nn_utils.CNN_ARCHITECTURE_KEY],
        temporal_cnn_architecture_dict=
        mmd[nn_utils.TEMPORAL_CNN_ARCHITECTURE_KEY],
        temporal_convnext_architecture_dict=
        mmd[nn_utils.TEMPORAL_CONVNEXT_ARCHITECTURE_KEY],
        u_net_architecture_dict=mmd[nn_utils.U_NET_ARCHITECTURE_KEY]
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        template_file_name=getattr(
            INPUT_ARG_OBJECT, training_args.TEMPLATE_FILE_ARG_NAME
        ),
        output_dir_name=getattr(
            INPUT_ARG_OBJECT, training_args.OUTPUT_DIR_ARG_NAME
        ),
        lag_times_minutes=numpy.array(
            getattr(INPUT_ARG_OBJECT, training_args.LAG_TIMES_ARG_NAME),
            dtype=int
        ),
        num_examples_per_batch=getattr(
            INPUT_ARG_OBJECT, training_args.BATCH_SIZE_ARG_NAME
        ),
        max_examples_per_cyclone=getattr(
            INPUT_ARG_OBJECT, training_args.MAX_EXAMPLES_PER_CYCLONE_ARG_NAME
        ),
        num_rows_low_res=getattr(
            INPUT_ARG_OBJECT, training_args.NUM_GRID_ROWS_ARG_NAME
        ),
        num_columns_low_res=getattr(
            INPUT_ARG_OBJECT, training_args.NUM_GRID_COLUMNS_ARG_NAME
        ),
        data_aug_num_translations=getattr(
            INPUT_ARG_OBJECT, training_args.DATA_AUG_NUM_TRANS_ARG_NAME
        ),
        data_aug_mean_translation_low_res_px=getattr(
            INPUT_ARG_OBJECT, training_args.DATA_AUG_MEAN_TRANS_ARG_NAME
        ),
        data_aug_stdev_translation_low_res_px=getattr(
            INPUT_ARG_OBJECT, training_args.DATA_AUG_STDEV_TRANS_ARG_NAME
        ),
        synoptic_times_only=bool(getattr(
            INPUT_ARG_OBJECT, training_args.SYNOPTIC_TIMES_ONLY_ARG_NAME
        )),
        a_deck_file_name=getattr(
            INPUT_ARG_OBJECT, training_args.A_DECK_FILE_ARG_NAME
        ),
        scalar_a_deck_field_names=getattr(
            INPUT_ARG_OBJECT, training_args.SCALAR_A_DECK_FIELDS_ARG_NAME
        ),
        remove_nontropical_systems=bool(getattr(
            INPUT_ARG_OBJECT, training_args.REMOVE_NONTROPICAL_ARG_NAME
        )),
        satellite_dir_name_for_training=getattr(
            INPUT_ARG_OBJECT, training_args.SATELLITE_DIR_FOR_TRAINING_ARG_NAME
        ),
        training_years=numpy.array(
            getattr(INPUT_ARG_OBJECT, training_args.TRAINING_YEARS_ARG_NAME),
            dtype=int
        ),
        satellite_dir_name_for_validation=getattr(
            INPUT_ARG_OBJECT,
            training_args.SATELLITE_DIR_FOR_VALIDATION_ARG_NAME
        ),
        validation_years=numpy.array(
            getattr(INPUT_ARG_OBJECT, training_args.VALIDATION_YEARS_ARG_NAME),
            dtype=int
        ),
        num_epochs=getattr(INPUT_ARG_OBJECT, training_args.NUM_EPOCHS_ARG_NAME),
        num_training_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, training_args.NUM_TRAINING_BATCHES_ARG_NAME
        ),
        num_validation_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, training_args.NUM_VALIDATION_BATCHES_ARG_NAME
        ),
        plateau_patience_epochs=getattr(
            INPUT_ARG_OBJECT, training_args.PLATEAU_PATIENCE_ARG_NAME
        ),
        plateau_learning_rate_multiplier=getattr(
            INPUT_ARG_OBJECT, training_args.PLATEAU_MULTIPLIER_ARG_NAME
        ),
        early_stopping_patience_epochs=getattr(
            INPUT_ARG_OBJECT, training_args.EARLY_STOPPING_PATIENCE_ARG_NAME
        )
    )
