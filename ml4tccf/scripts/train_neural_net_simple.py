"""Trains neural net."""

import os
import argparse
import numpy
from ml4tccf.machine_learning import neural_net
from ml4tccf.scripts import training_args_simple as training_args

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = training_args.add_input_args(parser_object=INPUT_ARG_PARSER)


def _run(template_file_name, output_dir_name, lag_times_minutes,
         num_examples_per_batch, max_examples_per_cyclone,
         num_rows_low_res, num_columns_low_res, data_aug_num_translations,
         data_aug_mean_translation_low_res_px,
         data_aug_stdev_translation_low_res_px,
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

    # TODO(thunderhoser): This is a HACK.
    if '2bands' in satellite_dir_name_for_training:
        low_res_wavelengths_microns = numpy.array([3.9, 11.2])
    else:
        low_res_wavelengths_microns = numpy.array([11.2])

    training_option_dict = {
        neural_net.SATELLITE_DIRECTORY_KEY: satellite_dir_name_for_training,
        neural_net.YEARS_KEY: training_years,
        neural_net.LAG_TIMES_KEY: lag_times_minutes,
        neural_net.HIGH_RES_WAVELENGTHS_KEY: numpy.array([]),
        neural_net.LOW_RES_WAVELENGTHS_KEY: low_res_wavelengths_microns,
        neural_net.BATCH_SIZE_KEY: num_examples_per_batch,
        neural_net.MAX_EXAMPLES_PER_CYCLONE_KEY: max_examples_per_cyclone,
        neural_net.NUM_GRID_ROWS_KEY: num_rows_low_res,
        neural_net.NUM_GRID_COLUMNS_KEY: num_columns_low_res,
        neural_net.DATA_AUG_NUM_TRANS_KEY: data_aug_num_translations,
        neural_net.DATA_AUG_MEAN_TRANS_KEY:
            data_aug_mean_translation_low_res_px,
        neural_net.DATA_AUG_STDEV_TRANS_KEY:
            data_aug_stdev_translation_low_res_px,
        neural_net.LAG_TIME_TOLERANCE_KEY: 0,
        neural_net.MAX_MISSING_LAG_TIMES_KEY: 0,
        neural_net.MAX_INTERP_GAP_KEY: 0,
        neural_net.SENTINEL_VALUE_KEY: -10.,
        neural_net.TARGET_SMOOOTHER_STDEV_KEY: 1e-6
    }

    validation_option_dict = {
        neural_net.SATELLITE_DIRECTORY_KEY: satellite_dir_name_for_validation,
        neural_net.YEARS_KEY: validation_years,
        neural_net.LAG_TIME_TOLERANCE_KEY: 0,
        neural_net.MAX_MISSING_LAG_TIMES_KEY: 0,
        neural_net.MAX_INTERP_GAP_KEY: 0
    }

    print('Reading model template from: "{0:s}"...'.format(template_file_name))
    model_object = neural_net.read_model(hdf5_file_name=template_file_name)

    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(template_file_name)[0],
        raise_error_if_missing=True
    )
    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    training_option_dict[neural_net.SEMANTIC_SEG_FLAG_KEY] = (
        model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY][
            neural_net.SEMANTIC_SEG_FLAG_KEY
        ]
    )

    neural_net.train_model_simple(
        model_object=model_object, output_dir_name=output_dir_name,
        num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_option_dict=training_option_dict,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_option_dict=validation_option_dict,
        loss_function_string=model_metadata_dict[neural_net.LOSS_FUNCTION_KEY],
        optimizer_function_string=
        model_metadata_dict[neural_net.OPTIMIZER_FUNCTION_KEY],
        plateau_patience_epochs=plateau_patience_epochs,
        plateau_learning_rate_multiplier=plateau_learning_rate_multiplier,
        early_stopping_patience_epochs=early_stopping_patience_epochs,
        architecture_dict=model_metadata_dict[neural_net.ARCHITECTURE_KEY],
        is_model_bnn=model_metadata_dict[neural_net.IS_MODEL_BNN_KEY]
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
