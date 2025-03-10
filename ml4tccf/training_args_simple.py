"""Contains list of input arguments for training a neural net."""

TEMPLATE_FILE_ARG_NAME = 'input_template_file_name'
OUTPUT_DIR_ARG_NAME = 'output_model_dir_name'
LAG_TIMES_ARG_NAME = 'lag_times_minutes'
LOW_RES_WAVELENGTHS_ARG_NAME = 'low_res_wavelengths_microns'
BATCH_SIZE_ARG_NAME = 'num_examples_per_batch'
MAX_EXAMPLES_PER_CYCLONE_ARG_NAME = 'max_examples_per_cyclone'
NUM_GRID_ROWS_ARG_NAME = 'num_rows_low_res'
NUM_GRID_COLUMNS_ARG_NAME = 'num_columns_low_res'

SHORT_TRACK_DIR_ARG_NAME = 'input_short_track_dir_name'
SHORT_TRACK_MAX_LEAD_ARG_NAME = 'short_track_max_lead_minutes'
SHORT_TRACK_DIFF_CENTERS_ARG_NAME = 'short_track_center_each_lag_diffly'
NUM_TRANSLATIONS_ARG_NAME = 'data_aug_num_translations'
MEAN_TRANSLATION_DIST_ARG_NAME = 'data_aug_mean_translation_low_res_px'
STDEV_TRANSLATION_DIST_ARG_NAME = 'data_aug_stdev_translation_low_res_px'
MEAN_TRANS_DIST_WITHIN_ARG_NAME = 'data_aug_within_mean_trans_px'
STDEV_TRANS_DIST_WITHIN_ARG_NAME = 'data_aug_within_stdev_trans_px'

SYNOPTIC_TIMES_ONLY_ARG_NAME = 'synoptic_times_only'
A_DECK_FILE_ARG_NAME = 'a_deck_file_name'
SCALAR_A_DECK_FIELDS_ARG_NAME = 'scalar_a_deck_field_names'
A_DECKS_AT_LEAST_6H_OLD_ARG_NAME = 'a_decks_at_least_6h_old'
REMOVE_NONTROPICAL_ARG_NAME = 'remove_nontropical_systems'
USE_SHUFFLED_DATA_ARG_NAME = 'use_shuffled_data'

SATELLITE_DIR_FOR_TRAINING_ARG_NAME = 'satellite_dir_name_for_training'
TRAINING_YEARS_ARG_NAME = 'training_years'
SATELLITE_DIR_FOR_VALIDATION_ARG_NAME = 'satellite_dir_name_for_validation'
VALIDATION_YEARS_ARG_NAME = 'validation_years'

NUM_EPOCHS_ARG_NAME = 'num_epochs'
NUM_TRAINING_BATCHES_ARG_NAME = 'num_training_batches_per_epoch'
NUM_VALIDATION_BATCHES_ARG_NAME = 'num_validation_batches_per_epoch'

PLATEAU_PATIENCE_ARG_NAME = 'plateau_patience_epochs'
PLATEAU_MULTIPLIER_ARG_NAME = 'plateau_learning_rate_multiplier'
EARLY_STOPPING_PATIENCE_ARG_NAME = 'early_stopping_patience_epochs'

TEMPLATE_FILE_HELP_STRING = (
    'Path to template file, containing model architecture.  This will be read '
    'by `neural_net.read_model`.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Trained model will be saved here.'
)
LAG_TIMES_HELP_STRING = 'List of lag times for predictors (satellite images).'
LOW_RES_WAVELENGTHS_HELP_STRING = (
    'List of wavelengths for low-resolution (infrared) satellite data.'
)
BATCH_SIZE_HELP_STRING = 'Batch size before data augmentation.'
MAX_EXAMPLES_PER_CYCLONE_HELP_STRING = (
    'Max number of examples per cyclone in one batch -- again, before data '
    'augmentation.'
)
NUM_GRID_ROWS_HELP_STRING = (
    'Number of grid rows to retain in low-resolution (infrared) satellite data.'
)
NUM_GRID_COLUMNS_HELP_STRING = (
    'Number of grid columns to retain in low-resolution (infrared) satellite '
    'data.'
)

SHORT_TRACK_DIR_HELP_STRING = (
    'Path to directory with short-track data (files therein will be found by '
    '`short_track_io.find_file` and read by `short_track_io.read_file`).  If '
    'you do not want the first guess centered on short track, make this '
    'argument the empty string "".'
)
SHORT_TRACK_MAX_LEAD_HELP_STRING = (
    '[used only if `{0:s}` is specified] Max lead time for short-track '
    'forecasts (any longer-lead forecast will not be used in first guess).'
).format(
    SHORT_TRACK_DIR_ARG_NAME
)
SHORT_TRACK_DIFF_CENTERS_HELP_STRING = (
    '[used only if `{0:s}` is specified] Boolean flag.  If 1 (0), for a given '
    'data sample, the first guess will involve a different (the same) lat/long '
    'center for each lag time.'
).format(
    SHORT_TRACK_DIR_ARG_NAME
)
NUM_TRANSLATIONS_HELP_STRING = (
    'Number of translations for each TC object.  Total number of data samples '
    'will be num_tc_objects * {0:s}.'
).format(
    NUM_TRANSLATIONS_ARG_NAME
)
MEAN_TRANSLATION_DIST_HELP_STRING = (
    'Mean whole-track translation distance (units of IR pixels, or low-res '
    'pixels).'
)
STDEV_TRANSLATION_DIST_HELP_STRING = (
    'Standard deviation of whole-track translation distance (units of IR '
    'pixels, or low-res pixels).'
)
MEAN_TRANS_DIST_WITHIN_HELP_STRING = (
    'Mean within-track translation distance (units of IR pixels, or low-res '
    'pixels).'
)
STDEV_TRANS_DIST_WITHIN_HELP_STRING = (
    'Standard deviation of within-track translation distance (units of IR '
    'pixels, or low-res pixels).'
)

SYNOPTIC_TIMES_ONLY_HELP_STRING = (
    '[used only if model is trained with Robert/Galina data] Boolean flag.  If '
    '1, only synoptic times can be target times.  If 0, any time can be a '
    'target time.'
)
A_DECK_FILE_HELP_STRING = (
    'Path to A-deck file, from which scalar predictors will be read.  If you '
    'do not want to use scalar predictors, leave this alone.'
)
SCALAR_A_DECK_FIELDS_HELP_STRING = (
    'List of scalar fields to use in predictors.  Each field must be a KEY '
    'listed at the top of a_deck_io.py.  If you do not want to use scalar '
    'predictors, leave this alone.'
)
A_DECKS_AT_LEAST_6H_OLD_HELP_STRING = (
    'Boolean flag.  If 1, when A-deck scalars are used in predictors, they '
    'will always be 6-12 h old.  If 0, they will be 0-6 h old.'
)
REMOVE_NONTROPICAL_HELP_STRING = (
    'Boolean flag.  If 1 (0), will train with only tropical systems (all '
    'systems).'
)
USE_SHUFFLED_DATA_HELP_STRING = (
    'Boolean flag.  If 1, with train with shuffled files.  If 0, will train '
    'with organized files (one file per cyclone-day or per cyclone).'
)

SATELLITE_DIR_FOR_TRAINING_HELP_STRING = (
    'Name of directory with training data.  Files therein will be found by '
    '`satellite_io.find_file` and read by `satellite_io.read_file`.'
)
TRAINING_YEARS_HELP_STRING = 'List of training years'
SATELLITE_DIR_FOR_VALIDATION_HELP_STRING = (
    'Name of directory with validation data.  Files therein will be found by '
    '`satellite_io.find_file` and read by `satellite_io.read_file`.'
)
VALIDATION_YEARS_HELP_STRING = 'List of validation years'

NUM_EPOCHS_HELP_STRING = 'Number of epochs.'
NUM_TRAINING_BATCHES_HELP_STRING = 'Number of training batches per epoch.'
NUM_VALIDATION_BATCHES_HELP_STRING = 'Number of validation batches per epoch.'

PLATEAU_PATIENCE_HELP_STRING = (
    'Training will be deemed to have reached "plateau" if validation loss has '
    'not decreased in the last N epochs, where N = {0:s}.'
).format(PLATEAU_PATIENCE_ARG_NAME)

PLATEAU_MULTIPLIER_HELP_STRING = (
    'If training reaches "plateau," learning rate will be multiplied by this '
    'value in range (0, 1).'
)
EARLY_STOPPING_PATIENCE_HELP_STRING = (
    'Training will be stopped early if validation loss has not decreased in '
    'the last N epochs, where N = {0:s}.'
).format(EARLY_STOPPING_PATIENCE_ARG_NAME)


def add_input_args(parser_object):
    """Adds input args to ArgumentParser object.

    :param parser_object: Instance of `argparse.ArgumentParser` (may already
        contain some input args).
    :return: parser_object: Same as input but with new args added.
    """

    parser_object.add_argument(
        '--' + TEMPLATE_FILE_ARG_NAME, type=str, required=True,
        help=TEMPLATE_FILE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
        help=OUTPUT_DIR_HELP_STRING
    )
    parser_object.add_argument(
        '--' + LAG_TIMES_ARG_NAME, type=int, nargs='+', required=True,
        help=LAG_TIMES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + LOW_RES_WAVELENGTHS_ARG_NAME, type=float, nargs='+',
        required=True, help=LOW_RES_WAVELENGTHS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + BATCH_SIZE_ARG_NAME, type=int, required=False, default=8,
        help=BATCH_SIZE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + MAX_EXAMPLES_PER_CYCLONE_ARG_NAME, type=int, required=False,
        default=2, help=MAX_EXAMPLES_PER_CYCLONE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NUM_GRID_ROWS_ARG_NAME, type=int, required=False, default=626,
        help=NUM_GRID_ROWS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NUM_GRID_COLUMNS_ARG_NAME, type=int, required=False, default=626,
        help=NUM_GRID_COLUMNS_HELP_STRING
    )

    parser_object.add_argument(
        '--' + SHORT_TRACK_DIR_ARG_NAME, type=str, required=True,
        help=SHORT_TRACK_DIR_HELP_STRING
    )
    parser_object.add_argument(
        '--' + SHORT_TRACK_MAX_LEAD_ARG_NAME, type=int, required=False,
        default=350, help=SHORT_TRACK_MAX_LEAD_HELP_STRING
    )
    parser_object.add_argument(
        '--' + SHORT_TRACK_DIFF_CENTERS_ARG_NAME, type=int, required=False,
        default=0, help=SHORT_TRACK_DIFF_CENTERS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NUM_TRANSLATIONS_ARG_NAME, type=int, required=False, default=8,
        help=NUM_TRANSLATIONS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + MEAN_TRANSLATION_DIST_ARG_NAME, type=float, required=True,
        help=MEAN_TRANSLATION_DIST_HELP_STRING
    )
    parser_object.add_argument(
        '--' + STDEV_TRANSLATION_DIST_ARG_NAME, type=float, required=True,
        help=STDEV_TRANSLATION_DIST_HELP_STRING
    )
    parser_object.add_argument(
        '--' + MEAN_TRANS_DIST_WITHIN_ARG_NAME, type=float, required=True,
        help=MEAN_TRANS_DIST_WITHIN_HELP_STRING
    )
    parser_object.add_argument(
        '--' + STDEV_TRANS_DIST_WITHIN_ARG_NAME, type=float, required=True,
        help=STDEV_TRANS_DIST_WITHIN_HELP_STRING
    )

    parser_object.add_argument(
        '--' + SYNOPTIC_TIMES_ONLY_ARG_NAME, type=int, required=False,
        default=1, help=SYNOPTIC_TIMES_ONLY_HELP_STRING
    )
    parser_object.add_argument(
        '--' + A_DECK_FILE_ARG_NAME, type=str, required=False, default='',
        help=A_DECK_FILE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + SCALAR_A_DECK_FIELDS_ARG_NAME, type=str, nargs='+',
        required=False, default=[''], help=SCALAR_A_DECK_FIELDS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + A_DECKS_AT_LEAST_6H_OLD_ARG_NAME, type=int, required=False,
        default=1, help=A_DECKS_AT_LEAST_6H_OLD_HELP_STRING
    )
    parser_object.add_argument(
        '--' + REMOVE_NONTROPICAL_ARG_NAME, type=int, required=False, default=0,
        help=REMOVE_NONTROPICAL_HELP_STRING
    )
    parser_object.add_argument(
        '--' + USE_SHUFFLED_DATA_ARG_NAME, type=int, required=True,
        help=USE_SHUFFLED_DATA_HELP_STRING
    )

    parser_object.add_argument(
        '--' + SATELLITE_DIR_FOR_TRAINING_ARG_NAME, type=str, required=True,
        help=SATELLITE_DIR_FOR_TRAINING_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TRAINING_YEARS_ARG_NAME, type=int, nargs='+', required=True,
        help=TRAINING_YEARS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + SATELLITE_DIR_FOR_VALIDATION_ARG_NAME, type=str, required=True,
        help=SATELLITE_DIR_FOR_VALIDATION_HELP_STRING
    )
    parser_object.add_argument(
        '--' + VALIDATION_YEARS_ARG_NAME, type=int, nargs='+', required=True,
        help=VALIDATION_YEARS_HELP_STRING
    )

    parser_object.add_argument(
        '--' + NUM_EPOCHS_ARG_NAME, type=int, required=False, default=1000,
        help=NUM_EPOCHS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NUM_TRAINING_BATCHES_ARG_NAME, type=int, required=False,
        default=32, help=NUM_TRAINING_BATCHES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NUM_VALIDATION_BATCHES_ARG_NAME, type=int, required=False,
        default=16, help=NUM_VALIDATION_BATCHES_HELP_STRING
    )

    parser_object.add_argument(
        '--' + PLATEAU_PATIENCE_ARG_NAME, type=int, required=False,
        default=10, help=PLATEAU_PATIENCE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + PLATEAU_MULTIPLIER_ARG_NAME, type=float, required=False,
        default=0.6, help=PLATEAU_MULTIPLIER_HELP_STRING
    )
    parser_object.add_argument(
        '--' + EARLY_STOPPING_PATIENCE_ARG_NAME, type=int, required=False,
        default=50, help=EARLY_STOPPING_PATIENCE_HELP_STRING
    )

    return parser_object
