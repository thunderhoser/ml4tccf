"""Plots generator output for TC-structure prediction."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import matplotlib.colors

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import file_system_utils
import imagemagick_utils
import a_deck_io
import neural_net_utils as nn_utils
import neural_net_training_structure as nn_training
import satellite_plotting
import plotting_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

A_DECK_FIELD_TO_FANCY_NAME = {
    a_deck_io.ABSOLUTE_LATITUDE_KEY: r'|$\phi$|',
    a_deck_io.LONGITUDE_SINE_KEY: r'sin($\lambda$)',
    a_deck_io.LONGITUDE_COSINE_KEY: r'cos($\lambda$)',
    a_deck_io.INTENSITY_KEY: r'$V_{max}$',
    a_deck_io.SEA_LEVEL_PRESSURE_KEY: r'$p_{min}$',
    a_deck_io.UNNORM_TROPICAL_FLAG_KEY: r'$\mathcal{I}$(trop)',
    a_deck_io.UNNORM_SUBTROPICAL_FLAG_KEY: r'$\mathcal{I}$(sub)',
    a_deck_io.UNNORM_EXTRATROPICAL_FLAG_KEY: r'$\mathcal{I}$(extra)',
    a_deck_io.UNNORM_DISTURBANCE_FLAG_KEY: r'$\mathcal{I}$(disturb)',
    a_deck_io.WIND_RADIUS_34KT_KEY: 'R34',
    a_deck_io.WIND_RADIUS_50KT_KEY: 'R50',
    a_deck_io.WIND_RADIUS_64KT_KEY: 'R64',
    a_deck_io.MAX_WIND_RADIUS_KEY: 'RMW'
}

TARGET_FIELD_TO_FANCY_NAME = {
    nn_training.INTENSITY_FIELD_NAME: r'$V_{max}$',
    nn_training.R34_FIELD_NAME: 'R34',
    nn_training.R50_FIELD_NAME: 'R50',
    nn_training.R64_FIELD_NAME: 'R64',
    nn_training.RMW_FIELD_NAME: 'RMW'
}

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

PANEL_SIZE_PX = int(2.5e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

MODEL_FILE_ARG_NAME = 'input_model_file_name'
SATELLITE_DIR_ARG_NAME = 'input_satellite_dir_name'
A_DECK_FILE_ARG_NAME = 'input_a_deck_file_name'
ARE_DATA_NORMALIZED_ARG_NAME = 'are_data_normalized'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to neural-net model (readable by `neural_net_utils.read_model`).'
)
SATELLITE_DIR_HELP_STRING = (
    'Path to directory with satellite (image) predictors.  Files therein will '
    'be found by `satellite_io.find_file` and read by `satellite_io.read_file`.'
)
A_DECK_FILE_HELP_STRING = (
    'Path to file with A-deck (scalar) predictors.  Will be read by '
    '`a_deck_io.read_file`.'
)
ARE_DATA_NORMALIZED_HELP_STRING = (
    'Boolean flag.  If 1 (0), this script will assume that input data are '
    '(un)normalized.'
)
NUM_EXAMPLES_HELP_STRING = 'Number of data examples to plot.'
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SATELLITE_DIR_ARG_NAME, type=str, required=True,
    help=SATELLITE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + A_DECK_FILE_ARG_NAME, type=str, required=True,
    help=A_DECK_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ARE_DATA_NORMALIZED_ARG_NAME, type=int, required=True,
    help=ARE_DATA_NORMALIZED_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=True,
    help=NUM_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(model_file_name, satellite_dir_name, a_deck_file_name,
         are_data_normalized, num_examples, output_dir_name):
    """Plots generator output for TC-structure prediction.

    This is effectively the main method.

    :param model_file_name: See documentation at top of this script.
    :param satellite_dir_name: Same.
    :param a_deck_file_name: Same.
    :param are_data_normalized: Same.
    :param num_examples: Same.
    :param output_dir_name: Same.
    """

    model_metafile_name = nn_utils.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = nn_utils.read_metafile(model_metafile_name)

    tod = model_metadata_dict[nn_utils.TRAINING_OPTIONS_KEY]
    tod[nn_training.SATELLITE_DIRECTORY_KEY] = satellite_dir_name
    tod[nn_training.BATCH_SIZE_KEY] = 1
    tod[nn_training.A_DECK_FILE_KEY] = a_deck_file_name
    model_metadata_dict[nn_utils.TRAINING_OPTIONS_KEY] = tod

    wavelengths_microns = tod[nn_training.LOW_RES_WAVELENGTHS_KEY]
    lag_times_minutes = tod[nn_training.LAG_TIMES_KEY]
    a_deck_field_names = tod[nn_training.SCALAR_A_DECK_FIELDS_KEY]
    target_field_names = tod[nn_training.TARGET_FIELDS_KEY]

    generator_handle = nn_training.data_generator_shuffled(
        tod, return_cyclone_ids=True
    )
    print(SEPARATOR_STRING)

    for i in range(num_examples):
        (
            predictor_matrices,
            target_matrix,
            cyclone_id_strings,
            target_times_unix_sec
        ) = next(generator_handle)

        print(SEPARATOR_STRING)

        base_title_string = '{0:s} at {1:s}'.format(
            cyclone_id_strings[0],
            time_conversion.unix_sec_to_string(
                target_times_unix_sec[0], '%Y-%m-%d-%H%M'
            )
        )

        for f in range(len(a_deck_field_names)):
            if f == 0:
                base_title_string += '\n'
            else:
                if numpy.mod(f, 4) == 0:
                    base_title_string += '\n'
                else:
                    base_title_string += '; '

            base_title_string += '{0:s} = {1:.2f}'.format(
                A_DECK_FIELD_TO_FANCY_NAME[a_deck_field_names[f]],
                predictor_matrices[1][0, f]
            )

        base_title_string += '\n'

        for f in range(len(target_field_names)):
            base_title_string += '{0:s} = {1:.2f}'.format(
                TARGET_FIELD_TO_FANCY_NAME[target_field_names[f]],
                target_matrix[0, f] *
                nn_training.TARGET_NAME_TO_CONV_FACTOR[target_field_names[f]]
            )

        for k in range(len(lag_times_minutes)):
            panel_file_names = [''] * len(wavelengths_microns)

            for j in range(len(wavelengths_microns)):
                figure_object, axes_object = pyplot.subplots(
                    1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
                )

                if are_data_normalized:
                    colour_map_object = pyplot.get_cmap('seismic', lut=1001)
                    colour_norm_object = matplotlib.colors.Normalize(
                        vmin=-3., vmax=3.
                    )
                else:
                    colour_map_object, colour_norm_object = (
                        satellite_plotting.get_colour_scheme_for_brightness_temp()
                    )

                satellite_plotting.plot_2d_grid_no_coords(
                    data_matrix=predictor_matrices[0][0, ..., k, j],
                    axes_object=axes_object,
                    plotting_brightness_temp=True,
                    cbar_orientation_string=None,
                    colour_map_object=colour_map_object,
                    colour_norm_object=colour_norm_object
                )

                if j == 0:
                    title_string = (
                        'Lag time = {0:.0f} min; wavelength = {1:.3f} microns'
                        '\n{2:s}'
                    ).format(
                        lag_times_minutes[k],
                        wavelengths_microns[j],
                        base_title_string
                    )
                else:
                    title_string = (
                        'Lag time = {0:.0f} min; wavelength = {1:.3f} microns'
                    ).format(
                        lag_times_minutes[k],
                        wavelengths_microns[j]
                    )

                axes_object.set_title(title_string)

                panel_file_names[j] = (
                    '{0:s}/{1:s}_{2:s}_{3:03d}minutes_{4:06.3f}microns.jpg'
                ).format(
                    output_dir_name,
                    cyclone_id_strings[0],
                    time_conversion.unix_sec_to_string(
                        target_times_unix_sec[0], '%Y-%m-%d-%H%M'
                    ),
                    lag_times_minutes[k],
                    wavelengths_microns[j]
                )

                file_system_utils.mkdir_recursive_if_necessary(
                    file_name=panel_file_names[j]
                )

                print('Saving figure to file: "{0:s}"...'.format(
                    panel_file_names[j]
                ))
                figure_object.savefig(
                    panel_file_names[j],
                    dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
                )
                pyplot.close(figure_object)

                imagemagick_utils.resize_image(
                    input_file_name=panel_file_names[j],
                    output_file_name=panel_file_names[j],
                    output_size_pixels=PANEL_SIZE_PX
                )

            concat_figure_file_name = (
                '{0:s}/{1:s}_{2:s}_{3:03d}minutes.jpg'
            ).format(
                output_dir_name,
                cyclone_id_strings[0],
                time_conversion.unix_sec_to_string(
                    target_times_unix_sec[0], '%Y-%m-%d-%H%M'
                ),
                lag_times_minutes[k]
            )

            print('Concatenating panels to file: "{0:s}"...'.format(
                concat_figure_file_name
            ))
            plotting_utils.concat_panels(
                panel_file_names=panel_file_names,
                concat_figure_file_name=concat_figure_file_name
            )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        satellite_dir_name=getattr(INPUT_ARG_OBJECT, SATELLITE_DIR_ARG_NAME),
        a_deck_file_name=getattr(INPUT_ARG_OBJECT, A_DECK_FILE_ARG_NAME),
        are_data_normalized=bool(getattr(
            INPUT_ARG_OBJECT, ARE_DATA_NORMALIZED_ARG_NAME
        )),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
