"""Plots comparison among all models for WAF 2024 paper."""

import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.plotting import imagemagick_utils

OUTPUT_DIR_NAME = '/home/ralager/tccf_paper_2024/model_comparisons'

DEFAULT_FONT_SIZE = 30
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

OPACITY = 0.8

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 9
FIGURE_RESOLUTION_DPI = 300

MEDIAN_ERROR_DICT = {
    r'$<$ 64 kt': {
        'ARCHER-2: SWIR': 46,
        'ARCHER-2: ASCAT': 26,
        'ST21': 23.5,
        'GeoCenter: tropical': (25.9, 0.5, 0.7),
        'GeoCenter: all sys': (26.1, 0.6, 0.5)
    },
    '[64, 83) kt': {
        'ARCHER-2: SWIR': 41,
        'ARCHER-2: ASCAT': 18,
        'ST21': 21.4,
        'GeoCenter: tropical': (18.6, 1.1, 1.3),
        'GeoCenter: all sys': (18.7, 1.0, 1.1)
    },
    r'$\geq$ 83 kt': {
        'ARCHER-2: SWIR': 21,
        'ARCHER-2: ASCAT': 16,
        'ST21': 15.3,
        'GeoCenter: tropical': (13.6, 0.6, 0.6),
        'GeoCenter: all sys': (13.6, 0.6, 0.7)
    },
    'All': {
        'ARCHER-2: SWIR': 43,
        'ARCHER-2: ASCAT': 24,
        'ST21': 19.3,
        'GeoCenter: tropical': (22.3, 0.4, 0.4),
        'GeoCenter: all sys': (23.3, 0.5, 0.4)
    }
}

MEAN_ERROR_DICT = {
    r'$<$ 64 kt': {
        'W19': (54, 71),
        'Y19': numpy.nan,
        'W23': 37,
        'GeoCenter: tropical': (29.0, 0.4, 0.5),
        'GeoCenter: all sys': (29.3, 0.4, 0.5)
    },
    '[64, 83) kt': {
        'W19': 36,
        'Y19': numpy.nan,
        'W23': 25,
        'GeoCenter: tropical': (21.1, 1.0, 0.9),
        'GeoCenter: all sys': (21.4, 0.8, 0.8)
    },
    r'$\geq$ 83 kt': {
        'W19': (20, 33),
        'Y19': numpy.nan,
        'W23': (13, 17),
        'GeoCenter: tropical': (15.7, 0.5, 0.5),
        'GeoCenter: all sys': (15.7, 0.5, 0.5)
    },
    'All': {
        'W19': 54,
        'Y19': 28.6,
        'W23': 29.3,
        'GeoCenter: tropical': (25.7, 0.4, 0.4),
        'GeoCenter: all sys': (26.9, 0.3, 0.3)
    }
}

MEDIAN_ERROR_DICT_ARCHER = {
    r'$<$ 64 kt': {
        'ARCHER-2: IR': 56,
        'ARCHER-2: SWIR': 46,
        'ARCHER-2: visible': 36,
        'ARCHER-2: microwave': 35,
        'ARCHER-2: ASCAT': 26,
        'GeoCenter: tropical': (25.9, 0.5, 0.7),
        'GeoCenter: all sys': (26.1, 0.6, 0.5)
    },
    '[64, 83) kt': {
        'ARCHER-2: IR': 43,
        'ARCHER-2: SWIR': 41,
        'ARCHER-2: visible': 26,
        'ARCHER-2: microwave': 26,
        'ARCHER-2: ASCAT': 18,
        'GeoCenter: tropical': (18.6, 1.1, 1.3),
        'GeoCenter: all sys': (18.7, 1.0, 1.1)
    },
    r'$\geq$ 83 kt': {
        'ARCHER-2: IR': 21,
        'ARCHER-2: SWIR': 21,
        'ARCHER-2: visible': 22,
        'ARCHER-2: microwave': 11,
        'ARCHER-2: ASCAT': 16,
        'GeoCenter: tropical': (13.6, 0.6, 0.6),
        'GeoCenter: all sys': (13.6, 0.6, 0.7)
    },
    'All': {
        'ARCHER-2: IR': 49,
        'ARCHER-2: SWIR': 43,
        'ARCHER-2: visible': 32,
        'ARCHER-2: microwave': 31,
        'ARCHER-2: ASCAT': 24,
        'GeoCenter: tropical': (22.3, 0.4, 0.4),
        'GeoCenter: all sys': (23.3, 0.5, 0.4)
    }
}


def _plot_grouped_bar_chart(data_dict, title_string):
    """Plots grouped bar chart with either mean or median errors.

    :param data_dict: One of the two constant dictionaries defined at the top of
        this script.
    :param title_string: Figure title.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    intensity_bin_strings = list(data_dict.keys())
    model_names = list(data_dict[intensity_bin_strings[0]].keys())

    if len(model_names) == 5:
        bar_width = 0.15
    else:
        bar_width = 0.75 / 7

    num_bins = len(intensity_bin_strings)
    x_tick_values = numpy.linspace(0, num_bins - 1, num=num_bins, dtype=int)
    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    for i in range(len(model_names)):
        mean_values = []
        lower_errors = []
        upper_errors = []
        min_values = []
        max_values = []

        for this_bin in intensity_bin_strings:
            data_for_this_bin = data_dict[this_bin][model_names[i]]

            if (
                    isinstance(data_for_this_bin, tuple)
                    and len(data_for_this_bin) == 3
            ):

                # Plotting mean value with error bar.
                this_mean, this_lower_error, this_upper_error = (
                    data_for_this_bin
                )
                mean_values.append(this_mean)
                lower_errors.append(this_lower_error)
                upper_errors.append(this_upper_error)
                min_values.append(numpy.nan)
                max_values.append(numpy.nan)

            elif (
                    isinstance(data_for_this_bin, tuple)
                    and len(data_for_this_bin) == 2
            ):

                # Plotting mean with range.
                this_min, this_max = data_for_this_bin
                mean_values.append(0.5 * (this_min + this_max))
                min_values.append(this_min)
                max_values.append(this_max)
                lower_errors.append(numpy.nan)
                upper_errors.append(numpy.nan)
            else:
                mean_values.append(data_for_this_bin)
                lower_errors.append(numpy.nan)
                upper_errors.append(numpy.nan)
                min_values.append(numpy.nan)
                max_values.append(numpy.nan)

        bar_positions = x_tick_values + i * bar_width

        mean_values = numpy.array(mean_values, dtype=float)
        lower_errors = numpy.array(lower_errors, dtype=float)
        upper_errors = numpy.array(upper_errors, dtype=float)
        min_values = numpy.array(min_values, dtype=float)
        max_values = numpy.array(max_values, dtype=float)

        good_indices = numpy.where(numpy.isfinite(mean_values))[0]
        bar_graph_handle = axes_object.bar(
            bar_positions[good_indices],
            mean_values[good_indices],
            bar_width,
            alpha=OPACITY,
            label=model_names[i],
            zorder=2
        )

        for this_position, this_min, this_max in zip(
                bar_positions, min_values, max_values
        ):
            if this_min is None:
                continue

            axes_object.bar(
                this_position,
                this_max - this_min,
                bar_width,
                bottom=this_min,
                color=bar_graph_handle.patches[0].get_facecolor(),
                alpha=1.,
                edgecolor='black',
                linewidth=3,
                zorder=1e12
            )

        good_indices = numpy.where(numpy.isfinite(lower_errors))[0]
        axes_object.errorbar(
            bar_positions[good_indices],
            mean_values[good_indices],
            yerr=[lower_errors[good_indices], upper_errors[good_indices]],
            fmt='none',
            ecolor='black',
            capsize=5,
            zorder=3
        )

    num_models = len(model_names)

    axes_object.set_xticks(x_tick_values + 0.5 * (num_models - 1) * bar_width)
    axes_object.set_xticklabels(intensity_bin_strings)
    axes_object.set_ylabel('Error (km)')
    axes_object.set_title(title_string)
    axes_object.legend(fontsize=20)
    axes_object.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    pyplot.xticks(rotation=45)

    return figure_object, axes_object


def _run():
    """Plots comparison among all models for WAF 2024 paper.

    This is effectively the main method.
    """

    figure_object = _plot_grouped_bar_chart(
        data_dict=MEAN_ERROR_DICT,
        title_string='(a) Mean distance error for all models'
    )[0]
    panel_file_names = ['{0:s}/mean_error_all_models.jpg'.format(OUTPUT_DIR_NAME)]

    print('Saving figure to file: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1],
        dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object = _plot_grouped_bar_chart(
        data_dict=MEDIAN_ERROR_DICT,
        title_string='(b) Median distance error for all models'
    )[0]
    panel_file_names.append(
        '{0:s}/median_error_all_models.jpg'.format(OUTPUT_DIR_NAME)
    )

    print('Saving figure to file: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1],
        dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object = _plot_grouped_bar_chart(
        data_dict=MEDIAN_ERROR_DICT_ARCHER,
        title_string='(c) Median distance error for GeoCenter and ARCHER-2'
    )[0]
    panel_file_names.append(
        '{0:s}/median_error_archer.jpg'.format(OUTPUT_DIR_NAME)
    )

    print('Saving figure to file: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1],
        dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    concat_file_name = '{0:s}/model_comparison.jpg'.format(OUTPUT_DIR_NAME)

    print('Concatenating panels to: "{0:s}"...'.format(concat_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_file_name,
        num_panel_rows=2,
        num_panel_columns=2
    )

    imagemagick_utils.trim_whitespace(
        input_file_name=concat_file_name,
        output_file_name=concat_file_name,
        border_width_pixels=0
    )


if __name__ == '__main__':
    _run()
