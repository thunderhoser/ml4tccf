"""Plots results of single-channel experiment."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import scalar_evaluation as evaluation
import spread_skill_utils as ss_utils
import discard_test_utils as dt_utils
import pit_utils

DEFAULT_FONT_SIZE = 30
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 9
FIGURE_RESOLUTION_DPI = 300

ALL_WAVELENGTHS_MICRONS = numpy.array([
    3.9, 6.185, 6.95, 7.34, 8.5, 9.61, 10.35, 11.2, 12.3, 13.3
])
METRIC_NAMES = [
    'Mean distance (km)',
    'Median distance (km)',
    'RMS distance (km)',
    r'Absolute bias $\times$ 10 (km)',
    r'Reliability (km$^2$)',
    'CRPS (km)',
    r'SSREL $\times$ 10 (km)',
    r'SSRAT $\times$ 10',
    r'RHD $\times$ 10 000',
    r'DTMF $\times$ 10'
]

INPUT_DIR_ARG_NAME = 'input_experiment_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_figure_dir_name'

INPUT_DIR_HELP_STRING = 'Path to top-level directory for experiment.'
OUTPUT_DIR_HELP_STRING = 'Path to output directory, where figure will be saved.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(experiment_dir_name, output_dir_name):
    """Plots results of single-channel experiment.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of this script.
    :param output_dir_name: Same.
    """

    num_wavelengths = len(ALL_WAVELENGTHS_MICRONS)
    num_metrics = len(METRIC_NAMES)
    metric_matrix = numpy.full((num_wavelengths, num_metrics), numpy.nan)

    for i in range(num_wavelengths):
        this_file_name = '{0:s}/wavelength-microns={1:.3f}/isotonic_regression_gaussian_dist/uncertainty_calibration/testing/evaluation_no_bootstrap.nc'.format(
            experiment_dir_name, ALL_WAVELENGTHS_MICRONS[i]
        )
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        evaluation_table_xarray = evaluation.read_file(this_file_name)
        etx = evaluation_table_xarray

        metric_matrix[i, 0] = 0.001 * numpy.mean(
            etx[evaluation.MEAN_DISTANCE_KEY].values
        )
        metric_matrix[i, 1] = 0.001 * numpy.mean(
            etx[evaluation.MEDIAN_DISTANCE_KEY].values
        )
        metric_matrix[i, 2] = 0.001 * numpy.sqrt(numpy.mean(
            etx[evaluation.MEAN_SQUARED_DISTANCE_KEY].values
        ))
        metric_matrix[i, 3] = 0.001 * numpy.mean(numpy.absolute(
            numpy.mean(etx[evaluation.BIAS_KEY].values, axis=-1)
        ))
        metric_matrix[i, 4] = 1e-6 * numpy.mean(
            numpy.mean(etx[evaluation.RELIABILITY_KEY].values, axis=-1)
        )
        metric_matrix[i, 5] = 0.001 * numpy.mean(
            numpy.mean(etx[evaluation.CRPS_KEY].values, axis=-1)
        )

        this_file_name = '{0:s}/wavelength-microns={1:.3f}/isotonic_regression_gaussian_dist/uncertainty_calibration/testing/spread_vs_skill.nc'.format(
            experiment_dir_name, ALL_WAVELENGTHS_MICRONS[i]
        )
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        spread_skill_table_xarray = ss_utils.read_results(this_file_name)
        sstx = spread_skill_table_xarray

        metric_matrix[i, 6] = 0.001 * numpy.mean(
            sstx[ss_utils.XY_SSREL_KEY].values
        )
        metric_matrix[i, 7] = numpy.mean(sstx[ss_utils.XY_SSRAT_KEY].values)

        this_file_name = '{0:s}/wavelength-microns={1:.3f}/isotonic_regression_gaussian_dist/uncertainty_calibration/testing/pit_histograms.nc'.format(
            experiment_dir_name, ALL_WAVELENGTHS_MICRONS[i]
        )
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        pit_table_xarray = pit_utils.read_results(this_file_name)
        ptx = pit_table_xarray
        
        metric_matrix[i, 8] = numpy.mean(
            ptx[pit_utils.PIT_DEVIATION_KEY].values[:2]
        )

        this_file_name = '{0:s}/wavelength-microns={1:.3f}/isotonic_regression_gaussian_dist/uncertainty_calibration/testing/discard_test.nc'.format(
            experiment_dir_name, ALL_WAVELENGTHS_MICRONS[i]
        )
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        discard_test_table_xarray = dt_utils.read_results(this_file_name)
        dttx = discard_test_table_xarray

        metric_matrix[i, 9] = numpy.mean(
            dttx[dt_utils.MONO_FRACTION_KEY].values[:2]
        )

    metric_matrix[:, 3] *= 10
    metric_matrix[:, 6] *= 10
    metric_matrix[:, 7] *= 10
    metric_matrix[:, 8] *= 10000
    metric_matrix[:, 9] *= 10

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    bar_width = 0.08
    x_tick_values = numpy.linspace(
        0, num_metrics - 1, num=num_metrics, dtype=int
    )
    
    for i in range(num_wavelengths):
        axes_object.bar(
            x_tick_values + i * bar_width,
            metric_matrix[i, :],
            bar_width,
            label='{0:f} microns'.format(ALL_WAVELENGTHS_MICRONS[i])
        )
    
    axes_object.set_xlabel('Evaluation metrics')
    axes_object.set_ylabel('Value')
    axes_object.set_xticks(x_tick_values + 4.5 * bar_width)
    axes_object.set_xticklabels(METRIC_NAMES, rotation=30, ha='right')
    axes_object.legend(loc='upper right', fontsize=20, ncol=2)

    output_file_name = '{0:s}/1channel_experiment_results.jpg'.format(
        output_dir_name
    )
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name,
        dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        experiment_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
