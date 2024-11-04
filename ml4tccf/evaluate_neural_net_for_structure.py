"""Evaluates neural net for TC-structure parameters."""

import os
import sys
import glob
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
import structure_prediction_io
import structure_prediction_utils
import scalar_evaluation
import scalar_evaluation_plotting as scalar_eval_plotting
import neural_net_utils as nn_utils
import neural_net_training_structure as nn_training

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_PATTERN_ARG_NAME = 'input_prediction_file_pattern'
EVAL_RECENT_CHANGES_ARG_NAME = 'eval_recent_changes'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_PATTERN_HELP_STRING = (
    'Glob pattern for input files.  Each input file will be read by '
    '`structure_prediction_io.read_file`.'
)
EVAL_RECENT_CHANGES_HELP_STRING = (
    'Boolean flag.  If 1, will evaluate recent changes (departure from A-deck '
    '6-12 hours ago).  If 0, will evaluate absolute values.'
)
OUTPUT_DIR_HELP_STRING = 'Path to output directory.  Stuff will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_PATTERN_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_PATTERN_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EVAL_RECENT_CHANGES_ARG_NAME, type=int, required=False, default=0,
    help=EVAL_RECENT_CHANGES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(prediction_file_pattern, eval_recent_changes, output_dir_name):
    """Evaluates neural net for TC-structure parameters.

    This is effectively the main method.

    :param prediction_file_pattern: See documentation at top of file.
    :param eval_recent_changes: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if no prediction files could be found.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    prediction_file_names = glob.glob(prediction_file_pattern)
    if len(prediction_file_names) == 0:
        error_string = (
            'Could not find any prediction file with the following pattern: '
            '"{0:s}"'
        ).format(prediction_file_pattern)

        raise ValueError(error_string)

    prediction_file_names.sort()

    target_matrix = numpy.array([], dtype=float)
    prediction_matrix = numpy.array([], dtype=float)
    baseline_prediction_matrix = numpy.array([], dtype=float)
    model_file_name = None

    for i in range(len(prediction_file_names)):
        print('Reading data from: "{0:s}"...'.format(prediction_file_names[i]))
        this_prediction_table_xarray = structure_prediction_io.read_file(
            prediction_file_names[i]
        )
        this_prediction_table_xarray = (
            structure_prediction_utils.get_ensemble_mean(
                this_prediction_table_xarray
            )
        )
        tptx = this_prediction_table_xarray

        if i == 0:
            model_file_name = tptx.attrs[
                structure_prediction_utils.MODEL_FILE_KEY
            ]

        this_target_matrix = (
            tptx[structure_prediction_utils.TARGET_KEY].values
        )
        this_prediction_matrix = numpy.mean(
            tptx[structure_prediction_utils.PREDICTION_KEY].values, axis=-1
        )

        try:
            this_baseline_prediction_matrix = (
                tptx[structure_prediction_utils.BASELINE_PREDICTION_KEY].values
            )
        except:
            this_baseline_prediction_matrix = numpy.array([])

        if i == 0:
            target_matrix = this_target_matrix + 0.
            prediction_matrix = this_prediction_matrix + 0.
            baseline_prediction_matrix = this_baseline_prediction_matrix + 0.
        else:
            target_matrix = numpy.concatenate([
                target_matrix, this_target_matrix
            ])
            prediction_matrix = numpy.concatenate([
                prediction_matrix, this_prediction_matrix
            ])
            baseline_prediction_matrix = numpy.concatenate([
                baseline_prediction_matrix, this_baseline_prediction_matrix
            ])

    if eval_recent_changes:
        prediction_matrix = prediction_matrix - baseline_prediction_matrix
        print('NAN FRACTIONS')
        print(numpy.mean(numpy.isnan(target_matrix)))
        target_matrix = target_matrix - baseline_prediction_matrix
        print(numpy.mean(numpy.isnan(target_matrix)))
        baseline_prediction_matrix = numpy.array([])

    print(SEPARATOR_STRING)

    model_metafile_name = nn_utils.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )
    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = nn_utils.read_metafile(model_metafile_name)

    training_option_dict = model_metadata_dict[nn_utils.TRAINING_OPTIONS_KEY]
    target_field_names = training_option_dict[nn_training.TARGET_FIELDS_KEY]
    num_target_fields = len(target_field_names)

    for f in range(num_target_fields):
        baseline_predicted_values = numpy.array([])

        if target_field_names[f] == nn_training.INTENSITY_FIELD_NAME:
            if eval_recent_changes:
                min_bin_edge = -30
                max_bin_edge = 30
                num_bins = 30
            else:
                min_bin_edge = 15
                max_bin_edge = 180
                num_bins = 33

            target_values = target_matrix[:, f]
            predicted_values = prediction_matrix[:, f]
            if numpy.size(baseline_prediction_matrix) > 0:
                baseline_predicted_values = baseline_prediction_matrix[:, f]

        elif target_field_names[f] == nn_training.R34_FIELD_NAME:
            if eval_recent_changes:
                min_bin_edge = -200
                max_bin_edge = 200
                num_bins = 40
            else:
                min_bin_edge = 0
                max_bin_edge = 1700
                num_bins = 170

            try:
                f_intensity = target_field_names.index(
                    nn_training.INTENSITY_FIELD_NAME
                )
                good_indices = numpy.where(numpy.logical_and(
                    target_matrix[:, f_intensity] >= 34.,
                    prediction_matrix[:, f_intensity] >= 34.
                ))[0]
                target_values = target_matrix[good_indices, f]
                predicted_values = prediction_matrix[good_indices, f]
                if numpy.size(baseline_prediction_matrix) > 0:
                    baseline_predicted_values = baseline_prediction_matrix[
                        good_indices, f
                    ]
            except:
                target_values = target_matrix[:, f]
                predicted_values = prediction_matrix[:, f]
                if numpy.size(baseline_prediction_matrix) > 0:
                    baseline_predicted_values = baseline_prediction_matrix[:, f]

        elif target_field_names[f] == nn_training.R50_FIELD_NAME:
            if eval_recent_changes:
                min_bin_edge = -150
                max_bin_edge = 150
                num_bins = 30
            else:
                min_bin_edge = 0
                max_bin_edge = 1000
                num_bins = 100

            try:
                f_intensity = target_field_names.index(
                    nn_training.INTENSITY_FIELD_NAME
                )
                good_indices = numpy.where(numpy.logical_and(
                    target_matrix[:, f_intensity] >= 50.,
                    prediction_matrix[:, f_intensity] >= 50.
                ))[0]
                target_values = target_matrix[good_indices, f]
                predicted_values = prediction_matrix[good_indices, f]
                if numpy.size(baseline_prediction_matrix) > 0:
                    baseline_predicted_values = baseline_prediction_matrix[
                        good_indices, f
                    ]
            except:
                target_values = target_matrix[:, f]
                predicted_values = prediction_matrix[:, f]
                if numpy.size(baseline_prediction_matrix) > 0:
                    baseline_predicted_values = baseline_prediction_matrix[:, f]

        elif target_field_names[f] == nn_training.R64_FIELD_NAME:
            if eval_recent_changes:
                min_bin_edge = -100
                max_bin_edge = 100
                num_bins = 20
            else:
                min_bin_edge = 0
                max_bin_edge = 500
                num_bins = 50

            try:
                f_intensity = target_field_names.index(
                    nn_training.INTENSITY_FIELD_NAME
                )
                good_indices = numpy.where(numpy.logical_and(
                    target_matrix[:, f_intensity] >= 64.,
                    prediction_matrix[:, f_intensity] >= 64.
                ))[0]
                target_values = target_matrix[good_indices, f]
                predicted_values = prediction_matrix[good_indices, f]
                if numpy.size(baseline_prediction_matrix) > 0:
                    baseline_predicted_values = baseline_prediction_matrix[
                        good_indices, f
                    ]
            except:
                target_values = target_matrix[:, f]
                predicted_values = prediction_matrix[:, f]
                if numpy.size(baseline_prediction_matrix) > 0:
                    baseline_predicted_values = baseline_prediction_matrix[:, f]

        else:
            if eval_recent_changes:
                min_bin_edge = -100
                max_bin_edge = 100
                num_bins = 20
            else:
                min_bin_edge = 0
                max_bin_edge = 1000
                num_bins = 100

            target_values = target_matrix[:, f]
            predicted_values = prediction_matrix[:, f]
            if numpy.size(baseline_prediction_matrix) > 0:
                baseline_predicted_values = baseline_prediction_matrix[:, f]

        if numpy.size(baseline_prediction_matrix) > 0:
            print((
                'Mean difference between NN and baseline for {0:s} = {1:.4f}'
            ).format(
                target_field_names[f],
                numpy.mean(numpy.absolute(
                    predicted_values - baseline_predicted_values
                ))
            ))

        (
            mean_predictions, mean_observations, example_counts
        ) = scalar_evaluation._get_reliability_curve_one_variable(
            target_values=target_values,
            predicted_values=predicted_values,
            is_var_direction=False,
            num_bins=num_bins,
            min_bin_edge=min_bin_edge,
            max_bin_edge=max_bin_edge,
            invert=False
        )

        (
            _, inv_mean_observations, inv_example_counts
        ) = scalar_evaluation._get_reliability_curve_one_variable(
            target_values=target_values,
            predicted_values=predicted_values,
            is_var_direction=False,
            num_bins=num_bins,
            min_bin_edge=min_bin_edge,
            max_bin_edge=max_bin_edge,
            invert=True
        )

        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )
        scalar_eval_plotting.plot_attributes_diagram(
            figure_object=figure_object,
            axes_object=axes_object,
            mean_predictions=mean_predictions,
            mean_observations=mean_observations,
            mean_value_in_training=numpy.mean(target_values),
            min_value_to_plot=min_bin_edge,
            max_value_to_plot=max_bin_edge
        )
        scalar_eval_plotting.plot_inset_histogram(
            figure_object=figure_object,
            bin_centers=inv_mean_observations,
            bin_counts=example_counts,
            has_predictions=True,
            bar_colour=scalar_eval_plotting.RELIABILITY_LINE_COLOUR
        )
        scalar_eval_plotting.plot_inset_histogram(
            figure_object=figure_object,
            bin_centers=inv_mean_observations,
            bin_counts=inv_example_counts,
            has_predictions=False,
            bar_colour=scalar_eval_plotting.RELIABILITY_LINE_COLOUR
        )

        reliability = scalar_evaluation._get_reliability(
            binned_mean_predictions=mean_predictions,
            binned_mean_observations=mean_observations,
            binned_example_counts=example_counts,
            is_var_direction=False
        )

        mean_absolute_error = numpy.mean(
            numpy.absolute(predicted_values - target_values)
        )
        bias = numpy.mean(predicted_values - target_values)
        rmse = numpy.sqrt(
            numpy.mean((predicted_values - target_values) ** 2)
        )

        numerator = numpy.sum(
            (target_values - numpy.mean(target_values)) *
            (predicted_values - numpy.mean(predicted_values))
        )
        sum_squared_target_diffs = numpy.sum(
            (target_values - numpy.mean(target_values)) ** 2
        )
        sum_squared_prediction_diffs = numpy.sum(
            (predicted_values - numpy.mean(predicted_values)) ** 2
        )
        correlation = (
            numerator /
            numpy.sqrt(sum_squared_target_diffs * sum_squared_prediction_diffs)
        )

        title_string = (
            'NN for {0:s}\n'
            'MAE = {1:.2f}; bias = {2:.2f}; RMSE = {3:.2f}; REL = {4:.2f}; '
            'corr = {5:.2f}'
        ).format(
            target_field_names[f],
            mean_absolute_error,
            bias,
            rmse,
            reliability,
            correlation
        )

        print(title_string)
        axes_object.set_title(title_string)

        figure_file_name = '{0:s}/nn_attributes_diagram_{1:s}.jpg'.format(
            output_dir_name,
            target_field_names[f].replace('_', '-')
        )
        print('Saving figure to file: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        if numpy.size(baseline_prediction_matrix) == 0:
            continue

        if target_field_names[f] == nn_training.INTENSITY_FIELD_NAME:
            baseline_predicted_values = baseline_prediction_matrix[:, f]
        elif target_field_names[f] == nn_training.R34_FIELD_NAME:
            try:
                f_intensity = target_field_names.index(
                    nn_training.INTENSITY_FIELD_NAME
                )
                good_indices = numpy.where(numpy.logical_and(
                    target_matrix[:, f_intensity] >= 34.,
                    baseline_prediction_matrix[:, f_intensity] >= 34.
                ))[0]

                target_values = target_matrix[good_indices, f]
                baseline_predicted_values = baseline_prediction_matrix[
                    good_indices, f
                ]
            except:
                baseline_predicted_values = baseline_prediction_matrix[:, f]

        elif target_field_names[f] == nn_training.R50_FIELD_NAME:
            try:
                f_intensity = target_field_names.index(
                    nn_training.INTENSITY_FIELD_NAME
                )
                good_indices = numpy.where(numpy.logical_and(
                    target_matrix[:, f_intensity] >= 50.,
                    baseline_prediction_matrix[:, f_intensity] >= 50.
                ))[0]

                target_values = target_matrix[good_indices, f]
                baseline_predicted_values = baseline_prediction_matrix[
                    good_indices, f
                ]
            except:
                baseline_predicted_values = baseline_prediction_matrix[:, f]

        elif target_field_names[f] == nn_training.R64_FIELD_NAME:
            try:
                f_intensity = target_field_names.index(
                    nn_training.INTENSITY_FIELD_NAME
                )
                good_indices = numpy.where(numpy.logical_and(
                    target_matrix[:, f_intensity] >= 64.,
                    baseline_prediction_matrix[:, f_intensity] >= 64.
                ))[0]

                target_values = target_matrix[good_indices, f]
                baseline_predicted_values = baseline_prediction_matrix[
                    good_indices, f
                ]
            except:
                baseline_predicted_values = baseline_prediction_matrix[:, f]

        else:
            baseline_predicted_values = baseline_prediction_matrix[:, f]

        (
            mean_predictions, mean_observations, example_counts
        ) = scalar_evaluation._get_reliability_curve_one_variable(
            target_values=target_values,
            predicted_values=baseline_predicted_values,
            is_var_direction=False,
            num_bins=num_bins,
            min_bin_edge=min_bin_edge,
            max_bin_edge=max_bin_edge,
            invert=False
        )

        (
            _, inv_mean_observations, inv_example_counts
        ) = scalar_evaluation._get_reliability_curve_one_variable(
            target_values=target_values,
            predicted_values=baseline_predicted_values,
            is_var_direction=False,
            num_bins=num_bins,
            min_bin_edge=min_bin_edge,
            max_bin_edge=max_bin_edge,
            invert=True
        )

        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )
        scalar_eval_plotting.plot_attributes_diagram(
            figure_object=figure_object,
            axes_object=axes_object,
            mean_predictions=mean_predictions,
            mean_observations=mean_observations,
            mean_value_in_training=numpy.mean(target_values),
            min_value_to_plot=min_bin_edge,
            max_value_to_plot=max_bin_edge
        )
        scalar_eval_plotting.plot_inset_histogram(
            figure_object=figure_object,
            bin_centers=inv_mean_observations,
            bin_counts=example_counts,
            has_predictions=True,
            bar_colour=scalar_eval_plotting.RELIABILITY_LINE_COLOUR
        )
        scalar_eval_plotting.plot_inset_histogram(
            figure_object=figure_object,
            bin_centers=inv_mean_observations,
            bin_counts=inv_example_counts,
            has_predictions=False,
            bar_colour=scalar_eval_plotting.RELIABILITY_LINE_COLOUR
        )

        reliability = scalar_evaluation._get_reliability(
            binned_mean_predictions=mean_predictions,
            binned_mean_observations=mean_observations,
            binned_example_counts=example_counts,
            is_var_direction=False
        )

        mean_absolute_error = numpy.mean(numpy.absolute(
            baseline_predicted_values - target_values
        ))
        bias = numpy.mean(
            baseline_predicted_values - target_values
        )
        rmse = numpy.sqrt(numpy.mean(
            (baseline_predicted_values - target_values) ** 2
        ))

        numerator = numpy.sum(
            (target_values - numpy.mean(target_values)) *
            (baseline_predicted_values -
             numpy.mean(baseline_predicted_values))
        )
        sum_squared_target_diffs = numpy.sum(
            (target_values - numpy.mean(target_values)) ** 2
        )
        sum_squared_prediction_diffs = numpy.sum(
            (baseline_predicted_values -
             numpy.mean(baseline_predicted_values)) ** 2
        )
        correlation = (
            numerator /
            numpy.sqrt(sum_squared_target_diffs * sum_squared_prediction_diffs)
        )

        title_string = (
            'Baseline for {0:s}\n'
            'MAE = {1:.2f}; bias = {2:.2f}; RMSE = {3:.2f}; REL = {4:.2f}; '
            'corr = {5:.2f}'
        ).format(
            target_field_names[f],
            mean_absolute_error,
            bias,
            rmse,
            reliability,
            correlation
        )

        print(title_string)
        axes_object.set_title(title_string)

        figure_file_name = '{0:s}/baseline_attributes_diagram_{1:s}.jpg'.format(
            output_dir_name,
            target_field_names[f].replace('_', '-')
        )
        print('Saving figure to file: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_pattern=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_PATTERN_ARG_NAME
        ),
        eval_recent_changes=bool(
            getattr(INPUT_ARG_OBJECT, EVAL_RECENT_CHANGES_ARG_NAME)
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
