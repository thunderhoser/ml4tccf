"""Plots evaluation metrics on lat-long grid."""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import error_checking
from ml4tccf.io import border_io
from ml4tccf.utils import scalar_evaluation
from ml4tccf.plotting import plotting_utils
from ml4tccf.plotting import scalar_evaluation_plotting as scalar_eval_plotting
from ml4tccf.scripts import split_predictions_by_latlng as split_predictions

TOLERANCE = 1e-6

BORDER_COLOUR = numpy.full(3, 152. / 255)

BIAS_COLOUR_MAP_NAME = 'seismic'
MAIN_COLOUR_MAP_NAME = 'viridis'
MIN_COLOUR_PERCENTILE = 1.
MAX_COLOUR_PERCENTILE = 99.

# TODO(thunderhoser): Remember this!
# NUM_EXAMPLES_COLOUR = numpy.full(3, 0.)
# NUM_EXAMPLES_FONT_SIZE = 20

FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 20
FIGURE_HEIGHT_INCHES = 8

INPUT_DIR_ARG_NAME = 'input_evaluation_dir_name'
LATITUDE_SPACING_ARG_NAME = 'latitude_spacing_deg'
LONGITUDE_SPACING_ARG_NAME = 'longitude_spacing_deg'
MAIN_FONT_SIZE_ARG_NAME = 'main_font_size'
LABEL_FONT_SIZE_ARG_NAME = 'label_font_size'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level directory with evaluation files (one per grid cell).'
)
LATITUDE_SPACING_HELP_SPACING = 'Meridional grid spacing (degrees).'
LONGITUDE_SPACING_HELP_SPACING = 'Zonal grid spacing (degrees).'
MAIN_FONT_SIZE_HELP_STRING = (
    'Main font size (for everything except text labels in grid cells).'
)
LABEL_FONT_SIZE_HELP_STRING = (
    'Font size for text labels in grid cells.  If you do not want labels, make '
    'this negative.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LATITUDE_SPACING_ARG_NAME, type=float, required=True,
    help=LATITUDE_SPACING_HELP_SPACING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LONGITUDE_SPACING_ARG_NAME, type=float, required=True,
    help=LONGITUDE_SPACING_HELP_SPACING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAIN_FONT_SIZE_ARG_NAME, type=float, required=True,
    help=MAIN_FONT_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LABEL_FONT_SIZE_ARG_NAME, type=float, required=True,
    help=LABEL_FONT_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(evaluation_dir_name, latitude_spacing_deg, longitude_spacing_deg,
         main_font_size, label_font_size, output_dir_name):
    """Plots evaluation metrics on lat-long grid.

    This is effectively the main method.

    :param evaluation_dir_name: See documentation at top of file.
    :param latitude_spacing_deg: Same.
    :param longitude_spacing_deg: Same.
    :param main_font_size: Same.
    :param label_font_size: Same.
    :param output_dir_name: Same.
    """

    error_checking.assert_is_greater(main_font_size, 0.)
    if label_font_size < 0:
        label_font_size = None

    pyplot.rc('font', size=main_font_size)
    pyplot.rc('axes', titlesize=main_font_size)
    pyplot.rc('axes', labelsize=main_font_size)
    pyplot.rc('xtick', labelsize=main_font_size)
    pyplot.rc('ytick', labelsize=main_font_size)
    pyplot.rc('legend', fontsize=main_font_size)
    pyplot.rc('figure', titlesize=main_font_size)

    # Housekeeping.
    (
        grid_latitudes_deg_n, grid_longitudes_deg_e,
        grid_edge_latitudes_deg_n, grid_edge_longitudes_deg_e
    ) = split_predictions.get_grid_coords(
        latitude_spacing_deg=latitude_spacing_deg,
        longitude_spacing_deg=longitude_spacing_deg
    )

    num_grid_rows = len(grid_latitudes_deg_n)
    num_grid_columns = len(grid_longitudes_deg_e)

    # Do actual stuff.
    eval_table_by_latlng_listlist = [
        [None] * num_grid_columns for _ in range(num_grid_rows)
    ]
    num_bootstrap_reps = -1

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            this_file_name = (
                '{0:s}/latitude-deg-n={1:+05.1f}_{2:+05.1f}_'
                'longitude-deg-e={3:+06.1f}_{4:+06.1f}/evaluation.nc'
            ).format(
                evaluation_dir_name,
                grid_edge_latitudes_deg_n[i],
                grid_edge_latitudes_deg_n[i + 1],
                grid_edge_longitudes_deg_e[j],
                grid_edge_longitudes_deg_e[j + 1]
            )

            if not os.path.isfile(this_file_name):
                continue

            print('Reading data from: "{0:s}"...'.format(this_file_name))

            try:
                eval_table_by_latlng_listlist[i][j] = (
                    scalar_evaluation.read_file(this_file_name)
                )
                assert (
                    scalar_evaluation.TARGET_FIELD_DIM
                    in eval_table_by_latlng_listlist[i][j].coords
                )
            except:
                raise ValueError(
                    'This script does not yet work for gridded predictions.'
                )

            etbll = eval_table_by_latlng_listlist

            if num_bootstrap_reps == -1:
                num_bootstrap_reps = len(
                    etbll[i][j].coords[
                        scalar_evaluation.BOOTSTRAP_REP_DIM
                    ].values
                )

            this_num_bootstrap_reps = len(
                etbll[i][j].coords[scalar_evaluation.BOOTSTRAP_REP_DIM].values
            )
            assert num_bootstrap_reps == this_num_bootstrap_reps

    etbll = eval_table_by_latlng_listlist
    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    for metric_name in scalar_eval_plotting.BASIC_METRIC_NAMES:
        for target_field_name in scalar_eval_plotting.BASIC_TARGET_FIELD_NAMES:
            these_dim = (num_grid_rows, num_grid_columns, num_bootstrap_reps)
            metric_matrix = numpy.full(these_dim, numpy.nan)

            for i in range(num_grid_rows):
                for j in range(num_grid_columns):
                    if etbll[i][j] is None:
                        continue

                    k = numpy.where(
                        etbll[i][j].coords[
                            scalar_evaluation.TARGET_FIELD_DIM
                        ].values
                        == target_field_name
                    )[0][0]

                    metric_matrix[i, j, :] = (
                        etbll[i][j][metric_name].values[k, :]
                    )

            figure_object, axes_object = pyplot.subplots(
                1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
            )
            plotting_utils.plot_borders(
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                line_width=1., line_colour=BORDER_COLOUR,
                axes_object=axes_object
            )

            scalar_eval_plotting.plot_metric_by_latlng(
                axes_object=axes_object,
                metric_matrix=numpy.nanmean(metric_matrix, axis=-1),
                metric_name=metric_name,
                target_field_name=target_field_name,
                grid_edge_latitudes_deg_n=grid_edge_latitudes_deg_n,
                grid_edge_longitudes_deg_e=grid_edge_longitudes_deg_e,
                colour_map_name=(
                    BIAS_COLOUR_MAP_NAME
                    if metric_name == scalar_evaluation.BIAS_KEY
                    else MAIN_COLOUR_MAP_NAME
                ),
                min_colour_percentile=MIN_COLOUR_PERCENTILE,
                max_colour_percentile=MAX_COLOUR_PERCENTILE,
                label_font_size=label_font_size
            )

            plotting_utils.plot_grid_lines(
                plot_latitudes_deg_n=grid_edge_latitudes_deg_n,
                plot_longitudes_deg_e=grid_edge_longitudes_deg_e,
                axes_object=axes_object,
                parallel_spacing_deg=10., meridian_spacing_deg=20.
            )

            axes_object.set_ylim(bottom=split_predictions.MIN_LATITUDE_DEG_N)
            axes_object.set_ylim(
                top=split_predictions.MAX_LATITUDE_DEG_N +
                    latitude_spacing_deg / 2
            )

            output_file_name = '{0:s}/{1:s}_{2:s}_by-latlng.jpg'.format(
                output_dir_name,
                target_field_name.replace('_', '-'),
                metric_name.replace('_', '-')
            )

            print('Saving figure to: "{0:s}"...'.format(output_file_name))
            figure_object.savefig(
                output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                bbox_inches='tight'
            )
            pyplot.close(figure_object)

    for metric_name in scalar_eval_plotting.ADVANCED_METRIC_NAMES:
        these_dim = (num_grid_rows, num_grid_columns, num_bootstrap_reps)
        metric_matrix = numpy.full(these_dim, numpy.nan)

        for i in range(num_grid_rows):
            for j in range(num_grid_columns):
                if etbll[i][j] is None:
                    continue

                if metric_name == scalar_eval_plotting.NUM_EXAMPLES_KEY:
                    metric_matrix[i, j, :] = numpy.sum(
                        etbll[i][j][
                            scalar_evaluation.OFFSET_DIST_BIN_COUNT_KEY
                        ].values
                    )
                else:
                    metric_matrix[i, j, :] = etbll[i][j][metric_name].values[:]

        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )
        plotting_utils.plot_borders(
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            line_width=1., line_colour=BORDER_COLOUR,
            axes_object=axes_object
        )

        scalar_eval_plotting.plot_metric_by_latlng(
            axes_object=axes_object,
            metric_matrix=numpy.nanmean(metric_matrix, axis=-1),
            metric_name=metric_name,
            target_field_name=scalar_evaluation.OFFSET_DISTANCE_NAME,
            grid_edge_latitudes_deg_n=grid_edge_latitudes_deg_n,
            grid_edge_longitudes_deg_e=grid_edge_longitudes_deg_e,
            colour_map_name=(
                BIAS_COLOUR_MAP_NAME
                if metric_name == scalar_evaluation.BIAS_KEY
                else MAIN_COLOUR_MAP_NAME
            ),
            min_colour_percentile=MIN_COLOUR_PERCENTILE,
            max_colour_percentile=MAX_COLOUR_PERCENTILE,
            label_font_size=label_font_size
        )

        plotting_utils.plot_grid_lines(
            plot_latitudes_deg_n=grid_edge_latitudes_deg_n,
            plot_longitudes_deg_e=grid_edge_longitudes_deg_e,
            axes_object=axes_object,
            parallel_spacing_deg=10., meridian_spacing_deg=20.
        )

        axes_object.set_ylim(bottom=split_predictions.MIN_LATITUDE_DEG_N)
        axes_object.set_ylim(
            top=split_predictions.MAX_LATITUDE_DEG_N +
                latitude_spacing_deg / 2
        )

        output_file_name = '{0:s}/{1:s}_by-latlng.jpg'.format(
            output_dir_name,
            metric_name.replace('_', '-')
        )

        print('Saving figure to: "{0:s}"...'.format(output_file_name))
        figure_object.savefig(
            output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
            bbox_inches='tight'
        )
        pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        evaluation_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        latitude_spacing_deg=getattr(
            INPUT_ARG_OBJECT, LATITUDE_SPACING_ARG_NAME
        ),
        longitude_spacing_deg=getattr(
            INPUT_ARG_OBJECT, LONGITUDE_SPACING_ARG_NAME
        ),
        main_font_size=getattr(INPUT_ARG_OBJECT, MAIN_FONT_SIZE_ARG_NAME),
        label_font_size=getattr(INPUT_ARG_OBJECT, LABEL_FONT_SIZE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
