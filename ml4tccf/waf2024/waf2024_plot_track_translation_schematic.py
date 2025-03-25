"""Plots second translation schematic for WAF 2024 paper."""

import numpy
from geopy.distance import distance
from geopy.point import Point
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.plotting import imagemagick_utils

OUTPUT_DIR_NAME = '/home/ralager/tccf_paper_2024/track_translation_schematic'

ORIG_LATITUDES_DEG_N = numpy.array([
    24.5, 24.56372218, 24.62371688,
    24.67988752, 24.73213754, 24.78037037,
    24.82448943, 24.86439816, 24.9
])
ORIG_LONGITUDES_DEG_E = numpy.array([
    -80.11975098, -80.17385864, -80.2243042,
    -80.27587891, -80.33343506, -80.40179443,
    -80.48577881, -80.59017944, -80.71981812
])
VALID_TIME_STRINGS = ['00Z', '', '01Z', '', '02Z', '', '03Z', '', '04Z']

ORIG_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
WHOLE_TRACK_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
TRACK_SHAPE_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

DEFAULT_FONT_SIZE = 30
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)


def _apply_whole_track_error(latitudes_deg_n, longitudes_deg_e, x_error_km,
                             y_error_km):
    """Applies whole-track error, i.e., shifts the TC track as a whole.

    P = number of points in track

    :param latitudes_deg_n: length-P numpy array of latitudes.
    :param longitudes_deg_e: length-P numpy array of longitudes.
    :param x_error_km: Zonal error.
    :param y_error_km: Meridional error.
    :return: latitudes_deg_n: Shifted version of input.
    :return: longitudes_deg_e: Shifted version of input.
    """

    num_points = len(latitudes_deg_n)

    for i in range(num_points):
        orig_point_object = Point(latitudes_deg_n[i], longitudes_deg_e[i])
        new_point_object = distance(kilometers=y_error_km).destination(
            point=orig_point_object, bearing=0.
        )
        latitudes_deg_n[i] = new_point_object.latitude

        new_point_object = distance(kilometers=x_error_km).destination(
            point=orig_point_object, bearing=90.
        )
        longitudes_deg_e[i] = new_point_object.longitude

    return latitudes_deg_n, longitudes_deg_e


def _apply_track_shape_error(latitudes_deg_n, longitudes_deg_e, mean_error_km,
                             stdev_error_km):
    """Applies track-shape error, i.e., changes the shape of the track.

    P = number of points in track

    :param latitudes_deg_n: length-P numpy array of latitudes.
    :param longitudes_deg_e: length-P numpy array of longitudes.
    :param mean_error_km: Mean error.
    :param stdev_error_km: Standard deviation of error.
    :return: latitudes_deg_n: Shifted version of input.
    :return: longitudes_deg_e: Shifted version of input.
    """

    num_points = len(latitudes_deg_n)

    for i in range(num_points - 1):
        this_distance_km = numpy.random.normal(
            loc=mean_error_km, scale=stdev_error_km, size=1
        )[0]
        this_direction_deg = numpy.random.uniform(
            low=0., high=360 - 1e-6, size=1
        )[0]

        orig_point_object = Point(latitudes_deg_n[i], longitudes_deg_e[i])
        new_point_object = distance(kilometers=this_distance_km).destination(
            point=orig_point_object, bearing=this_direction_deg
        )
        latitudes_deg_n[i] = new_point_object.latitude
        longitudes_deg_e[i] = new_point_object.longitude

    return latitudes_deg_n, longitudes_deg_e


def _run():
    """Plots second translation schematic for WAF 2024 paper.

    This is effectively the main method.
    """

    whole_track_latitudes_deg_n, whole_track_longitudes_deg_e = (
        _apply_whole_track_error(
            latitudes_deg_n=ORIG_LATITUDES_DEG_N + 0.,
            longitudes_deg_e=ORIG_LONGITUDES_DEG_E + 0.,
            x_error_km=48. / numpy.sqrt(2),
            y_error_km=48. / numpy.sqrt(2)
        )
    )

    track_shape_latitudes_deg_n, track_shape_longitudes_deg_e = (
        _apply_track_shape_error(
            latitudes_deg_n=whole_track_latitudes_deg_n + 0.,
            longitudes_deg_e=whole_track_longitudes_deg_e + 0.,
            mean_error_km=10.,
            stdev_error_km=5.
        )
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.plot(
        ORIG_LONGITUDES_DEG_E,
        ORIG_LATITUDES_DEG_N,
        color=ORIG_COLOUR,
        linewidth=3,
        marker='o',
        markersize=12,
        markeredgewidth=0,
        markerfacecolor=ORIG_COLOUR,
        label='Original track'
    )
    axes_object.plot(
        whole_track_longitudes_deg_e,
        whole_track_latitudes_deg_n,
        color=WHOLE_TRACK_COLOUR,
        linewidth=3,
        marker='o',
        markersize=12,
        markeredgewidth=0,
        markerfacecolor=WHOLE_TRACK_COLOUR,
        label='Perturbed track'
    )

    for k in range(len(ORIG_LATITUDES_DEG_N)):
        axes_object.text(
            ORIG_LONGITUDES_DEG_E[k],
            ORIG_LATITUDES_DEG_N[k],
            VALID_TIME_STRINGS[k],
            color=ORIG_COLOUR,
            fontsize=30,
            fontweight='bold',
            verticalalignment='bottom',
            horizontalalignment='left'
        )
        axes_object.text(
            whole_track_longitudes_deg_e[k],
            whole_track_latitudes_deg_n[k],
            VALID_TIME_STRINGS[k],
            color=WHOLE_TRACK_COLOUR,
            fontsize=30,
            fontweight='bold',
            verticalalignment='bottom',
            horizontalalignment='left'
        )

    axes_object.set_title('(a) Whole-track error')
    axes_object.set_xlabel(r'Longitude ($^{\circ}$E)')
    axes_object.set_ylabel(r'Latitude ($^{\circ}$N)')
    axes_object.legend(fontsize=20)

    left_panel_file_name = '{0:s}/whole_track_only.jpg'.format(OUTPUT_DIR_NAME)
    print('Saving figure to file: "{0:s}"...'.format(left_panel_file_name))
    figure_object.savefig(
        left_panel_file_name,
        dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.plot(
        ORIG_LONGITUDES_DEG_E,
        ORIG_LATITUDES_DEG_N,
        color=ORIG_COLOUR,
        linewidth=3,
        marker='o',
        markersize=12,
        markeredgewidth=0,
        markerfacecolor=ORIG_COLOUR,
        label='Original track'
    )
    axes_object.plot(
        whole_track_longitudes_deg_e,
        whole_track_latitudes_deg_n,
        color=WHOLE_TRACK_COLOUR,
        linewidth=3,
        marker='o',
        markersize=12,
        markeredgewidth=0,
        markerfacecolor=WHOLE_TRACK_COLOUR,
        label='Whole-track error'
    )
    axes_object.plot(
        track_shape_longitudes_deg_e,
        track_shape_latitudes_deg_n,
        color=TRACK_SHAPE_COLOUR,
        linewidth=3,
        marker='o',
        markersize=12,
        markeredgewidth=0,
        markerfacecolor=TRACK_SHAPE_COLOUR,
        label='Track-shape error'
    )

    for k in range(len(ORIG_LATITUDES_DEG_N)):
        axes_object.text(
            ORIG_LONGITUDES_DEG_E[k],
            ORIG_LATITUDES_DEG_N[k],
            VALID_TIME_STRINGS[k],
            color=ORIG_COLOUR,
            fontsize=30,
            fontweight='bold',
            verticalalignment='bottom',
            horizontalalignment='left'
        )
        axes_object.text(
            track_shape_longitudes_deg_e[k],
            track_shape_latitudes_deg_n[k],
            VALID_TIME_STRINGS[k],
            color=TRACK_SHAPE_COLOUR,
            fontsize=30,
            fontweight='bold',
            verticalalignment='bottom',
            horizontalalignment='left'
        )

    axes_object.set_title('(b) Both errors')
    axes_object.set_xlabel(r'Longitude ($^{\circ}$E)')
    axes_object.set_ylabel(r'Latitude ($^{\circ}$N)')
    axes_object.legend(fontsize=20)

    right_panel_file_name = '{0:s}/both_errors.jpg'.format(OUTPUT_DIR_NAME)
    print('Saving figure to file: "{0:s}"...'.format(right_panel_file_name))
    figure_object.savefig(
        right_panel_file_name,
        dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    concat_file_name = '{0:s}/track_translation_schematic.jpg'.format(
        OUTPUT_DIR_NAME
    )

    print('Concatenating panels to: "{0:s}"...'.format(concat_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=[left_panel_file_name, right_panel_file_name],
        output_file_name=concat_file_name,
        num_panel_rows=1,
        num_panel_columns=2
    )


if __name__ == '__main__':
    _run()
