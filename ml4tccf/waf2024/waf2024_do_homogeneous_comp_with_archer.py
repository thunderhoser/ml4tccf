"""Performs homogeneous comparison between GeoCenter and ARCHER-2."""

import os
import numpy
import xarray
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from geopy.distance import geodesic
from gewittergefahr.plotting import imagemagick_utils
from ml4tccf.io import prediction_io
from ml4tccf.io import raw_archer_io
from ml4tccf.utils import scalar_prediction_utils as prediction_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_BOOTSTRAP_REPS = 1000
TIME_TOLERANCE_SEC = 1800

ARCHER_CYCLONE_ID_KEY = 'cyclone_id_string'
GEOCENTER_INTENSITY_KEY = 'geocenter_intensity_m_s01'

TESTING_CYCLONE_IDS_ONE_STRING = (
    '"2022EP01" "2022EP02" "2022EP03" "2022EP04" "2022EP05" "2022EP06" '
    '"2022EP07" "2022EP08" "2022EP09" "2022EP10" "2022EP11" "2022EP12" '
    '"2022EP13" "2022EP14" "2022EP15" "2022EP16" "2022EP17" "2022EP18" '
    '"2022EP19" "2021AL01" "2021AL02" "2021AL03" "2021AL04" "2021AL05" '
    '"2021AL06" "2021AL07" "2021AL08" "2021AL09" "2021AL10" "2021AL11" '
    '"2021AL12" "2021AL13" "2021AL14" "2021AL15" "2021AL16" "2021AL17" '
    '"2021AL18" "2021AL19" "2021WP01" "2021WP02" "2021WP03" "2021WP04" '
    '"2021WP05" "2021WP06" "2021WP07" "2021WP08" "2021WP09" "2021WP10" '
    '"2021WP11" "2021WP12" "2021WP13" "2021WP14" "2021WP15" "2021WP16" '
    '"2021WP17" "2021WP18" "2021WP19" "2021WP20" "2021WP21" "2021WP22" '
    '"2021WP23" "2021WP24" "2021WP25" "2021WP26" "2021WP27" "2021WP28" '
    '"2021WP29"'
)

EBTRK_FILE_NAME = (
    '/home/ralager/condo/swatwork/ralager/scratch1/RDARCH/rda-ghpcs/'
    'Ryan.Lagerquist/ml4tccf_project/'
    'fake_extended_best_track_robert_with_intensity.nc'
)

GEOCENTER_PREDICTION_DIR_NAME = (
    '/home/ralager/condo/swatwork/ralager/scratch1/RDARCH/rda-ghpcs/'
    'Ryan.Lagerquist/ml4tccf_models/'
    'paper_experiment05_domain_video_distribution/'
    'num-grid-rows=300_use-uniform-dist=0_num-lag-times=09/'
    'isotonic_regression_gaussian_dist/uncertainty_calibration/testing'
)

KT_TO_METRES_PER_SECOND = 1.852 / 3.6
MIN_INTENSITY_CUTOFFS_M_S01 = KT_TO_METRES_PER_SECOND * numpy.array(
    [0, 64, 83, 0]
)
MAX_INTENSITY_CUTOFFS_M_S01 = KT_TO_METRES_PER_SECOND * numpy.array(
    [64, 83, numpy.inf, numpy.inf]
)
INTENSITY_CUTOFF_STRINGS = [r'$<$ 64 kt', '[64, 83) kt', r'$\geq$ 83 kt', 'All']

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

OUTPUT_DIR_NAME = (
    '/home/ralager/tccf_paper_2024/homogeneous_comparison_with_archer2'
)


def _read_archer_predictions(testing_cyclone_id_strings):
    """Reads ARCHER-2 predictions.

    :param testing_cyclone_id_strings: 1-D list of cyclone IDs in testing set.
    :return: archer_table_xarray: xarray table in format returned by
        `raw_archer_io.read_file`, except with an additional key for cyclone ID.
    """

    num_cyclones = len(testing_cyclone_id_strings)
    archer_tables_xarray = [xarray.Dataset()] * num_cyclones
    archer_cyclone_id_strings = []

    for i in range(num_cyclones):
        url_string = raw_archer_io.find_url(testing_cyclone_id_strings[i])
        local_file_name = '/home/ralager/{0:s}_archer2.txt'.format(
            testing_cyclone_id_strings[i]
        )

        try:
            raw_archer_io.download_file(
                url_string=url_string, local_file_name=local_file_name
            )
        except:
            continue

        archer_tables_xarray[i] = raw_archer_io.read_file(
            ascii_file_name=local_file_name,
            cyclone_id_string=testing_cyclone_id_strings[i],
            ebtrk_file_name=EBTRK_FILE_NAME,
            time_tolerance_sec=TIME_TOLERANCE_SEC,
            match_to_synoptic_times_only=True
        )

        this_num_examples = len(
            archer_tables_xarray[i].coords[raw_archer_io.TIME_DIM].values
        )
        if this_num_examples == 0:
            continue

        archer_cyclone_id_strings += (
            [testing_cyclone_id_strings[i]] * this_num_examples
        )

    archer_tables_xarray = [atx for atx in archer_tables_xarray if atx]
    archer_table_xarray = xarray.concat(
        archer_tables_xarray, dim=raw_archer_io.TIME_DIM, data_vars='all',
        coords='minimal', compat='identical', join='exact'
    )

    return archer_table_xarray.assign({
        ARCHER_CYCLONE_ID_KEY: (
            (raw_archer_io.TIME_DIM,), archer_cyclone_id_strings
        )
    })


def _match_archer_to_geocenter(archer_table_xarray, geocenter_table_xarray):
    """Removes ARCHER-2 fixes that cannot be matched with a GeoCenter fix.

    :param archer_table_xarray: xarray table in format returned by
        `raw_archer_io.read_file`, except with an additional key for cyclone ID.
    :param geocenter_table_xarray: xarray table in format returned by
        `prediction_io.read_file`.
    :return: archer_table_xarray: Same as input, except maybe with fewer data
        samples.
    """

    geocenter_times_unix_sec = (
        geocenter_table_xarray[prediction_utils.TARGET_TIME_KEY].values
    )
    num_archer_examples = len(
        archer_table_xarray.coords[raw_archer_io.TIME_DIM].values
    )
    archer_to_geocenter_indices = numpy.full(num_archer_examples, -1, dtype=int)

    for i in range(num_archer_examples):
        time_diffs_sec = numpy.absolute(
            archer_table_xarray.coords[raw_archer_io.TIME_DIM].values[i] -
            geocenter_times_unix_sec
        )
        same_cyclone_flags = (
            geocenter_table_xarray[prediction_utils.CYCLONE_ID_KEY].values ==
            archer_table_xarray[ARCHER_CYCLONE_ID_KEY].values[i]
        )
        time_diffs_sec[same_cyclone_flags == False] = int(1e12)

        min_diff_index = numpy.argmin(time_diffs_sec)
        if time_diffs_sec[min_diff_index] > TIME_TOLERANCE_SEC:
            continue

        archer_to_geocenter_indices[i] = min_diff_index + 0

    good_archer_indices = numpy.where(archer_to_geocenter_indices > -1)[0]
    return archer_table_xarray.isel(
        {raw_archer_io.TIME_DIM: good_archer_indices}
    )


def _match_geocenter_to_archer(archer_table_xarray, geocenter_table_xarray):
    """Removes GeoCenter fixes that cannot be matched with an ARCHER-2 fix.

    :param archer_table_xarray: xarray table in format returned by
        `raw_archer_io.read_file`, except with an additional key for cyclone ID.
    :param geocenter_table_xarray: xarray table in format returned by
        `prediction_io.read_file`.
    :return: geocenter_table_xarray: Same as input, except [1] with extra key
        for intensity and [2] maybe with fewer samples.
    """

    archer_times_unix_sec = (
        archer_table_xarray.coords[raw_archer_io.TIME_DIM].values
    )
    num_geocenter_examples = len(
        geocenter_table_xarray.coords[prediction_utils.EXAMPLE_DIM_KEY].values
    )
    geocenter_to_archer_indices = numpy.full(
        num_geocenter_examples, -1, dtype=int
    )
    geocenter_intensities_m_s01 = numpy.full(num_geocenter_examples, numpy.nan)

    for i in range(num_geocenter_examples):
        time_diffs_sec = numpy.absolute(
            geocenter_table_xarray[prediction_utils.TARGET_TIME_KEY].values[i] -
            archer_times_unix_sec
        )
        same_cyclone_flags = (
            geocenter_table_xarray[prediction_utils.CYCLONE_ID_KEY].values[i] ==
            archer_table_xarray[ARCHER_CYCLONE_ID_KEY].values
        )
        time_diffs_sec[same_cyclone_flags == False] = int(1e12)

        min_diff_index = numpy.argmin(time_diffs_sec)
        if time_diffs_sec[min_diff_index] > TIME_TOLERANCE_SEC:
            continue

        geocenter_to_archer_indices[i] = min_diff_index + 0
        geocenter_intensities_m_s01[i] = archer_table_xarray[
            raw_archer_io.EBTRK_INTENSITY_KEY
        ].values[min_diff_index]

    geocenter_table_xarray = geocenter_table_xarray.assign({
        GEOCENTER_INTENSITY_KEY: (
            (prediction_utils.EXAMPLE_DIM_KEY,),
            geocenter_intensities_m_s01
        )
    })

    good_geocenter_indices = numpy.where(geocenter_to_archer_indices > -1)[0]
    return geocenter_table_xarray.isel(
        {prediction_utils.EXAMPLE_DIM_KEY: good_geocenter_indices}
    )


def _compute_archer_errors(archer_table_xarray):
    """Computes error statistics for ARCHER-2.

    B = number of bootstrap replicates

    :param archer_table_xarray: xarray table in format returned by
        `raw_archer_io.read_file`, except with an additional key for cyclone ID.
    :return: archer_table_xarray: Same as input, except with errors > 200 km
        removed.
    :return: mean_errors_km: length-B numpy array of mean errors.
    :return: median_errors_km: length-B numpy array of median errors.
    """

    sample_errors_km = [
        geodesic((y1, x1), (y2, x2)).kilometers
        for y1, x1, y2, x2 in zip(
            archer_table_xarray[raw_archer_io.LATITUDE_KEY],
            archer_table_xarray[raw_archer_io.LONGITUDE_KEY],
            archer_table_xarray[raw_archer_io.EBTRK_LATITUDE_KEY],
            archer_table_xarray[raw_archer_io.EBTRK_LONGITUDE_KEY]
        )
    ]
    sample_errors_km = numpy.array(sample_errors_km, dtype=float)

    good_indices = numpy.where(sample_errors_km < 200)[0]
    sample_errors_km = sample_errors_km[good_indices]
    archer_table_xarray = archer_table_xarray.isel(
        {raw_archer_io.TIME_DIM: good_indices}
    )

    mean_errors_km = numpy.full(NUM_BOOTSTRAP_REPS, numpy.nan)
    median_errors_km = numpy.full(NUM_BOOTSTRAP_REPS, numpy.nan)
    num_examples = len(sample_errors_km)
    all_example_indices = numpy.linspace(
        0, num_examples - 1, num=num_examples, dtype=int
    )

    for k in range(NUM_BOOTSTRAP_REPS):
        these_indices = numpy.random.choice(
            all_example_indices, size=num_examples, replace=True
        )
        mean_errors_km[k] = numpy.mean(sample_errors_km[these_indices])
        median_errors_km[k] = numpy.median(sample_errors_km[these_indices])

    print((
        '95% confidence interval for mean ARCHER-2 error = '
        '[{0:.1f}, {1:.1f}] km\n'
        'For median ARCHER-2 error = [{2:.1f}, {3:.1f}] km'
    ).format(
        numpy.percentile(mean_errors_km, 2.5),
        numpy.percentile(mean_errors_km, 97.5),
        numpy.percentile(median_errors_km, 2.5),
        numpy.percentile(median_errors_km, 97.5)
    ))

    return archer_table_xarray, mean_errors_km, median_errors_km


def _compute_geocenter_errors(geocenter_table_xarray):
    """Computes error statistics for GeoCenter.

    B = number of bootstrap replicates

    :param geocenter_table_xarray: xarray table in format returned by
        `prediction_io.read_file`.
    :return: mean_errors_km: length-B numpy array of mean errors.
    :return: median_errors_km: length-B numpy array of median errors.
    """

    gtx = geocenter_table_xarray
    grid_spacings_km = gtx[prediction_utils.GRID_SPACING_KEY].values

    prediction_matrix = numpy.transpose(numpy.vstack((
        grid_spacings_km *
        gtx[prediction_utils.PREDICTED_ROW_OFFSET_KEY].values[:, 0],
        grid_spacings_km *
        gtx[prediction_utils.PREDICTED_COLUMN_OFFSET_KEY].values[:, 0]
    )))

    target_matrix = numpy.transpose(numpy.vstack((
        grid_spacings_km * gtx[prediction_utils.ACTUAL_ROW_OFFSET_KEY].values,
        grid_spacings_km * gtx[prediction_utils.ACTUAL_COLUMN_OFFSET_KEY].values
    )))

    sample_errors_km = numpy.sqrt(
        (target_matrix[:, 0] - prediction_matrix[:, 0]) ** 2
        + (target_matrix[:, 1] - prediction_matrix[:, 1]) ** 2
    )

    mean_errors_km = numpy.full(NUM_BOOTSTRAP_REPS, numpy.nan)
    median_errors_km = numpy.full(NUM_BOOTSTRAP_REPS, numpy.nan)
    num_examples = len(sample_errors_km)
    all_example_indices = numpy.linspace(
        0, num_examples - 1, num=num_examples, dtype=int
    )

    for k in range(NUM_BOOTSTRAP_REPS):
        these_indices = numpy.random.choice(
            all_example_indices, size=num_examples, replace=True
        )
        mean_errors_km[k] = numpy.mean(sample_errors_km[these_indices])
        median_errors_km[k] = numpy.median(sample_errors_km[these_indices])

    print((
        '95% confidence interval for mean GeoCenter error = '
        '[{0:.1f}, {1:.1f}] km\n'
        'For median GeoCenter error = [{2:.1f}, {3:.1f}] km'
    ).format(
        numpy.percentile(mean_errors_km, 2.5),
        numpy.percentile(mean_errors_km, 97.5),
        numpy.percentile(median_errors_km, 2.5),
        numpy.percentile(median_errors_km, 97.5)
    ))

    return mean_errors_km, median_errors_km


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

    bar_width = 0.4

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
    """Performs homogeneous comparison between GeoCenter and ARCHER-2.

    This is effectively the main method.
    """

    # Read ARCHER-2 predictions.
    testing_cyclone_id_strings = TESTING_CYCLONE_IDS_ONE_STRING.split(' ')
    testing_cyclone_id_strings = [
        c.replace('"', '') for c in testing_cyclone_id_strings
    ]

    archer_table_xarray = _read_archer_predictions(testing_cyclone_id_strings)
    print(SEPARATOR_STRING)

    # Subset ARCHER-2 data to microwave-based fixes.  For the entire testing
    # set, there is only one scatterometer-based fix and no IR-based fixes.
    archer_table_xarray = raw_archer_io.subset_by_sensor_type(
        archer_table_xarray=archer_table_xarray, get_microwave=True
    )

    # Read GeoCenter predictions.
    geocenter_tables_xarray = []

    for this_cyclone_id_string in testing_cyclone_id_strings:
        this_prediction_file_name = prediction_io.find_file(
            directory_name=GEOCENTER_PREDICTION_DIR_NAME,
            cyclone_id_string=this_cyclone_id_string,
            raise_error_if_missing=False
        )
        if not os.path.isfile(this_prediction_file_name):
            continue

        print('Reading data from: "{0:s}"...'.format(this_prediction_file_name))
        geocenter_tables_xarray.append(
            prediction_io.read_file(this_prediction_file_name)
        )

    geocenter_table_xarray = xarray.concat(
        geocenter_tables_xarray, dim=prediction_utils.EXAMPLE_DIM_KEY,
        data_vars='all', coords='minimal', compat='identical', join='exact'
    )

    # Remove ARCHER-2 predictions that cannot be matched with a GeoCenter
    # prediction.
    archer_table_xarray = _match_archer_to_geocenter(
        archer_table_xarray=archer_table_xarray,
        geocenter_table_xarray=geocenter_table_xarray
    )

    # Remove GeoCenter predictions that cannot be matched with an ARCHER-2
    # prediction.
    geocenter_table_xarray = _match_geocenter_to_archer(
        archer_table_xarray=archer_table_xarray,
        geocenter_table_xarray=geocenter_table_xarray
    )

    archer_table_xarray = _compute_archer_errors(archer_table_xarray)[0]
    print(archer_table_xarray)
    print('\n\n')

    geocenter_table_xarray = prediction_utils.get_ensemble_mean(
        geocenter_table_xarray
    )
    print(geocenter_table_xarray)
    print('\n\n')

    _compute_geocenter_errors(geocenter_table_xarray)
    mean_error_dict = dict()
    median_error_dict = dict()

    for j in range(len(MIN_INTENSITY_CUTOFFS_M_S01)):
        mean_error_dict[INTENSITY_CUTOFF_STRINGS[j]] = dict()
        median_error_dict[INTENSITY_CUTOFF_STRINGS[j]] = dict()

        atx = archer_table_xarray
        these_indices = numpy.where(numpy.logical_and(
            atx[raw_archer_io.EBTRK_INTENSITY_KEY].values >=
            MIN_INTENSITY_CUTOFFS_M_S01[j],
            atx[raw_archer_io.EBTRK_INTENSITY_KEY].values <
            MAX_INTENSITY_CUTOFFS_M_S01[j]
        ))[0]

        this_table_xarray = archer_table_xarray.isel(
            {raw_archer_io.TIME_DIM: these_indices}
        )
        _, these_mean_errors_km, these_median_errors_km = (
            _compute_archer_errors(this_table_xarray)
        )

        this_tuple = (
            numpy.mean(these_mean_errors_km),
            numpy.mean(these_mean_errors_km) -
            numpy.percentile(these_mean_errors_km, 2.5),
            numpy.percentile(these_mean_errors_km, 97.5) -
            numpy.mean(these_mean_errors_km)
        )
        mean_error_dict[INTENSITY_CUTOFF_STRINGS[j]]['ARCHER-2: microwave'] = (
            this_tuple
        )

        this_tuple = (
            numpy.mean(these_median_errors_km),
            numpy.mean(these_median_errors_km) -
            numpy.percentile(these_median_errors_km, 2.5),
            numpy.percentile(these_median_errors_km, 97.5) -
            numpy.mean(these_median_errors_km)
        )
        median_error_dict[INTENSITY_CUTOFF_STRINGS[j]][
            'ARCHER-2: microwave'
        ] = this_tuple

        gtx = geocenter_table_xarray
        these_indices = numpy.where(numpy.logical_and(
            gtx[GEOCENTER_INTENSITY_KEY].values >= MIN_INTENSITY_CUTOFFS_M_S01[j],
            gtx[GEOCENTER_INTENSITY_KEY].values < MAX_INTENSITY_CUTOFFS_M_S01[j]
        ))[0]

        this_table_xarray = geocenter_table_xarray.isel(
            {prediction_utils.EXAMPLE_DIM_KEY: these_indices}
        )
        these_mean_errors_km, these_median_errors_km = (
            _compute_geocenter_errors(this_table_xarray)
        )

        this_tuple = (
            numpy.mean(these_mean_errors_km),
            numpy.mean(these_mean_errors_km) -
            numpy.percentile(these_mean_errors_km, 2.5),
            numpy.percentile(these_mean_errors_km, 97.5) -
            numpy.mean(these_mean_errors_km)
        )
        mean_error_dict[INTENSITY_CUTOFF_STRINGS[j]]['GeoCenter'] = (
            this_tuple
        )

        this_tuple = (
            numpy.mean(these_median_errors_km),
            numpy.mean(these_median_errors_km) -
            numpy.percentile(these_median_errors_km, 2.5),
            numpy.percentile(these_median_errors_km, 97.5) -
            numpy.mean(these_median_errors_km)
        )
        median_error_dict[INTENSITY_CUTOFF_STRINGS[j]]['GeoCenter'] = (
            this_tuple
        )

    figure_object = _plot_grouped_bar_chart(
        data_dict=mean_error_dict, title_string='(a) Mean distance error'
    )[0]
    panel_file_names = ['{0:s}/mean_error.jpg'.format(OUTPUT_DIR_NAME)]

    print('Saving figure to file: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1],
        dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object = _plot_grouped_bar_chart(
        data_dict=median_error_dict, title_string='(b) Median distance error'
    )[0]
    panel_file_names.append(
        '{0:s}/median_error.jpg'.format(OUTPUT_DIR_NAME)
    )

    print('Saving figure to file: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1],
        dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    concat_file_name = '{0:s}/homogeneous_comparison_with_archer2.jpg'.format(
        OUTPUT_DIR_NAME
    )

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
