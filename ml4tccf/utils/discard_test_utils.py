"""Helper methods for running discard test."""

import numpy
import xarray
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4tccf.io import prediction_io
from ml4tccf.utils import scalar_evaluation
from ml4tccf.utils import scalar_prediction_utils as prediction_utils
from ml4tccf.utils import spread_skill_utils as ss_utils
from ml4tccf.outside_code import angular_utils

KM_TO_METRES = 1000.

X_OFFSET_NAME = 'x_offset_metres'
Y_OFFSET_NAME = 'y_offset_metres'
OFFSET_DIRECTION_NAME = 'offset_direction_deg'
OFFSET_DISTANCE_NAME = 'offset_distance_metres'
TARGET_FIELD_NAMES = [
    Y_OFFSET_NAME, X_OFFSET_NAME, OFFSET_DIRECTION_NAME, OFFSET_DISTANCE_NAME
]

TARGET_FIELD_DIM = 'target_field'
DISCARD_FRACTION_DIM = 'discard_fraction'

POST_DISCARD_MAE_KEY = 'post_discard_mae'
MONO_FRACTION_KEY = 'monotonicity_fraction'
MEAN_MAE_IMPROVEMENT_KEY = 'mean_mae_improvement'
MEAN_MEAN_PREDICTION_KEY = 'mean_mean_prediction'
MEAN_TARGET_KEY = 'mean_target_value'

MODEL_FILE_KEY = 'model_file_name'
PREDICTION_FILES_KEY = 'prediction_file_names'


def run_discard_test(prediction_file_names, discard_fractions):
    """Runs the discard test.

    :param prediction_file_names: 1-D list of paths to prediction files (will be
        read by `prediction_io.read_file`).
    :param discard_fractions: 1-D numpy array of discard fractions.
    :return: result_table_xarray: xarray table with results (variable and
        dimension names should make the table self-explanatory).
    """

    # Check input args.
    error_checking.assert_is_numpy_array(discard_fractions, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(discard_fractions, 0.)
    error_checking.assert_is_less_than_numpy_array(discard_fractions, 1.)

    discard_fractions = numpy.concatenate((
        numpy.array([0.]),
        numpy.sort(discard_fractions)
    ))

    num_discard_fractions = len(discard_fractions)
    assert num_discard_fractions >= 2

    # Read predictions.
    num_files = len(prediction_file_names)
    prediction_tables_xarray = [None] * num_files

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(prediction_file_names[i]))
        prediction_tables_xarray[i] = prediction_io.read_file(
            prediction_file_names[i]
        )

    prediction_table_xarray = prediction_utils.concat_over_examples(
        prediction_tables_xarray
    )
    pt = prediction_table_xarray

    grid_spacings_km = pt[prediction_utils.GRID_SPACING_KEY].values

    prediction_matrix = numpy.stack((
        pt[prediction_utils.PREDICTED_ROW_OFFSET_KEY].values,
        pt[prediction_utils.PREDICTED_COLUMN_OFFSET_KEY].values
    ), axis=1)
    prediction_matrix *= numpy.expand_dims(grid_spacings_km, axis=(1, 2))

    target_matrix = numpy.transpose(numpy.vstack((
        grid_spacings_km * pt[prediction_utils.ACTUAL_ROW_OFFSET_KEY].values,
        grid_spacings_km * pt[prediction_utils.ACTUAL_COLUMN_OFFSET_KEY].values
    )))

    prediction_matrix *= KM_TO_METRES
    target_matrix *= KM_TO_METRES

    # Add vector directions.
    prediction_matrix = numpy.concatenate((
        prediction_matrix,
        scalar_evaluation.get_offset_angles(
            x_offsets=prediction_matrix[:, [1], :],
            y_offsets=prediction_matrix[:, [0], :]
        )
    ), axis=1)

    target_matrix = numpy.hstack((
        target_matrix,
        scalar_evaluation.get_offset_angles(
            x_offsets=target_matrix[:, [1]],
            y_offsets=target_matrix[:, [0]]
        )
    ))

    # Create result table.
    num_targets = len(TARGET_FIELD_NAMES)

    main_data_dict = {
        POST_DISCARD_MAE_KEY: (
            (TARGET_FIELD_DIM, DISCARD_FRACTION_DIM),
            numpy.full((num_targets, num_discard_fractions), numpy.nan)
        ),
        MEAN_MEAN_PREDICTION_KEY: (
            (TARGET_FIELD_DIM, DISCARD_FRACTION_DIM),
            numpy.full((num_targets, num_discard_fractions), numpy.nan)
        ),
        MEAN_TARGET_KEY: (
            (TARGET_FIELD_DIM, DISCARD_FRACTION_DIM),
            numpy.full((num_targets, num_discard_fractions), numpy.nan)
        ),
        MONO_FRACTION_KEY: (
            (TARGET_FIELD_DIM,),
            numpy.full(num_targets, numpy.nan)
        ),
        MEAN_MAE_IMPROVEMENT_KEY: (
            (TARGET_FIELD_DIM,),
            numpy.full(num_targets, numpy.nan)
        )
    }

    metadata_dict = {
        TARGET_FIELD_DIM: TARGET_FIELD_NAMES,
        DISCARD_FRACTION_DIM: discard_fractions
    }

    result_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )
    result_table_xarray.attrs[MODEL_FILE_KEY] = (
        prediction_table_xarray.attrs[prediction_utils.MODEL_FILE_KEY]
    )
    result_table_xarray.attrs[PREDICTION_FILES_KEY] = ' '.join([
        '{0:s}'.format(f) for f in prediction_file_names
    ])

    # Do actual stuff.
    xy_indices = numpy.array([
        TARGET_FIELD_NAMES.index(X_OFFSET_NAME),
        TARGET_FIELD_NAMES.index(Y_OFFSET_NAME)
    ], dtype=int)

    euclidean_uncertainty_values = ss_utils._get_predictive_stdistdevs(
        prediction_matrix[:, xy_indices, :]
    )
    mean_xy_prediction_matrix = numpy.mean(
        prediction_matrix[:, xy_indices, :], axis=-1
    )
    xy_target_matrix = target_matrix[:, xy_indices]
    euclidean_squared_errors = (
        (mean_xy_prediction_matrix[:, 0] - xy_target_matrix[:, 0]) ** 2 +
        (mean_xy_prediction_matrix[:, 1] - xy_target_matrix[:, 1]) ** 2
    )

    num_examples = len(euclidean_uncertainty_values)
    use_example_flags = numpy.full(num_examples, 1, dtype=bool)
    t = result_table_xarray

    for k in range(num_discard_fractions):
        this_percentile_level = 100 * (1 - discard_fractions[k])
        this_inverted_mask = (
            euclidean_uncertainty_values >
            numpy.percentile(
                euclidean_uncertainty_values, this_percentile_level
            )
        )
        use_example_flags[this_inverted_mask] = False
        use_example_indices = numpy.where(use_example_flags)[0]

        for j in range(num_targets):
            if TARGET_FIELD_NAMES[j] == OFFSET_DIRECTION_NAME:
                these_ensemble_mean_preds = numpy.array([
                    angular_utils.mean(
                        prediction_matrix[i, j, :][
                            numpy.invert(numpy.isnan(prediction_matrix[i, j, :]))
                        ],
                        deg=True
                    )
                    for i in use_example_indices
                ])

                these_target_values = target_matrix[use_example_indices, j]

                t[MEAN_MEAN_PREDICTION_KEY].values[j, k] = angular_utils.mean(
                    these_ensemble_mean_preds[
                        numpy.invert(numpy.isnan(these_ensemble_mean_preds))
                    ],
                    deg=True
                )

                t[MEAN_TARGET_KEY].values[j, k] = angular_utils.mean(
                    these_target_values[
                        numpy.invert(numpy.isnan(these_target_values))
                    ],
                    deg=True
                )

                t[POST_DISCARD_MAE_KEY].values[j, k] = numpy.nanmean(
                    numpy.absolute(
                        scalar_evaluation.get_angular_diffs(
                            these_ensemble_mean_preds, these_target_values
                        )
                    )
                )

                continue

            if TARGET_FIELD_NAMES[j] == OFFSET_DISTANCE_NAME:
                t[POST_DISCARD_MAE_KEY].values[j, k] = numpy.sqrt(numpy.mean(
                    euclidean_squared_errors[use_example_indices]
                ))

                t[MEAN_MEAN_PREDICTION_KEY].values[j, k] = numpy.sqrt(
                    numpy.mean(
                        mean_xy_prediction_matrix[use_example_indices, 0] ** 2 +
                        mean_xy_prediction_matrix[use_example_indices, 1] ** 2
                    )
                )

                t[MEAN_TARGET_KEY].values[j, k] = numpy.sqrt(numpy.mean(
                    xy_target_matrix[use_example_indices, 0] ** 2 +
                    xy_target_matrix[use_example_indices, 1] ** 2
                ))

                continue

            these_ensemble_mean_preds = numpy.mean(
                prediction_matrix[use_example_indices, j, :], axis=-1
            )
            these_target_values = target_matrix[use_example_indices, j]

            t[MEAN_MEAN_PREDICTION_KEY].values[j, k] = numpy.mean(
                these_ensemble_mean_preds
            )
            t[MEAN_TARGET_KEY].values[j, k] = numpy.mean(these_target_values)
            t[POST_DISCARD_MAE_KEY].values[j, k] = numpy.mean(numpy.absolute(
                these_ensemble_mean_preds - these_target_values
            ))

    for j in range(num_targets):
        t[MONO_FRACTION_KEY].values[j] = numpy.mean(
            numpy.diff(t[POST_DISCARD_MAE_KEY].values[j, :]) < 0
        )
        t[MEAN_MAE_IMPROVEMENT_KEY].values[j] = numpy.nanmean(
            -1 * numpy.diff(t[POST_DISCARD_MAE_KEY].values[j, :]) /
            numpy.diff(discard_fractions)
        )

    result_table_xarray = t
    return result_table_xarray


def write_results(result_table_xarray, netcdf_file_name):
    """Writes discard-test results to NetCDF file.

    :param result_table_xarray: xarray table in format returned by
        `run_discard_test`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    result_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def read_results(netcdf_file_name):
    """Reads discard-test results from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: result_table_xarray: xarray table.  Documentation in the
        xarray table should make values self-explanatory.
    """

    return xarray.open_dataset(netcdf_file_name)
