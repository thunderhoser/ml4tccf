"""Custom loss functions."""

from tensorflow.keras import backend as K


def mean_squared_distance_kilometres():
    """Creates mean-squared-distance loss function.

    :return: loss: Loss function (defined below).
    """

    def loss(target_tensor, prediction_tensor):
        """Computes mean squared distance btw predicted and actual TC centers.

        E = number of examples
        S = ensemble size

        :param target_tensor: E-by-3 Keras tensor.  target_tensor[:, 0] contains
            row positions of TC centers; target_tensor[:, 1] contains column
            positions of TC centers; and target_tensor[:, 2] contains grid
            spacing in km, used to convert row-column distances to actual
            distances.
        :param prediction_tensor: E-by-2-by-S Keras tensor.
            prediction_tensor[:, 0, :] contains predicted row positions of TC
            centers, and prediction_tensor[:, 1, :] contains predicted column
            positions of TC centers.
        :return: mean_squared_distance_km: Mean squared distance.
        """

        mean_prediction_tensor = K.mean(prediction_tensor, axis=-1)
        row_distances_km = (
            target_tensor[:, 2] *
            (mean_prediction_tensor[:, 0] - target_tensor[:, 0])
        )
        column_distances_km = (
            target_tensor[:, 2] *
            (mean_prediction_tensor[:, 1] - target_tensor[:, 1])
        )

        return K.mean(
            row_distances_km ** 2 + column_distances_km ** 2
        )

    return loss


def mean_distance_kilometres():
    """Creates mean-distance loss function.

    :return: loss: Loss function (defined below).
    """

    def loss(target_tensor, prediction_tensor):
        """Computes mean distance between predicted and actual TC centers.


        :param target_tensor: See doc for `mean_squared_distance_km`.
        :param prediction_tensor: Same.
        :return: mean_distance_km: Mean distance.
        """

        mean_prediction_tensor = K.mean(prediction_tensor, axis=-1)
        row_distances_km = (
            target_tensor[:, 2] *
            (mean_prediction_tensor[:, 0] - target_tensor[:, 0])
        )
        column_distances_km = (
            target_tensor[:, 2] *
            (mean_prediction_tensor[:, 1] - target_tensor[:, 1])
        )

        return K.mean(K.sqrt(
            row_distances_km ** 2 + column_distances_km ** 2
        ))

    return loss


def crps_kilometres():
    """Creates CRPS (continuous ranked probability score) loss function.

    :return: loss: Loss function (defined below).
    """

    def loss(target_tensor, prediction_tensor):
        """Computes CRPS between predicted and actual TC centers.

        :param target_tensor: See doc for `mean_squared_distance_km`.
        :param prediction_tensor: Same.
        :return: crps_km: CRPS.
        """

        mean_prediction_tensor = K.mean(prediction_tensor, axis=-1)
        row_distances_km = (
            target_tensor[:, 2] *
            (mean_prediction_tensor[:, 0] - target_tensor[:, 0])
        )
        column_distances_km = (
            target_tensor[:, 2] *
            (mean_prediction_tensor[:, 1] - target_tensor[:, 1])
        )
        mean_error_tensor_km = K.sqrt(
            row_distances_km ** 2 + column_distances_km ** 2
        )

        prediction_diff_tensor = K.abs(
            K.expand_dims(prediction_tensor, axis=-1) -
            K.expand_dims(prediction_tensor, axis=-2)
        )

        all_rowcol_diff_tensor = K.sqrt(
            prediction_diff_tensor[:, 0, ...] ** 2
            + prediction_diff_tensor[:, 1, ...] ** 2
        )
        all_diff_tensor_km = (
            K.expand_dims(K.expand_dims(target_tensor[:, 2], axis=-1), axis=-1)
            * all_rowcol_diff_tensor
        )
        mean_prediction_diff_tensor_km = K.mean(
            all_diff_tensor_km, axis=(-2, -1)
        )

        return K.mean(
            mean_error_tensor_km - 0.5 * mean_prediction_diff_tensor_km
        )

    return loss