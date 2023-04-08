"""Tests generator with CIRA IR data."""

import os
import sys
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import neural_net


def _run():
    """Tests generator with CIRA IR data.

    This is effectively the main method.
    """

    option_dict = {
        neural_net.SATELLITE_DIRECTORY_KEY:
            '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tc_project/'
            'learning_examples/rotated_with_storm_motion/imputed/normalized',
        neural_net.YEARS_KEY: numpy.array([2004], dtype=int),
        neural_net.LAG_TIMES_KEY: numpy.array([0, 30, 60], dtype=int),
        neural_net.BATCH_SIZE_KEY: 8,
        neural_net.MAX_EXAMPLES_PER_CYCLONE_KEY: 2,
        neural_net.NUM_GRID_ROWS_KEY: 280,
        neural_net.NUM_GRID_COLUMNS_KEY: 440,
        neural_net.DATA_AUG_NUM_TRANS_KEY: 8,
        neural_net.DATA_AUG_MEAN_TRANS_KEY: 11.25,
        neural_net.DATA_AUG_STDEV_TRANS_KEY: 6.125,
        neural_net.HIGH_RES_WAVELENGTHS_KEY: numpy.array([]),
        neural_net.LOW_RES_WAVELENGTHS_KEY: numpy.array([11.2]),
        neural_net.LAG_TIME_TOLERANCE_KEY: 0,
        neural_net.MAX_MISSING_LAG_TIMES_KEY: 0,
        neural_net.MAX_INTERP_GAP_KEY: 0,
        neural_net.SENTINEL_VALUE_KEY: -10.,
        neural_net.SEMANTIC_SEG_FLAG_KEY: False
    }

    generator_handle = neural_net.data_generator_cira_ir(option_dict)
    predictor_matrices, target_matrix = next(generator_handle)

    print(predictor_matrices[0].shape)
    print(numpy.mean(predictor_matrices[0]))
    print(numpy.min(predictor_matrices[0]))
    print(numpy.max(predictor_matrices[0]))
    print('\n\n')
    print(target_matrix[:, 0])
    print('\n\n')
    print(target_matrix[:, 1])
    print('\n\n')
    print(target_matrix[:, 2])
    print('\n\n')
    print(target_matrix[:, 3])
    print('\n\n')


if __name__ == '__main__':
    _run()
