#!/bin/sh

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_standalone/ml4tccf"
TOP_MODEL_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/paper_experiment02_ensemble/tropical_and_nontropical"

python3 -u "${CODE_DIR_NAME}/evaluate_neural_net.py" \
--input_prediction_file_pattern="${TOP_MODEL_DIR_NAME}/dumb_ensemble/isotonic_regression/testing_by_storm_type/storm-type=tropical/predictions*.nc" \
--num_bootstrap_reps=100 \
--num_xy_offset_bins=100 \
--xy_offset_limits_metres -100000 100000 \
--num_offset_distance_bins=50 \
--offset_distance_limits_metres 0 100000 \
--num_offset_direction_bins=36 \
--output_file_name="${TOP_MODEL_DIR_NAME}/dumb_ensemble/isotonic_regression/testing_by_storm_type/storm-type=tropical/evaluation.nc"
