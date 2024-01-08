#!/bin/sh

model_dir_name=$1

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_standalone/ml4tccf"

python3 -u "${CODE_DIR_NAME}/evaluate_neural_net.py" \
--input_prediction_file_pattern="${model_dir_name}/isotonic_regression/validation/predictions*.nc" \
--num_bootstrap_reps=1 \
--num_xy_offset_bins=100 \
--xy_offset_limits_metres -100000 100000 \
--num_offset_distance_bins=50 \
--offset_distance_limits_metres 0 100000 \
--num_offset_direction_bins=36 \
--output_file_name="${model_dir_name}/isotonic_regression/validation/evaluation_no_bootstrap.nc"
