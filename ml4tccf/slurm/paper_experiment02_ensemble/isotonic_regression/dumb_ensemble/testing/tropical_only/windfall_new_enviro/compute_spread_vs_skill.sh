#!/bin/sh

model_dir_name=$1

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_standalone/ml4tccf"

python3 -u "${CODE_DIR_NAME}/compute_spread_vs_skill.py" \
--input_prediction_file_pattern="${model_dir_name}/dumb_ensemble/isotonic_regression/testing_by_storm_type/storm-type=tropical/predictions*.nc" \
--num_xy_offset_bins=30 \
--xy_offset_limits_percentile 0.5 99.5 \
--num_offset_distance_bins=30 \
--offset_distance_limits_percentile 0.5 99.5 \
--num_offset_direction_bins=30 \
--offset_direction_limits_percentile 0.5 99.5 \
--output_file_name="${model_dir_name}/dumb_ensemble/isotonic_regression/testing_by_storm_type/storm-type=tropical/spread_vs_skill.nc"
