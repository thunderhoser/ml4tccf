#!/bin/sh

model_dir_name=$1

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_standalone/ml4tccf"

python3 -u "${CODE_DIR_NAME}/make_pit_histograms.py" \
--input_prediction_file_pattern="${model_dir_name}/isotonic_regression/validation/predictions*.nc" \
--num_bins=51 \
--output_file_name="${model_dir_name}/isotonic_regression/validation/pit_histograms.nc"
