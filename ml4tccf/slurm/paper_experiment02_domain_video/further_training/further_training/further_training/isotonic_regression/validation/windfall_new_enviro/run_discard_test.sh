#!/bin/sh

model_dir_name=$1

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_standalone/ml4tccf"

python3 -u "${CODE_DIR_NAME}/run_discard_test.py" \
--input_prediction_file_pattern="${model_dir_name}/isotonic_regression/validation/predictions*.nc" \
--discard_fractions 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 \
--output_file_name="${model_dir_name}/isotonic_regression/validation/discard_test.nc"
